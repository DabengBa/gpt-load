"use strict";

const fs = require("node:fs");
const path = require("node:path");

const SKIPPED_DIRS = new Set(["db/dist"]);
const SKIPPED_FILES = new Set(["AGENTS.md"]);

function toPosix(value) {
  return value.split(path.sep).join("/");
}

function stripQuotes(value) {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith("\"") && trimmed.endsWith("\"")) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1);
  }
  return trimmed;
}

function parseListValue(value) {
  const trimmed = value.trim();
  if (trimmed === "[]") return [];
  if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) return [stripQuotes(trimmed)];
  return trimmed
    .slice(1, -1)
    .split(",")
    .map(stripQuotes)
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseFrontmatter(raw) {
  if (!raw.startsWith("---\n") && !raw.startsWith("---\r\n")) {
    return { metadata: {}, body: raw };
  }

  const lines = raw.split(/\r?\n/);
  const end = lines.indexOf("---", 1);
  if (end === -1) {
    throw new Error("Unclosed YAML frontmatter");
  }

  const metadata = {};
  let section = null;
  for (const rawLine of lines.slice(1, end)) {
    if (!rawLine.trim() || rawLine.trim().startsWith("#")) continue;
    const indent = rawLine.match(/^\s*/)[0].length;
    const line = rawLine.trim();
    if (indent === 0) {
      const separator = line.indexOf(":");
      if (separator === -1) throw new Error(`Invalid frontmatter line: ${rawLine}`);
      const key = line.slice(0, separator).trim();
      const value = line.slice(separator + 1).trim();
      if (value) {
        if (value === ">" || value === "|" || value.startsWith(">") || value.startsWith("|")) {
          throw new Error("Block scalar frontmatter values are not supported");
        }
        metadata[key] = key === "code.paths" || value.startsWith("[")
          ? parseListValue(value)
          : stripQuotes(value);
        section = null;
      } else {
        metadata[key] = {};
        section = key;
      }
      continue;
    }

    if (!section) continue;
    const separator = line.indexOf(":");
    if (separator !== -1) {
      const key = line.slice(0, separator).trim();
      const value = line.slice(separator + 1).trim();
      metadata[section][key] = value ? parseListValue(value) : [];
      continue;
    }
    if (line.startsWith("- ")) {
      const currentKey = Object.keys(metadata[section]).at(-1);
      if (currentKey) metadata[section][currentKey].push(stripQuotes(line.slice(2)));
    }
  }

  return {
    metadata,
    body: lines.slice(end + 1).join("\n").replace(/^\n+|\n+$/g, ""),
  };
}

function walkMarkdownFiles(directory, relativeParts = []) {
  if (!fs.existsSync(directory)) return [];
  const files = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const nextParts = relativeParts.concat(entry.name);
    const relativePath = nextParts.join("/");
    if (entry.isDirectory()) {
      if (SKIPPED_DIRS.has(relativePath)) continue;
      files.push(...walkMarkdownFiles(path.join(directory, entry.name), nextParts));
    } else if (
      entry.isFile() &&
      entry.name.toLowerCase().endsWith(".md") &&
      !SKIPPED_FILES.has(relativePath)
    ) {
      files.push({ filePath: path.join(directory, entry.name), relativePath: toPosix(relativePath) });
    }
  }
  return files.sort((left, right) => left.relativePath.localeCompare(right.relativePath));
}

function inferOwner(relativePath) {
  if (relativePath.startsWith("db/pages/")) return "semantic-pages";
  if (relativePath.startsWith("db/features/")) return "semantic-features";
  if (relativePath.startsWith("db/terms/")) return "semantic-terms";
  if (relativePath.startsWith("tech/")) return "technical";
  if (relativePath.startsWith("workflows/")) return "workflow";
  if (relativePath.startsWith("adr/")) return "adr";
  return "document";
}

function inferKind(relativePath) {
  const owner = inferOwner(relativePath);
  const semanticKinds = {
    "semantic-pages": "page",
    "semantic-features": "feature",
    "semantic-terms": "term",
  };
  if (semanticKinds[owner]) return semanticKinds[owner];
  if (owner === "adr") return "adr";
  if (owner === "technical") return "technical";
  if (owner === "workflow") return "workflow";
  return "document";
}

function inferTopic(relativePath) {
  const parts = relativePath.split("/");
  if (parts[0] === "db" && parts[1]) return parts[1];
  if (parts[0] === "adr") return "adr";
  if (parts.length > 2) return parts[1];
  return parts[0];
}

function inferDescription(body, relativePath) {
  const paragraph = body
    .split(/\n\s*\n/)
    .map((part) => part.replace(/^#+\s+/gm, "").replace(/[`*_]/g, "").trim())
    .find(Boolean);
  return paragraph || relativePath;
}

function expandSemanticContent(doc, docsById, stack = []) {
  const id = doc.semanticId;
  if (!id) return doc.content;
  if (stack.includes(id)) throw new Error(`Semantic include cycle: ${stack.concat(id).join(" -> ")}`);
  return doc.content.replace(/!include\s+([\w.-]+)/g, (_, targetId) => {
    const target = docsById.get(targetId);
    if (!target) throw new Error(`Missing semantic include target "${targetId}" in ${doc.relativePath}`);
    return expandSemanticContent(target, docsById, stack.concat(id));
  });
}

function loadDocumentation(repoRoot) {
  const docsRoot = path.resolve(repoRoot, ".docs");
  const docs = walkMarkdownFiles(docsRoot).map(({ filePath, relativePath }) => {
    const raw = fs.readFileSync(filePath, "utf8");
    const { metadata, body } = parseFrontmatter(raw);
    const relations = metadata.relations || {};
    const code = metadata.code || {};
    return {
      filePath,
      relativePath,
      content: body,
      title: (body.match(/^#\s+(.+)$/m) || [null, relativePath])[1].trim(),
      description: metadata.description || inferDescription(body, relativePath),
      kind: metadata.kind || inferKind(relativePath),
      owner: inferOwner(relativePath),
      topic: metadata.topic || inferTopic(relativePath),
      requires: Array.isArray(relations.requires) ? relations.requires : [],
      related: Array.isArray(metadata.related)
        ? metadata.related
        : Array.isArray(relations.related)
          ? relations.related
          : [],
      codePaths: Array.isArray(code.paths) ? code.paths : [],
      semanticId: typeof metadata.id === "string" ? metadata.id : null,
    };
  });

  const semanticDocs = new Map(
    docs.filter((doc) => doc.semanticId).map((doc) => [doc.semanticId, doc]),
  );
  for (const doc of docs) {
    doc.content = expandSemanticContent(doc, semanticDocs);
  }
  return docs;
}

function documentSummary(doc) {
  return {
    relativePath: doc.relativePath,
    title: doc.title,
    description: doc.description,
    kind: doc.kind,
    owner: doc.owner,
    topic: doc.topic,
    requires: doc.requires,
    related: doc.related,
    codePaths: doc.codePaths,
  };
}

function buildTree(docs) {
  const topics = new Map();
  for (const doc of docs) {
    if (!topics.has(doc.topic)) topics.set(doc.topic, []);
    topics.get(doc.topic).push(doc);
  }
  return {
    total: docs.length,
    topics: [...topics.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([topic, topicDocs]) => ({
      topic,
      owners: [...new Set(topicDocs.map((doc) => doc.owner))].sort(),
      count: topicDocs.length,
      documents: topicDocs.sort((left, right) => left.relativePath.localeCompare(right.relativePath)).map(documentSummary),
    })),
  };
}

function buildIndex(docs, filters = {}) {
  return docs
    .filter((doc) => !filters.topic || doc.topic === filters.topic)
    .filter((doc) => !filters.kind || doc.kind === filters.kind)
    .sort((left, right) => left.relativePath.localeCompare(right.relativePath))
    .map(documentSummary);
}

function tokenize(value) {
  const normalized = value.toLocaleLowerCase();
  const words = normalized.match(/[a-z0-9][a-z0-9._/-]*/g) || [];
  const cjkRuns = normalized.match(/[\u3400-\u9fff]+/gu) || [];
  const cjk = cjkRuns.flatMap((run) => {
    if (run.length === 1) return [run];
    return [...Array(run.length - 1)].map((_, index) => run.slice(index, index + 2));
  });
  return [...new Set([...words, ...cjk])];
}

function scoreDocument(doc, query) {
  const haystacks = [
    [doc.title, 8],
    [doc.description, 6],
    [doc.relativePath, 5],
    [doc.content, 1],
  ];
  return tokenize(query).reduce((score, token) => {
    return score + haystacks.reduce((subtotal, [value, weight]) => subtotal + (value.toLocaleLowerCase().includes(token) ? weight : 0), 0);
  }, 0);
}

function snippet(doc, query) {
  const terms = tokenize(query).sort((left, right) => right.length - left.length);
  const lower = doc.content.toLocaleLowerCase();
  const positions = terms
    .map((term) => lower.indexOf(term))
    .filter((index) => index >= 0);
  if (positions.length === 0) return doc.description;
  const position = Math.min(...positions);
  const start = Math.max(0, lower.lastIndexOf("\n", position - 1) + 1);
  const end = doc.content.indexOf("\n", position + 150);
  return doc.content
    .slice(start, end === -1 ? Math.min(doc.content.length, position + 200) : end)
    .replace(/\s+/g, " ")
    .trim();
}

function searchDocumentation(docs, query, filters = {}) {
  return buildIndex(docs, filters)
    .map((summary) => {
      const doc = docs.find((candidate) => candidate.relativePath === summary.relativePath);
      return { ...summary, score: scoreDocument(doc, query), snippet: snippet(doc, query) };
    })
    .filter((result) => result.score > 0)
    .sort((left, right) => right.score - left.score || left.relativePath.localeCompare(right.relativePath));
}

function showDocumentation(docs, relativePaths) {
  const byPath = new Map(docs.map((doc) => [doc.relativePath, doc]));
  const missing = [];
  const selected = [];
  for (const relativePath of relativePaths) {
    const doc = byPath.get(relativePath);
    if (!doc) missing.push(relativePath);
    else selected.push({ ...documentSummary(doc), content: doc.content });
  }
  return { documents: selected, missing };
}

module.exports = {
  buildIndex,
  buildTree,
  loadDocumentation,
  parseFrontmatter,
  searchDocumentation,
  showDocumentation,
};
