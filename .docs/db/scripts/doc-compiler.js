"use strict";

const fs = require("node:fs");
const path = require("node:path");

const INCLUDE_RE = /!include\s+([\w.-]+)/g;
const LINK_RE = /\[[^\]]*?\]\(([\w.-]+)\)/g;
const DOC_TYPE_DIRS = {
  page: "pages",
  feature: "features",
  term: "terms",
};
const ID_RE = /^(page|feature|term)\.[A-Za-z0-9._-]+$/;
const ID_EXPLANATION_RE = /^##\s+(?:ID Explanation|ID 解释)\s*$/im;
const REQUIRED_SECTION_PATTERNS = {
  page: [
    /^##\s+(?:Page Purpose|页面目的|Purpose)\s*$/im,
    /^##\s+(?:Page Structure|页面结构|UI Layout)\s*$/im,
  ],
  feature: [
    /^##\s+(?:Purpose|目的|功能目的)\s*$/im,
    /^##\s+(?:User-Visible Contract|用户可见契约|用户可见行为)\s*$/im,
    /^##\s+(?:Boundaries|边界)\s*$/im,
  ],
  term: [
    /^##\s+(?:Business Definition|业务定义|Definition)\s*$/im,
  ],
};

function fail(message) {
  const error = new Error(message);
  error.isDocCompilerError = true;
  throw error;
}

function parseArgs(argv) {
  const mode = argv[0] || "build";
  if (mode !== "check" && mode !== "build") {
    fail(`Unsupported mode: ${mode}. Use "check" or "build".`);
  }

  let root = path.join(process.cwd(), ".docs", "db");
  for (let index = 1; index < argv.length; index += 1) {
    if (argv[index] === "--root") {
      root = path.resolve(argv[index + 1] || "");
      index += 1;
    }
  }

  return { mode, docsRoot: path.resolve(root) };
}

function walkMarkdownFiles(rootDir) {
  const entries = fs.readdirSync(rootDir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(rootDir, entry.name);

    if (entry.isDirectory()) {
      if (entry.name === "dist" || entry.name === ".git") {
        continue;
      }
      files.push(...walkMarkdownFiles(fullPath));
      continue;
    }

    if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
      files.push(fullPath);
    }
  }

  return files;
}

function splitFrontmatter(raw, filePath) {
  if (!raw.startsWith("---\n") && !raw.startsWith("---\r\n")) {
    fail(`Missing YAML frontmatter in ${filePath}`);
  }

  const lines = raw.split(/\r?\n/);
  let endIndex = -1;
  for (let index = 1; index < lines.length; index += 1) {
    if (lines[index] === "---") {
      endIndex = index;
      break;
    }
  }

  if (endIndex === -1) {
    fail(`Unclosed YAML frontmatter in ${filePath}`);
  }

  const frontmatter = lines.slice(1, endIndex);
  const body = lines.slice(endIndex + 1).join("\n").trim();
  return { frontmatter, body };
}

function parseScalar(rawValue) {
  const value = rawValue.trim();
  if (value === "[]") {
    return [];
  }
  if (value.startsWith("[") && value.endsWith("]")) {
    const inner = value.slice(1, -1).trim();
    if (!inner) {
      return [];
    }
    return inner
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => item.replace(/^['"]|['"]$/g, ""));
  }
  return value.replace(/^['"]|['"]$/g, "");
}

function parseFrontmatter(lines, filePath) {
  const data = {};

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) {
      continue;
    }

    const separator = line.indexOf(":");
    if (separator === -1) {
      fail(`Invalid frontmatter line in ${filePath}: ${rawLine}`);
    }

    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1);
    data[key] = parseScalar(value);
  }

  for (const requiredField of ["id", "type", "name"]) {
    if (!data[requiredField] || Array.isArray(data[requiredField])) {
      fail(`Missing required frontmatter field "${requiredField}" in ${filePath}`);
    }
  }

  if (!Object.prototype.hasOwnProperty.call(data, "related")) {
    data.related = [];
  } else if (!Array.isArray(data.related)) {
    data.related = [String(data.related)];
  }

  return data;
}

function validateDocumentContract(doc, docsRoot) {
  if (!Object.prototype.hasOwnProperty.call(DOC_TYPE_DIRS, doc.type)) {
    fail(`Unsupported semantic doc type "${doc.type}" in ${doc.filePath}`);
  }

  if (!ID_RE.test(doc.id) || !doc.id.startsWith(`${doc.type}.`)) {
    fail(`Invalid semantic id "${doc.id}" for type "${doc.type}" in ${doc.filePath}`);
  }

  const relativePath = path.relative(docsRoot, doc.filePath).split(path.sep);
  if (relativePath[0] !== DOC_TYPE_DIRS[doc.type]) {
    fail(
      `Semantic doc type "${doc.type}" must live under "${DOC_TYPE_DIRS[doc.type]}/": ${doc.filePath}`,
    );
  }

  if (!ID_EXPLANATION_RE.test(doc.content)) {
    fail(`Missing "ID Explanation" section in ${doc.filePath}`);
  }

  for (const sectionPattern of REQUIRED_SECTION_PATTERNS[doc.type]) {
    if (!sectionPattern.test(doc.content)) {
      fail(`Missing required ${doc.type} section in ${doc.filePath}`);
    }
  }
}

function extractMatches(regex, content) {
  const matches = [];
  let match;
  regex.lastIndex = 0;
  while ((match = regex.exec(content)) !== null) {
    matches.push(match[1]);
  }
  return matches;
}

function loadDocuments(docsRoot) {
  if (!fs.existsSync(docsRoot)) {
    fail(`Docs root does not exist: ${docsRoot}`);
  }

  const markdownFiles = walkMarkdownFiles(docsRoot);
  const docsById = new Map();

  for (const filePath of markdownFiles) {
    const raw = fs.readFileSync(filePath, "utf8");
    const { frontmatter, body } = splitFrontmatter(raw, filePath);
    const meta = parseFrontmatter(frontmatter, filePath);

    if (docsById.has(meta.id)) {
      fail(`Duplicate id "${meta.id}" found in ${filePath} and ${docsById.get(meta.id).filePath}`);
    }

    const doc = {
      ...meta,
      filePath: path.resolve(filePath),
      content: body,
      includes: extractMatches(INCLUDE_RE, body),
      links: extractMatches(LINK_RE, body),
    };

    validateDocumentContract(doc, docsRoot);
    docsById.set(doc.id, doc);
  }

  return docsById;
}

function validateReferences(docsById) {
  for (const doc of docsById.values()) {
    for (const targetId of doc.includes) {
      if (!docsById.has(targetId)) {
        fail(`Missing include target "${targetId}" referenced from ${doc.filePath}`);
      }
    }

    for (const targetId of doc.links) {
      if (!docsById.has(targetId)) {
        fail(`Missing semantic link target "${targetId}" referenced from ${doc.filePath}`);
      }
    }

    for (const targetId of doc.related) {
      if (!docsById.has(targetId)) {
        fail(`Missing related target "${targetId}" referenced from ${doc.filePath}`);
      }
    }
  }
}

function validateIncludeCycles(docsById) {
  const visiting = new Set();
  const visited = new Set();

  function visit(docId, stack) {
    if (visiting.has(docId)) {
      const cycleStart = stack.indexOf(docId);
      const cycle = stack.slice(cycleStart).concat(docId).join(" -> ");
      fail(`Include cycle detected: ${cycle}`);
    }

    if (visited.has(docId)) {
      return;
    }

    visiting.add(docId);
    stack.push(docId);

    const doc = docsById.get(docId);
    for (const targetId of doc.includes) {
      visit(targetId, stack);
    }

    stack.pop();
    visiting.delete(docId);
    visited.add(docId);
  }

  for (const docId of docsById.keys()) {
    visit(docId, []);
  }
}

function expandDoc(docId, docsById, stack = []) {
  if (stack.includes(docId)) {
    fail(`Include cycle detected during expansion: ${stack.concat(docId).join(" -> ")}`);
  }

  const doc = docsById.get(docId);
  return doc.content.replace(INCLUDE_RE, (_, targetId) => expandDoc(targetId, docsById, stack.concat(docId)));
}

function buildIdMap(docsById, docsRoot) {
  return Object.fromEntries(
    Array.from(docsById.values())
      .sort((left, right) => left.id.localeCompare(right.id))
      .map((doc) => [
        doc.id,
        path.relative(docsRoot, doc.filePath).split(path.sep).join("/"),
      ]),
  );
}

function buildBundle(docsById) {
  return Array.from(docsById.values())
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((doc) => {
      const entry = {
        id: doc.id,
        type: doc.type,
        name: doc.name,
        content: expandDoc(doc.id, docsById),
      };

      if (doc.route) {
        entry.route = doc.route;
      }

      return entry;
    });
}

function pushEdge(edges, seen, source, target, relation) {
  const key = `${source}|${target}|${relation}`;
  if (seen.has(key)) {
    return;
  }
  seen.add(key);
  edges.push({ source, target, relation });
}

function buildTopology(docsById) {
  const nodes = Array.from(docsById.values())
    .sort((left, right) => left.id.localeCompare(right.id))
    .map((doc) => ({
      id: doc.id,
      type: doc.type,
      name: doc.name,
    }));

  const edges = [];
  const seen = new Set();

  for (const doc of Array.from(docsById.values()).sort((left, right) => left.id.localeCompare(right.id))) {
    for (const targetId of doc.includes) {
      pushEdge(edges, seen, doc.id, targetId, "includes");
    }
    for (const targetId of doc.related) {
      pushEdge(edges, seen, doc.id, targetId, "related");
    }
    for (const targetId of doc.links) {
      pushEdge(edges, seen, doc.id, targetId, "references");
    }
  }

  return { nodes, edges };
}

function writeJson(filePath, data) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

function run(mode, docsRoot) {
  const docsById = loadDocuments(docsRoot);
  validateReferences(docsById);
  validateIncludeCycles(docsById);

  if (mode === "check") {
    console.log(`Validated ${docsById.size} semantic docs in ${docsRoot}`);
    return;
  }

  const distDir = path.join(docsRoot, "dist");
  writeJson(path.join(distDir, "id-map.json"), buildIdMap(docsById, docsRoot));
  writeJson(path.join(distDir, "bundle.json"), buildBundle(docsById));
  writeJson(path.join(distDir, "topology.json"), buildTopology(docsById));
  console.log(`Built semantic docs bundle for ${docsById.size} documents in ${docsRoot}`);
}

function main() {
  try {
    const { mode, docsRoot } = parseArgs(process.argv.slice(2));
    run(mode, docsRoot);
  } catch (error) {
    const prefix = error && error.isDocCompilerError ? "Doc compiler error" : "Unexpected error";
    const message = error && error.message ? error.message : String(error);
    console.error(`${prefix}: ${message}`);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = {
  buildIdMap,
  buildTopology,
  loadDocuments,
  run,
  validateDocumentContract,
  validateIncludeCycles,
  validateReferences,
};
