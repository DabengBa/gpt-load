"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { parseFrontmatter } = require("./doc-runtime.js");

const MARKDOWN_LINK_RE = /(!?)\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)/g;
const FENCE_RE = /^\s{0,3}(`{3,}|~{3,})/;
const INLINE_CODE_RE = /(`+)[^\n]*?\1/g;
const ADR_FILENAME_RE = /^\d{4}-[a-z0-9]+(?:-[a-z0-9]+)*\.md$/;
const SEMANTIC_ID_RE = /^(?:page|feature|term)\.[A-Za-z0-9._-]+$/;
const PROTOCOL_RE = /^[A-Za-z][A-Za-z0-9+.-]*:/;

function toPosix(value) {
  return value.split(path.sep).join("/");
}

function relativeDocsPath(filePath, docsRoot) {
  return toPosix(path.relative(docsRoot, filePath));
}

function findMarkdownFiles(docsRoot, { skipDirs = ["db"] } = {}) {
  if (!fs.existsSync(docsRoot)) {
    return [];
  }

  const files = [];

  function visit(directory, relativeParts) {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (entry.name === ".git") {
        continue;
      }

      const nextParts = relativeParts.concat(entry.name);
      if (entry.isDirectory()) {
        if (skipDirs.includes(nextParts[0])) {
          continue;
        }
        visit(path.join(directory, entry.name), nextParts);
        continue;
      }

      if (entry.isFile() && entry.name.toLowerCase().endsWith(".md")) {
        files.push(path.join(directory, entry.name));
      }
    }
  }

  visit(docsRoot, []);
  return files.sort();
}

function collectSemanticIds(docsRoot) {
  const ids = new Map();

  for (const filePath of findMarkdownFiles(path.join(docsRoot, "db"), { skipDirs: ["dist"] })) {
    let metadata;
    try {
      metadata = parseFrontmatter(fs.readFileSync(filePath, "utf8")).metadata;
    } catch {
      continue;
    }

    if (typeof metadata.id === "string" && SEMANTIC_ID_RE.test(metadata.id.trim())) {
      ids.set(metadata.id.trim(), filePath);
    }
  }

  return ids;
}

function ownerFor(relativePath) {
  const parts = relativePath.split("/");

  if (relativePath === "AGENTS.md") {
    return "Schema";
  }
  if (["README.md", "PROJECT_HISTORY.md"].includes(relativePath) || parts[0] === "product-changelog") {
    return "navigation";
  }
  if (parts[0] === "communications" || relativePath.startsWith("tech/briefs/")) {
    return "raw evidence";
  }
  if (relativePath === "project-overview.md") {
    return "project overview";
  }
  if (parts[0] === "tech") {
    return "technical facts";
  }
  if (parts[0] === "workflows") {
    return "reusable workflow";
  }
  if (parts[0] === "specs" && parts.length >= 3) {
    return "activity package";
  }
  if (parts[0] === "adr") {
    return "ADR";
  }

  return null;
}

function lineNumber(content, offset) {
  return content.slice(0, offset).split(/\r?\n/).length;
}

function isExternalTarget(target) {
  return target.startsWith("#") || PROTOCOL_RE.test(target) || target.startsWith("//");
}

function maskCode(content) {
  let fence = null;

  return content
    .split("\n")
    .map((line) => {
      const match = line.match(FENCE_RE);
      if (fence) {
        if (match && match[1][0] === fence[0] && match[1].length >= fence.length) {
          fence = null;
        }
        return " ".repeat(line.length);
      }
      if (match) {
        fence = match[1];
        return " ".repeat(line.length);
      }
      return line.replace(INLINE_CODE_RE, (code) => " ".repeat(code.length));
    })
    .join("\n");
}

function isWithin(rootPath, candidatePath) {
  const root = fs.realpathSync(rootPath);
  const candidate = fs.existsSync(candidatePath)
    ? fs.realpathSync(candidatePath)
    : path.resolve(candidatePath);
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== "..");
}

function checkLocalLinks(filePath, content, repoRoot, docsRoot, semanticIds) {
  const problems = [];
  const maskedContent = maskCode(content);
  MARKDOWN_LINK_RE.lastIndex = 0;
  let match;

  while ((match = MARKDOWN_LINK_RE.exec(maskedContent)) !== null) {
    if (match[1] === "!") {
      continue;
    }

    const rawTarget = match[2].startsWith("<")
      ? match[2].slice(1, -1)
      : match[2];
    if (isExternalTarget(rawTarget)) {
      continue;
    }

    const target = rawTarget.replace(/[?#].*$/, "");
    if (!target) {
      continue;
    }
    if (SEMANTIC_ID_RE.test(target)) {
      if (!semanticIds.has(target)) {
        problems.push(
          `${relativeDocsPath(filePath, docsRoot)}:${lineNumber(content, match.index)}: ` +
            `broken semantic link "${rawTarget}"`,
        );
      }
      continue;
    }

    let decodedTarget;
    try {
      decodedTarget = decodeURIComponent(target);
    } catch {
      decodedTarget = target;
    }

    const targetPath = target.startsWith("/")
      ? path.resolve(repoRoot, `.${decodedTarget}`)
      : path.resolve(path.dirname(filePath), decodedTarget);
    if (!isWithin(repoRoot, targetPath)) {
      problems.push(
        `${relativeDocsPath(filePath, docsRoot)}:${lineNumber(content, match.index)}: ` +
          `local link escapes repository "${rawTarget}"`,
      );
      continue;
    }
    if (!fs.existsSync(targetPath)) {
      problems.push(
        `${relativeDocsPath(filePath, docsRoot)}:${lineNumber(content, match.index)}: ` +
          `broken local link "${rawTarget}"`,
      );
    }
  }

  return problems;
}

function checkRelationTargets(relativePath, metadata, docsRoot) {
  const relations = metadata.relations;
  if (!relations || typeof relations !== "object" || Array.isArray(relations)) return [];

  const problems = [];
  for (const relationName of ["requires", "related"]) {
    const targets = relations[relationName] || [];
    if (!Array.isArray(targets)) {
      problems.push(`${relativePath}: relations.${relationName} must be a list`);
      continue;
    }
    for (const target of targets) {
      if (typeof target !== "string" || !target.trim()) {
        problems.push(`${relativePath}: relations.${relationName} contains an invalid target`);
        continue;
      }
      const targetPath = path.resolve(docsRoot, target);
      if (!isWithin(docsRoot, targetPath) || !fs.existsSync(targetPath)) {
        problems.push(`${relativePath}: missing relation target "${target}"`);
      }
    }
  }
  return problems;
}

function checkRetrievalMetadata(relativePath, content, owner, docsRoot) {
  if (!["technical facts", "reusable workflow", "ADR"].includes(owner)) {
    return [];
  }

  const problems = [];
  let metadata;
  try {
    metadata = parseFrontmatter(content).metadata;
  } catch (error) {
    return [`${relativePath}: invalid retrieval frontmatter: ${error.message}`];
  }

  for (const field of ["description", "kind", "topic"]) {
    if (typeof metadata[field] !== "string" || !metadata[field].trim()) {
      problems.push(`${relativePath}: missing retrieval frontmatter field "${field}"`);
    }
  }

  const expectedKind = {
    "technical facts": "technical",
    "reusable workflow": "workflow",
    ADR: "adr",
  }[owner];
  if (metadata.kind && metadata.kind !== expectedKind) {
    problems.push(`${relativePath}: kind must be "${expectedKind}" for ${owner}`);
  }

  const title = (parseFrontmatter(content).body.match(/^#\s+(.+)$/m) || [null, ""])[1].trim();
  if (metadata.description && title && metadata.description.trim() === title) {
    problems.push(`${relativePath}: description must add routing information beyond the title`);
  }
  problems.push(...checkRelationTargets(relativePath, metadata, docsRoot));
  return problems;
}

function checkAdrPath(relativePath, owner) {
  if (owner !== "ADR") {
    return [];
  }

  const parts = relativePath.split("/");
  if (parts.length !== 2) {
    return [`ADR must be a direct Markdown file under .docs/adr: ${relativePath}`];
  }
  if (!ADR_FILENAME_RE.test(parts[1])) {
    return [`ADR filename must match NNNN-slug.md: ${relativePath}`];
  }
  return [];
}

function checkDuplicateAdrNumbers(relativePaths) {
  const numbers = new Map();
  const problems = [];

  for (const relativePath of relativePaths) {
    if (!relativePath.startsWith("adr/") || relativePath.split("/").length !== 2) {
      continue;
    }

    const match = relativePath.match(/^(\d{4})-/);
    if (!match) {
      continue;
    }

    const previous = numbers.get(match[1]);
    if (previous) {
      problems.push(
        `duplicate ADR number ${match[1]}: ${previous} and ${relativePath}`,
      );
    } else {
      numbers.set(match[1], relativePath);
    }
  }

  return problems;
}

function findDocumentationProblems(repoRoot) {
  const resolvedRepoRoot = path.resolve(repoRoot);
  const docsRoot = path.join(resolvedRepoRoot, ".docs");
  const files = findMarkdownFiles(docsRoot);
  const relativePaths = files.map((filePath) => relativeDocsPath(filePath, docsRoot));
  const semanticIds = collectSemanticIds(docsRoot);
  const problems = [];

  for (const [index, filePath] of files.entries()) {
    const relativePath = relativePaths[index];
    const owner = ownerFor(relativePath);
    if (!owner) {
      problems.push(`unowned Markdown under .docs: ${relativePath}`);
      continue;
    }

    problems.push(...checkAdrPath(relativePath, owner));
    const content = fs.readFileSync(filePath, "utf8");
    problems.push(...checkRetrievalMetadata(relativePath, content, owner, docsRoot));
    problems.push(...checkLocalLinks(filePath, content, resolvedRepoRoot, docsRoot, semanticIds));
  }

  const semanticFiles = findMarkdownFiles(path.join(docsRoot, "db"), { skipDirs: ["dist"] });
  for (const filePath of semanticFiles) {
    const content = fs.readFileSync(filePath, "utf8");
    problems.push(...checkLocalLinks(filePath, content, resolvedRepoRoot, docsRoot, semanticIds));
  }

  problems.push(...checkDuplicateAdrNumbers(relativePaths));
  return problems;
}

function main(argv = process.argv.slice(2)) {
  let repoRoot = process.cwd();
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--repo") {
      repoRoot = argv[index + 1] || "";
      index += 1;
    }
  }

  const problems = findDocumentationProblems(repoRoot);
  if (problems.length > 0) {
    console.error("Non-semantic documentation check failed:");
    for (const problem of problems) {
      console.error(`  - ${problem}`);
    }
    return 1;
  }

  console.log("Checked non-semantic documentation");
  return 0;
}

if (require.main === module) {
  process.exitCode = main();
}

module.exports = {
  findDocumentationProblems,
  ownerFor,
  checkRetrievalMetadata,
};
