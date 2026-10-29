const { afterEach, describe, test } = require("node:test");
const assert = require("node:assert/strict");
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {
  buildIndex,
  buildTree,
  loadDocumentation,
  searchDocumentation,
  showDocumentation,
} = require("./doc-runtime.js");

const tempRoots = [];

function makeDocs() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "docs-runtime-"));
  tempRoots.push(root);
  fs.mkdirSync(path.join(root, ".docs", "tech"), { recursive: true });
  fs.mkdirSync(path.join(root, ".docs", "adr"), { recursive: true });
  fs.mkdirSync(path.join(root, ".docs", "db", "pages"), { recursive: true });
  fs.mkdirSync(path.join(root, ".docs", "db", "dist"), { recursive: true });
  fs.writeFileSync(path.join(root, ".docs", "db", "pages", "home.md"), "---\nid: page.home\ntype: page\nname: Home\nrelated: [term.home-label]\n---\n\n# Home\n\n## Page Purpose\n\n!include term.home-label\n\n## Page Structure\n\nMain content.\n", "utf8");
  fs.mkdirSync(path.join(root, ".docs", "db", "terms"), { recursive: true });
  fs.writeFileSync(path.join(root, ".docs", "db", "terms", "home-label.md"), "---\nid: term.home-label\ntype: term\nname: Home Label\n---\n\n# Home Label\n", "utf8");
  fs.writeFileSync(
    path.join(root, ".docs", "tech", "search.md"),
    "---\n" +
      "description: 中文检索与渐进读取规则。\n" +
      "kind: technical\n" +
      "topic: retrieval\n" +
      "code:\n" +
      "  paths:\n" +
      "    - src/search.js\n" +
      "---\n\n" +
      "# 文档检索\n\n先定位 owner，再读取源码。\n",
    "utf8",
  );
  fs.writeFileSync(
    path.join(root, ".docs", "adr", "0001-runtime.md"),
    "---\ndescription: 选择确定性文本检索作为第一阶段方案。\nkind: adr\ntopic: retrieval\n---\n\n# 确定性检索\n\n不引入 embedding。\n",
    "utf8",
  );
  fs.writeFileSync(path.join(root, ".docs", "db", "dist", "ignored.md"), "ignored\n", "utf8");
  return root;
}

afterEach(() => {
  while (tempRoots.length > 0) {
    fs.rmSync(tempRoots.pop(), { recursive: true, force: true });
  }
});

describe("documentation runtime", () => {
  test("loads metadata and excludes schema and generated files", () => {
    const root = makeDocs();
    const docs = loadDocumentation(root);

    assert.deepEqual(docs.map((doc) => doc.relativePath), [
      "adr/0001-runtime.md",
      "db/pages/home.md",
      "db/terms/home-label.md",
      "tech/search.md",
    ]);
    assert.deepEqual({ description: docs[3].description, kind: docs[3].kind, topic: docs[3].topic, codePaths: docs[3].codePaths }, {
      description: "中文检索与渐进读取规则。",
      kind: "technical",
      topic: "retrieval",
      codePaths: ["src/search.js"],
    });
    assert.ok(docs[1].content.includes("# Home Label"));
    assert.deepEqual(docs[1].related, ["term.home-label"]);
  });

  test("builds a topic tree and filtered index", () => {
    const root = makeDocs();

    const tree = buildTree(loadDocumentation(root));
    assert.equal(tree.total, 4);
    assert.deepEqual(tree.topics.map((topic) => [topic.topic, topic.count]), [
      ["pages", 1],
      ["retrieval", 2],
      ["terms", 1],
    ]);
    const index = buildIndex(loadDocumentation(root), { topic: "retrieval", kind: "adr" });
    assert.equal(index.length, 1);
    assert.equal(index[0].relativePath, "adr/0001-runtime.md");
    assert.equal(index[0].kind, "adr");
  });

  test("searches Chinese terms and returns snippets", () => {
    const root = makeDocs();
    const results = searchDocumentation(loadDocumentation(root), "中文检索");

    assert.equal(results[0].relativePath, "tech/search.md");
    assert.ok(results[0].snippet.includes("文档检索"));
  });

  test("rejects unsupported block scalar metadata", () => {
    assert.throws(() => require("./doc-runtime.js").parseFrontmatter("---\ndescription: >\n  two lines\n---\n# Doc\n"), /Block scalar/);
  });

  test("returns non-zero for missing JSON show paths and parses filter order", () => {
    const root = makeDocs();
    const script = path.resolve(__dirname, "docs.js");
    const missing = spawnSync(process.execPath, [script, "show", "missing.md", "--json"], { cwd: root, encoding: "utf8" });
    assert.equal(missing.status, 1);

    const filtered = spawnSync(process.execPath, [script, "search", "确定性", "--kind", "adr", "--json"], { cwd: root, encoding: "utf8" });
    assert.equal(filtered.status, 0);
    assert.equal(JSON.parse(filtered.stdout)[0].relativePath, "adr/0001-runtime.md");
  });

  test("shows selected documents and reports missing paths", () => {
    const root = makeDocs();
    const result = showDocumentation(loadDocumentation(root), ["tech/search.md", "missing.md"]);

    assert.equal(result.documents.length, 1);
    assert.deepEqual(result.missing, ["missing.md"]);
    assert.ok(result.documents[0].content.includes("# 文档检索"));
  });
});
