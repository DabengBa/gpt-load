"use strict";

const assert = require("node:assert/strict");
const { afterEach, describe, test } = require("node:test");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { findDocumentationProblems } = require("./doc-integrity-check.js");

const tempRoots = [];

function makeRepo() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "docs-integrity-"));
  tempRoots.push(root);
  fs.mkdirSync(path.join(root, ".docs"), { recursive: true });
  return root;
}

function writeFile(root, relativePath, content) {
  const filePath = path.join(root, relativePath);
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content, "utf8");
}

afterEach(() => {
  while (tempRoots.length > 0) {
    fs.rmSync(tempRoots.pop(), { recursive: true, force: true });
  }
});

const TECH_FRONTMATTER = "---\ndescription: 模块运行边界。\nkind: technical\ntopic: module\n---\n";

describe("non-semantic documentation integrity", () => {
  test("recognizes this repository's brief, history, communications and product changes", () => {
    const root = makeRepo();
    for (const file of [
      "README.md",
      "PROJECT_HISTORY.md",
      "communications/source.md",
      "product-changelog/change.md",
      "tech/briefs/legacy.md",
    ]) {
      writeFile(root, `.docs/${file}`, "# Original record\n");
    }

    assert.deepEqual(findDocumentationProblems(root), []);
  });

  test("accepts configured owners and valid local links", () => {
    const root = makeRepo();
    writeFile(root, ".docs/AGENTS.md", "# Rules\n");
    writeFile(root, ".docs/project-overview.md", "# Overview\n");
    writeFile(
      root,
      ".docs/tech/module.md",
      `${TECH_FRONTMATTER}\n# Module\n\n[Workflow](../workflows/runtime.md)\n`,
    );
    writeFile(
      root,
      ".docs/workflows/runtime.md",
      "---\ndescription: 运行时验证步骤和失败边界。\nkind: workflow\ntopic: runtime\n---\n\n# Runtime\n",
    );
    writeFile(root, ".docs/specs/release/plan.md", "# Plan\n");
    writeFile(
      root,
      ".docs/adr/0001-use-runtime.md",
      "---\ndescription: 说明运行时选择及其验证取舍。\nkind: adr\ntopic: runtime\n---\n\n# Use Runtime\n",
    );

    assert.deepEqual(findDocumentationProblems(root), []);
  });

  test("rejects unowned Markdown and invalid ADR names", () => {
    const root = makeRepo();
    writeFile(root, ".docs/notes.md", "# Notes\n");
    writeFile(root, ".docs/adr/decision.md", "# Decision\n");

    const problems = findDocumentationProblems(root);

    assert.ok(problems.some((problem) => problem.includes("unowned Markdown")));
    assert.ok(problems.some((problem) => problem.includes("ADR filename")));
  });

  test("rejects missing retrieval metadata", () => {
    const root = makeRepo();
    writeFile(root, ".docs/tech/module.md", "# Module\n");

    const problems = findDocumentationProblems(root);

    assert.ok(
      problems.some((problem) =>
        problem.includes('missing retrieval frontmatter field "description"'),
      ),
    );
    assert.ok(
      problems.some((problem) => problem.includes('missing retrieval frontmatter field "kind"')),
    );
    assert.ok(
      problems.some((problem) => problem.includes('missing retrieval frontmatter field "topic"')),
    );
  });

  test("rejects broken local links while ignoring external links", () => {
    const root = makeRepo();
    writeFile(
      root,
      ".docs/tech/module.md",
      `${TECH_FRONTMATTER}\n# Module\n\n[Missing](missing.md)\n\n[External](https://example.com)\n`,
    );

    const problems = findDocumentationProblems(root);

    assert.equal(problems.length, 1);
    assert.ok(problems[0].includes("missing.md"));
  });

  test("rejects relation targets and links outside the repository", () => {
    const root = makeRepo();
    writeFile(
      root,
      ".docs/tech/module.md",
      `---\ndescription: 模块边界和关联 owner。\nkind: technical\ntopic: module\nrelations:\n  requires:\n    - tech/missing.md\n---\n\n# Module\n\n[Outside](../../../${path.basename(root)}-outside.md)\n`,
    );
    writeFile(root, `../${path.basename(root)}-outside.md`, "outside\n");

    const problems = findDocumentationProblems(root);

    assert.ok(problems.some((problem) => problem.includes('missing relation target "tech/missing.md"')));
    assert.ok(problems.some((problem) => problem.includes("local link escapes repository")));
  });

  test("ignores link syntax inside fenced code blocks and inline code", () => {
    const root = makeRepo();
    writeFile(
      root,
      ".docs/tech/module.md",
      [
        TECH_FRONTMATTER,
        "# Module",
        "",
        "Inline `[Inline](inline-missing.md)` stays literal.",
        "",
        "```md",
        "[Fenced](fenced-missing.md)",
        "```",
        "",
        "~~~~",
        "[Tilde](tilde-missing.md)",
        "~~~~",
        "",
        "[Real](../tech/missing-real.md)",
        "",
      ].join("\n"),
    );

    const problems = findDocumentationProblems(root);

    assert.equal(problems.length, 1);
    assert.ok(problems[0].includes("missing-real.md"));
    assert.ok(problems[0].includes("tech/module.md:19"));
  });

  test("rejects unknown semantic links and accepts declared semantic ids", () => {
    const root = makeRepo();
    writeFile(
      root,
      ".docs/db/features/known.md",
      "---\nid: feature.known\ntype: feature\nname: Known\n---\n# Known\n",
    );
    writeFile(
      root,
      ".docs/tech/module.md",
      `${TECH_FRONTMATTER}\n# Module\n\n[Known](feature.known)\n\n[Unknown](feature.unknown)\n`,
    );

    const problems = findDocumentationProblems(root);

    assert.equal(problems.length, 1);
    assert.ok(problems[0].includes('broken semantic link "feature.unknown"'));
  });

  test("checks plain path links inside semantic documents", () => {
    const root = makeRepo();
    writeFile(root, ".docs/tech/module.md", `${TECH_FRONTMATTER}\n# Module\n`);
    writeFile(
      root,
      ".docs/db/features/known.md",
      [
        "---",
        "id: feature.known",
        "type: feature",
        "name: Known",
        "---",
        "# Known",
        "",
        "[Tech](../../tech/module.md)",
        "",
        "[Missing](../../tech/missing.md)",
        "",
        "[Self](feature.known)",
        "",
      ].join("\n"),
    );

    const problems = findDocumentationProblems(root);

    assert.equal(problems.length, 1);
    assert.ok(problems[0].includes("db/features/known.md"));
    assert.ok(problems[0].includes("broken local link"));
    assert.ok(problems[0].includes("tech/missing.md"));
  });
});
