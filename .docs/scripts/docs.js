#!/usr/bin/env bun
"use strict";

const { buildIndex, buildTree, loadDocumentation, searchDocumentation, showDocumentation } = require("./doc-runtime.js");

function usage() {
  console.error("Usage: bun .docs/scripts/docs.js <tree|index|search|show> [options]");
  console.error("  tree [--json]");
  console.error("  index [--topic <topic>] [--kind <kind>] [--json]");
  console.error("  search <query> [--topic <topic>] [--kind <kind>] [--json]");
  console.error("  show <path...> [--json]");
}

function parseArguments(argv) {
  const positional = [];
  const options = {};
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--json") {
      options.json = true;
      continue;
    }
    if (value === "--topic" || value === "--kind") {
      const optionValue = argv[index + 1];
      if (!optionValue || optionValue.startsWith("--")) {
        throw new Error(`${value} requires a value`);
      }
      options[value.slice(2)] = optionValue;
      index += 1;
      continue;
    }
    positional.push(value);
  }
  return { positional, options };
}

function printTree(tree) {
  console.log(`.docs/ (${tree.total} documents)`);
  for (const topic of tree.topics) {
    console.log(`\n  ${topic.topic}/ (${topic.count}; ${topic.owners.join(", ")})`);
    for (const doc of topic.documents) {
      console.log(`    ${doc.relativePath} [${doc.kind}]`);
      console.log(`      ${doc.description}`);
    }
  }
  console.log("\nhint: docs index --topic <topic>; docs search \"<concept>\"; docs show <path>");
}

function printIndex(docs) {
  for (const doc of docs) {
    console.log(`${doc.relativePath} [${doc.kind}] (${doc.topic})`);
    console.log(`  ${doc.description}`);
    if (doc.codePaths.length > 0) console.log(`  code: ${doc.codePaths.join(", ")}`);
  }
}

function printSearch(results) {
  for (const result of results) {
    console.log(`${result.relativePath} [${result.kind}] score=${result.score}`);
    console.log(`  ${result.description}`);
    console.log(`  ${result.snippet}`);
  }
}

function printShow(result) {
  for (const doc of result.documents) {
    console.log(`--- ${doc.relativePath} [${doc.kind}] ---`);
    console.log(doc.content);
  }
  if (result.missing.length > 0) {
    console.error(`Missing documents: ${result.missing.join(", ")}`);
    process.exitCode = 1;
  }
}

const argv = process.argv.slice(2);
const command = argv.shift();
if (!command || !["tree", "index", "search", "show"].includes(command)) {
  usage();
  process.exit(1);
}

try {
  const { positional, options } = parseArguments(argv);
  const docs = loadDocumentation(process.cwd());
  const filters = { topic: options.topic, kind: options.kind };
  let result;
  if (command === "tree") result = buildTree(docs);
  if (command === "index") result = buildIndex(docs, filters);
  if (command === "search") {
    const query = positional[0];
    if (!query) throw new Error("search requires a query");
    result = searchDocumentation(docs, query, filters);
  }
  if (command === "show") result = showDocumentation(docs, positional);

  if (options.json) console.log(JSON.stringify(result, null, 2));
  else if (command === "tree") printTree(result);
  else if (command === "show") printShow(result);
  else if (command === "search") printSearch(result);
  else printIndex(result);

  if (command === "show" && result.missing.length > 0) {
    process.exitCode = 1;
  }
} catch (error) {
  console.error(`Documentation runtime error: ${error.message}`);
  process.exit(1);
}
