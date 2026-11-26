#!/usr/bin/env python3
"""Focused integration proof for the L5 wrap-up scripts (filesystem + git boundary)."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def run_script(script: str, *args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        text=True,
        capture_output=True,
        check=False,
    )


class Fixture:
    def __init__(self, root: Path):
        self.root = root

    def write(self, rel: str, text: str) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def git(self, *args: str):
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            text=True,
            capture_output=True,
            check=False,
        )


def make_repo(f: Fixture) -> None:
    f.write(".gitignore", ".tmp/\n**/.tmp/\n")
    f.write("AGENTS.MD", "# Boot\n- build: bun run build\n")
    f.write("src/app.py", "def stable_symbol():\n    return 1\n")
    f.write(
        ".agents/skills/demo/SKILL.md",
        "---\nname: demo\ndescription: Does a thing.\n---\n# Demo\nStep.\n",
    )
    f.write(
        ".knowledge/decisions/ADR-0001.md",
        "---\nid: ADR-0001\nstatus: active\n"
        "affects_symbols: [src/app.py#stable_symbol]\n---\nContext.\n",
    )
    f.write(".tmp/notes.md", "tradeoff: chose X over Y because Z\n")
    f.git("init", "-q")
    f.git("add", ".")
    f.git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    f.git("checkout", "-qb", "feature")
    f.write("src/app.py", "def stable_symbol():\n    return 2\n")


class ScriptTests(unittest.TestCase):
    def test_clean_repo_passes_and_accepts_uppercase_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            make_repo(f)
            result = run_script("check_doc_budgets.py", "--repo", str(f.root))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "pass")
            self.assertFalse(
                any("not found" in w for w in payload["warnings"]),
                "uppercase AGENTS.MD must satisfy the root boot-contract check",
            )

    def test_budget_dangling_symbol_and_tmp_leak_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            make_repo(f)
            f.write(
                ".agents/skills/demo/SKILL.md",
                "---\nname: demo\ndescription: Does a thing.\n---\n"
                + "# Demo\n" * 510,
            )
            f.write(
                ".knowledge/decisions/ADR-0001.md",
                "---\nid: ADR-0001\nstatus: active\n"
                "affects_symbols: [src/app.py#missing_symbol]\n---\nContext.\n",
            )
            f.write(".tmp/leak.md", "process doc\n")
            f.git("add", "-f", ".tmp/leak.md")
            result = run_script("check_doc_budgets.py", "--repo", str(f.root))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            errors = " ".join(json.loads(result.stdout)["errors"])
            self.assertIn("exceeds 500 lines", errors)
            self.assertIn("Dangling symbol 'missing_symbol'", errors)
            self.assertIn("Tracked files found under .tmp/", errors)

    def test_distill_bundle_contains_ephemeral_docs_diff_and_owner_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            make_repo(f)
            out = f.root / ".tmp" / "distill-bundle.json"
            result = run_script(
                "prepare_distill_bundle.py",
                "--worktree", str(f.root),
                "--base-ref", "HEAD",
                "--out", str(out),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["status"], "ok")
            self.assertGreaterEqual(summary["ephemeral_files"], 1)
            bundle = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                bundle["existing_knowledge_index"]["decisions"], ["ADR-0001.md"]
            )
            self.assertTrue(any(".tmp/notes.md" == doc["path"] for doc in bundle["ephemeral_tmp_files"]))
            self.assertIn("UNCOMMITTED WORKTREE DIFF", bundle["git_diff"])
            self.assertTrue(bundle["git_diff_stat"])

    def test_mapped_owner_symbol_refs_are_garbage_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Fixture(Path(tmp))
            make_repo(f)
            f.write(
                ".docs/adr/0001-choice.md",
                "---\ndescription: Why X was chosen over Y\nkind: adr\ntopic: architecture\n"
                "id: ADR-0002\nstatus: active\n"
                "affects_symbols: [src/app.py#missing_owner_symbol]\n---\nContext.\n",
            )
            f.write(
                ".docs/tech/plain.md",
                "---\ndescription: Plain tech doc\nkind: technical\ntopic: ops\n---\nBody.\n",
            )
            f.write(
                ".docs/adr/0002-valid.md",
                "---\ndescription: Valid symbol binding\nkind: adr\ntopic: architecture\n"
                "id: ADR-0003\nstatus: active\naffects_symbols: [src/app.py#stable_symbol]\n---\nContext.\n",
            )
            f.write(
                ".docs/adr/0003-archived.md",
                "---\ndescription: Archived decision\nkind: adr\ntopic: architecture\n"
                "id: ADR-0004\nstatus: superseded\n"
                "affects_symbols: [src/app.py#missing_archived_symbol]\n---\nContext.\n",
            )
            result = run_script("check_doc_budgets.py", "--repo", str(f.root))
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            errors = " ".join(json.loads(result.stdout)["errors"])
            self.assertIn("Dangling symbol 'missing_owner_symbol'", errors)
            self.assertIn(".docs/adr/0001-choice.md", errors)
            self.assertNotIn("missing_archived_symbol", errors)
            self.assertNotIn("missing frontmatter field", errors)

    def test_init_creates_l5_layout_ignore_rules_and_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            holdout = Path(tmp) / "holdout"
            root.mkdir()
            result = run_script(
                "init_l5_docs.py",
                "--repo", str(root),
                "--with-templates",
                "--init-holdout", str(holdout),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["status"], "ok")
            for rel in (
                ".knowledge/decisions", ".knowledge/invariants",
                ".knowledge/quirks", ".agents/skills",
            ):
                self.assertTrue((root / rel).is_dir(), rel)
            ignore = (root / ".gitignore").read_text(encoding="utf-8")
            self.assertIn(".tmp/", ignore)
            self.assertIn("**/.tmp/", ignore)
            self.assertTrue((root / "AGENTS.md").is_file())
            for sub in ("scenarios", "digital-twins", "seeds"):
                self.assertTrue((holdout / sub).is_dir(), sub)


if __name__ == "__main__":
    unittest.main(verbosity=2)
