#!/usr/bin/env python3
"""Integration checks for the lm-detector updater using isolated local repositories."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("update-lm-detector.py")


def run(*args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, check=True).stdout.strip()


class UpdateTest(unittest.TestCase):
    def test_updates_sdk_and_data_together_and_skips_unrelated_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upstream = root / "upstream"
            target = root / "target"
            upstream.mkdir()
            target.mkdir()
            run("git", "init", "-q", "-b", "main", cwd=upstream)
            run("git", "config", "user.email", "test@example.com", cwd=upstream)
            run("git", "config", "user.name", "Test", cwd=upstream)
            (upstream / "data").mkdir()
            (upstream / "sdk/go").mkdir(parents=True)
            (upstream / "LICENSE").write_text("MIT\n")
            (upstream / "sdk/go/go.mod").write_text("module example\n")
            (upstream / "data/unified_bank.json").write_text('{"old":true}\n')
            (upstream / "data/shared_detector.bin").write_bytes(b"old")
            run("git", "add", ".", cwd=upstream)
            run("git", "commit", "-qm", "old", cwd=upstream)
            old = run("git", "rev-parse", "HEAD", cwd=upstream)
            (target / "internal/control/lmdata").mkdir(parents=True)
            (target / "internal/control/modeltrace_detector.go").write_text(f"// Data and SDK are pinned to upstream commit {old}.\n")
            (target / "THIRD_PARTY_NOTICES.md").write_text(f"## lm-detector\n\n- Go SDK: `github.com/Ikaleio/lm-detector/sdk/go` version\n  `v0.1.0`\n- Embedded data snapshot: `{old}`,\n")
            (target / "go.mod").write_text("module example\n")
            (target / "go.sum").write_text("")
            (target / "internal/control/lmdata/unified_bank.json").write_text('{"old":true}\n')
            (target / "internal/control/lmdata/shared_detector.bin").write_bytes(b"old")
            fakebin = root / "bin"
            fakebin.mkdir()
            (fakebin / "go").write_text('#!/bin/sh\nif [ "$1" != list ]; then exit 0; fi\nprintf "v0.2.1-test\\n"\n')
            (fakebin / "go").chmod(0o755)
            env = dict(os.environ, PATH=f"{fakebin}:{os.environ['PATH']}")
            (upstream / "README.md").write_text("unrelated\n")
            run("git", "add", ".", cwd=upstream)
            run("git", "commit", "-qm", "unrelated", cwd=upstream)
            run("python3", str(SCRIPT), "--repo", str(target), "--upstream", str(upstream), env=env)
            self.assertIn(old, (target / "internal/control/modeltrace_detector.go").read_text())
            (upstream / "data/shared_detector.bin").write_bytes(b"new")
            run("git", "add", ".", cwd=upstream)
            run("git", "commit", "-qm", "new data", cwd=upstream)
            new = run("git", "rev-parse", "HEAD", cwd=upstream)
            run("python3", str(SCRIPT), "--repo", str(target), "--upstream", str(upstream), env=env)
            self.assertEqual((target / "internal/control/lmdata/shared_detector.bin").read_bytes(), b"new")
            self.assertIn(new, (target / "internal/control/modeltrace_detector.go").read_text())
            notice = (target / "THIRD_PARTY_NOTICES.md").read_text()
            self.assertIn(new, notice)
            self.assertIn("v0.2.1-test", notice)
            (upstream / "LICENSE").write_text("changed license\n")
            run("git", "add", ".", cwd=upstream)
            run("git", "commit", "-qm", "license", cwd=upstream)
            failed = subprocess.run(["python3", str(SCRIPT), "--repo", str(target), "--upstream", str(upstream)], env=env, text=True, capture_output=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("upstream LICENSE changed", failed.stderr)
            self.assertIn(new, (target / "internal/control/modeltrace_detector.go").read_text())


if __name__ == "__main__":
    unittest.main()
