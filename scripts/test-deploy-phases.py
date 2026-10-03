"""Verify deployment preparation and activation without Docker or SSH."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("deploy.sh")
COMMIT = "a" * 40


class DeployPhasesTest(unittest.TestCase):
    def run_deploy(self, *args):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "stub"
            stub.write_text("""#!/usr/bin/env python3
import os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as log:
    log.write(name + " " + " ".join(args) + "\\n")
if name == "git" and "rev-parse" in args:
    print("aaaaaaaa" if "--short" in args else "a" * 40)
elif name == "docker":
    if args[:2] == ["context", "inspect"]: print("unix:///var/run/docker.sock")
    if "build" in args: sys.stdin.buffer.read()
    if "save" in args: sys.stdout.buffer.write(b"image")
elif name == "ssh":
    if "uname -m" in args: print("x86_64")
    elif "docker load" in args: sys.stdin.buffer.read()
    else: pathlib.Path(os.environ["REMOTE"]).write_text(sys.stdin.read())
""")
            stub.chmod(0o755)
            for name in ("git", "docker", "ssh", "curl"):
                (root / name).symlink_to(stub)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                       DOCKER_HOST="unix:///var/run/docker.sock",
                       CALLS=str(root / "calls"), REMOTE=str(root / "remote"))
            env.pop("DOCKER_CONTEXT", None)
            result = subprocess.run(["bash", str(SCRIPT), *args], env=env,
                                    capture_output=True, text=True, timeout=10)
            calls = (root / "calls").read_text() if (root / "calls").exists() else ""
            remote = (root / "remote").read_text() if (root / "remote").exists() else ""
            return result, calls, remote

    def test_prepare_uploads_without_switching(self):
        result, calls, remote = self.run_deploy("dev", "--prepare-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("docker build", calls)
        self.assertIn("docker load", calls)
        self.assertNotIn("docker compose", remote)
        self.assertNotIn("curl", calls)

    def test_activate_pins_commit_without_build_or_upload(self):
        result, calls, remote = self.run_deploy("dev", "--activate-only", COMMIT)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("docker build", calls)
        self.assertNotIn("docker load", calls)
        self.assertIn("docker compose", remote)
        self.assertIn("docker image inspect", remote)
        self.assertIn("curl", calls)

    def test_changed_branch_refuses_activation(self):
        result, calls, remote = self.run_deploy("dev", "--activate-only", "b" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(remote, "")
        self.assertNotIn("docker build", calls)

    def test_activate_requires_expected_commit(self):
        result, _, remote = self.run_deploy("dev", "--activate-only")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(remote, "")

    def test_unknown_mode_is_rejected(self):
        result, _, remote = self.run_deploy("dev", "--unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(remote, "")


if __name__ == "__main__":
    unittest.main()
