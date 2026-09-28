"""Command-boundary deployment tests; no Docker daemon or SSH server required."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("deploy.sh")


class DeployTest(unittest.TestCase):
    def run_deploy(self, fail="", endpoint="unix:///var/run/docker.sock"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "stub"
            stub.write_text('''#!/usr/bin/env python3
import os, sys, pathlib
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as log:
    log.write(name + " " + " ".join(args) + "\\n")
if name == "git":
    if "rev-parse" in args: print("abc1234")
elif name == "docker":
    if args[:2] == ["context", "inspect"]: print("unix:///var/run/docker.sock")
    if "build" in args: sys.stdin.buffer.read()
    if "build" in args and os.environ["FAIL"] == "build": sys.exit(1)
    if "save" in args: sys.stdout.buffer.write(b"image")
elif name == "ssh":
    if "uname -m" in args: print("x86_64")
    elif "docker load" in args:
        sys.stdin.buffer.read()
        if os.environ["FAIL"] == "load": sys.exit(1)
    else:
        pathlib.Path(os.environ["REMOTE"]).write_text(sys.stdin.read())
''')
            stub.chmod(0o755)
            for name in ("git", "docker", "ssh", "curl"):
                (root / name).symlink_to(stub)
            env = dict(os.environ, DOCKER_HOST=endpoint, PATH=f"{root}:{os.environ['PATH']}",
                       CALLS=str(root / "calls"), REMOTE=str(root / "remote"), FAIL=fail)
            env.pop("DOCKER_CONTEXT", None)
            result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, timeout=10)
            calls = (root / "calls").read_text()
            remote = (root / "remote").read_text() if (root / "remote").exists() else ""
            return result, calls, remote

    def test_local_build_then_load_then_switch(self):
        result, calls, remote = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertIn("--platform linux/amd64", calls)
        self.assertLess(calls.index("docker build"), calls.index("docker load"))
        self.assertIn("docker compose", remote)
        self.assertNotIn("docker build", remote)
        self.assertNotIn("git ", remote)
        self.assertIn("--no-build --pull never", remote)
        syntax = subprocess.run(["bash", "-n"], input=remote, text=True, capture_output=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_build_failure_does_not_upload_or_switch(self):
        result, calls, remote = self.run_deploy("build")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("docker load", calls)
        self.assertEqual(remote, "")

    def test_load_failure_does_not_switch(self):
        result, calls, remote = self.run_deploy("load")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("docker load", calls)
        self.assertEqual(remote, "")

    def test_remote_docker_daemon_is_rejected(self):
        result, calls, remote = self.run_deploy(endpoint="ssh://vps-kl")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("docker build", calls)
        self.assertEqual(remote, "")


if __name__ == "__main__":
    unittest.main()
