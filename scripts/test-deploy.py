"""Command-boundary deployment tests; no Docker daemon or SSH server required."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("deploy.sh").resolve()
REPO = SCRIPT.parent.parent
COMMIT = "a" * 40
SHORT = "abc1234"
VERSION = "dev-" + SHORT

STUB = f'''#!{sys.executable}
import io, os, sys, pathlib, tarfile

name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["CALLS"], "a") as log:
    log.write(name + " " + " ".join(args) + "\\n")
if name == "git":
    if args and args[0] == "archive":
        # Stand in for the fetched commit: the extracted tree is what must run.
        archive = tarfile.open(fileobj=sys.stdout.buffer, mode="w|")
        for relative in ("Dockerfile", "scripts/deploy.sh", "scripts/local-rehearsal.py",
                         "scripts/verify-built-image.py"):
            path = pathlib.Path(os.environ["REPO"]) / relative
            content = path.read_bytes()
            if relative == "scripts/deploy.sh":
                content = content.replace(b"set -euo pipefail",
                                          b"set -euo pipefail\\necho bundle-script-ran", 1)
            info = archive.gettarinfo(str(path), arcname=relative)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        archive.close()
    elif "rev-parse" in args:
        print("{SHORT}" if "--short" in args else "{COMMIT}")
elif name == "docker":
    if args[:2] == ["context", "inspect"]: print("unix:///var/run/docker.sock")
    if "build" in args: sys.stdin.buffer.read()
    if "build" in args and os.environ["FAIL"] == "build": sys.exit(1)
    if args[:2] == ["image", "inspect"]:
        # The previously deployed image is not cached locally: only its ID is known.
        if any(arg.startswith("sha256:") for arg in args): sys.exit(1)
        if os.environ["MISSING_IMAGE"] == "1": sys.exit(1)
        print("sha256:" + "c" * 64)
    if "save" in args: sys.stdout.buffer.write(b"image")
elif name == "ssh":
    if "uname -m" in args: print("x86_64")
    elif any(".Image" in arg for arg in args): print("sha256:" + "b" * 64)
    elif "docker save" in args: sys.stdout.buffer.write(b"image")
    elif "docker load" in args:
        sys.stdin.buffer.read()
        if os.environ["FAIL"] == "load": sys.exit(1)
    else:
        pathlib.Path(os.environ["REMOTE"]).write_text(sys.stdin.read())
elif name == "curl":
    print('{{"status":"ok","version":"{VERSION}"}}')
elif name == "python3":
    if os.environ["FAIL"] == "verify": sys.exit(1)
'''


class DeployTest(unittest.TestCase):
    def run_deploy(self, fail="", endpoint="unix:///var/run/docker.sock"):
        return self.run_script(fail, endpoint, [])

    def run_script(self, fail="", endpoint="unix:///var/run/docker.sock", args=(),
                   missing_image=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "stub"
            stub.write_text(STUB)
            stub.chmod(0o755)
            for name in ("git", "docker", "ssh", "curl", "python3"):
                (root / name).symlink_to(stub)
            env = dict(
                os.environ,
                DOCKER_HOST=endpoint,
                PATH=f"{root}:{os.environ['PATH']}",
                CALLS=str(root / "calls"),
                REMOTE=str(root / "remote"),
                REPO=str(REPO),
                FAIL=fail,
                MISSING_IMAGE="1" if missing_image else "0",
            )
            env.pop("DOCKER_CONTEXT", None)
            env.pop("DEPLOY_COMMIT", None)
            result = subprocess.run(
                ["bash", str(SCRIPT), *args], env=env, capture_output=True, text=True, timeout=60
            )
            log = root / "calls"
            calls = log.read_text() if log.exists() else ""
            remote = (root / "remote").read_text() if (root / "remote").exists() else ""
            return result, calls, remote

    def test_local_build_then_load_then_switch(self):
        result, calls, remote = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--platform linux/amd64", calls)
        self.assertLess(calls.index("docker build"), calls.index("docker load"))
        self.assertIn("docker compose", remote)
        self.assertNotIn("docker build", remote)
        self.assertNotIn("git ", remote)
        self.assertIn("--no-build --pull never", remote)
        syntax = subprocess.run(
            ["bash", "-n"], input=remote, text=True, capture_output=True
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_build_failure_does_not_upload_or_switch(self):
        result, calls, remote = self.run_deploy("build")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("docker load", calls)
        self.assertEqual(remote, "")

    def test_verification_failure_does_not_upload_or_stop_production(self):
        result, calls, remote = self.run_deploy("verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verify-built-image.py", calls)
        self.assertNotIn("docker save gpt-load:", calls)
        self.assertEqual(remote, "")

    def test_verification_precedes_upload(self):
        result, calls, _ = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        # The old image is fetched from the server and loaded before verification.
        self.assertIn("docker save 'sha256:", calls)
        self.assertIn("docker load", calls)
        self.assertIn("--previous-image sha256:" + "b" * 64, calls)
        self.assertLess(calls.index("docker build"), calls.index("verify-built-image.py"))
        # Only the verified image is uploaded to the server.
        self.assertLess(calls.index("verify-built-image.py"), calls.index("docker save gpt-load:"))

    def test_validation_runs_the_fetched_commit_scripts(self):
        result, calls, _ = self.run_deploy()
        self.assertEqual(result.returncode, 0, result.stderr)
        # The archive of the fetched commit is extracted and executed; the working
        # tree copy of deploy.sh never runs the build, so a dirty tree cannot ship.
        self.assertIn("bundle-script-ran", result.stdout)
        self.assertLess(calls.index("git archive"), calls.index("docker build"))
        verify = [line.split() for line in calls.splitlines() if "verify-built-image.py" in line]
        self.assertEqual(len(verify), 1, calls)
        self.assertNotIn(str(REPO), verify[0][1])
        self.assertIn("--expected-version " + VERSION, calls)

    def test_prepare_only_verifies_and_uploads_but_does_not_switch(self):
        result, calls, remote = self.run_script(args=("dev", "--prepare-only"))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(calls.index("docker build"), calls.index("verify-built-image.py"))
        self.assertLess(calls.index("verify-built-image.py"), calls.index("docker save gpt-load:"))
        self.assertEqual(remote, "")

    def test_activate_only_still_verifies_before_switching(self):
        # 切换不是绕开验证的捷径：必须重新验证本地镜像，且不上传、不构建。
        result, calls, remote = self.run_script(args=("dev", "--activate-only", COMMIT))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("docker build", calls)
        self.assertNotIn("docker save gpt-load:", calls)
        self.assertIn("verify-built-image.py", calls)
        self.assertIn("docker compose", remote)

    def test_activate_only_refuses_a_commit_that_is_not_the_remote_head(self):
        result, calls, remote = self.run_script(args=("dev", "--activate-only", "b" * 40))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("远端分支已变化", result.stderr)
        # 只允许读取分支信息，不构建、不上传、不切换。
        self.assertEqual([line for line in calls.splitlines() if not line.startswith("git")], [])
        self.assertEqual(remote, "")

    def test_activate_only_refuses_when_the_image_is_missing_locally(self):
        result, calls, remote = self.run_script(args=("dev", "--activate-only", COMMIT),
                                                missing_image=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("本地没有待切换镜像", result.stderr)
        self.assertNotIn("verify-built-image.py", calls)
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
