#!/usr/bin/env python3
"""Release orchestration tests; no GitHub, deployment, or network access."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

spec = importlib.util.spec_from_file_location(
    "release_with_lm_detector", Path(__file__).with_name("release-with-lm-detector.py")
)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class ReleaseTest(unittest.TestCase):
    def backend(self, changed=True):
        backend = Mock()
        backend.base.return_value = "a" * 40
        backend.update.return_value = changed
        backend.open_pr.return_value = (42, "b" * 40)
        backend.merge.return_value = "c" * 40
        return backend

    def test_unchanged_deploys_exact_checked_commit(self):
        backend = self.backend(False)
        release.run_release(backend, "all", None)
        self.assertEqual(backend.mock_calls, [
            call.base(), call.update(), call.confirm_base("a" * 40),
            call.deploy("a" * 40, "all", None),
        ])

    def test_changed_validates_creates_merges_verifies_then_deploys(self):
        backend = self.backend()
        release.run_release(backend, "--prepare-only", None)
        self.assertEqual(backend.mock_calls, [
            call.base(), call.update(), call.validate(),
            call.confirm_base("a" * 40), call.open_pr("a" * 40),
            call.merge(42, "b" * 40, "a" * 40),
            call.confirm_merged("c" * 40, "a" * 40, "b" * 40),
            call.confirm_base("c" * 40),
            call.deploy("c" * 40, "--prepare-only", None),
        ])

    def test_failure_never_deploys(self):
        for step in ("base", "update", "validate", "confirm_base", "open_pr", "merge", "confirm_merged"):
            with self.subTest(step=step):
                backend = self.backend()
                getattr(backend, step).side_effect = RuntimeError(step)
                with self.assertRaises(RuntimeError):
                    release.run_release(backend, "all", None)
                backend.deploy.assert_not_called()

    def test_activate_rejects_different_commit_before_deploy(self):
        backend = self.backend(False)
        with self.assertRaisesRegex(RuntimeError, "commit"):
            release.run_release(backend, "--activate-only", "d" * 40)
        backend.deploy.assert_not_called()

    def test_activate_with_upstream_changes_does_not_create_pr(self):
        backend = self.backend()
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            release.run_release(backend, "--activate-only", "a" * 40)
        backend.open_pr.assert_not_called()
        backend.deploy.assert_not_called()


class BackendTest(unittest.TestCase):
    def setUp(self):
        self.backend = release.Backend("/tmp/release-test")
        self.backend.upstream = "d" * 40
        self.base = "a" * 40
        self.head = "b" * 40
        self.commit = "c" * 40

    def test_merge_uses_real_gh_fields_and_confirms_repo_and_head(self):
        pr = {"state": "OPEN", "isDraft": False, "baseRefName": "dev",
              "headRefName": release.PREFIX + "d" * 12, "headRefOid": self.head,
              "isCrossRepository": False,
              "headRepository": {"name": "gpt-load"},
              "headRepositoryOwner": {"login": "DabengBa"}}
        self.backend.gh = Mock(side_effect=[json.dumps(pr), json.dumps({
            "merged": True, "sha": self.commit}), json.dumps({
            "state": "MERGED", "mergeCommit": {"oid": self.commit}})])
        self.backend.wait_checks = Mock()
        self.backend.confirm_base = Mock()
        self.assertEqual(self.backend.merge(42, self.head, self.base), self.commit)
        self.backend.wait_checks.assert_called_once_with(42, self.head, self.base)
        api = self.backend.gh.call_args_list[1].args
        self.assertIn("sha=" + self.head, api)
        self.assertIn("merge_method=merge", api)

    def test_failed_checks_do_not_merge(self):
        pr = {"state": "OPEN", "isDraft": False, "baseRefName": "dev",
              "headRefName": release.PREFIX + "d" * 12, "headRefOid": self.head,
              "isCrossRepository": False, "headRepository": {"name": "gpt-load"},
              "headRepositoryOwner": {"login": "DabengBa"}}
        self.backend.gh = Mock(return_value=json.dumps(pr))
        self.backend.wait_checks = Mock(side_effect=RuntimeError("failed checks"))
        with self.assertRaises(RuntimeError):
            self.backend.merge(42, self.head, self.base)
        self.assertEqual(self.backend.gh.call_count, 1)

    def test_existing_pr_and_auth_failure_stop_before_push(self):
        for result in ([{"number": 1}], RuntimeError("authentication failed")):
            with self.subTest(result=result):
                self.backend.update_prs = Mock(side_effect=result if isinstance(result, Exception) else None,
                                               return_value=result)
                self.backend.run = Mock()
                with self.assertRaises(RuntimeError):
                    self.backend.open_pr(self.base)
                self.backend.run.assert_not_called()

    def test_checks_wait_for_registration_and_pending_then_require_success(self):
        def status(items):
            return json.dumps({"headRefOid": self.head, "baseRefOid": self.base,
                               "statusCheckRollup": items})
        check = lambda state, conclusion: {"__typename": "CheckRun", "name": "deployment-tests",
                                            "status": state, "conclusion": conclusion}
        self.backend.gh = Mock(side_effect=[status([]), status([check("IN_PROGRESS", "")]),
                                           status([check("COMPLETED", "SUCCESS")])])
        with patch.object(release.time, "sleep") as sleep:
            self.backend.wait_checks(42, self.head, self.base)
        self.assertEqual(sleep.call_count, 2)

    def test_cancelled_and_other_non_success_checks_never_pass(self):
        for conclusion in ("CANCELLED", "FAILURE", "SKIPPED", "NEUTRAL", "TIMED_OUT"):
            with self.subTest(conclusion=conclusion):
                self.backend.gh = Mock(return_value=json.dumps({"headRefOid": self.head, "baseRefOid": self.base,
                    "statusCheckRollup": [{"__typename": "CheckRun", "name": "deployment-tests",
                                           "status": "COMPLETED", "conclusion": conclusion}]}))
                with self.assertRaisesRegex(RuntimeError, "check"):
                    self.backend.wait_checks(42, self.head, self.base)

    def test_missing_checks_timeout_and_auth_errors_fail_closed(self):
        response = json.dumps({"headRefOid": self.head, "baseRefOid": self.base, "statusCheckRollup": []})
        self.backend.gh = Mock(return_value=response)
        with patch.object(release.time, "monotonic", side_effect=[0, 1801]), patch.object(release.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                self.backend.wait_checks(42, self.head, self.base)
        self.backend.gh = Mock(side_effect=subprocess.CalledProcessError(1, "gh"))
        with self.assertRaises(subprocess.CalledProcessError):
            self.backend.wait_checks(42, self.head, self.base)

    def test_status_context_cannot_replace_expected_actions_job(self):
        response = {"headRefOid": self.head, "baseRefOid": self.base,
                    "statusCheckRollup": [{"__typename": "StatusContext",
                                           "context": "deployment-tests", "state": "SUCCESS"}]}
        self.backend.gh = Mock(return_value=json.dumps(response))
        with patch.object(release.time, "monotonic", side_effect=[0, 1801]), patch.object(release.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                self.backend.wait_checks(42, self.head, self.base)

    def test_checks_ref_changes_and_additional_failure_stop(self):
        success = {"__typename": "CheckRun", "name": "deployment-tests", "status": "COMPLETED", "conclusion": "SUCCESS"}
        cases = {"head": {"headRefOid": "f" * 40}, "base": {"baseRefOid": "f" * 40},
                 "failure": {"statusCheckRollup": [success, {"__typename": "StatusContext", "context": "other", "state": "FAILURE"}]},
                 "full page": {"statusCheckRollup": [success] * 100}}
        for name, changes in cases.items():
            with self.subTest(case=name):
                response = {"headRefOid": self.head, "baseRefOid": self.base, "statusCheckRollup": [success]}
                response.update(changes)
                self.backend.gh = Mock(return_value=json.dumps(response))
                with self.assertRaises(RuntimeError):
                    self.backend.wait_checks(42, self.head, self.base)

    @patch.object(release.subprocess, "run")
    def test_branch_create_is_atomic_even_if_another_writer_races(self, subprocess_run):
        subprocess_run.return_value.returncode = 2
        self.backend.update_prs = Mock(side_effect=[[], [{"head": {"ref": release.PREFIX + "d" * 12}, "number": 42}]])
        self.backend.run = Mock()
        self.backend.git = Mock(return_value=self.head)
        self.backend.confirm_base = Mock()
        self.backend.open_pr(self.base)
        push = next(c.args for c in self.backend.run.call_args_list if c.args[:2] == ("git", "push"))
        self.assertIn("--force-with-lease=refs/heads/" + release.PREFIX + "d" * 12 + ":", push)

    def test_changed_merge_base_never_passes_verification(self):
        self.backend.remote_dev = Mock(return_value=self.commit)
        self.backend.git = Mock(return_value="e" * 40 + " " + self.head)
        with self.assertRaisesRegex(RuntimeError, "different base"):
            self.backend.confirm_merged(self.commit, self.base, self.head)


class LocalGitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="release-test-")
        self.addCleanup(self.tmp.cleanup)
        self.backend = release.Backend(self.tmp.name)
        self.backend.root.mkdir()
        self.git("init", "-q", "-b", "dev")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "user.name", "Test")

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.backend.root, check=True,
                              text=True, stdout=subprocess.PIPE).stdout.strip()

    def test_update_recognizes_real_unstaged_git_status(self):
        source = self.backend.root / "internal/control/modeltrace_detector.go"
        source.parent.mkdir(parents=True)
        source.write_text("// Data and SDK are pinned to upstream commit " + "a" * 40 + ".\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        source.write_text("// Data and SDK are pinned to upstream commit " + "b" * 40 + ".\n")
        real_run = self.backend.run
        def run(*args, **kwargs):
            if args[0] == "python3":
                return None
            return real_run(*args, **kwargs)
        self.backend.run = run
        self.assertTrue(self.backend.update())
        self.assertEqual(self.backend.upstream, "b" * 40)

    def test_resume_existing_matching_pr_without_push_or_rewrite(self):
        for relative in release.FILES:
            fixture = self.backend.root / relative
            fixture.parent.mkdir(parents=True, exist_ok=True)
            fixture.write_text("fixture\n")
        path = self.backend.root / "go.mod"
        path.write_text("base\n")
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        base = self.git("rev-parse", "HEAD")
        path.write_text("updated\n")
        self.git("add", ".")
        self.git("commit", "-qm", "existing update")
        head = self.git("rev-parse", "HEAD")
        branch = release.PREFIX + "d" * 12
        self.git("branch", branch)
        remote = Path(self.tmp.name) / "remote.git"
        subprocess.run(["git", "clone", "--bare", "-q", str(self.backend.root), str(remote)], check=True)
        subprocess.run(["git", "--git-dir", str(remote), "update-ref", "refs/heads/dev", base], check=True)
        self.git("remote", "add", "origin", str(remote))
        self.git("reset", "--hard", base)
        path.write_text("updated\n")
        self.backend.upstream = "d" * 40
        self.backend.update_prs = Mock(return_value=[{"number": 42, "head": {"ref": branch}}])
        pr = {"state": "OPEN", "isDraft": False, "baseRefName": "dev", "baseRefOid": base,
              "headRefName": branch, "headRefOid": head, "isCrossRepository": False,
              "headRepository": {"name": "gpt-load"}, "headRepositoryOwner": {"login": "DabengBa"},
              "author": {"login": "operator"}, "title": "Update lm-detector to " + "d" * 12,
              "body": release.pr_body("d" * 40)}
        self.backend.gh = Mock(side_effect=lambda *args: json.dumps({"login": "operator"} if args == ("api", "user") else pr))
        self.assertEqual(self.backend.open_pr(base), (42, head))
        self.assertEqual(self.git("rev-parse", "HEAD"), base)
        self.assertEqual(self.git("ls-remote", "origin", "refs/heads/" + branch).split()[0], head)
        for field, value in (("body", "not ours"), ("baseRefOid", "e" * 40),
                             ("author", {"login": "someone-else"}), ("headRefOid", "f" * 40)):
            with self.subTest(field=field):
                original = pr[field]
                pr[field] = value
                with self.assertRaises(RuntimeError):
                    self.backend.open_pr(base)
                pr[field] = original
        path.write_text("different content\n")
        with self.assertRaisesRegex(RuntimeError, "tree"):
            self.backend.open_pr(base)

    def test_deploy_uses_archived_commit_not_dirty_checkout(self):
        script = self.backend.root / "scripts/deploy.sh"
        script.parent.mkdir()
        script.write_text('#!/bin/sh\nprintf "%s %s %s" "$DEPLOY_COMMIT" "$DEPLOY_SHA" "$*" > "$RESULT"\n')
        self.git("add", ".")
        self.git("commit", "-qm", "base")
        commit = self.git("rev-parse", "HEAD")
        short = self.git("rev-parse", "--short", commit)
        script.write_text("exit 99\n")
        result = Path(self.tmp.name) / "result"
        with patch.dict(os.environ, {"RESULT": str(result), "DEPLOY_COMMIT": "wrong", "DEPLOY_SHA": "wrong"}):
            self.backend.deploy(commit, "--prepare-only", None)
        self.assertEqual(result.read_text(), f"{commit} {short} dev --prepare-only")

    def test_shell_entry_reaches_gate_then_real_archived_deploy_without_recursion(self):
        scripts = self.backend.root / "scripts"
        scripts.mkdir()
        source = Path(__file__).parent
        shutil.copy(source / "deploy.sh", scripts / "deploy.sh")
        shutil.copy(source / "release-with-lm-detector.py", scripts / "release-with-lm-detector.py")
        (scripts / "verify-built-image.py").write_text("# intercepted by python3 stub\n")
        self.git("add", ".")
        self.git("commit", "-qm", "target")
        commit = self.git("rev-parse", "HEAD")
        bindir = Path(self.tmp.name) / "bin"
        bindir.mkdir()
        stub = bindir / "stub"
        stub.write_text(f'''#!{sys.executable}
import importlib.util, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["EVENTS"], "a") as log:
    log.write(name + " " + " ".join(args) + "\\n")
if name == "python3" and args and args[0].endswith("release-with-lm-detector.py"):
    spec = importlib.util.spec_from_file_location("release_gate", args[0])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def base(self):
        self.root = pathlib.Path(os.environ["TARGET_REPO"])
        return os.environ["TARGET_COMMIT"]
    def update(self):
        with open(os.environ["EVENTS"], "a") as log: log.write("upstream checked\\n")
        return False
    def confirm(self, commit):
        assert commit == os.environ["TARGET_COMMIT"]
    module.Backend.base = base
    module.Backend.update = update
    module.Backend.confirm_base = confirm
    sys.argv = [args[0], *args[1:]]
    module.main()
elif name == "docker":
    if args[:2] == ["image", "inspect"]:
        if any(x.startswith("sha256:") for x in args): sys.exit(1)
        print("sha256:" + "c" * 64)
    elif "load" in args: sys.stdin.buffer.read()
    elif "save" in args: sys.stdout.buffer.write(b"image")
elif name == "ssh":
    if "uname -m" in args: print("x86_64")
    elif any(".Image" in x for x in args): print("sha256:" + "b" * 64)
    elif "docker save" in args: sys.stdout.buffer.write(b"image")
    elif "docker load" in args: sys.stdin.buffer.read()
''')
        stub.chmod(0o755)
        for name in ("python3", "docker", "ssh"):
            (bindir / name).symlink_to(stub)
        events = Path(self.tmp.name) / "events"
        env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
                   TARGET_REPO=str(self.backend.root), TARGET_COMMIT=commit,
                   EVENTS=str(events), DOCKER_HOST="unix:///var/run/docker.sock")
        for key in ("DOCKER_CONTEXT", "DEPLOY_COMMIT", "DEPLOY_SHA", "DEPLOY_GATE_FD"):
            env.pop(key, None)
        result = subprocess.run(["bash", str(scripts / "deploy.sh"), "dev", "--prepare-only"],
                                env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = events.read_text()
        self.assertEqual(calls.count("release-with-lm-detector.py"), 1, calls)
        self.assertLess(calls.index("upstream checked"), calls.index("docker build"))
        self.assertIn("verify-built-image.py", calls)
        self.assertIn("docker load", calls)
        self.assertNotIn("bash -s", calls)

if __name__ == "__main__":
    unittest.main()
