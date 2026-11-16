#!/usr/bin/env python3
"""Sync lm-detector into canonical dev before deploying a pinned dev commit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

REPO = "DabengBa/gpt-load"
URL = "https://github.com/DabengBa/gpt-load.git"
PREFIX = "automation/lm-detector-"
FILES = ("go.mod", "go.sum", "THIRD_PARTY_NOTICES.md",
         "internal/control/modeltrace_detector.go",
         "internal/control/lmdata/unified_bank.json",
         "internal/control/lmdata/shared_detector.bin")
SHA = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_CHECKS = {"deployment-tests"}  # .github/workflows/deployment-tests.yml, PRs to dev


def pr_body(upstream):
    # Keep the original body so a PR created before an interrupted run is resumable.
    return (f"Updates the SDK and fingerprint data to https://github.com/Ikaleio/lm-detector/commit/{upstream}. "
            "Validated by the release entry point.")


def run_release(backend, mode, expected):
    base = backend.base()
    changed = backend.update()
    if changed:
        if mode == "--activate-only":
            raise RuntimeError("upstream changed; prepare a new release before activation")
        backend.validate()
        backend.confirm_base(base)
        number, head = backend.open_pr(base)
        commit = backend.merge(number, head, base)
        backend.confirm_merged(commit, base, head)
        backend.confirm_base(commit)
    else:
        backend.confirm_base(base)
        commit = base
    if mode == "--activate-only" and expected != commit:
        raise RuntimeError("prepared commit differs from current dev; prepare again")
    backend.deploy(commit, mode, expected)


class Backend:
    def __init__(self, directory):
        self.root = Path(directory) / "repo"
        self.archive = Path(directory) / "release"

    def run(self, *args, capture=False, cwd=None, env=None, timeout=None, pass_fds=()):
        return subprocess.run(args, cwd=cwd or self.root, env=env, check=True,
                              text=True, timeout=timeout, pass_fds=pass_fds,
                              stdout=subprocess.PIPE if capture else None).stdout

    def git(self, *args):
        return self.run("git", *args, capture=True).strip()

    def gh(self, *args):
        return self.run("gh", *args, capture=True, timeout=60).strip()

    def base(self):
        self.run("git", "clone", "--quiet", "--single-branch", "--branch", "dev",
                 "--no-tags", URL, str(self.root), cwd=self.root.parent)
        return self.git("rev-parse", "HEAD")

    def update(self):
        self.run("python3", "scripts/test_update_lm_detector.py")
        self.run("python3", "scripts/update-lm-detector.py")
        status = self.run("git", "status", "--porcelain=v1", "--untracked-files=all", capture=True).rstrip("\n")
        if not status:
            return False
        for line in status.splitlines():
            if line[3:] not in FILES or line[:2] not in (" M", "??"):
                raise RuntimeError(f"unexpected update file/status: {line}")
        source = (self.root / "internal/control/modeltrace_detector.go").read_text()
        matches = re.findall(r"Data and SDK are pinned to upstream commit ([0-9a-f]{40})\.", source)
        if len(matches) != 1:
            raise RuntimeError("cannot identify unique updated upstream commit")
        self.upstream = matches[0]
        return True

    def validate(self):
        for args in (("go", "mod", "tidy", "-diff"), ("go", "vet", "./..."),
                     ("go", "test", "-count=1", "./internal/control", "./internal/requestlog"),
                     ("go", "build", "-o", str(self.root.parent / "gpt-load"), "."),
                     ("git", "diff", "--check")):
            self.run(*args)

    def remote_dev(self):
        self.git("fetch", "--quiet", "origin", "refs/heads/dev")
        return self.git("rev-parse", "FETCH_HEAD")

    def confirm_base(self, base):
        if self.remote_dev() != base:
            raise RuntimeError("remote dev changed; restart release")

    def update_prs(self):
        pages = json.loads(self.gh("api", "--paginate", "--slurp",
                                   f"repos/{REPO}/pulls?state=open&per_page=100"))
        return [pr for page in pages for pr in page
                if pr["head"]["ref"].startswith(PREFIX)]

    def open_pr(self, base):
        branch = PREFIX + self.upstream[:12]
        prs = self.update_prs()
        if prs:
            if len(prs) != 1 or prs[0].get("head", {}).get("ref") != branch:
                raise RuntimeError("existing lm-detector PR; resolve it manually and restart")
            return self.resume_pr(prs[0]["number"], base, branch)
        probe = subprocess.run(["git", "ls-remote", "--exit-code", "--heads", "origin",
                                "refs/heads/" + branch], cwd=self.root, stdout=subprocess.DEVNULL)
        if probe.returncode == 0:
            raise RuntimeError(f"remote branch {branch} exists; resolve it manually")
        if probe.returncode != 2:
            raise RuntimeError("failed to check remote branch")
        self.run("git", "switch", "-c", branch)
        self.run("git", "add", "--", *FILES)
        self.run("git", "-c", "user.name=gpt-load release", "-c",
                 "user.email=gpt-load-release@users.noreply.github.com",
                 "commit", "-m", f"chore: update lm-detector to {self.upstream[:12]}")
        head = self.git("rev-parse", "HEAD")
        self.confirm_base(base)
        # Empty expected value is a create-only compare-and-swap, including
        # when a concurrent writer creates a fast-forwardable branch.
        self.run("git", "push", f"--force-with-lease=refs/heads/{branch}:",
                 "origin", f"HEAD:refs/heads/{branch}")
        print(f"Pushed {branch}; if a later step fails, inspect this branch and any PR manually.", flush=True)
        self.run("gh", "pr", "create", "--repo", REPO, "--base", "dev", "--head", branch,
                 "--title", f"Update lm-detector to {self.upstream[:12]}", "--body",
                 pr_body(self.upstream))
        prs = self.update_prs()
        if len(prs) != 1 or prs[0]["head"]["ref"] != branch:
            raise RuntimeError("concurrent lm-detector PR detected; resolve manually")
        return prs[0]["number"], head

    def resume_pr(self, number, base, branch):
        pr = json.loads(self.gh("pr", "view", str(number), "--repo", REPO, "--json",
            "state,isDraft,baseRefName,baseRefOid,headRefName,headRefOid,isCrossRepository,headRepository,headRepositoryOwner,author,title,body"))
        login = json.loads(self.gh("api", "user"))["login"]
        head = pr["headRefOid"]
        if (pr["state"] != "OPEN" or pr["isDraft"] or pr["baseRefName"] != "dev"
                or pr["baseRefOid"] != base or pr["headRefName"] != branch
                or not SHA.fullmatch(head) or pr["isCrossRepository"]
                or (pr["headRepositoryOwner"]["login"] + "/" + pr["headRepository"]["name"]).lower() != REPO.lower()
                or pr["author"]["login"].lower() != login.lower()
                or pr["title"] != f"Update lm-detector to {self.upstream[:12]}"
                or pr["body"] != pr_body(self.upstream)):
            raise RuntimeError("existing PR identity or content does not match; resolve manually")
        # Compare the independently regenerated and validated complete tree, not just paths.
        self.run("git", "add", "--", *FILES)
        expected_tree = self.git("write-tree")
        self.git("fetch", "--quiet", "origin", "refs/heads/" + branch)
        if self.git("rev-parse", "FETCH_HEAD") != head:
            raise RuntimeError("existing PR head changed; resolve manually")
        if self.git("show", "-s", "--format=%P", head).split() != [base]:
            raise RuntimeError("existing PR parent differs from validated base")
        if self.git("rev-parse", head + "^{tree}") != expected_tree:
            raise RuntimeError("existing PR tree differs from validated update")
        self.confirm_base(base)
        return number, head

    def wait_checks(self, number, head, base):
        deadline = time.monotonic() + 1800
        while True:
            pr = json.loads(self.gh("pr", "view", str(number), "--repo", REPO, "--json",
                                    "headRefOid,baseRefOid,statusCheckRollup"))
            if pr["headRefOid"] != head or pr["baseRefOid"] != base:
                raise RuntimeError("PR head or base changed while waiting for checks")
            checks = pr["statusCheckRollup"] or []
            # gh pr view queries only the first 100 contexts; a full page may
            # hide further failures. Refuse rather than assume it is complete.
            if len(checks) >= 100:
                raise RuntimeError("too many check contexts to verify without pagination")
            seen = set()
            pending = False
            for check in checks:
                kind = check["__typename"]
                if kind == "CheckRun":
                    seen.add(check["name"])
                    if check["status"] != "COMPLETED":
                        pending = True
                    elif check["conclusion"] != "SUCCESS":
                        raise RuntimeError(f"check {check['name']} did not succeed: {check['conclusion']}")
                elif kind == "StatusContext":
                    if check["state"] in ("EXPECTED", "PENDING"):
                        pending = True
                    elif check["state"] != "SUCCESS":
                        raise RuntimeError(f"check {check['context']} did not succeed: {check['state']}")
                else:
                    raise RuntimeError(f"unknown check type: {kind}")
            if not pending and EXPECTED_CHECKS <= seen:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for checks to register and succeed")
            time.sleep(10)

    def merge(self, number, head, base):
        pr = json.loads(self.gh("pr", "view", str(number), "--repo", REPO, "--json",
                                "state,isDraft,baseRefName,headRefName,headRefOid,isCrossRepository,headRepository,headRepositoryOwner"))
        if (pr["state"] != "OPEN" or pr["isDraft"] or pr["baseRefName"] != "dev"
                or pr["headRefName"] != PREFIX + self.upstream[:12]
                or pr["headRefOid"] != head or pr["isCrossRepository"]
                or (pr["headRepositoryOwner"]["login"] + "/" + pr["headRepository"]["name"]).lower() != REPO.lower()):
            raise RuntimeError("PR identity or head changed; resolve manually")
        self.wait_checks(number, head, base)
        self.confirm_base(base)
        # The REST endpoint does not silently enable auto-merge or enqueue a PR.
        merged = json.loads(self.gh("api", "--method", "PUT", f"repos/{REPO}/pulls/{number}/merge",
                                    "-f", f"sha={head}", "-f", "merge_method=merge"))
        if not merged.get("merged"):
            raise RuntimeError("GitHub did not merge the PR")
        pr = json.loads(self.gh("pr", "view", str(number), "--repo", REPO,
                                "--json", "state,mergeCommit"))
        commit = (pr.get("mergeCommit") or {}).get("oid", "")
        if pr["state"] != "MERGED" or not SHA.fullmatch(commit) or merged.get("sha") != commit:
            raise RuntimeError("PR not confirmed merged; do not deploy")
        return commit

    def confirm_merged(self, commit, base, head):
        if self.remote_dev() != commit:
            raise RuntimeError("dev changed after merge; restart release")
        parents = self.git("show", "-s", "--format=%P", commit).split()
        if parents != [base, head]:
            raise RuntimeError("merged PR used a different base or head; restart release")

    def deploy(self, commit, mode, expected):
        # Execute the exact checked commit's own deploy script, as deploy.sh does
        # after its fetch. Do not re-fetch a different tip between gate and build.
        self.archive.mkdir()
        with subprocess.Popen(["git", "archive", commit], cwd=self.root, stdout=subprocess.PIPE) as archive:
            try:
                subprocess.run(["tar", "-x", "-C", str(self.archive)], stdin=archive.stdout, check=True)
            finally:
                archive.stdout.close()
            if archive.wait() != 0:
                raise RuntimeError("git archive failed")
        env = os.environ.copy()
        env["DEPLOY_COMMIT"] = commit
        env["DEPLOY_SHA"] = self.git("rev-parse", "--short", commit)
        args = ["bash", str(self.archive / "scripts/deploy.sh"), "dev"]
        if mode != "all":
            args.append(mode)
        if expected:
            args.append(expected)
        print(f"Deploying verified dev commit {commit}", flush=True)
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, f"{commit} {env['DEPLOY_SHA']}\n".encode())
        finally:
            os.close(write_fd)
        try:
            env["DEPLOY_GATE_FD"] = str(read_fd)
            self.run(*args, cwd=self.archive, env=env, pass_fds=(read_fd,))
        finally:
            os.close(read_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare-only", action="store_true")
    group.add_argument("--activate-only", metavar="COMMIT")
    args = parser.parse_args()
    if args.activate_only and not SHA.fullmatch(args.activate_only):
        parser.error("--activate-only requires the full 40-character commit SHA")
    mode = "--prepare-only" if args.prepare_only else "--activate-only" if args.activate_only else "all"
    lock = Path(tempfile.gettempdir()) / f"gpt-load-release-{os.getuid()}.lock"
    fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "r+") as file:
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tempfile.TemporaryDirectory(prefix="gpt-load-release-") as directory:
            run_release(Backend(directory), mode, args.activate_only)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, subprocess.SubprocessError, ValueError, KeyError) as error:
        print(f"Release stopped: {error}", file=sys.stderr)
        sys.exit(1)
