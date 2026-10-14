#!/usr/bin/env python3
"""Clean /opt/gpt-load deployment artifacts with retention quotas.

Dry-run by default; pass --apply to delete. Fail-closed: if Docker mount
source inspection fails, abort without deleting anything.

Quotas: newest 1 rehearsal-*, newest 2 pre-upgrade.* dirs, newest 2
backups/gpt-load-offline-*.tar.gz. Entries younger than 24h are never
deleted. Never touches symlinks, Docker mount sources (running or stopped),
or paths opened/cwd'd by running processes (/proc scan).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import time

REHEARSAL_RE = re.compile(r"^rehearsal-.+$")
PRE_UPGRADE_RE = re.compile(r"^pre-upgrade\..+$")
OFFLINE_BACKUP_RE = re.compile(r"^gpt-load-offline-.+\.tar\.gz$")
MIN_AGE_SECONDS = 24 * 3600
KEEP = {"rehearsal": 1, "pre_upgrade": 2, "offline_backup": 2}


class ProtectionError(RuntimeError):
    pass


def docker_mount_sources():
    """Return host mount source paths of all containers (running+stopped).

    Fail-closed: any docker error raises ProtectionError.
    """
    try:
        ids = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}}"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout.split()
        mounts = set()
        for cid in ids:
            data = subprocess.run(
                ["docker", "inspect", cid],
                capture_output=True, text=True, check=True, timeout=30,
            ).stdout
            import json
            info = json.loads(data)[0]
            for m in info.get("Mounts", []):
                src = m.get("Source") or m.get("Name")
                if src:
                    mounts.add(os.path.abspath(src))
        return mounts
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as e:
        raise ProtectionError(f"docker inspection failed: {e}") from e


def proc_protected(_root):
    """Paths held open or used as cwd by any process, plus their prefixes."""
    protected = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        for sub in ("cwd", "root"):
            try:
                protected.add(os.path.realpath(f"/proc/{pid}/{sub}"))
            except OSError:
                pass
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                protected.add(os.path.realpath(os.path.join(fd_dir, fd)))
            except OSError:
                pass
    return protected


def is_protected(path, protected_roots):
    real = os.path.realpath(path)
    if os.path.islink(path):
        return True
    for root in protected_roots:
        if root == os.sep:
            continue  # generic process root protects nothing specifically
        # ancestor: candidate contains a protected path (e.g. open file inside)
        if real == root or real.startswith(root + os.sep):
            return True
        # descendant: protected path contains the candidate (mount child, cwd)
        if root == real or root.startswith(real + os.sep):
            return True
    return False


def newest_mtime(path):
    """Newest mtime of files within path's tree (directories excluded,
    since dir mtimes change whenever any file lands inside)."""
    newest = None
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in filenames:
            try:
                m = os.lstat(os.path.join(dirpath, name)).st_mtime
                newest = m if newest is None else max(newest, m)
            except OSError:
                continue
    if newest is None:
        newest = os.lstat(path).st_mtime
    return newest


def collect_candidates(root, now, docker_mounts, protected):
    """Return (dirs, files) to delete, sorted oldest-first, applying quotas."""
    candidates = []

    def ok(name, path):
        if is_protected(path, docker_mounts | protected):
            return False
        return now - newest_mtime(path) >= MIN_AGE_SECONDS

    rehearsal_dirs = []
    pre_upgrade_dirs = []
    with os.scandir(root) as it:
        for entry in it:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if REHEARSAL_RE.match(entry.name) and ok(entry.name, entry.path):
                rehearsal_dirs.append((entry.path, entry.stat(follow_symlinks=False).st_mtime))
            elif PRE_UPGRADE_RE.match(entry.name) and ok(entry.name, entry.path):
                pre_upgrade_dirs.append((entry.path, entry.stat(follow_symlinks=False).st_mtime))

    backup_files = []
    backups_dir = os.path.join(root, "backups")
    if os.path.isdir(backups_dir) and not os.path.islink(backups_dir):
        with os.scandir(backups_dir) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and OFFLINE_BACKUP_RE.match(entry.name) \
                        and ok(entry.name, entry.path):
                    backup_files.append((entry.path, entry.stat(follow_symlinks=False).st_mtime))

    # Oldest-first so quota keep = newest N.
    dirs, files = [], []
    for items, keep in ((sorted(rehearsal_dirs, key=lambda x: x[1]), KEEP["rehearsal"]),
                        (sorted(pre_upgrade_dirs, key=lambda x: x[1]), KEEP["pre_upgrade"]),
                        (sorted(backup_files, key=lambda x: x[1]), KEEP["offline_backup"])):
        stale = items[:-keep] if keep else items
        for path, _ in stale:
            (files if path.endswith((".tar.gz",)) and os.path.isfile(path) else dirs).append(path)
    return sorted(dirs), sorted(files)


def run(root, apply, docker_mounts, protected, log=print):
    dirs, files = collect_candidates(root, time.time(), docker_mounts, protected)
    freed = 0
    for d in dirs:
        if apply:
            shutil.rmtree(d)
            log(f"deleted dir: {d}")
        else:
            log(f"would delete dir: {d}")
    for f in files:
        size = os.path.getsize(f)
        if apply:
            os.remove(f)
            log(f"deleted file: {f}")
        else:
            log(f"would delete file: {f}")
        freed += size
    return {"deleted_dirs": dirs, "deleted_files": files, "freed_bytes": freed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/opt/gpt-load")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete (default: dry run)")
    args = parser.parse_args()
    mounts = docker_mount_sources()  # fail-closed
    proc = proc_protected(args.root)
    result = run(args.root, args.apply, mounts, proc)
    print(f"dirs: {len(result['deleted_dirs'])}, "
          f"files: {len(result['deleted_files'])}, "
          f"freed: {result['freed_bytes'] / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
