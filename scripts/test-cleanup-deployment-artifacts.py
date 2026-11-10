#!/usr/bin/env python3
"""Integration tests for scripts/cleanup-deployment-artifacts.py.

Real temp filesystem fixtures; docker and /proc boundaries are stubbed via
injected callables, keeping the fast lane hermetic.
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "cleanup_deployment_artifacts",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cleanup-deployment-artifacts.py"
    ),
)
cda = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cda)


def touch(path, size=1024, age_days=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(size)
    old = time.time() - age_days * 86400
    os.utime(path, (old, old))


class Base(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="gptl-cleanup-test-")
        self.addCleanup(shutil.rmtree, self.root, True)

    def run_clean(
        self,
        apply_=False,
        docker_mounts=None,
        proc=None,
    ):
        return cda.run(
            root=self.root,
            apply=apply_,
            docker_mounts=set() if docker_mounts is None else docker_mounts,
            protected=set() if proc is None else proc,
            log=lambda *_: None,
        )


class RetentionPolicy(Base):
    def test_keeps_newest_one_rehearsal(self):
        touch(f"{self.root}/rehearsal-old/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-new/gpt-load.db", age_days=4)
        result = self.run_clean()
        self.assertEqual(result["deleted_dirs"], [f"{self.root}/rehearsal-old"])
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-new"))

    def test_keeps_one_pre_upgrade(self):
        touch(f"{self.root}/pre-upgrade.1/data.tar", age_days=5)
        touch(f"{self.root}/pre-upgrade.2/data.tar", age_days=4)
        touch(f"{self.root}/pre-upgrade.3/data.tar", age_days=3)
        result = self.run_clean(apply_=True)
        self.assertEqual(result["deleted_dirs"], [f"{self.root}/pre-upgrade.1", f"{self.root}/pre-upgrade.2"])
        self.assertFalse(os.path.exists(f"{self.root}/pre-upgrade.1"))
        self.assertFalse(os.path.exists(f"{self.root}/pre-upgrade.2"))
        self.assertTrue(os.path.isdir(f"{self.root}/pre-upgrade.3"))

    def test_keeps_two_offline_backups(self):
        touch(f"{self.root}/backups/gpt-load-offline-a.tar.gz", age_days=6)
        touch(f"{self.root}/backups/gpt-load-offline-b.tar.gz", age_days=5)
        touch(f"{self.root}/backups/gpt-load-offline-c.tar.gz", age_days=4)
        result = self.run_clean()
        self.assertEqual(
            result["deleted_files"],
            [f"{self.root}/backups/gpt-load-offline-a.tar.gz"],
        )


class YoungerThan24h(Base):
    def test_recent_dir_never_deleted(self):
        touch(f"{self.root}/rehearsal-today/gpt-load.db", age_days=0)
        result = self.run_clean()
        self.assertEqual(result["deleted_dirs"], [])
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-today"))

    def test_recent_backup_never_deleted_even_if_over_quota(self):
        touch(f"{self.root}/backups/gpt-load-offline-1.tar.gz", age_days=3)
        touch(f"{self.root}/backups/gpt-load-offline-2.tar.gz", age_days=2)
        touch(f"{self.root}/backups/gpt-load-offline-3.tar.gz", age_days=0)
        result = self.run_clean()
        self.assertEqual(result["deleted_files"], [])


class Protection(Base):
    def test_docker_mount_source_not_deleted(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        result = self.run_clean(docker_mounts={f"{self.root}/rehearsal-a"})
        self.assertEqual(result["deleted_dirs"], [])
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-a"))

    def test_symlink_never_deleted(self):
        touch(f"{self.root}/real/gpt-load.db", age_days=5)
        os.symlink(f"{self.root}/real", f"{self.root}/rehearsal-link")
        self.run_clean()
        self.assertTrue(os.path.islink(f"{self.root}/rehearsal-link"))
        self.assertTrue(os.path.isdir(f"{self.root}/real"))

    def test_proc_protected_not_deleted(self):
        touch(f"{self.root}/rehearsal-open/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-closed/gpt-load.db", age_days=4)
        result = self.run_clean(proc={f"{self.root}/rehearsal-open"})
        # open is protected; closed is the newest eligible and stays per quota
        self.assertEqual(result["deleted_dirs"], [])
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-open"))

    def test_cwd_protection_covers_subtree(self):
        touch(f"{self.root}/rehearsal-cwd/data/gpt-load.db", age_days=5)
        result = self.run_clean(proc={f"{self.root}/rehearsal-cwd"})
        self.assertEqual(result["deleted_dirs"], [])

    def test_open_file_inside_candidate_protects_candidate(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        result = self.run_clean(proc={f"{self.root}/rehearsal-a/gpt-load.db"})
        # a holds an open file → protected; b is newest eligible, kept by quota
        self.assertEqual(result["deleted_dirs"], [])
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-a"))
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-b"))

    def test_docker_mount_child_protects_candidate(self):
        touch(f"{self.root}/rehearsal-a/data/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        result = self.run_clean(docker_mounts={f"{self.root}/rehearsal-a/data"})
        self.assertEqual(result["deleted_dirs"], [])

    def test_generic_process_root_slash_not_protecting_everything(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        result = self.run_clean(proc={"/"})
        self.assertEqual(result["deleted_dirs"], [f"{self.root}/rehearsal-a"])

    def test_open_file_inside_backup_protects_file(self):
        touch(f"{self.root}/backups/gpt-load-offline-a.tar.gz", age_days=6)
        touch(f"{self.root}/backups/gpt-load-offline-b.tar.gz", age_days=5)
        touch(f"{self.root}/backups/gpt-load-offline-c.tar.gz", age_days=4)
        result = self.run_clean(proc={f"{self.root}/backups/gpt-load-offline-a.tar.gz"})
        self.assertEqual(result["deleted_files"], [])


class FailClosed(Base):
    def test_literal_deleted_suffix_in_cwd_is_protected(self):
        candidate = self.root + "/rehearsal-a (deleted)"
        touch(candidate + "/gpt-load.db", age_days=5)
        touch(self.root + "/rehearsal-b/gpt-load.db", age_days=4)
        with (
            patch.object(
                cda.os,
                "listdir",
                side_effect=lambda path: ["123"] if path == "/proc" else [],
            ),
            patch.object(
                cda.os,
                "readlink",
                side_effect=lambda path: candidate if path.endswith("/cwd") else "/",
            ),
        ):
            protected = cda.proc_protected(self.root)
        self.run_clean(apply_=True, proc=protected)
        self.assertTrue(os.path.isdir(candidate))

    def test_proc_open_and_deleted_paths_protect_candidates(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        with tempfile.TemporaryDirectory() as proc:
            os.makedirs(f"{proc}/123/fd")
            os.symlink("/", f"{proc}/123/root")
            os.symlink(self.root + "/rehearsal-a", f"{proc}/123/cwd")
            os.symlink(self.root + "/rehearsal-a/gone (deleted)", f"{proc}/123/fd/3")
            os.symlink("pipe:[123]", f"{proc}/123/fd/4")
            original_listdir, original_readlink = os.listdir, os.readlink

            def map_proc(path):
                return proc + path[5:] if str(path).startswith("/proc") else path

            with (
                patch.object(
                    cda.os,
                    "listdir",
                    side_effect=lambda p: original_listdir(map_proc(p)),
                ),
                patch.object(
                    cda.os,
                    "readlink",
                    side_effect=lambda p: original_readlink(map_proc(p)),
                ),
            ):
                protected = cda.proc_protected(self.root)
            self.assertIn(self.root + "/rehearsal-a/gone", protected)
            self.assertFalse(any("pipe:" in path for path in protected))
            result = self.run_clean(apply_=True, proc=protected)
            self.assertEqual(result["deleted_dirs"], [])
            self.assertTrue(os.path.exists(self.root + "/rehearsal-a"))

    def test_proc_fd_permission_denied_aborts(self):
        with (
            patch.object(
                cda.os,
                "listdir",
                side_effect=lambda path: (
                    ["123"]
                    if path == "/proc"
                    else (_ for _ in ()).throw(PermissionError(13, "denied"))
                ),
            ),
            patch.object(cda.os, "readlink", return_value="/"),
        ):
            with self.assertRaises(cda.ProtectionError):
                cda.proc_protected(self.root)

    def test_proc_permission_denied_aborts_before_deletion(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        with (
            patch.object(cda, "docker_mount_sources", return_value=set()),
            patch.object(sys, "argv", ["cleanup", "--root", self.root, "--apply"]),
            patch.object(
                cda.os,
                "listdir",
                side_effect=lambda path: ["123"] if path == "/proc" else [],
            ),
            patch.object(cda.os, "readlink", side_effect=PermissionError(13, "denied")),
        ):
            with self.assertRaises(cda.ProtectionError):
                cda.main()
        self.assertTrue(os.path.exists(f"{self.root}/rehearsal-a"))

    def test_vanished_process_is_ignored(self):
        with (
            patch.object(
                cda.os,
                "listdir",
                side_effect=lambda path: (
                    ["123"]
                    if path == "/proc"
                    else (_ for _ in ()).throw(FileNotFoundError(2, "gone"))
                ),
            ),
            patch.object(cda.os, "readlink", side_effect=FileNotFoundError(2, "gone")),
        ):
            self.assertEqual(cda.proc_protected(self.root), set())

    def test_unmatched_names_never_deleted(self):
        touch(f"{self.root}/random-dir/gpt-load.db", age_days=9)
        touch(f"{self.root}/backups/other.tar.gz", age_days=9)
        result = self.run_clean()
        self.assertEqual(result["deleted_dirs"], [])
        self.assertEqual(result["deleted_files"], [])
        self.assertTrue(os.path.isdir(f"{self.root}/random-dir"))

    def test_docker_error_aborts_fail_closed(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        original = cda.docker_mount_sources
        self.addCleanup(setattr, cda, "docker_mount_sources", original)
        cda.docker_mount_sources = _raise_protection_error
        with self.assertRaises(cda.ProtectionError):
            cda.docker_mount_sources()
        self.assertTrue(os.path.exists(f"{self.root}/rehearsal-a"))


def _raise_protection_error():
    raise cda.ProtectionError("docker inspect failed")


class DryRunDefault(Base):
    def test_dry_run_reports_but_deletes_nothing(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        result = self.run_clean(apply_=False)
        self.assertEqual(result["deleted_dirs"], [f"{self.root}/rehearsal-a"])
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-a"))

    def test_apply_deletes(self):
        touch(f"{self.root}/rehearsal-a/gpt-load.db", age_days=5)
        touch(f"{self.root}/rehearsal-b/gpt-load.db", age_days=4)
        self.run_clean(apply_=True)
        self.assertFalse(os.path.exists(f"{self.root}/rehearsal-a"))
        self.assertTrue(os.path.isdir(f"{self.root}/rehearsal-b"))


if __name__ == "__main__":
    unittest.main()
