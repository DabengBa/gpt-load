#!/usr/bin/env python3
"""Offline regression tests; no Docker daemon or SSH connection is used."""
import importlib.util
import io
import hashlib
import json
import os
import pathlib
import tarfile
import tempfile
import unittest
from unittest.mock import patch, Mock

spec = importlib.util.spec_from_file_location(
    "local_rehearsal", pathlib.Path(__file__).with_name("local-rehearsal.py")
)
rehearsal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rehearsal)


class CredentialProofTest(unittest.TestCase):
    def test_reveal_requires_success_and_matching_credential(self):
        rehearsal.validate_reveal({"code": 0, "data": {
            "credential_id": 7, "credential": {"api_key": "secret"}}}, 7)
        for response in (
            {"code": 1, "data": {"credential_id": 7, "credential": {"api_key": "secret"}}},
            {"code": 0, "data": {"credential_id": 8, "credential": {"api_key": "secret"}}},
            {"code": 0, "data": {"credential_id": 7, "credential": {}}},
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                rehearsal.validate_reveal(response, 7)

    def test_reveal_requires_x_credential_id(self):
        commands = []
        def call(path, method="GET", credential_id=None):
            commands.append((path, method, credential_id))
            return {"code": 0, "data": {"credential_id": 7, "credential": {"api_key": "secret"}}}
        rehearsal.reveal(call, 12, 7)
        self.assertEqual(commands, [("/api/groups/12/credential/reveal", "POST", 7)])

    def test_verify_pages_and_reveals_every_credential(self):
        calls = []
        def api(name, path, method='GET', credential_id=None):
            calls.append((path, method, credential_id))
            if path.startswith('/api/groups?'):
                page = 1 if 'page=1&' in path else 2
                return {'code': 0, 'data': {'items': [{'id': page}],
                        'pagination': {'total_pages': 2, 'total_items': 2}}}
            if path.endswith('/reveal'):
                return {'code': 0, 'data': {'credential_id': credential_id, 'credential': {'api_key': 'hidden'}}}
            return {'code': 0, 'data': {'credential': {'credential_id': int(path.split('/')[3])}}}
        with patch.object(rehearsal, 'command', return_value=Mock(stdout=b'{"status":"ok","version":"test"}')), \
             patch.object(rehearsal, 'docker_api', side_effect=api):
            rehearsal.verify_container('isolated')
        self.assertIn(('/api/groups/1/credential/reveal', 'POST', 1), calls)
        self.assertIn(('/api/groups/2/credential/reveal', 'POST', 2), calls)


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def test_two_starts_same_work_and_cleanup_preserves_snapshot(self):
        events = []
        def start(work, name):
            events.append(("start", work, name))
        def stop(name):
            events.append(("stop", name))
        snapshot = self.root / "snapshot.db"
        snapshot.write_bytes(b"snapshot")
        work = self.root / "work"
        work.mkdir()
        (work / "gpt-load.db").write_bytes(b"work")
        rehearsal.two_starts(work, "rehearsal-test", start, lambda name: events.append(("verify", name)), stop)
        self.assertEqual([x[0] for x in events], ["start", "verify", "stop", "start", "verify", "stop"])
        self.assertEqual(events[0][1], events[3][1])
        self.assertEqual(snapshot.read_bytes(), b"snapshot")

    def test_failure_stops_container(self):
        events = []
        def fail(name):
            raise RuntimeError("verification failed")
        with self.assertRaises(RuntimeError):
            rehearsal.two_starts(self.root, "rehearsal-test", lambda *args: events.append("start"), fail,
                                 lambda *args: events.append("stop"))
        self.assertEqual(events, ["start", "stop"])

    def test_unsafe_root_rejected(self):
        link = self.root / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            rehearsal.check_root(link)

    def test_inspect_db_escapes_uri_path_and_rejects_empty_database(self):
        odd = self.root / 'data?#space'
        odd.mkdir()
        db = odd / 'valid.db'
        connection = rehearsal.sqlite3.connect(db)
        connection.executescript('CREATE TABLE groups(id INTEGER); CREATE TABLE schema_migrations(id TEXT);')
        connection.close()
        self.assertEqual(rehearsal.inspect_db(db)['groups'], 0)
        empty = self.root / 'empty.db'
        rehearsal.sqlite3.connect(empty).close()
        with self.assertRaises(ValueError):
            rehearsal.inspect_db(empty)

    def test_remote_size_uses_sqlite_logical_pages_not_main_file(self):
        self.assertIn('PRAGMA page_count', rehearsal.REMOTE)
        self.assertIn('PRAGMA page_size', rehearsal.REMOTE)

    def test_docker_isolation_flags(self):
        args = rehearsal.docker_start_args("gpt-load:version", self.root, "rehearsal-test")
        self.assertEqual(args[args.index("--network") + 1], "none")
        self.assertIn("MODELS_DEV_AUTO_SYNC_ENABLED=false", args)
        self.assertNotIn("-p", args)

    def test_remote_docker_daemon_is_rejected(self):
        with patch.dict(os.environ, {"DOCKER_HOST": "ssh://production"}):
            with self.assertRaises(ValueError): rehearsal.check_docker_local()

    def test_explicit_keys_are_used_without_exposing_them_in_arguments(self):
        work = self.root / "work"
        work.mkdir()
        (work / "auth.key").write_bytes(b"auth-value")
        (work / "encryption.key").write_bytes(b"explicit-password-not-hex")
        env = rehearsal.make_env_file(work)
        self.assertEqual(env.read_bytes(), b"AUTH_KEY=auth-value\nENCRYPTION_KEY=explicit-password-not-hex\n")
        args = rehearsal.docker_start_args("image", work, "test", env)
        self.assertIn("--env-file", args)
        self.assertNotIn("explicit-password-not-hex", str(args))

    def test_key_with_newline_rejected(self):
        (self.root / "auth.key").write_bytes(b"a\nb")
        (self.root / "encryption.key").write_bytes(b"secret")
        with self.assertRaises(ValueError):
            rehearsal.make_env_file(self.root)

    def test_archive_integrity_and_member_validation(self):
        db = b"sqlite snapshot"
        archive = self.root / "transfer.tar"
        work = self.root / "work"
        work.mkdir()
        with tarfile.open(archive, "w") as tar:
            for name, data in (("manifest.json", json.dumps({"sha256": hashlib.sha256(db).hexdigest(), "size": len(db)}).encode()),
                               ("auth.key", b"auth"), ("encryption.key", b"key"), ("snapshot.db", db)):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        snapshot = self.root / "snapshot.db"
        rehearsal.receive_archive(archive, snapshot, work)
        self.assertEqual(snapshot.read_bytes(), db)
        self.assertEqual(snapshot.stat().st_mode & 0o777, 0o400)
        with tarfile.open(archive, "a") as tar:
            info = tarfile.TarInfo("../escape")
            info.size = 1
            tar.addfile(info, io.BytesIO(b"x"))
        with self.assertRaises(ValueError):
            rehearsal.receive_archive(archive, self.root / "second", work)

    def test_failure_cleans_sensitive_files_after_stopping_container(self):
        events = []
        args = Mock(output=str(self.root), image="image", host="example", timeout=60)
        def command(cmd, **kwargs):
            if cmd[0] == "findmnt": return Mock(stdout="ext4\n")
            if cmd[0:3] == ["docker", "image", "inspect"]: return Mock(stdout="amd64\n")
            if cmd[0] == "ssh": return Mock(stdout="x86_64\n")
            raise AssertionError(cmd)
        def ssh(host, op, token, timeout, **kwargs):
            if op == "size": return Mock(stdout="1024")
            kwargs["stdout"].write(b"sensitive partial archive")
            raise TimeoutError("test")
        def stop(name):
            events.append("stop")
            self.assertTrue(list(self.root.glob("rehearsal-*/transfer.tar")))
        with patch.object(rehearsal.os, "geteuid", return_value=0), \
             patch.object(rehearsal, "check_root", return_value=self.root), \
             patch.object(rehearsal, "check_docker_local"), \
             patch.object(rehearsal, "command", side_effect=command), \
             patch.object(rehearsal, "ssh", side_effect=ssh), \
             patch.object(rehearsal, "stop_container", side_effect=stop):
            with self.assertRaises(TimeoutError): rehearsal.run(args)
        self.assertEqual(events, ["stop"])
        self.assertFalse(list(self.root.glob("rehearsal-*/transfer.tar")))
        self.assertFalse(list(self.root.glob("rehearsal-*/work")))
        self.assertFalse(list(self.root.glob(".active-*")))


if __name__ == "__main__":
    unittest.main()
