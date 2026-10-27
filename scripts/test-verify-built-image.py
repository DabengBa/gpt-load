"""Fast filesystem/SQLite and orchestration proofs; no daemon or SSH required."""
from contextlib import closing
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location("verification", Path(__file__).with_name("verify-built-image.py"))
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)

# The old image owns these tables; the fixture only adds data rows that use the
# columns the released schema already has.
OLD_SCHEMA = """
CREATE TABLE groups (id INTEGER PRIMARY KEY);
INSERT INTO groups VALUES (1);
CREATE TABLE credentials (id INTEGER PRIMARY KEY, data TEXT);
INSERT INTO credentials VALUES (1, 'cipher');
CREATE TABLE schema_migrations (id TEXT);
INSERT INTO schema_migrations VALUES ('0001_initial');
CREATE TABLE debug_captures (
  id varchar(64) PRIMARY KEY NOT NULL,
  request_id varchar(255) NOT NULL DEFAULT '',
  access_key_id integer NOT NULL DEFAULT 0,
  protocol varchar(32) NOT NULL DEFAULT '',
  operation varchar(64) NOT NULL DEFAULT '',
  metadata json,
  created_at_ms integer NOT NULL,
  expires_at_ms integer NOT NULL,
  state varchar(16) NOT NULL DEFAULT 'active'
    CHECK (state IN ('active','completed','failed','expired')),
  error text NOT NULL,
  terminal_at_ms integer
);
CREATE TABLE debug_capture_attempts (
  id varchar(64) PRIMARY KEY NOT NULL,
  capture_id varchar(64) NOT NULL REFERENCES debug_captures(id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  sequence integer NOT NULL CHECK (sequence > 0),
  metadata json,
  started_at_ms integer NOT NULL,
  completed_at_ms integer,
  state varchar(16) NOT NULL DEFAULT 'active'
    CHECK (state IN ('active','completed','failed')),
  error text NOT NULL
);
CREATE TABLE debug_capture_chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  capture_id varchar(64) NOT NULL REFERENCES debug_captures(id)
    ON DELETE CASCADE ON UPDATE CASCADE,
  attempt_id varchar(64) NOT NULL,
  part varchar(16) NOT NULL,
  direction varchar(16) NOT NULL,
  data blob NOT NULL,
  created_at_ms integer NOT NULL
);
"""

class VerificationTests(unittest.TestCase):
    def test_fixture_is_created_by_old_image_then_target_started_twice(self):
        events = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(verification, "start", side_effect=lambda image, work, name: events.append(("start", image, work))), \
             patch.object(verification, "stop", side_effect=lambda name: events.append(("stop",))), \
             patch.object(verification, "seed", side_effect=lambda name: events.append(("seed",))), \
             patch.object(verification, "check_api", side_effect=lambda name: events.append(("api",))), \
             patch.object(verification, "seed_capture", side_effect=lambda work: events.append(("capture-seed",))), \
             patch.object(verification, "check_capture", side_effect=lambda name: events.append(("capture",))), \
             patch.object(verification, "check_version", side_effect=lambda name, version: events.append(("version", version))), \
             patch.object(verification, "inspect", side_effect=lambda work: {"migration_ids": ["0001_initial"], "groups": 1, "credentials": [(1, "ciphertext")], "encoding_column": True, "chunks": [(1, verification.CAPTURE_ID, verification.ATTEMPT_ID, "body", "request", b"raw", "identity")]}):
            verification.scenario("target", "old", Path(directory), "isolated", "dev-abc1234",
                                  ["0001_initial"])
        self.assertEqual([e[:2] for e in events],
                         [("start", "old"), ("seed",), ("api",), ("stop",), ("capture-seed",),
                          ("start", "target"), ("api",), ("version", "dev-abc1234"), ("capture",), ("stop",),
                          ("start", "target"), ("api",), ("version", "dev-abc1234"), ("capture",), ("stop",)])
        self.assertEqual(sum(e[0] == "seed" for e in events), 1)
        self.assertEqual(sum(e[0] == "stop" for e in events), 3)
        self.assertEqual(sum(e[0] == "api" for e in events), 3)

    def test_served_version_must_be_the_expected_commit(self):
        served = lambda version: SimpleNamespace(
            stdout=json.dumps({"status": "ok", "version": version}).encode())
        with patch.object(verification.rehearsal, "command", return_value=served("dev-abc1234")):
            verification.check_version("target", "dev-abc1234")
        with patch.object(verification.rehearsal, "command", return_value=served("dev-0000000")):
            with self.assertRaisesRegex(ValueError, "dev-abc1234"):
                verification.check_version("target", "dev-abc1234")

    def test_failed_api_stops_and_does_not_continue(self):
        stops = []
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(verification, "start"), patch.object(verification, "seed"), \
             patch.object(verification, "stop", side_effect=stops.append), \
             patch.object(verification, "check_api", side_effect=ValueError("bad endpoint")):
            with self.assertRaisesRegex(ValueError, "bad endpoint"):
                verification.scenario("target", "old", Path(directory), "isolated", [])
        self.assertEqual(stops, ["isolated"])

    def test_ledger_must_match_fresh_and_preserve_old_entries_and_data(self):
        before = {"migration_ids": ["0001_initial"], "groups": 1, "credentials": [(1, "encrypted")]}
        after = {**before, "migration_ids": ["0001_initial", "0002_next"]}
        verification.check_database(before, after, after["migration_ids"])
        for broken in ({**after, "migration_ids": ["0001_initial"]},
                       {**after, "groups": 0}, {**after, "credentials": [(1, "changed")]},
                       {**after, "migration_ids": ["0002_next"]}):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                verification.check_database(before, broken, after["migration_ids"])

    def test_real_sqlite_fixture_checks_integrity_and_encrypted_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with closing(sqlite3.connect(work / "gpt-load.db")) as db:
                db.executescript("CREATE TABLE groups (id INTEGER PRIMARY KEY); INSERT INTO groups VALUES (1);"
                                 "CREATE TABLE credentials (id INTEGER PRIMARY KEY, data TEXT); INSERT INTO credentials VALUES (1, 'cipher');"
                                 "CREATE TABLE schema_migrations (id TEXT); INSERT INTO schema_migrations VALUES ('0001_initial');")
            result = verification.inspect(work)
            self.assertEqual(result["credentials"], [(1, "cipher")])
            self.assertEqual(result["migration_ids"], ["0001_initial"])

    def test_capture_fixture_is_data_only_and_survives_the_encoding_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with closing(sqlite3.connect(work / "gpt-load.db")) as db:
                db.executescript(OLD_SCHEMA)
            verification.seed_capture(work)
            before = verification.inspect(work)
            self.assertFalse(before["encoding_column"])
            # Several rows per part: the append order, not one row per part.
            self.assertGreater(len(before["chunks"]), len(verification.CAPTURE_PARTS))
            rebuilt = {}
            for _, _, _, part, direction, data, _ in before["chunks"]:
                rebuilt[(part, direction)] = rebuilt.get((part, direction), b"") + data
            self.assertEqual(rebuilt, {(part, direction): data
                                       for part, direction, data in verification.CAPTURE_PARTS})
            with closing(sqlite3.connect(work / "gpt-load.db")) as db:
                db.execute("ALTER TABLE debug_capture_chunks ADD COLUMN encoding VARCHAR(16) "
                           "NOT NULL DEFAULT 'identity'")
            after = verification.inspect(work)
            self.assertTrue(after["encoding_column"])
            self.assertEqual([c[6] for c in after["chunks"]], ["identity"] * len(before["chunks"]))
            verification.check_database(before, after, ["0001_initial"])

    def test_chunk_data_order_and_encoding_regressions_are_rejected(self):
        first = (1, "capture", "attempt", "body", "request", b"raw", "identity")
        second = (2, "capture", "attempt", "body", "request", b"tail", "identity")
        before = {"migration_ids": ["0001_initial"], "groups": 1, "credentials": [(1, "cipher")],
                  "encoding_column": False, "chunks": [first[:6], second[:6]]}
        after = {"migration_ids": ["0001_initial"], "groups": 1, "credentials": [(1, "cipher")],
                 "encoding_column": True, "chunks": [first, second]}
        verification.check_database(before, after, ["0001_initial"])
        for broken in ({**after, "chunks": [first[:5] + (b"change",) + first[6:], second]},
                       {**after, "chunks": [first[:6] + ("gzip",), second]},
                       {**after, "chunks": [second, first]},
                       {**after, "encoding_column": False},
                       {**after, "chunks": []}):
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                verification.check_database(before, broken, ["0001_initial"])

    def test_capture_export_must_reproduce_the_captured_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with closing(sqlite3.connect(work / "gpt-load.db")) as db:
                db.executescript(OLD_SCHEMA)
            verification.seed_capture(work)
        parts = verification.exported_parts()
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr("manifest.json", json.dumps({"id": verification.CAPTURE_ID}))
            for (direction, part), data in parts.items():
                bundle.writestr(f"attempts/{verification.ATTEMPT_ID}/parts/{direction}.{part}", data)
        verification.check_export(archive.getvalue())
        corrupt = io.BytesIO()
        with zipfile.ZipFile(corrupt, "w") as bundle:
            bundle.writestr("manifest.json", json.dumps({"id": verification.CAPTURE_ID}))
            for (direction, part), data in parts.items():
                payload = data + b"!" if (direction, part) == ("request", "body") else data
                bundle.writestr(f"attempts/{verification.ATTEMPT_ID}/parts/{direction}.{part}", payload)
        with self.assertRaisesRegex(ValueError, "differs"):
            verification.check_export(corrupt.getvalue())

    def test_capture_fixture_requires_the_released_debug_capture_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            with closing(sqlite3.connect(work / "gpt-load.db")) as db:
                db.executescript("CREATE TABLE groups (id INTEGER PRIMARY KEY);"
                                 "CREATE TABLE schema_migrations (id TEXT);")
            with self.assertRaisesRegex(ValueError, "debug capture"):
                verification.seed_capture(work)


if __name__ == "__main__":
    unittest.main()
