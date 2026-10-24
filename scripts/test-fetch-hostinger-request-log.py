#!/usr/bin/env python3
"""Exercise the fetch helper's real analyzer with temporary JSON/ZIP fixtures.

Run with python3 scripts/test-fetch-hostinger-request-log.py. No SSH, service,
or dependencies are needed; TemporaryDirectory owns cleanup for each test.
"""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile


SCRIPT = Path(__file__).with_name("fetch-hostinger-request-log.sh")
ANALYZER = SCRIPT.read_text().split(
    'python3 - "$request_json" "$captures_json" "$output_dir" "$analysis" <<\'PY\'\n',
    1,
)[1].split("\nPY\n", 1)[0]


def attempt(sequence=1, outcome="eof", state="completed", error=""):
    events = [] if outcome is None else [
        {"kind": "response_termination", "outcome": outcome, "error": error}
    ]
    return {
        "id": f"attempt-{sequence}",
        "sequence": sequence,
        "state": state,
        "error": error,
        "metadata": {"events": events},
    }


def capture(attempts, state="completed", error="", capture_id="capture-1"):
    return {"id": capture_id, "state": state, "error": error, "attempts": attempts}


class CaptureEvidenceTests(unittest.TestCase):
    def analyze(self, captures, sequences=(1,), missing=(), overrides=None):
        overrides = overrides or {}
        with tempfile.TemporaryDirectory(prefix="fetch-capture-test-") as temporary:
            root = Path(temporary)
            (root / "captures").mkdir()
            request_path = root / "request.json"
            captures_path = root / "captures.json"
            report_path = root / "evidence.txt"
            request_path.write_text(json.dumps({
                "code": 0,
                "data": {
                    "request_id": "fixture-request",
                    "status": "error",
                    "attempts": [{"sequence": value} for value in sequences],
                },
            }))
            captures_path.write_text(json.dumps({"code": 0, "data": {"items": captures}}))
            for item in captures:
                with zipfile.ZipFile(root / "captures" / f"{item['id']}.zip", "w") as archive:
                    archive.writestr("manifest.json", json.dumps(item))
                    for record in item["attempts"]:
                        prefix = f"attempts/{record['id']}/"
                        parts = {
                            "metadata.json": json.dumps(record).encode(),
                            "parts/request.headers": b"Content-Type: application/json\r\n",
                            "parts/request.body": b'{"model":"fixture"}',
                            "parts/response.headers": b"HTTP 200\r\nContent-Type: text/event-stream\r\n",
                            "parts/response.body": b'event: message\r\ndata: {"partial":true}\r\n\r\n',
                        }
                        for name, content in parts.items():
                            key = (item["id"], record["id"], name)
                            if name not in missing and key not in missing:
                                archive.writestr(prefix + name, overrides.get(key, content))
            result = subprocess.run(
                [sys.executable, "-", str(request_path), str(captures_path),
                 str(root), str(report_path)],
                input=ANALYZER, text=True, capture_output=True, timeout=5,
            )
            self.assertTrue(report_path.exists(), result.stderr)
            return result.returncode, result.stdout, report_path.read_text()

    def assert_evidence(self, result, complete):
        code, stdout, report = result
        expected = "COMPLETE" if complete else "INCOMPLETE"
        self.assertEqual(code, 0 if complete else 3)
        self.assertIn(f"raw_evidence_status: {expected}\n", stdout)
        self.assertIn(f"raw_evidence_status: {expected}\n", report)
        self.assertNotIn('"partial":true', stdout + report)

    def test_completed_capture(self):
        self.assert_evidence(self.analyze([capture([attempt()])]), True)

    def test_no_response_with_recorded_transport_failure(self):
        item = capture([attempt(outcome="no_response", state="failed", error="dial failed")],
                       state="failed", error="dial failed")
        self.assert_evidence(self.analyze([item], missing=(
            "parts/response.headers", "parts/response.body",
        )), True)

    def test_stream_error_preserves_complete_observed_evidence(self):
        for outcome in ("read_error", "unexpected_eof", "timeout", "idle_timeout",
                        "cancel", "closed", "close", "transport_error"):
            with self.subTest(outcome=outcome):
                item = capture([attempt(outcome=outcome, state="failed", error="stream_error")],
                               state="failed", error="stream_error")
                self.assert_evidence(self.analyze([item]), True)

    def test_no_observation_source_with_partial_parts_is_incomplete(self):
        record = attempt(outcome="no_observation_source", state="failed", error="no completion source")
        item = capture([record], state="failed", error="no completion source")
        self.assert_evidence(self.analyze([item]), False)

    def test_valid_termination_does_not_hide_no_observation_source(self):
        record = attempt()
        record["metadata"]["events"].append({
            "kind": "response_termination", "outcome": "no_observation_source",
        })
        self.assert_evidence(self.analyze([capture([record])]), False)

    def test_missing_or_unknown_termination_is_incomplete(self):
        for outcome in (None, "", "session_limit_reached", "unknown"):
            with self.subTest(outcome=outcome):
                self.assert_evidence(self.analyze([capture([attempt(outcome=outcome)])]), False)

    def test_capture_failures_are_not_transport_failures(self):
        for error in ("session_limit_reached", "capture queue is full",
                      "capture body byte limit exceeded", "disk write failed", ""):
            for owner in ("session", "attempt", "archive"):
                with self.subTest(error=error, owner=owner):
                    item = capture([attempt()])
                    overrides = {}
                    if owner == "archive":
                        archived = copy.deepcopy(item["attempts"][0])
                        archived.update(state="failed", error=error)
                        overrides[(item["id"], archived["id"], "metadata.json")] = json.dumps(archived).encode()
                    else:
                        record = item if owner == "session" else item["attempts"][0]
                        record.update(state="failed", error=error)
                    self.assert_evidence(self.analyze([item], overrides=overrides), False)

    def test_nonterminal_or_expired_states_are_incomplete(self):
        for state in ("active", "expired", "unknown", None):
            for owner in ("session", "attempt", "archive"):
                with self.subTest(state=state, owner=owner):
                    item = capture([attempt()])
                    overrides = {}
                    if owner == "archive":
                        archived = copy.deepcopy(item["attempts"][0])
                        archived["state"] = state
                        overrides[(item["id"], archived["id"], "metadata.json")] = json.dumps(archived).encode()
                    else:
                        (item if owner == "session" else item["attempts"][0])["state"] = state
                    self.assert_evidence(self.analyze([item], overrides=overrides), False)

    def test_missing_raw_parts_or_metadata_are_incomplete(self):
        for part in ("metadata.json", "parts/request.headers", "parts/request.body",
                     "parts/response.headers", "parts/response.body"):
            with self.subTest(part=part):
                self.assert_evidence(self.analyze([capture([attempt()])], missing=(part,)), False)

    def test_empty_response_body_is_valid_with_observed_eof(self):
        self.assert_evidence(self.analyze([capture([attempt()])], overrides={
            ("capture-1", "attempt-1", "parts/response.body"): b"",
        }), True)

    def test_no_response_cannot_cover_orphan_response_bytes(self):
        item = capture([attempt(outcome="no_response", state="failed", error="dial failed")],
                       state="failed", error="dial failed")
        self.assert_evidence(self.analyze([item], missing=("parts/response.headers",)), False)

    def test_session_limit_without_capture_is_incomplete(self):
        self.assert_evidence(self.analyze([]), False)

    def test_all_attempts_are_required_even_after_successful_retry(self):
        records = [attempt(1, "read_error", "failed", "stream_error"), attempt(2)]
        item = capture(records, state="failed", error="stream_error")
        self.assert_evidence(self.analyze([item], sequences=(1, 2)), True)
        self.assert_evidence(self.analyze([item], sequences=(1, 2), missing=(
            ("capture-1", "attempt-1", "parts/response.body"),
        )), False)
        self.assert_evidence(self.analyze([capture([attempt(2)])], sequences=(1, 2)), False)

    def test_transport_error_propagated_to_later_attempt_is_not_capture_loss(self):
        # Gateway finishAttempt can propagate the session's earlier transport
        # failure even when a later attempt reaches EOF with all its bytes.
        later = attempt(2)
        later.update(state="failed", error="stream_error")
        item = capture([attempt(1, "read_error", "failed", "stream_error"), later],
                       state="failed", error="stream_error")
        self.assert_evidence(self.analyze([item], sequences=(1, 2)), True)

    def test_all_captures_are_required(self):
        items = [capture([attempt(1)]), capture([attempt(2)], capture_id="capture-2")]
        self.assert_evidence(self.analyze(items, sequences=(1, 2)), True)
        items[1]["attempts"][0]["metadata"]["events"] = []
        self.assert_evidence(self.analyze(items, sequences=(1, 2)), False)


if __name__ == "__main__":
    unittest.main()
