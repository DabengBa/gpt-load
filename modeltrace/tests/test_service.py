"""Contract tests for the ModelTrace analysis service.

Surface: HTTP integration over a real server instance started in-process on a
loopback port (`ThreadingHTTPServer`). The scorer is the real vendored
fingerprint core, so analysis assertions are end-to-end for the service contract.

Lane: focused fast lane (`python -m unittest discover -s modeltrace/tests`).
No temporary directories or files are written; the service must not persist input.
"""

from __future__ import annotations

import json
import random
import threading
import unittest
import urllib.error
import urllib.request

from modeltrace import server

TOKEN = "test-token-0123456789abcdef"


class ServiceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = server.build_server(host="127.0.0.1", port=0, token=TOKEN)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.thread.join(timeout=5)

    def request(self, method, path, body=None, headers=None, content_type="application/json"):
        data = body if isinstance(body, bytes) else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None and content_type:
            req.add_header("Content-Type", content_type)
        for key, value in (headers or {}).items():
            req.add_header(key, value)
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b"{}")

    def auth(self, token=TOKEN):
        return {"Authorization": f"Bearer {token}"}

    def analyze(self, outputs, headers=None, **kwargs):
        return self.request("POST", "/api/analyze", {"outputs": outputs}, headers or self.auth(), **kwargs)


class HealthTests(ServiceTestCase):
    def test_health_is_unauthenticated_and_reports_ready_bank_summary(self):
        status, body = self.request("GET", "/health")

        self.assertEqual(status, 200)
        self.assertEqual(body["status"], "ready")
        self.assertEqual(body["bank"]["models"], 18)
        self.assertEqual(body["bank"]["bank_sha256"][:12], "f267d0513aa0")
        self.assertEqual(body["bank"]["built_at"], "2026-10-09T06:24:27.469312+00:00")
        self.assertEqual(body["upstream"]["commit"], "6cd4a8ff0f3f4e49d6d6969dd2fbadfa1569fde2")
        self.assertNotIn("prompts", json.dumps(body))


class ChallengeTests(ServiceTestCase):
    def test_challenges_require_bearer_token(self):
        status, body = self.request("GET", "/api/challenges")

        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_challenges_reject_wrong_token(self):
        status, body = self.request("GET", "/api/challenges", headers=self.auth("nope"))

        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_challenges_return_three_well_formed_items(self):
        status, body = self.request("GET", "/api/challenges", headers=self.auth())

        self.assertEqual(status, 200)
        self.assertEqual(len(body["challenges"]), 3)
        ids = set()
        for challenge in body["challenges"]:
            self.assertTrue(challenge["id"].startswith("probe-"))
            self.assertIsInstance(challenge["prompt"], str)
            self.assertTrue(292 <= challenge["expected_count"] <= 332)
            ids.add(challenge["id"])
        self.assertEqual(len(ids), 3)


class AnalyzeValidationTests(ServiceTestCase):
    def synthetic(self, expected_count):
        # Odd answer: raw numbers instead of model output. Proves plumbing only.
        rng = random.Random(7)
        return " ".join(str(rng.randint(1, 355)) for _ in range(expected_count))

    def test_analyze_requires_bearer_token(self):
        status, body = self.request("POST", "/api/analyze", {"outputs": []})

        self.assertEqual(status, 401)
        self.assertEqual(body["error"], "unauthorized")

    def test_analyze_rejects_non_json_content_type(self):
        status, body = self.analyze(
            [], content_type="text/plain"
        )

        self.assertEqual(status, 415)
        self.assertEqual(body["error"], "unsupported_media_type")

    def test_analyze_rejects_body_over_limit(self):
        oversized = b'{"outputs": [{"text": "' + b"1 " * (server.MAX_BODY_BYTES // 2 + 16) + b'"}]}'
        status, body = self.request("POST", "/api/analyze", oversized, self.auth())

        self.assertEqual(status, 413)
        self.assertEqual(body["error"], "payload_too_large")

    def test_analyze_rejects_wrong_output_count(self):
        single = {"prompt_id": "probe-1", "expected_count": 300, "text": self.synthetic(300)}
        too_many = [dict(single, prompt_id=f"probe-{index}") for index in range(1, 5)]
        for outputs in ([], too_many):
            with self.subTest(count=len(outputs)):
                status, body = self.analyze(outputs)
                self.assertEqual(status, 400)
                self.assertEqual(body["error"], "invalid_outputs")

    def test_analyze_rejects_duplicate_prompt_ids(self):
        single = {"prompt_id": "probe-1", "expected_count": 300, "text": self.synthetic(300)}
        status, body = self.analyze([single, single])

        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_outputs")

    def test_analyze_rejects_malformed_item(self):
        for item in (
            {"prompt_id": "p1", "expected_count": 300, "text": 42},
            {"prompt_id": "", "expected_count": 300, "text": "1"},
            {"prompt_id": "p1", "expected_count": 300},
            {"expected_count": 300, "text": "1"},
            {"prompt_id": "p1", "expected_count": 300, "text": "1" * (server.MAX_OUTPUT_CHARS + 1)},
            "not-an-object",
        ):
            with self.subTest(item=str(item)[:40]):
                status, body = self.analyze([item])
                self.assertEqual(status, 400)
                self.assertEqual(body["error"], "invalid_outputs")

    def test_analyze_rejects_expected_count_out_of_range(self):
        for expected_count in (1, 291, 333, 300.0, True):
            with self.subTest(expected_count=expected_count):
                status, body = self.analyze(
                    [{"prompt_id": "p1", "expected_count": expected_count, "text": self.synthetic(5)}]
                )
                self.assertEqual(status, 400)
                self.assertEqual(body["error"], "invalid_outputs")

    def test_analyze_rejects_corrupt_json(self):
        status, body = self.request("POST", "/api/analyze", b'{"outputs": [', self.auth())

        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "invalid_json")

    def test_analyze_reports_unusable_when_all_outputs_rejected(self):
        status, body = self.analyze(
            [{"prompt_id": "p1", "expected_count": 300, "text": "I cannot help with that."}]
        )

        self.assertEqual(status, 422)
        self.assertEqual(body["error"], "no_usable_outputs")
        self.assertEqual(len(body["diagnostics"]), 1)


class AnalyzeSuccessTests(ServiceTestCase):
    def test_analyze_returns_attribution_without_echoing_input(self):
        """Analysis accepts the complete set of GPT-Load-executed outputs."""
        rng = random.Random(11)
        texts = [" ".join(str(rng.randint(1, 355)) for _ in range(300)) for _ in range(3)]
        outputs = [
            {"prompt_id": f"probe-{index}", "expected_count": 300, "text": text}
            for index, text in enumerate(texts, start=1)
        ]

        status, body = self.analyze(outputs)

        self.assertEqual(status, 200)
        self.assertEqual(body["used_outputs"], 3)
        self.assertEqual(body["calibration"]["queries"], "3")
        self.assertEqual(body["bank"]["models"], 18)
        self.assertIn(body["prediction"], {item["model"] for item in body["candidates"]})
        self.assertAlmostEqual(sum(item["probability"] for item in body["candidates"]), 1.0, places=6)
        self.assertEqual(
            [item["probability"] for item in body["candidates"]],
            sorted((item["probability"] for item in body["candidates"]), reverse=True),
        )
        self.assertEqual(len(body["diagnostics"]), 3)
        serialized = json.dumps(body)
        for text in texts:
            self.assertNotIn(text[:40], serialized)
        self.assertNotIn("prompt", serialized.replace("prompt_id", ""))


if __name__ == "__main__":
    unittest.main()
