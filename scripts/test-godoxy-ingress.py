#!/usr/bin/env python3
"""Focused real-network integration tests for the GoDoxy ingress probe."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "godoxy-ingress-probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("godoxy_ingress_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClosingSocket:
    """Fake connected socket: serves one canned response, close() fails."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def settimeout(self, _timeout: float) -> None:
        return

    def sendall(self, _data: bytes) -> None:
        return

    def recv(self, _size: int) -> bytes:
        payload, self.payload = self.payload, b""
        return payload

    def close(self) -> None:
        raise OSError("simulated close failure")


class FailingSendSocket(ClosingSocket):
    def sendall(self, _data: bytes) -> None:
        raise OSError("simulated send failure")


class ProbeCleanupTests(unittest.TestCase):
    def test_close_failure_logs_warning_with_cleanup_context(self) -> None:
        probe = load_probe()
        args = probe.build_parser().parse_args(["request", "http://127.0.0.1:9/health"])
        fake = ClosingSocket(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
        with mock.patch.object(probe.socket, "create_connection", return_value=fake):
            with self.assertLogs(level="WARNING") as captured:
                result = probe.run_request(args)
        self.assertEqual(result["status"], 200)
        warning = "\n".join(captured.output)
        self.assertIn("close", warning)
        self.assertIn("simulated close failure", warning)

    def test_probe_error_survives_close_failure_and_still_logs_warning(self) -> None:
        probe = load_probe()
        args = probe.build_parser().parse_args(["request", "http://127.0.0.1:9/health"])
        fake = FailingSendSocket(b"")
        with mock.patch.object(probe.socket, "create_connection", return_value=fake):
            with self.assertLogs(level="WARNING") as captured:
                with self.assertRaises(probe.ProbeError) as caught:
                    probe.run_request(args)
        self.assertEqual(caught.exception.error_type, "write_error")
        warning = "\n".join(captured.output)
        self.assertIn("close", warning)
        self.assertIn("simulated close failure", warning)


class FixtureProcess:
    def __init__(self, *extra: str) -> None:
        self.proc = subprocess.Popen(
            [
                os.fspath(Path(os.sys.executable)),
                os.fspath(PROBE),
                "serve",
                "--port",
                "0",
                *extra,
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline()
        if not ready.startswith("READY "):
            self.close()
            raise AssertionError(f"fixture did not become ready: {ready!r}")
        self.ready = json.loads(ready.removeprefix("READY "))
        self.base_url = self.ready["base_url"]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)
        if self.proc.stdout is not None:
            self.proc.stdout.close()
        if self.proc.stderr is not None:
            self.proc.stderr.close()


class GoDoxyIngressProbeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = FixtureProcess()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.close()

    def run_client(
        self, *args: str, timeout: float = 5
    ) -> tuple[subprocess.CompletedProcess[str], dict]:
        result = subprocess.run(
            [os.fspath(Path(os.sys.executable)), os.fspath(PROBE), "request", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            self.fail(
                f"client did not emit JSON: stdout={result.stdout!r} stderr={result.stderr!r}: {exc}"
            )
        return result, payload

    def test_header_timeout_is_separate_from_successful_response(self) -> None:
        url = f"{self.fixture.base_url}/delay-headers?case=delay&delay=0.15"
        failed, failure = self.run_client(url, "--header-timeout", "0.05")
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual(failure["error"]["type"], "header_timeout")

        passed, response = self.run_client(
            url,
            "--header-timeout",
            "1",
            "--expect-status",
            "200",
            "--expect-header",
            "x-fixture-case=delay",
        )
        self.assertEqual(passed.returncode, 0, passed.stderr)
        self.assertEqual(response["response"]["status"], 200)
        self.assertEqual(response["response"]["body"], "delayed")

    def test_sse_events_are_received_before_eof(self) -> None:
        url = f"{self.fixture.base_url}/sse?case=timed&events=2&interval=0.05"
        result, payload = self.run_client(
            url,
            "--sse",
            "--expect-events",
            "2",
            "--expect-status",
            "200",
            "--read-timeout",
            "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        response = payload["response"]
        self.assertTrue(response["eof_observed"])
        self.assertTrue(response["events_before_eof"])
        self.assertEqual(len(response["events"]), 2)
        self.assertLess(
            response["header_received_at"], response["events"][0]["received_at"]
        )
        self.assertLess(response["events"][-1]["received_at"], response["eof_at"])

    def test_cancel_after_first_event_is_observed_by_backend(self) -> None:
        url = f"{self.fixture.base_url}/sse?case=cancel&events=3&interval=0.1"
        result, payload = self.run_client(
            url,
            "--sse",
            "--cancel-after-events",
            "1",
            "--expect-status",
            "200",
            "--read-timeout",
            "2",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(payload["response"]["cancelled_after_events"], 1)
        self.assertFalse(payload["response"]["eof_observed"])

        deadline = time.monotonic() + 2
        status = None
        while time.monotonic() < deadline:
            status_result, status_payload = self.run_client(
                f"{self.fixture.base_url}/status?case=cancel"
            )
            self.assertEqual(status_result.returncode, 0, status_result.stderr)
            status = json.loads(status_payload["response"]["body"])
            if status.get("disconnect_observed"):
                break
            time.sleep(0.05)
        self.assertIsNotNone(status)
        self.assertTrue(status["disconnect_observed"], status)
        self.assertGreaterEqual(status["events_sent"], 1)

    def test_non_2xx_preserves_status_headers_and_json_body(self) -> None:
        url = f"{self.fixture.base_url}/json-error?case=json&status=429"
        result, payload = self.run_client(
            url,
            "--expect-status",
            "429",
            "--expect-header",
            "x-fixture-error=synthetic",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        response = payload["response"]
        self.assertEqual(response["status"], 429)
        self.assertEqual(response["headers"]["content-type"], "application/json")
        self.assertEqual(json.loads(response["body"])["error"], "synthetic")

    def test_tls_requires_trusted_ca_and_matching_explicit_server_name(self) -> None:
        if shutil.which("openssl") is None:
            self.skipTest("openssl is required for the local TLS fixture")
        with tempfile.TemporaryDirectory(prefix="godoxy-ingress-tls-") as directory:
            cert = Path(directory) / "fixture.crt"
            key = Path(directory) / "fixture.key"
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    os.fspath(key),
                    "-out",
                    os.fspath(cert),
                    "-days",
                    "1",
                    "-subj",
                    "/CN=localhost",
                    "-addext",
                    "subjectAltName=DNS:localhost,IP:127.0.0.1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            fixture = FixtureProcess(
                "--tls-cert", os.fspath(cert), "--tls-key", os.fspath(key)
            )
            try:
                url = fixture.base_url.replace("http://", "https://", 1) + "/health"
                untrusted, untrusted_payload = self.run_client(url)
                self.assertNotEqual(untrusted.returncode, 0)
                self.assertEqual(
                    untrusted_payload["error"]["type"], "tls_verification_error"
                )

                trusted, trusted_payload = self.run_client(
                    url,
                    "--ca-file",
                    os.fspath(cert),
                    "--server-name",
                    "localhost",
                    "--expect-status",
                    "200",
                )
                self.assertEqual(trusted.returncode, 0, trusted.stderr)
                self.assertEqual(trusted_payload["response"]["status"], 200)

                wrong_name, wrong_payload = self.run_client(
                    url,
                    "--ca-file",
                    os.fspath(cert),
                    "--server-name",
                    "wrong.localhost",
                )
                self.assertNotEqual(wrong_name.returncode, 0)
                self.assertEqual(
                    wrong_payload["error"]["type"], "tls_verification_error"
                )
            finally:
                fixture.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
