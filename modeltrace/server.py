"""ModelTrace analysis service.

Two bearer-protected endpoints and one open health endpoint. GPT-Load executes
the challenges against a real route, then posts the model text here for
attribution. This service never calls upstream APIs, never holds credentials,
never enrolls models, and never writes request bodies to disk.
"""

from __future__ import annotations


import hmac
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .vendor import modeltrace as vendored
from .vendor.modeltrace.fingerprint import analyze_global_outputs, generate_challenges

VENDOR_DIR = Path(vendored.__file__).resolve().parent
BANK_PATH = VENDOR_DIR / "unified_bank.json"

CHALLENGE_COUNT = 3
MAX_BODY_BYTES = 256 * 1024
MAX_OUTPUT_CHARS = 32 * 1024
EXPECTED_COUNT_MIN = 292
EXPECTED_COUNT_MAX = 332
MIN_TOKEN_LENGTH = 16


class RequestError(Exception):
    def __init__(self, status: int, error: str, **details) -> None:
        super().__init__(error)
        self.status = status
        self.payload = {"error": error, **details}


class Bank:
    """Read-only fingerprint library loaded once at startup."""

    def __init__(self, path: Path = BANK_PATH) -> None:
        self.raw = path.read_bytes()

        self.data = json.loads(self.raw)
        if not self.data.get("models"):
            raise RuntimeError("fingerprint bank has no models")

    def summary(self) -> dict:
        families: dict[str, int] = {}
        for model in self.data["models"]:
            family = model.get("family") or "models"
            families[family] = families.get(family, 0) + 1
        return {
            "schema": self.data.get("schema"),
            "built_at": self.data.get("built_at"),
            "models": len(self.data["models"]),
            "families": families,
            "range": self.data.get("method", {}).get("range"),
            "method": self.data.get("method", {}).get("name"),
            "bank_sha256": vendored.BANK_SHA256,
            "upstream_repository": vendored.UPSTREAM_REPOSITORY,
            "upstream_commit": vendored.UPSTREAM_COMMIT,
        }


def validate_outputs(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        raise RequestError(400, "invalid_request", detail="body must be a JSON object")
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise RequestError(400, "invalid_outputs", detail="outputs must be a list")
    if not 1 <= len(outputs) <= CHALLENGE_COUNT:
        raise RequestError(
            400,
            "invalid_outputs",
            detail=f"outputs must contain 1 to {CHALLENGE_COUNT} items",
        )
    validated = []
    for index, item in enumerate(outputs):
        if not isinstance(item, dict):
            raise RequestError(400, "invalid_outputs", detail=f"outputs[{index}] must be an object")
        prompt_id = item.get("prompt_id")
        expected = item.get("expected_count")
        text = item.get("text")
        if not isinstance(prompt_id, str) or not prompt_id.strip() or len(prompt_id) > 128:
            raise RequestError(400, "invalid_outputs", detail=f"outputs[{index}].prompt_id invalid")
        if (
            isinstance(expected, bool)
            or not isinstance(expected, int)
            or not EXPECTED_COUNT_MIN <= expected <= EXPECTED_COUNT_MAX
        ):
            raise RequestError(
                400,
                "invalid_outputs",
                detail=f"outputs[{index}].expected_count must be an integer in "
                f"[{EXPECTED_COUNT_MIN}, {EXPECTED_COUNT_MAX}]",
            )
        if not isinstance(text, str) or len(text) > MAX_OUTPUT_CHARS:
            raise RequestError(
                400,
                "invalid_outputs",
                detail=f"outputs[{index}].text must be a string under {MAX_OUTPUT_CHARS} chars",
            )
        validated.append({"prompt_id": prompt_id, "expected_count": expected, "text": text})
    if len({item["prompt_id"] for item in validated}) != len(validated):
        raise RequestError(400, "invalid_outputs", detail="prompt_id must be unique")
    return validated


def analyze(bank: Bank, outputs: list[dict]) -> dict:
    try:
        result = analyze_global_outputs(outputs, bank.data)
    except ValueError as error:
        raise RequestError(
            422,
            "no_usable_outputs",
            detail=str(error),
            diagnostics=[
                {
                    "prompt_id": outputs[index]["prompt_id"],
                    "accepted": False,
                }
                for index in range(len(outputs))
            ],
        ) from error
    return {
        "prediction": result["prediction"],
        "prediction_name": result["prediction_name"],
        "probability": result["probability"],
        "family_prediction": result["family_prediction"],
        "family_prediction_name": result["family_prediction_name"],
        "family_probability": result["family_probability"],
        "used_outputs": result["used_outputs"],
        "method": result["method"],
        "calibration": result["calibration"],
        "candidates": [
            {
                "model": item["model"],
                "display_name": item["display_name"],
                "family": item["family"],
                "family_name": item["family_name"],
                "probability": item["probability"],
                "conditional_probability": item["conditional_probability"],
                "profile_similarity": item["profile_similarity"],
            }
            for item in result["results"]
        ],
        "families": result["family_probabilities"],
        "diagnostics": [
            {
                "prompt_id": outputs[item["index"]]["prompt_id"],
                "expected_count": outputs[item["index"]]["expected_count"],
                "parsed_numbers": item["parsed_numbers"],
                "minimum_numbers": item["minimum_numbers"],
                "accepted": item["accepted"],
            }
            for item in result["diagnostics"]
        ],
        "bank": {
            "models": len(bank.data["models"]),
            "built_at": bank.data.get("built_at"),
            "bank_sha256": vendored.BANK_SHA256,
        },
        "upstream": {
            "repository": vendored.UPSTREAM_REPOSITORY,
            "commit": vendored.UPSTREAM_COMMIT,
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = f"ModelTrace/{__version__}"
    sys_version = ""

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(
                200,
                {
                    "status": "ready",
                    "version": __version__,
                    "bank": self.server.bank.summary(),
                    "upstream": {
                        "repository": vendored.UPSTREAM_REPOSITORY,
                        "commit": vendored.UPSTREAM_COMMIT,
                    },
                },
            )
            return
        if self.path == "/api/challenges":
            if not self.authorized():
                return
            self.send_json(200, {"challenges": generate_challenges(CHALLENGE_COUNT)})
            return
        self.send_error_json(404, "not_found")

    def do_POST(self) -> None:
        if self.path != "/api/analyze":
            self.send_error_json(404, "not_found")
            return
        if not self.authorized():
            return
        try:
            payload = self.read_json()
            outputs = validate_outputs(payload)
            self.send_json(200, analyze(self.server.bank, outputs))
        except RequestError as error:
            self.send_json(error.status, error.payload)

    def authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        expected = self.server.token
        if scheme.lower() != "bearer" or not token or not hmac.compare_digest(
            token.strip().encode(), expected.encode()
        ):
            self.send_error_json(401, "unauthorized")
            return False
        return True

    def read_json(self) -> object:
        content_type = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type != "application/json":
            raise RequestError(415, "unsupported_media_type", detail="Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            raise RequestError(400, "invalid_request", detail="Content-Length required") from None
        if length < 0:
            raise RequestError(400, "invalid_request", detail="Content-Length invalid")
        if length > MAX_BODY_BYTES:
            raise RequestError(413, "payload_too_large", detail=f"body limit is {MAX_BODY_BYTES} bytes")
        body = self.rfile.read(length)
        if len(body) != length:
            raise RequestError(400, "invalid_request", detail="body shorter than Content-Length")
        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise RequestError(400, "invalid_json") from None

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(self, status: int, error: str) -> None:
        self.send_json(status, {"error": error})

    def log_message(self, format: str, *args) -> None:
        sys.stderr.write("%s %s\n" % (self.address_string(), format % args))


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, token: str, bank: Bank) -> None:
        self.token = token
        self.bank = bank
        super().__init__(address, Handler)


def build_server(host: str, port: int, token: str, bank: Bank | None = None) -> Server:
    if len(token) < MIN_TOKEN_LENGTH:
        raise ValueError(f"MODELTRACE_TOKEN must be at least {MIN_TOKEN_LENGTH} characters")
    return Server((host, port), token=token, bank=bank or Bank())


def main() -> int:
    token = os.environ.get("MODELTRACE_TOKEN", "")
    if len(token) < MIN_TOKEN_LENGTH:
        print(
            f"MODELTRACE_TOKEN is required and must be at least {MIN_TOKEN_LENGTH} characters",
            file=sys.stderr,
        )
        return 1
    host = os.environ.get("MODELTRACE_HOST", "127.0.0.1")
    port = int(os.environ.get("MODELTRACE_PORT", "8080"))
    bank = Bank()
    httpd = build_server(host, port, token, bank)
    summary = bank.summary()
    print(
        f"modeltrace {__version__} listening on {host}:{port} "
        f"({summary['models']} models, bank {summary['built_at']})",
        file=sys.stderr,
    )
    # shutdown() waits for serve_forever() to leave its loop, so invoke it from
    # a helper thread rather than from the signal handler itself.
    signal.signal(
        signal.SIGTERM,
        lambda *_: threading.Thread(target=httpd.shutdown, daemon=True).start(),
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
