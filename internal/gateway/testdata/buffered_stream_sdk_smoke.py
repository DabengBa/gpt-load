#!/usr/bin/env python3
"""Run official OpenAI and Anthropic SDKs against a loopback SSE fixture only.

The fixture is a local SDK compatibility proof. The gateway proof is kept
separate: point the SDK at a local gateway only after a deterministic upstream
fixture has been configured, and record that gateway output independently.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import Anthropic
from openai import APIError as OpenAIAPIError
from openai import OpenAI


CHAT_SCENARIOS = ("success", "incomplete", "error")
RESPONSES_SCENARIOS = ("success", "incomplete", "failed", "stream_error")
ANTHROPIC_SCENARIOS = ("success", "incomplete", "error")


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        scenario = self.headers.get("X-Scenario", "success")
        if self.path.endswith("/chat/completions"):
            allowed = CHAT_SCENARIOS
        elif self.path.endswith("/responses"):
            allowed = RESPONSES_SCENARIOS
        elif self.path.endswith("/messages"):
            allowed = ANTHROPIC_SCENARIOS
        else:
            allowed = ()
        if scenario not in allowed:
            self.send_error(400, "unknown local fixture scenario")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.end_headers()

        self.wire: list[str] = []
        self._send_raw(b": keep-alive\n\n")
        time.sleep(0.02)
        if self.path.endswith("/chat/completions"):
            self._send_chat(scenario)
        elif self.path.endswith("/responses"):
            self._send_responses(scenario)
        else:
            self._send_anthropic(scenario)
        self.server.records.append({
            "path": self.path,
            "scenario": scenario,
            "wire": "".join(self.wire),
        })

    def _send_raw(self, value: bytes) -> None:
        self.wire.append(value.decode("utf-8", errors="replace"))
        try:
            self.wfile.write(value)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _send_json_event(self, name: str | None, value: dict[str, Any]) -> None:
        prefix = f"event: {name}\n" if name else ""
        self._send_raw((prefix + "data: " + json.dumps(value) + "\n\n").encode())

    def _send_chat(self, scenario: str) -> None:
        if scenario == "error":
            self._send_json_event(None, {"error": {"type": "server_error", "message": "local stream error"}})
            return
        chunk = {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": 1800000000,
            "model": "local-fixture",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
        }
        self._send_json_event(None, chunk)
        if scenario == "success":
            chunk["choices"][0]["delta"] = {}
            chunk["choices"][0]["finish_reason"] = "stop"
            self._send_json_event(None, chunk)
            self._send_raw(b"data: [DONE]\n\n")

    def _send_responses(self, scenario: str) -> None:
        response = {"id": "resp-local", "object": "response", "status": "in_progress", "model": "local-fixture"}
        self._send_json_event("response.created", {"type": "response.created", "response": response})
        if scenario == "stream_error":
            self._send_json_event("error", {"type": "error", "error": {"code": "local_stream_error", "message": "local stream error"}})
            return
        if scenario == "failed":
            response["status"] = "failed"
            response["error"] = {"code": "local_failed", "message": "local failed stream"}
            self._send_json_event("response.failed", {"type": "response.failed", "response": response})
            return
        self._send_json_event(
            "response.output_text.delta",
            {"type": "response.output_text.delta", "delta": "ok", "item_id": "item-local", "output_index": 0, "content_index": 0},
        )
        response["status"] = "completed" if scenario == "success" else "incomplete"
        if scenario == "success":
            self._send_json_event("response.completed", {"type": "response.completed", "response": response})
        else:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
            self._send_json_event("response.incomplete", {"type": "response.incomplete", "response": response})

    def _send_anthropic(self, scenario: str) -> None:
        message = {
            "id": "msg-local",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "local-fixture",
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
        if scenario == "error":
            self._send_json_event("error", {"type": "error", "error": {"type": "api_error", "message": "local stream error"}})
            return
        self._send_json_event("message_start", {"type": "message_start", "message": message})
        self._send_json_event("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
        self._send_json_event("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}})
        if scenario == "success":
            self._send_json_event("content_block_stop", {"type": "content_block_stop", "index": 0})
            self._send_json_event("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}})
            self._send_json_event("message_stop", {"type": "message_stop"})


def _records_for(server: ThreadingHTTPServer, path: str, scenario: str) -> dict[str, str]:
    deadline = time.monotonic() + 1.0
    while True:
        matches = [record for record in server.records if record["path"] == path and record["scenario"] == scenario]
        if len(matches) == 1:
            return matches[0]
        if time.monotonic() >= deadline:
            raise AssertionError(f"fixture records for {path}/{scenario}: {matches!r}; all={server.records!r}")
        time.sleep(0.005)


def run_openai(base_url: str, server: ThreadingHTTPServer) -> list[dict[str, str]]:
    client = OpenAI(api_key="local-test-key", base_url=base_url, timeout=5.0)
    results: list[dict[str, str]] = []

    for scenario in CHAT_SCENARIOS:
        observed: list[str] = []
        try:
            stream = client.chat.completions.create(
                model="local-fixture",
                messages=[{"role": "user", "content": scenario}],
                stream=True,
                extra_headers={"X-Scenario": scenario},
            )
            for event in stream:
                for choice in getattr(event, "choices", []) or []:
                    finish_reason = getattr(choice, "finish_reason", None)
                    if finish_reason is not None:
                        observed.append("finish_reason:" + str(finish_reason))
        except OpenAIAPIError:
            if scenario != "error":
                raise
            record = _records_for(server, "/v1/chat/completions", scenario)
            if "data: {\"error\": {\"type\": \"server_error\"" not in record["wire"]:
                raise AssertionError(f"Chat error lacked explicit server_error fixture code: {record['wire']!r}")
            results.append({"sdk": "openai", "protocol": "chat", "scenario": scenario, "status": "error", "observed": "wire:data.error,APIError"})
            continue
        if scenario == "success":
            record = _records_for(server, "/v1/chat/completions", scenario)
            if "finish_reason:stop" not in observed or "data: [DONE]\n\n" not in record["wire"]:
                raise AssertionError(f"Chat success lacks stop/DONE evidence: observed={observed!r} wire={record['wire']!r}")
            status = "success"
        else:
            record = _records_for(server, "/v1/chat/completions", scenario)
            if observed or "data: [DONE]" in record["wire"]:
                raise AssertionError(f"Chat incomplete wire/SDK evidence is not incomplete: observed={observed!r} wire={record['wire']!r}")
            status = "incomplete"
        results.append({"sdk": "openai", "protocol": "chat", "scenario": scenario, "status": status, "observed": ",".join(observed)})

    for scenario in RESPONSES_SCENARIOS:
        observed: list[str] = []
        try:
            stream = client.responses.create(
                model="local-fixture",
                input=scenario,
                stream=True,
                extra_headers={"X-Scenario": scenario},
            )
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type:
                    observed.append(event_type)
                response = getattr(event, "response", None)
                status_value = getattr(response, "status", None)
                if status_value:
                    observed.append("status:" + str(status_value))
        except OpenAIAPIError:
            if scenario != "stream_error":
                raise
            record = _records_for(server, "/v1/responses", scenario)
            if "event: error\n" not in record["wire"] or "local_stream_error" not in record["wire"]:
                raise AssertionError(f"Responses stream error lacked explicit fixture code: {record['wire']!r}")
            results.append({"sdk": "openai", "protocol": "responses", "scenario": scenario, "status": "stream_error", "observed": ",".join(observed + ["wire:event:error", "APIError"])})
            continue
        expected = {
            "success": ("response.completed", "status:completed"),
            "incomplete": ("response.incomplete", "status:incomplete"),
            "failed": ("response.failed", "status:failed"),
        }[scenario]
        if any(value not in observed for value in expected):
            raise AssertionError(f"Responses {scenario} missing exact terminal evidence: {observed!r}")
        results.append({"sdk": "openai", "protocol": "responses", "scenario": scenario, "status": scenario, "observed": ",".join(observed)})

    return results


def run_anthropic(base_url: str, server: ThreadingHTTPServer) -> list[dict[str, str]]:
    client = Anthropic(api_key="local-test-key", base_url=base_url, timeout=5.0)
    results: list[dict[str, str]] = []
    for scenario in ANTHROPIC_SCENARIOS:
        observed: list[str] = []
        try:
            with client.messages.stream(
                model="local-fixture",
                max_tokens=16,
                messages=[{"role": "user", "content": scenario}],
                extra_headers={"X-Scenario": scenario},
            ) as stream:
                for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type:
                        observed.append(event_type)
        except AnthropicAPIStatusError:
            if scenario != "error":
                raise
            record = _records_for(server, "/v1/messages", scenario)
            if "event: error\n" not in record["wire"] or "api_error" not in record["wire"]:
                raise AssertionError(f"Anthropic error lacked explicit api_error fixture type: {record['wire']!r}")
            results.append({"sdk": "anthropic", "protocol": "messages", "scenario": scenario, "status": "error", "observed": "wire:event:error,APIStatusError"})
            continue
        record = _records_for(server, "/v1/messages", scenario)
        if scenario == "success":
            required = ("message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop")
            if any(f"event: {value}\n" not in record["wire"] for value in required) or "message_stop" not in observed:
                raise AssertionError(f"Anthropic success lacks complete lifecycle: observed={observed!r} wire={record['wire']!r}")
            status = "success"
        else:
            required = ("message_start", "content_block_start", "content_block_delta")
            if any(f"event: {value}\n" not in record["wire"] for value in required) or "message_stop" in observed or "message_stop" in record["wire"]:
                raise AssertionError(f"Anthropic incomplete lifecycle is not incomplete: observed={observed!r} wire={record['wire']!r}")
            status = "incomplete"
        results.append({"sdk": "anthropic", "protocol": "messages", "scenario": scenario, "status": status, "observed": ",".join(observed)})
    return results


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    server.records: list[dict[str, str]] = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        results = run_openai(base_url, server) + run_anthropic(base_url.removesuffix("/v1"), server)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    print(json.dumps(results, indent=2))
    expected = {
        ("openai", "chat", "success", "success"),
        ("openai", "chat", "incomplete", "incomplete"),
        ("openai", "chat", "error", "error"),
        ("openai", "responses", "success", "success"),
        ("openai", "responses", "incomplete", "incomplete"),
        ("openai", "responses", "failed", "failed"),
        ("openai", "responses", "stream_error", "stream_error"),
        ("anthropic", "messages", "success", "success"),
        ("anthropic", "messages", "incomplete", "incomplete"),
        ("anthropic", "messages", "error", "error"),
    }
    actual = {(item["sdk"], item["protocol"], item["scenario"], item["status"]) for item in results}
    if actual != expected or len(results) != len(expected):
        raise AssertionError(f"protocol/scenario matrix mismatch: actual={actual!r} expected={expected!r}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    raise SystemExit(main())
