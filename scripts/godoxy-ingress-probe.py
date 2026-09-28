#!/usr/bin/env python3
"""Loopback fixture backend and stdlib HTTP/SSE/TLS probe client."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import ipaddress
import json
import signal
import socket
import ssl
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


MAX_HEADER_BYTES = 64 * 1024
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024


class ProbeError(Exception):
    def __init__(self, error_type: str, message: str, response: dict | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.response = response


class FixtureState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cases: dict[str, dict] = {}

    def begin_stream(self, case: str) -> None:
        with self._lock:
            self._cases[case] = {
                "case": case,
                "started_at": time.time(),
                "headers_sent_at": None,
                "events_sent": 0,
                "event_log": [],
                "disconnect_observed": False,
                "disconnect_reason": None,
                "active": True,
                "finished_at": None,
            }

    def _case(self, case: str) -> dict:
        return self._cases.setdefault(
            case,
            {
                "case": case,
                "started_at": time.time(),
                "headers_sent_at": None,
                "events_sent": 0,
                "event_log": [],
                "disconnect_observed": False,
                "disconnect_reason": None,
                "active": True,
                "finished_at": None,
            },
        )

    def headers_sent(self, case: str) -> None:
        with self._lock:
            self._case(case)["headers_sent_at"] = time.time()

    def event_sent(self, case: str, index: int) -> None:
        with self._lock:
            current = self._case(case)
            current["events_sent"] = index
            current["event_log"].append({"index": index, "sent_at": time.time()})

    def disconnect(self, case: str, reason: str) -> None:
        with self._lock:
            current = self._case(case)
            current["disconnect_observed"] = True
            current["disconnect_reason"] = reason
            current["active"] = False
            current["finished_at"] = time.time()

    def finish(self, case: str) -> None:
        with self._lock:
            current = self._case(case)
            current["active"] = False
            current["finished_at"] = time.time()

    def snapshot(self, case: str) -> dict | None:
        with self._lock:
            current = self._cases.get(case)
            return None if current is None else json.loads(json.dumps(current))


class FixtureHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> FixtureState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlsplit(self.path)
        query = parse_qs(parsed.query, keep_blank_values=True)
        try:
            if parsed.path == "/delay-headers":
                self._delay_headers(query)
            elif parsed.path == "/sse":
                self._sse(query)
            elif parsed.path == "/status":
                self._status(query)
            elif parsed.path == "/json-error":
                self._json_error(query)
            elif parsed.path == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_text(404, "not found")
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError):
            self.close_connection = True

    def _value(self, query: dict[str, list[str]], name: str, default: str) -> str:
        value = query.get(name, [default])[0]
        if "\r" in value or "\n" in value:
            raise ValueError(f"invalid {name}")
        return value

    def _float_value(self, query: dict[str, list[str]], name: str, default: float, maximum: float) -> float:
        value = float(self._value(query, name, str(default)))
        if value < 0 or value > maximum:
            raise ValueError(f"{name} must be between 0 and {maximum}")
        return value

    def _int_value(self, query: dict[str, list[str]], name: str, default: int, minimum: int, maximum: int) -> int:
        value = int(self._value(query, name, str(default)))
        if value < minimum or value > maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return value

    def _delay_headers(self, query: dict[str, list[str]]) -> None:
        delay = self._float_value(query, "delay", 0, 3600)
        case = self._value(query, "case", "delay")
        time.sleep(delay)
        self._send_text(200, "delayed", {"X-Fixture-Case": case})

    def _sse(self, query: dict[str, list[str]]) -> None:
        case = self._value(query, "case", "sse")
        event_count = self._int_value(query, "events", 3, 1, 100)
        interval = self._float_value(query, "interval", 0.25, 60)
        self.state.begin_stream(case)
        disconnected = False
        self.close_connection = True
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Fixture-Case", case)
            self.end_headers()
            self.wfile.flush()
            self.state.headers_sent(case)
            for index in range(1, event_count + 1):
                if index > 1:
                    time.sleep(interval)
                payload = (
                    f"id: {index}\n"
                    "event: fixture\n"
                    f'data: {{"event": {index}}}\n\n'
                ).encode("utf-8")
                self.wfile.write(payload)
                self.wfile.flush()
                self.state.event_sent(case, index)
        except (BrokenPipeError, ConnectionResetError, ssl.SSLError, OSError) as exc:
            disconnected = True
            self.state.disconnect(case, type(exc).__name__)
        finally:
            if not disconnected:
                self.state.finish(case)

    def _status(self, query: dict[str, list[str]]) -> None:
        case = self._value(query, "case", "")
        if not case:
            self._send_json(400, {"error": "case is required"})
            return
        snapshot = self.state.snapshot(case)
        if snapshot is None:
            self._send_json(404, {"error": "unknown case", "case": case})
            return
        self._send_json(200, snapshot)

    def _json_error(self, query: dict[str, list[str]]) -> None:
        status = self._int_value(query, "status", 429, 400, 599)
        case = self._value(query, "case", "json-error")
        body = {"error": "synthetic", "case": case}
        self._send_json(status, body, {"X-Fixture-Error": "synthetic", "X-Fixture-Case": case})

    def _send_text(self, status: int, body: str, extra_headers: dict[str, str] | None = None) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()
        self.close_connection = True

    def _send_json(self, status: int, body: dict, extra_headers: dict[str, str] | None = None) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)
        self.wfile.flush()
        self.close_connection = True


class FixtureServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], tls_context: ssl.SSLContext | None) -> None:
        super().__init__(address, FixtureHandler)
        self.state = FixtureState()
        if tls_context is not None:
            self.socket = tls_context.wrap_socket(self.socket, server_side=True)


def require_loopback(bind: str) -> None:
    try:
        address = ipaddress.ip_address(bind)
    except ValueError as exc:
        raise ProbeError("invalid_bind", "fixture bind must be a literal loopback address") from exc
    if not address.is_loopback:
        raise ProbeError("invalid_bind", "fixture bind must be loopback")


def run_server(args: argparse.Namespace) -> int:
    require_loopback(args.bind)
    if bool(args.tls_cert) != bool(args.tls_key):
        raise ProbeError("invalid_tls_config", "--tls-cert and --tls-key must be supplied together")
    tls_context = None
    if args.tls_cert:
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.minimum_version = ssl.TLSVersion.TLSv1_2
        tls_context.load_cert_chain(args.tls_cert, args.tls_key)
    server = FixtureServer((args.bind, args.port), tls_context)
    scheme = "https" if tls_context is not None else "http"
    host = f"[{args.bind}]" if ":" in args.bind else args.bind
    ready = {
        "scheme": scheme,
        "bind": args.bind,
        "port": server.server_port,
        "base_url": f"{scheme}://{host}:{server.server_port}",
    }
    print(f"READY {json.dumps(ready, separators=(',', ':'))}", flush=True)

    def stop(_signum: int, _frame: object) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
    return 0


class SocketReader:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock
        self.buffer = bytearray()

    def _receive(self, deadline: float, error_type: str, allow_eof: bool = False) -> bool:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProbeError(error_type, "read deadline exceeded")
            self.sock.settimeout(min(remaining, 1.0))
            try:
                data = self.sock.recv(65536)
            except socket.timeout as exc:
                if time.monotonic() >= deadline:
                    raise ProbeError(error_type, "read deadline exceeded") from exc
                continue
            except ssl.SSLWantReadError:
                continue
            except ssl.SSLError as exc:
                raise ProbeError("tls_read_error", "TLS read failed") from exc
            except OSError as exc:
                raise ProbeError("read_error", "socket read failed") from exc
            if not data:
                if allow_eof:
                    return False
                raise ProbeError("connection_closed", "connection closed before response was complete")
            self.buffer.extend(data)
            return True

    def read_until(self, delimiter: bytes, deadline: float, error_type: str) -> bytes:
        while True:
            index = self.buffer.find(delimiter)
            if index >= 0:
                end = index + len(delimiter)
                result = bytes(self.buffer[:end])
                del self.buffer[:end]
                return result
            if len(self.buffer) > MAX_HEADER_BYTES:
                raise ProbeError("header_too_large", "response headers exceed the probe limit")
            self._receive(deadline, error_type)

    def read_exact(self, size: int, deadline: float, error_type: str) -> bytes:
        while len(self.buffer) < size:
            self._receive(deadline, error_type)
        result = bytes(self.buffer[:size])
        del self.buffer[:size]
        return result

    def read_some(self, deadline: float, error_type: str) -> bytes:
        if self.buffer:
            result = bytes(self.buffer)
            self.buffer.clear()
            return result
        if not self._receive(deadline, error_type, allow_eof=True):
            return b""
        result = bytes(self.buffer)
        self.buffer.clear()
        return result


class SseParser:
    def __init__(self) -> None:
        self._buffer = ""
        self._event: OrderedDict[str, list[str]] = OrderedDict()

    def feed(self, chunk: bytes) -> list[dict]:
        self._buffer += chunk.decode("utf-8", errors="replace")
        events: list[dict] = []
        while True:
            separator = self._buffer.find("\n")
            if separator < 0:
                break
            line = self._buffer[:separator]
            self._buffer = self._buffer[separator + 1 :]
            if line.endswith("\r"):
                line = line[:-1]
            if line == "":
                event = self._finish_event()
                if event is not None:
                    events.append(event)
                continue
            if line.startswith(":"):
                continue
            field, separator, value = line.partition(":")
            if not separator:
                value = ""
            elif value.startswith(" "):
                value = value[1:]
            self._event.setdefault(field, []).append(value)
        return events

    def _finish_event(self) -> dict | None:
        if not self._event:
            return None
        event = {
            "event": self._event.get("event", ["message"])[-1],
            "data": "\n".join(self._event.get("data", [])),
        }
        if "id" in self._event:
            event["id"] = self._event["id"][-1]
        self._event.clear()
        return event


def parse_target(url: str) -> tuple[str, int, str, str, str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise ProbeError("invalid_url", "URL scheme must be http or https")
    if parsed.username is not None or parsed.password is not None:
        raise ProbeError("invalid_url", "URL credentials are not accepted")
    if parsed.fragment:
        raise ProbeError("invalid_url", "URL fragments are not sent to the server")
    if parsed.hostname is None:
        raise ProbeError("invalid_url", "URL hostname is required")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ProbeError("invalid_url", "URL port is invalid") from exc
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    host = parsed.hostname
    host_header = f"[{host}]" if ":" in host else host
    if parsed.port is not None:
        host_header += f":{parsed.port}"
    return parsed.scheme, port, host, path, host_header


def parse_header_option(value: str) -> tuple[str, str]:
    name, separator, header_value = value.partition(":")
    if not separator or not name.strip() or "\r" in value or "\n" in value:
        raise ProbeError("invalid_header", "--header must be NAME:VALUE")
    return name.strip(), header_value.strip()


def parse_expected_header(value: str) -> tuple[str, str]:
    name, separator, expected = value.partition("=")
    if not separator or not name.strip():
        raise ProbeError("invalid_expectation", "--expect-header must be NAME=VALUE")
    return name.strip().lower(), expected


def response_headers(block: bytes) -> tuple[int, str, dict[str, str]]:
    lines = block[:-4].split(b"\r\n")
    if not lines or not lines[0].startswith(b"HTTP/"):
        raise ProbeError("invalid_response", "invalid HTTP status line")
    status_parts = lines[0].decode("latin-1").split(" ", 2)
    if len(status_parts) < 2:
        raise ProbeError("invalid_response", "invalid HTTP status line")
    try:
        status = int(status_parts[1])
    except ValueError as exc:
        raise ProbeError("invalid_response", "invalid HTTP status code") from exc
    reason = status_parts[2] if len(status_parts) == 3 else ""
    headers: dict[str, list[str]] = {}
    for line in lines[1:]:
        if not line:
            continue
        try:
            raw_name, raw_value = line.split(b":", 1)
        except ValueError as exc:
            raise ProbeError("invalid_response", "invalid HTTP response header") from exc
        name = raw_name.decode("latin-1").strip().lower()
        value = raw_value.decode("latin-1").strip()
        headers.setdefault(name, []).append(value)
    collapsed: dict[str, str] = {}
    for name, values in headers.items():
        collapsed[name] = ", ".join(values)
    return status, reason, collapsed


def iter_body(reader: SocketReader, headers: dict[str, str], read_timeout: float):
    body_deadline = time.monotonic() + read_timeout
    transfer_encoding = headers.get("transfer-encoding", "").lower()
    if "chunked" in transfer_encoding:
        while True:
            line = reader.read_until(b"\r\n", body_deadline, "read_timeout")[:-2]
            try:
                size = int(line.split(b";", 1)[0].strip(), 16)
            except ValueError as exc:
                raise ProbeError("invalid_body", "invalid chunk size") from exc
            if size == 0:
                while True:
                    trailer = reader.read_until(b"\r\n", body_deadline, "read_timeout")
                    if trailer == b"\r\n":
                        return
            chunk = reader.read_exact(size, body_deadline, "read_timeout")
            terminator = reader.read_exact(2, body_deadline, "read_timeout")
            if terminator != b"\r\n":
                raise ProbeError("invalid_body", "invalid chunk terminator")
            yield chunk
        return

    if "content-length" in headers:
        try:
            remaining = int(headers["content-length"])
        except ValueError as exc:
            raise ProbeError("invalid_body", "invalid Content-Length") from exc
        if remaining < 0:
            raise ProbeError("invalid_body", "negative Content-Length")
        while remaining:
            chunk = reader.read_exact(min(65536, remaining), body_deadline, "read_timeout")
            remaining -= len(chunk)
            yield chunk
        return

    while True:
        chunk = reader.read_some(body_deadline, "read_timeout")
        if not chunk:
            return
        yield chunk


def make_tls_context(ca_file: str | None) -> ssl.SSLContext:
    try:
        context = ssl.create_default_context(cafile=ca_file)
    except (OSError, ssl.SSLError) as exc:
        raise ProbeError("tls_configuration_error", "unable to load the explicit CA file") from exc
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    return context


def run_request(args: argparse.Namespace) -> dict:
    scheme, port, hostname, path, default_host_header = parse_target(args.url)
    if args.ca_file and scheme != "https":
        raise ProbeError("invalid_tls_config", "--ca-file is only valid for https URLs")
    if args.server_name and scheme != "https":
        raise ProbeError("invalid_tls_config", "--server-name is only valid for https URLs")
    if args.header_timeout <= 0 or args.read_timeout <= 0 or args.connect_timeout <= 0:
        raise ProbeError("invalid_timeout", "timeouts must be positive")

    host_header = args.host_header or default_host_header
    if "\r" in host_header or "\n" in host_header or not host_header:
        raise ProbeError("invalid_header", "--host-header is invalid")
    extra_headers = [parse_header_option(value) for value in args.header]
    sock: socket.socket | ssl.SSLSocket | None = None
    response: dict | None = None
    try:
        try:
            sock = socket.create_connection((hostname, port), timeout=args.connect_timeout)
        except (OSError, socket.timeout) as exc:
            raise ProbeError("connect_error", "unable to connect to target") from exc
        if scheme == "https":
            context = make_tls_context(args.ca_file)
            server_name = args.server_name or hostname
            try:
                sock = context.wrap_socket(sock, server_hostname=server_name)
            except (ssl.SSLError, OSError) as exc:
                raise ProbeError("tls_verification_error", "TLS certificate or hostname verification failed") from exc
        request_headers = [
            f"GET {path} HTTP/1.1",
            f"Host: {host_header}",
            "User-Agent: godoxy-ingress-probe/1",
            f"Accept: {'text/event-stream' if args.sse else '*/*'}",
            "Connection: close",
        ]
        request_headers.extend(f"{name}: {value}" for name, value in extra_headers)
        request = ("\r\n".join(request_headers) + "\r\n\r\n").encode("latin-1")
        sock.settimeout(args.connect_timeout)
        try:
            sock.sendall(request)
        except (OSError, socket.timeout) as exc:
            raise ProbeError("write_error", "unable to send request") from exc

        reader = SocketReader(sock)
        try:
            header_block = reader.read_until(
                b"\r\n\r\n", time.monotonic() + args.header_timeout, "header_timeout"
            )
        except ProbeError as exc:
            raise exc
        header_received_at = time.time()
        status, reason, headers = response_headers(header_block)
        response = {
            "status": status,
            "reason": reason,
            "headers": headers,
            "header_received_at": header_received_at,
        }
        if args.expect_status is not None and status != args.expect_status:
            raise ProbeError(
                "assertion_failed",
                f"expected HTTP status {args.expect_status}, got {status}",
                response,
            )
        for expected in args.expect_header:
            name, expected_value = parse_expected_header(expected)
            if headers.get(name) != expected_value:
                raise ProbeError("assertion_failed", f"expected response header {name}", response)

        if args.sse:
            content_type = headers.get("content-type", "").lower()
            if "text/event-stream" not in content_type:
                raise ProbeError("assertion_failed", "response is not text/event-stream", response)
            parser = SseParser()
            events: list[dict] = []
            event_receive_times: list[float] = []
            eof_observed = False
            eof_at: float | None = None
            cancelled_after_events: int | None = None
            for chunk in iter_body(reader, headers, args.read_timeout):
                for event in parser.feed(chunk):
                    receive_time = time.time()
                    event["received_at"] = receive_time
                    event["elapsed_ms"] = round((receive_time - header_received_at) * 1000, 3)
                    events.append(event)
                    event_receive_times.append(receive_time)
                    if args.cancel_after_events and len(events) >= args.cancel_after_events:
                        cancelled_after_events = len(events)
                        break
                if cancelled_after_events is not None:
                    break
            else:
                eof_observed = True
                eof_at = time.time()
            if args.expect_events is not None and len(events) != args.expect_events:
                raise ProbeError(
                    "assertion_failed",
                    f"expected {args.expect_events} SSE events, got {len(events)}",
                    {**response, "events": events, "eof_observed": eof_observed},
                )
            if args.cancel_after_events and cancelled_after_events is None:
                raise ProbeError(
                    "assertion_failed",
                    "SSE stream ended before cancellation threshold",
                    {**response, "events": events, "eof_observed": eof_observed},
                )
            events_before_eof = (
                eof_observed and eof_at is not None and all(received < eof_at for received in event_receive_times)
            )
            if eof_observed and not events_before_eof:
                raise ProbeError(
                    "assertion_failed",
                    "SSE event was not observed before EOF",
                    {**response, "events": events, "eof_observed": eof_observed, "eof_at": eof_at},
                )
            response.update(
                {
                    "events": events,
                    "eof_observed": eof_observed,
                    "eof_at": eof_at,
                    "events_before_eof": events_before_eof if eof_observed else None,
                    "cancelled_after_events": cancelled_after_events,
                }
            )
        else:
            body = bytearray()
            eof_at = None
            for chunk in iter_body(reader, headers, args.read_timeout):
                body.extend(chunk)
                if len(body) > args.max_body_bytes:
                    raise ProbeError("body_too_large", "response body exceeds the probe limit", response)
            eof_at = time.time()
            body_text = bytes(body).decode("utf-8", errors="replace")
            response.update({"body": body_text, "body_bytes": len(body), "eof_observed": True, "eof_at": eof_at})
            if args.expect_body_contains and args.expect_body_contains not in body_text:
                raise ProbeError("assertion_failed", "expected response body text was not found", response)
        return response
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the loopback synthetic backend")
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=0)
    serve.add_argument("--tls-cert")
    serve.add_argument("--tls-key")

    request = subparsers.add_parser("request", help="probe one HTTP or HTTPS response")
    request.add_argument("url")
    request.add_argument("--header-timeout", type=float, default=60.0)
    request.add_argument("--connect-timeout", type=float, default=10.0)
    request.add_argument("--read-timeout", type=float, default=30.0)
    request.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    request.add_argument("--ca-file")
    request.add_argument("--server-name")
    request.add_argument("--host-header")
    request.add_argument("--header", action="append", default=[])
    request.add_argument("--expect-status", type=int)
    request.add_argument("--expect-header", action="append", default=[])
    request.add_argument("--expect-body-contains")
    request.add_argument("--sse", action="store_true")
    request.add_argument("--expect-events", type=int)
    request.add_argument("--cancel-after-events", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            if args.port < 0 or args.port > 65535:
                raise ProbeError("invalid_port", "port must be between 0 and 65535")
            return run_server(args)
        if args.expect_events is not None and args.expect_events < 1:
            raise ProbeError("invalid_expectation", "--expect-events must be positive")
        if args.cancel_after_events is not None and args.cancel_after_events < 1:
            raise ProbeError("invalid_expectation", "--cancel-after-events must be positive")
        response = run_request(args)
        print(json.dumps({"ok": True, "response": response}, separators=(",", ":")))
        return 0
    except ProbeError as exc:
        payload = {"ok": False, "error": {"type": exc.error_type, "message": str(exc)}}
        if exc.response is not None:
            payload["response"] = exc.response
        print(json.dumps(payload, separators=(",", ":")))
        return 1
    except (OSError, ValueError, ssl.SSLError) as exc:
        print(
            json.dumps(
                {"ok": False, "error": {"type": "configuration_error", "message": str(exc)}},
                separators=(",", ":"),
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
