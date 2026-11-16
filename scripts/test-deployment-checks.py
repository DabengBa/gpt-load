#!/usr/bin/env python3
"""Offline command-boundary tests; temporary files belong to unittest mocks."""
import contextlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import io
import os
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("checks", Path(__file__).with_name("deployment-checks.py"))
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


class ChecksTests(unittest.TestCase):
    def test_admin_secret_is_supplied_on_stdin_not_child_arguments(self):
        with tempfile.TemporaryDirectory(prefix="deployment-auth-test-") as directory:
            root = Path(directory)
            key = "test-auth-secret-5f1a"
            (root / "auth.key").write_text(key + "\n")
            docker = root / "docker"
            docker.write_text('''#!%s
import os, subprocess, sys
from pathlib import Path
Path(os.environ['DOCKER_ARGS']).write_text(repr(sys.argv))
script = sys.argv[-1].replace('/app/data/auth.key', os.environ['AUTH_FILE'])
sys.exit(subprocess.run(['/bin/sh', '-c', script]).returncode)
''' % sys.executable)
            curl = root / "curl"
            curl.write_text('''#!%s
import json, os, sys
from pathlib import Path
Path(os.environ['CURL_ARGS']).write_text(repr(sys.argv))
Path(os.environ['CURL_INPUT']).write_bytes(sys.stdin.buffer.read())
print(json.dumps({'code': 0, 'data': {}}))
''' % sys.executable)
            for binary in (docker, curl):
                binary.chmod(0o755)
            env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH'],
                   'AUTH_FILE': str(root / 'auth.key'), 'DOCKER_ARGS': str(root / 'docker-args'),
                   'CURL_ARGS': str(root / 'curl-args'), 'CURL_INPUT': str(root / 'curl-input')}
            def run_remote(args):
                self.assertEqual(args[:len(checks.SSH)], checks.SSH)
                return json.loads(subprocess.run(['/bin/sh', '-c', args[-1]], env=env,
                    capture_output=True, text=True, check=True, timeout=10).stdout)
            with patch.object(checks, 'read_json', side_effect=run_remote):
                self.assertEqual(checks.api('/api/settings')['code'], 0)
            self.assertNotIn(key, (root / 'docker-args').read_text())
            self.assertNotIn(key, (root / 'curl-args').read_text())
            self.assertIn('Authorization: Bearer ' + key, (root / 'curl-input').read_text())

    @unittest.skipUnless(shutil.which('curl'), 'curl unavailable')
    def test_real_curl_reads_header_from_stdin_and_sends_post_body(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                received.append((self.headers.get('Authorization'), self.headers.get('X-Credential-ID'),
                                 self.rfile.read(int(self.headers['Content-Length']))))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"code":0,"data":{}}')
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            # Only the remote boundary is stubbed; curl and its @- parser are real.
            def run_remote(args):
                command = args[-1].replace('http://127.0.0.1:3001',
                                            f'http://127.0.0.1:{server.server_port}')
                command = command.replace('docker exec gpt-load sh -c', 'sh -c')
                with tempfile.TemporaryDirectory() as directory:
                    keyfile = Path(directory) / 'key'
                    keyfile.write_text('local-test-key\n')
                    command = command.replace('/app/data/auth.key', str(keyfile))
                    return json.loads(subprocess.run(['/bin/sh', '-c', command], capture_output=True,
                                                     text=True, check=True, timeout=10).stdout)
            with patch.object(checks, 'read_json', side_effect=run_remote):
                checks.api('/api/groups/1/credential/reveal', 'POST', 7)
            thread.join(timeout=3)
            self.assertEqual(received, [('Bearer local-test-key', '7', b'{}')])
        finally:
            server.server_close()

    def test_admin_pipeline_fails_if_either_child_fails(self):
        for failed in ('docker', 'curl'):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ('docker', 'curl'):
                    executable = root / name
                    executable.write_text('#!/bin/sh\n' + ('exit 1\n' if name == failed else
                                          ('printf "Authorization: Bearer test\\n"\n' if name == 'docker' else 'printf "{}"\n')))
                    executable.chmod(0o755)
                env = {**os.environ, 'PATH': str(root) + os.pathsep + os.environ['PATH']}
                def run_remote(args):
                    return json.loads(subprocess.run(['/bin/sh', '-c', args[-1]], env=env,
                        capture_output=True, text=True, check=True, timeout=10).stdout)
                with patch.object(checks, 'read_json', side_effect=run_remote):
                    with self.assertRaises(subprocess.CalledProcessError):
                        checks.api('/api/settings')

    def test_group_pagination_uses_created_order_and_checks_all_pages_twice(self):
        pages = []
        def api(path, method='GET', credential_id=None):
            if path.startswith('/api/groups?'):
                pages.append(path)
                page = 1 if 'page=1&' in path else 2
                items = [{'id': i} for i in (range(1, 101) if page == 1 else range(101, 103))]
                return {'code': 0, 'data': {'items': items, 'pagination': {'page': page, 'page_size': 100, 'total_items': 102, 'total_pages': 2}}}
            if path.endswith('/credential'):
                return {'code': 0, 'data': {'credential': {'credential_id': 7}}}
            if path.endswith('/reveal'):
                return {'code': 0, 'data': {'credential_id': 7, 'credential': {'api_key': 'secret'}}}
            return {'code': 0, 'data': {}}
        with patch.object(checks, 'api', side_effect=api):
            self.assertIn('groups=102 decrypted=102', checks.check_admin())
        self.assertEqual(len(pages), 4)
        self.assertTrue(all('sort=created' in page for page in pages))

    def test_duplicate_and_moving_pagination_fail_closed(self):
        for kind in ('duplicate', 'moved', 'changed_total'):
            with self.subTest(kind=kind):
                calls = []
                def api(path, method='GET', credential_id=None):
                    if path.startswith('/api/groups?'):
                        calls.append(path)
                        page = 1 if 'page=1&' in path else 2
                        ids = list(range(1, 101)) if page == 1 else [101, 102]
                        if kind == 'duplicate' and page == 2:
                            ids = [100, 102]
                        if kind == 'moved' and len(calls) > 2 and page == 2:
                            ids = [101, 103]
                        total = 103 if kind == 'changed_total' and page == 2 else 102
                        return {'code': 0, 'data': {'items': [{'id': i} for i in ids],
                                'pagination': {'page': page, 'page_size': 100, 'total_items': total, 'total_pages': 2}}}
                    if path.endswith('/credential'):
                        return {'code': 0, 'data': {'credential': {'credential_id': 7}}}
                    if path.endswith('/reveal'):
                        return {'code': 0, 'data': {'credential_id': 7, 'credential': {'api_key': 'secret'}}}
                    return {'code': 0, 'data': {}}
                with patch.object(checks, 'api', side_effect=api):
                    with self.assertRaises(ValueError):
                        checks.check_admin()

    def test_preflight_runs_all_three_concurrently_and_reports_each_failure(self):
        barrier = threading.Barrier(3, timeout=3)
        def task(name):
            barrier.wait()
            if name == "host":
                raise RuntimeError("sensitive output")
            return "ok"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ok = checks.run_checks({name: lambda n=name: task(n) for name in ("git", "host", "tests")})
        self.assertFalse(ok)
        self.assertIn("host: FAIL", output.getvalue())
        self.assertIn("git: ok", output.getvalue())
        self.assertIn("tests: ok", output.getvalue())
        self.assertNotIn("sensitive output", output.getvalue())

    def test_admin_checks_paginate_and_reveal_without_exposing_secrets(self):
        calls = []
        def api(path, method="GET", credential_id=None):
            calls.append((path, method, credential_id))
            if path.startswith('/api/groups?'):
                return {"code": 0, "data": {"items": [{"id": 3}], "pagination": {"page": 1, "page_size": 100, "total_pages": 1, "total_items": 1}}}
            if path.endswith('/credential'):
                return {"code": 0, "data": {"credential": {"credential_id": 7}}}
            if path.endswith('/reveal'):
                return {"code": 0, "data": {"credential_id": 7, "credential": {"api_key": "never-print-this"}}}
            return {"code": 0, "data": {}}
        with patch.object(checks, "api", side_effect=api):
            summary = checks.check_admin()
        self.assertIn("decrypted=1", summary)
        self.assertNotIn("never-print-this", summary)
        self.assertIn(("/api/groups/3/credential/reveal", "POST", 7), calls)

    def test_no_credentials_is_reported_as_unverified(self):
        def api(path, *args):
            if path.startswith('/api/groups?'):
                return {"code": 0, "data": {"items": [], "pagination": {"page": 1, "page_size": 100, "total_pages": 0, "total_items": 0}}}
            return {"code": 0, "data": {}}
        with patch.object(checks, "api", side_effect=api):
            summary = checks.check_admin()
        self.assertIn("unverified", summary)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(checks.run_checks({"admin": lambda: summary}))

    def test_health_requires_exact_version(self):
        with patch.object(checks, "read_json", return_value={"status": "ok", "version": "dev-abcdef00-old"}):
            with self.assertRaises(ValueError):
                checks.check_health(["curl"], "dev-abcdef00")

    def test_backup_path_is_constrained(self):
        with self.assertRaises(ValueError):
            checks.check_backup('/opt/gpt-load/pre-upgrade.foo;echo secret')

    def test_git_reports_remote_head_without_printing_remote_or_branch(self):
        outputs = iter(["private-branch\n", "a" * 40 + "\n", "", "b" * 40 + "\trefs/heads/dev\n"])
        with patch.object(checks, "command", side_effect=lambda *a, **k: next(outputs)):
            result = checks.check_git()
        self.assertIn("remote dev differs", result)
        self.assertNotIn("private-branch", result)

    def test_logs_do_not_hide_failed_docker_command(self):
        with patch.object(checks, "command", return_value="0") as command:
            checks.check_logs()
        self.assertIn('docker logs --since 5m', command.call_args.args[0][-1])
        self.assertIn('|| exit 1', command.call_args.args[0][-1])

    def test_post_cli_uses_stubbed_boundaries_without_printing_api_data(self):
        stub = '''#!%s
import json, sys
s = ' '.join(sys.argv[1:])
if '/api/groups?' in s:
    value = {'code': 0, 'data': {'items': [{'id': 3}], 'pagination': {'page': 1, 'page_size': 100, 'total_pages': 1, 'total_items': 1}}}
elif '/credential/reveal' in s:
    value = {'code': 0, 'data': {'credential_id': 7, 'credential': {'api_key': 'never-print-this'}}}
elif '/credential' in s:
    value = {'code': 0, 'data': {'credential': {'credential_id': 7}}}
elif '/api/settings' in s:
    value = {'code': 0, 'data': {'secret': 'never-print-this'}}
elif '/health' in s:
    value = {'status': 'ok', 'version': 'dev-abcdef00'}
else:
    value = None
if value is not None: print(json.dumps(value))
elif 'docker inspect' in s: print('gpt-load:dev-abcdef00 healthy 0')
elif 'docker logs' in s: print('0')
''' % sys.executable
        with tempfile.TemporaryDirectory(prefix="deployment-check-test-") as directory:
            root = Path(directory)
            for name in ("ssh", "curl"):
                path = root / name
                path.write_text(stub)
                path.chmod(0o755)
            result = subprocess.run(
                [sys.executable, str(Path(checks.__file__)), "post", "--version", "dev-abcdef00",
                 "--backup", "/opt/gpt-load/pre-upgrade.Abc123"],
                env={**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]},
                capture_output=True, text=True, timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('admin ok groups=1 decrypted=1', result.stdout)
        self.assertIn('backup: archive and compose exist', result.stdout)
        self.assertNotIn('never-print-this', result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
