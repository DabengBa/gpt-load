#!/usr/bin/env python3
"""Offline, isolated rehearsal of a Hostinger SQLite online snapshot."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tarfile
import time
import uuid


REMOTE = r'''
import atexit, hashlib, io, json, os, pathlib, shutil, signal, sqlite3, subprocess, sys, tarfile
os.umask(0o077)
op, token = sys.argv[1:]
assert len(token) == 32 and all(c in '0123456789abcdef' for c in token)
root = pathlib.Path('/opt/gpt-load')
stage = root / ('.local-rehearsal-' + token)
src = pathlib.Path('/var/lib/docker/volumes/gpt-load_gpt-load-data/_data')
if root.is_symlink() or not root.is_dir(): raise SystemExit('unsafe remote root')
if subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(root)], text=True).strip() in ('tmpfs','ramfs','overlay'):
    raise SystemExit('remote root is not persistent storage')
if op == 'size':
    connection = sqlite3.connect('file:' + str(src / 'gpt-load.db') + '?mode=ro', uri=True, timeout=30)
    try: print(connection.execute('PRAGMA page_count').fetchone()[0] * connection.execute('PRAGMA page_size').fetchone()[0])
    finally: connection.close()
elif op == 'make':
    if shutil.disk_usage(root).free <= (src / 'gpt-load.db').stat().st_size * 2:
        raise SystemExit('insufficient remote disk space')
    info = json.loads(subprocess.check_output(['docker','inspect','gpt-load']))[0]
    if not info['State']['Running']: raise SystemExit('production container is not running')
    if not any(m.get('Name') == 'gpt-load_gpt-load-data' and m.get('Destination') == '/app/data' for m in info['Mounts']):
        raise SystemExit('unexpected production data mount')
    env = dict(x.split('=',1) for x in info['Config']['Env'] if '=' in x)
    if env.get('DATABASE_DSN'): raise SystemExit('custom DATABASE_DSN requires manual snapshot')
    if env.get('DATA_DIR', '/app/data') != '/app/data': raise SystemExit('unexpected production DATA_DIR')
    secrets = {}
    for key, filename in [('AUTH_KEY','auth.key'), ('ENCRYPTION_KEY','encryption.key')]:
        value = env.get(key, '').encode() or (src / filename).read_bytes().strip()
        if not value: raise SystemExit('missing required key')
        secrets[filename] = value
    stage.mkdir(mode=0o700)
    def cleanup():
        for name in ('snapshot.db', 'snapshot.db-journal', 'snapshot.db-wal', 'snapshot.db-shm'):
            path = stage / name
            if path.is_file() and not path.is_symlink(): path.unlink()
        stage.rmdir()
    atexit.register(cleanup)
    def terminate(signum, frame): raise SystemExit('remote snapshot interrupted')
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGHUP, terminate)
    db = stage / 'snapshot.db'
    source = sqlite3.connect('file:' + str(src / 'gpt-load.db') + '?mode=ro', uri=True, timeout=30)
    try:
        dest = sqlite3.connect(str(db))
        try: source.backup(dest)
        finally: dest.close()
    finally: source.close()
    os.chmod(db, 0o600)
    digest = hashlib.sha256()
    with db.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): digest.update(block)
    manifest = json.dumps({'sha256':digest.hexdigest(), 'size':db.stat().st_size}).encode()
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as tar:
        for name, value in [('manifest.json',manifest), *secrets.items()]:
            info = tarfile.TarInfo(name); info.size = len(value); info.mode = 0o600
            tar.addfile(info, io.BytesIO(value))
        tar.add(db, arcname='snapshot.db', recursive=False)
    cleanup()
    atexit.unregister(cleanup)
else: raise SystemExit('invalid remote operation')
'''


def validate_reveal(response, credential_id):
    if not isinstance(response, dict) or response.get("code") != 0:
        raise ValueError("credential reveal failed")
    data = response.get("data")
    if not isinstance(data, dict) or data.get("credential_id") != credential_id:
        raise ValueError("credential reveal identity mismatch")
    if not isinstance(data.get("credential"), dict) or not data["credential"]:
        raise ValueError("credential reveal is empty")


def reveal(call, group_id, credential_id):
    validate_reveal(call(f"/api/groups/{group_id}/credential/reveal", "POST", credential_id), credential_id)


def check_root(root):
    root = Path(root)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("output root must be an existing absolute directory")
    for path in (root, *root.parents):
        if path.is_symlink():
            raise ValueError("symlink in output root")
    if root.stat().st_uid != os.geteuid() or root.stat().st_mode & 0o077:
        raise ValueError("output root must be owned by the operator and mode 0700")
    return root


def docker_start_args(image, work, name, env_file=None):
    return ["docker", "run", "-d", "--name", name, "--label", "local-rehearsal=" + name,
            "--network", "none", "--pull", "never", "-e", "MODELS_DEV_AUTO_SYNC_ENABLED=false",
            "-e", "DATA_DIR=/app/data", "-e", "DATABASE_DSN=", "-e", "HOST=0.0.0.0", "-e", "PORT=3001",
            *(["--env-file", str(env_file)] if env_file else []),
            "--mount", f"type=bind,src={work},dst=/app/data", image]


def make_env_file(work):
    values = []
    for key, filename in (("AUTH_KEY", "auth.key"), ("ENCRYPTION_KEY", "encryption.key")):
        value = (work / filename).read_bytes()
        if not value or b'\n' in value or b'\r' in value or b'\0' in value:
            raise ValueError('key cannot be represented in Docker env file')
        values.append(key.encode() + b'=' + value + b'\n')
    path = work.parent / 'keys.env'
    with path.open('xb') as output:
        os.chmod(path, 0o600)
        output.write(b''.join(values))
    return path


def two_starts(work, name, start, verify, stop):
    for _ in range(2):
        try:
            start(work, name)
            verify(name)
        finally:
            stop(name)


def command(args, timeout=60, **kwargs):
    return subprocess.run(args, check=True, timeout=timeout, **kwargs)


def check_docker_local():
    host = os.environ.get('DOCKER_HOST')
    if not host:
        host = command(['docker', 'context', 'inspect', '--format', '{{.Endpoints.docker.Host}}'],
                       stdout=subprocess.PIPE, text=True).stdout.strip()
    if not host.startswith('unix:///'):
        raise ValueError('Docker daemon must use a local Unix socket')


def ssh(host, op, token, timeout, **kwargs):
    remote = "timeout --signal=TERM --kill-after=10s " + str(timeout) + "s python3 -c " + shlex.quote(REMOTE)
    remote += " " + shlex.quote(op) + " " + shlex.quote(token)
    return command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, remote],
                   timeout=timeout + 20, **kwargs)


def receive_archive(archive, snapshot, work):
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        expected = {'manifest.json', 'snapshot.db', 'auth.key', 'encryption.key'}
        if len(members) != 4 or {m.name for m in members} != expected or any(not m.isfile() for m in members):
            raise ValueError("invalid snapshot archive")
        manifest_file = tar.extractfile('manifest.json')
        if manifest_file is None or tar.getmember('manifest.json').size > 4096:
            raise ValueError("missing manifest")
        manifest = json.load(manifest_file)
        for name in ('auth.key', 'encryption.key'):
            member = tar.getmember(name)
            if not 0 < member.size <= 65536:
                raise ValueError("invalid key size")
            source = tar.extractfile(member)
            with (work / name).open('xb') as target:
                shutil.copyfileobj(source, target)
        member = tar.getmember('snapshot.db')
        if member.size <= 0 or member.size != manifest.get('size'):
            raise ValueError("snapshot size mismatch")
        if shutil.disk_usage(snapshot.parent).free <= member.size * 2:
            raise ValueError('insufficient local disk space for snapshot and work copy')
        digest = hashlib.sha256()
        with snapshot.open('xb') as target:
            source = tar.extractfile(member)
            for block in iter(lambda: source.read(1024 * 1024), b''):
                target.write(block)
                digest.update(block)
        if digest.hexdigest() != manifest.get('sha256'):
            snapshot.unlink()
            raise ValueError("snapshot checksum mismatch")
    os.chmod(snapshot, 0o400)


def envelope(value):
    if not isinstance(value, dict) or value.get('code') != 0 or 'data' not in value:
        raise ValueError('admin API failed')
    return value['data']


def docker_api(name, path, method='GET', credential_id=None):
    # Secrets are read inside the isolated container, never placed in command arguments.
    if not re.fullmatch(r'/[A-Za-z0-9_/?=&.-]+', path):
        raise ValueError('unsafe API path')
    script = 'wget -qO- -T 10 --header="Authorization: Bearer $(cat /app/data/auth.key)"'
    if method == 'POST':
        if not isinstance(credential_id, int) or credential_id <= 0:
            raise ValueError('invalid credential id')
        script += f' --header="X-Credential-ID: {credential_id}" --header="Content-Type: application/json" --post-data="{{}}"'
    elif method != 'GET':
        raise ValueError('invalid method')
    script += ' ' + shlex.quote('http://127.0.0.1:3001' + path)
    result = command(['docker', 'exec', name, 'sh', '-c', script], timeout=20, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return json.loads(result.stdout)


def verify_container(name):
    deadline = time.monotonic() + 120
    while True:
        try:
            result = command(['docker', 'exec', name, 'wget', '-qO-', '-T', '5', 'http://127.0.0.1:3001/health'],
                             timeout=10, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            health = json.loads(result.stdout)
            if health.get('status') != 'ok': raise ValueError('health not ok')
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            if time.monotonic() >= deadline: raise RuntimeError('container health timeout')
            time.sleep(1)
    call = lambda path, method='GET', credential_id=None: docker_api(name, path, method, credential_id)
    page, count, decrypted = 1, 0, 0
    while True:
        data = envelope(call(f'/api/groups?page={page}&page_size=100'))
        items = data['items']
        for group in items:
            group_id = group['id']
            if not isinstance(group_id, int) or group_id <= 0: raise ValueError('invalid group id')
            detail = envelope(call(f'/api/groups/{group_id}/credential'))
            credential = detail['credential']
            if credential is not None:
                cid = credential['credential_id']
                reveal(call, group_id, cid)
                decrypted += 1
            count += 1
        pagination = data['pagination']
        if page >= pagination['total_pages']: break
        page += 1
        if page > 100000: raise ValueError('invalid pagination')
    if count != pagination['total_items']: raise ValueError('incomplete group listing')
    if not decrypted: raise ValueError('no credential was available to verify decryption')
    print(f"health version={health.get('version')} groups={count} decrypted={decrypted}")


def stop_container(name):
    result = command(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$',
                      '--filter', 'label=local-rehearsal=' + name], stdout=subprocess.PIPE, text=True)
    ids = result.stdout.split()
    if len(ids) > 1: raise RuntimeError('ambiguous rehearsal containers')
    if ids:
        command(['docker', 'rm', '-f', ids[0]], stdout=subprocess.DEVNULL)
    result = command(['docker', 'ps', '-aq', '--filter', 'name=^/' + name + '$'], stdout=subprocess.PIPE, text=True)
    if result.stdout.strip(): raise RuntimeError('container still exists; work directory retained')


def inspect_db(path):
    db = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=30)
    try:
        if db.execute('pragma integrity_check').fetchone()[0] != 'ok': raise ValueError('SQLite integrity check failed')
        if db.execute('pragma foreign_key_check').fetchone(): raise ValueError('SQLite foreign key check failed')
        names = {x[0] for x in db.execute("select name from sqlite_master where type='table'")}
        if not {'groups', 'schema_migrations'}.issubset(names):
            raise ValueError('database is missing required application tables')
        counts = {table: db.execute('select count(*) from "' + table + '"').fetchone()[0]
                  for table in ('groups', 'credentials', 'access_keys', 'schema_migrations') if table in names}
        if 'schema_migrations' in names:
            counts['migration_ids'] = [row[0] for row in db.execute('select id from schema_migrations order by id')]
        return counts
    finally:
        db.close()


def run(args):
    os.umask(0o077)
    if os.geteuid() != 0: raise ValueError('run as root to match image UID 10001 and protect data')
    root = check_root(args.output)
    check_docker_local()
    if ',' in str(root): raise ValueError('output path must not contain comma')
    fs = command(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(root)], stdout=subprocess.PIPE, text=True).stdout.strip()
    if fs in ('tmpfs', 'ramfs', 'overlay'): raise ValueError('output must be on persistent storage')
    if list(root.glob('.active-*')): raise ValueError('previous active rehearsal exists; inspect manually')
    token = uuid.uuid4().hex
    name = 'local-rehearsal-' + token
    session = root / ('rehearsal-' + token)
    marker = root / ('.active-' + token)
    work = session / 'work'
    archive = session / 'transfer.tar'
    snapshot = session / 'snapshot.db'
    env_file = session / 'keys.env'
    clean = False
    try:
        session.mkdir(mode=0o700)
        marker.write_text(name + '\n', encoding='ascii')
        os.chmod(marker, 0o600)
        work.mkdir(mode=0o700)
        image_arch = command(['docker', 'image', 'inspect', args.image, '--format', '{{.Architecture}}'], stdout=subprocess.PIPE, text=True).stdout.strip()
        host_arch = command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', args.host,
                             "docker info --format '{{.Architecture}}'"], timeout=30, stdout=subprocess.PIPE, text=True).stdout.strip()
        if {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(host_arch, host_arch) != image_arch:
            raise ValueError('target image architecture differs from server')
        size = int(ssh(args.host, 'size', token, 30, stdout=subprocess.PIPE, text=True).stdout)
        if size <= 0 or shutil.disk_usage(root).free <= size * 4:
            raise ValueError('insufficient local persistent disk space')
        with archive.open('xb') as output:
            ssh(args.host, 'make', token, args.timeout, stdout=output)
        receive_archive(archive, snapshot, work)
        archive.unlink()
        make_env_file(work)
        print('original snapshot:', snapshot)
        print('before:', inspect_db(snapshot))
        shutil.copyfile(snapshot, work / 'gpt-load.db')
        for path in work.iterdir():
            os.chmod(path, 0o600)
            os.chown(path, 10001, 10001)
        os.chown(work, 10001, 10001)
        def start(directory, container):
            command(docker_start_args(args.image, directory, container, env_file), timeout=120, stdout=subprocess.DEVNULL)
        two_starts(work, name, start, verify_container, stop_container)
        print('after:', inspect_db(work / 'gpt-load.db'))
        clean = True
    finally:
        # Never remove a directory that might still be used by a container.
        stop_container(name)
        if archive.exists(): archive.unlink()
        if env_file.exists(): env_file.unlink()
        if work.exists(): shutil.rmtree(work)
        marker.unlink(missing_ok=True)
        if not clean:
            print(f'failure: inspect remote /opt/gpt-load/.local-rehearsal-{token} and local {session}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='existing private persistent directory')
    parser.add_argument('--image', required=True, help='prepared local target image')
    parser.add_argument('--host', default='vps-kl')
    parser.add_argument('--timeout', type=int, default=1800, help='remote snapshot/transfer timeout in seconds')
    args = parser.parse_args()
    if not 30 <= args.timeout <= 86400: parser.error('timeout must be between 30 and 86400 seconds')
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', args.host) or args.host.startswith('-'): parser.error('invalid SSH target')
    if not re.fullmatch(r'[A-Za-z0-9_./:@-]+', args.image) or args.image.startswith('-'): parser.error('invalid image')
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'rehearsal failed: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
