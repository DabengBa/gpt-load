#!/usr/bin/env python3
"""Verify fresh install and real old-image upgrade locally, without production data."""
import argparse
from contextlib import closing
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import tempfile
import time
import uuid
import zipfile

spec = importlib.util.spec_from_file_location("rehearsal", Path(__file__).with_name("local-rehearsal.py"))
rehearsal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rehearsal)

CREDENTIAL = "sk-local-upgrade-proof-not-a-real-key"

# Debug captures written by the old image as plain rows. Migration 0026 adds the
# encoding column, so these bytes and their order must survive it unchanged.
CAPTURE_ID = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
ATTEMPT_ID = "0a1b2c3d4e5f60718293a4b5c6d7e8f9"
CAPTURE_PARTS = (
    ("headers", "request",
     b"POST /v1/chat/completions HTTP/1.1\r\nHost: captured.invalid\r\n"
     b"Content-Type: application/json\r\n\r\n"),
    ("body", "request",
     b'{"model":"capture-upgrade-proof","stream":false,"messages":[{"role":"user",'
     b'"content":"raw captured bytes must survive the debug capture chunk encoding migration"}]}'),
    ("headers", "response",
     b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n"),
    ("body", "response",
     b'{"id":"capture-upgrade-proof","choices":[{"message":{"role":"assistant",'
     b'"content":"preserved through the upgrade"}}]}'),
)
CAPTURE_CHUNK_SIZE = 48


def capture_chunks():
    # Chunks are rows, one append operation each, and their ids define the order.
    for part, direction, data in CAPTURE_PARTS:
        for offset in range(0, len(data), CAPTURE_CHUNK_SIZE):
            yield part, direction, data[offset:offset + CAPTURE_CHUNK_SIZE]


def exported_parts():
    return {(direction, part): data for part, direction, data in CAPTURE_PARTS}


def start(image, work, name):
    # One env file per work directory; it is reused by both target starts.
    env_file = work.parent / "keys.env"
    if not env_file.exists():
        env_file = rehearsal.make_env_file(work)
    args = rehearsal.docker_start_args(image, work, name, env_file)
    # Bind-mounted fixtures belong to the local operator, not root or production.
    args[2:2] = ["--user", f"{os.getuid()}:{os.getgid()}"]
    rehearsal.command(args, timeout=120, stdout=subprocess.DEVNULL)


def stop(name):
    rehearsal.stop_container(name)


def seed(name):
    # Wait for readiness before creating a genuine encrypted credential through
    # the old version's management API. No schema is synthesized by this script.
    wait_health(name)
    payload = json.dumps({"channel_id": "openai", "models": [], "credential": CREDENTIAL})
    script = ('wget -qO- -T 10 --header="Authorization: Bearer $(cat /app/data/auth.key)" '
              '--header="Content-Type: application/json" '
              f'--header="Idempotency-Key: {uuid.uuid4()}" --post-data=' + shlex.quote(payload) +
              ' http://127.0.0.1:3001/api/groups')
    result = rehearsal.command(["docker", "exec", name, "sh", "-c", script], timeout=20,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    data = rehearsal.envelope(json.loads(result.stdout))
    if not data.get("group_id") or not data.get("credential_id"):
        raise ValueError("fixture group/credential was not created")


def seed_capture(work):
    # Only data rows are written, with the columns the old image already has. The
    # old image owns this schema; the fixture never creates or alters tables.
    now_ms = int(time.time() * 1000)
    with closing(sqlite3.connect(work / "gpt-load.db")) as db:
        columns = {row[1] for row in db.execute("pragma table_info(debug_capture_chunks)")}
        if not {"capture_id", "attempt_id", "part", "direction", "data", "created_at_ms"}.issubset(columns):
            raise ValueError("released debug capture schema is missing from the fixture database")
        db.execute("insert into debug_captures (id, request_id, access_key_id, protocol, operation, metadata,"
                   " created_at_ms, expires_at_ms, state, error) "
                   "values (?, '', 0, 'openai', 'chat', '{}', ?, ?, 'completed', '')",
                   (CAPTURE_ID, now_ms - 60_000, now_ms + 7 * 24 * 60 * 60 * 1000))
        db.execute("insert into debug_capture_attempts (id, capture_id, sequence, metadata,"
                   " started_at_ms, completed_at_ms, state, error) "
                   "values (?, ?, 1, '{}', ?, ?, 'completed', '')",
                   (ATTEMPT_ID, CAPTURE_ID, now_ms - 60_000, now_ms - 59_000))
        db.executemany("insert into debug_capture_chunks"
                       " (capture_id, attempt_id, part, direction, data, created_at_ms)"
                       " values (?, ?, ?, ?, ?, ?)",
                       [(CAPTURE_ID, ATTEMPT_ID, part, direction, data, now_ms - 60_000)
                        for part, direction, data in capture_chunks()])
        db.commit()
        stored = db.execute("select count(*) from debug_capture_chunks where capture_id = ?",
                            (CAPTURE_ID,)).fetchone()[0]
    if stored != len(list(capture_chunks())):
        raise ValueError("fixture debug capture chunks were not written")


def wait_health(name):
    # verify_container also checks readiness, but requires an existing credential.
    import time
    deadline = time.monotonic() + 120
    while True:
        try:
            result = rehearsal.command(["docker", "exec", name, "wget", "-qO-", "-T", "5",
                                        "http://127.0.0.1:3001/health"], timeout=10,
                                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if json.loads(result.stdout).get("status") != "ok":
                raise ValueError("health not ok")
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            if time.monotonic() >= deadline:
                raise RuntimeError("container health timeout")
            time.sleep(1)


def check_api(name):
    rehearsal.verify_container(name)
    for path in ("/api/settings", "/api/access-keys?page=1&page_size=100"):
        rehearsal.envelope(rehearsal.docker_api(name, path))
    groups = rehearsal.envelope(rehearsal.docker_api(name, "/api/groups?page=1&page_size=100"))
    if groups["pagination"]["total_items"] != 1:
        raise ValueError("fixture group count changed")
    group = groups["items"][0]
    detail = rehearsal.envelope(rehearsal.docker_api(name, f'/api/groups/{group["id"]}/credential'))
    credential_id = detail["credential"]["credential_id"]
    revealed = rehearsal.envelope(rehearsal.docker_api(name, f'/api/groups/{group["id"]}/credential/reveal', "POST", credential_id))
    if revealed["credential"].get("api_key") != CREDENTIAL:
        raise ValueError("upgraded credential plaintext differs")


def check_version(name, version):
    # A healthy container is not enough: it must serve the commit that was built.
    result = rehearsal.command(["docker", "exec", name, "wget", "-qO-", "-T", "5",
                                "http://127.0.0.1:3001/health"], timeout=10,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    served = json.loads(result.stdout).get("version")
    if served != version:
        raise ValueError(f"container serves version {served!r}, expected {version!r}")


def check_capture(name):
    listing = rehearsal.envelope(rehearsal.docker_api(name, "/api/debug-captures?limit=10"))
    if CAPTURE_ID not in {item["id"] for item in listing["items"]}:
        raise ValueError("fixture debug capture session is missing from the target")
    record = rehearsal.envelope(rehearsal.docker_api(name, "/api/debug-captures/" + CAPTURE_ID))
    if [attempt["id"] for attempt in record["attempts"]] != [ATTEMPT_ID]:
        raise ValueError("fixture debug capture attempt is missing from the target")
    check_export(download_capture(name))


def download_capture(name):
    # The ZIP is the only surface that decodes every chunk through the store.
    script = ('wget -qO- -T 30 --header="Authorization: Bearer $(cat /app/data/auth.key)" '
              + shlex.quote("http://127.0.0.1:3001/api/debug-captures/" + CAPTURE_ID + "/download"))
    result = rehearsal.command(["docker", "exec", name, "sh", "-c", script], timeout=60,
                               stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return result.stdout


def check_export(bundle):
    expected = exported_parts()
    try:
        archive = zipfile.ZipFile(io.BytesIO(bundle))
    except zipfile.BadZipFile as error:
        raise ValueError("debug capture export is not a ZIP archive") from error
    with archive:
        try:
            manifest = json.loads(archive.read("manifest.json"))
        except KeyError as error:
            raise ValueError("debug capture export has no manifest") from error
        if manifest.get("id") != CAPTURE_ID:
            raise ValueError("debug capture export manifest id differs")
        for (direction, part), data in expected.items():
            entry = f"attempts/{ATTEMPT_ID}/parts/{direction}.{part}"
            try:
                exported = archive.read(entry)
            except KeyError as error:
                raise ValueError("debug capture export is missing " + entry) from error
            if exported != data:
                raise ValueError("debug capture export part " + entry + " differs from the captured bytes")


def inspect(work):
    path = work / "gpt-load.db"
    result = rehearsal.inspect_db(path)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)) as db:
        result["credentials"] = db.execute("select id, data from credentials order by id").fetchall()
        tables = {row[0] for row in db.execute("select name from sqlite_master where type='table'")}
        result["chunks"] = []
        result["encoding_column"] = False
        if "debug_capture_chunks" in tables:
            columns = {row[1] for row in db.execute("pragma table_info(debug_capture_chunks)")}
            result["encoding_column"] = "encoding" in columns
            result["chunks"] = [
                (*row[:5], bytes(row[5]), row[6] if "encoding" in columns else None)
                for row in db.execute("select id, capture_id, attempt_id, part, direction, data"
                                      + (", encoding" if "encoding" in columns else ", null")
                                      + " from debug_capture_chunks order by id")]
    if result["groups"] != 1 or len(result["credentials"]) != 1:
        raise ValueError("fixture database is not populated")
    if "sk-local-upgrade-proof" in result["credentials"][0][1]:
        raise ValueError("fixture credential was not encrypted")
    return result


def check_database(before, after, fresh_ids):
    old_ids, new_ids = set(before["migration_ids"]), set(after["migration_ids"])
    if not old_ids.issubset(new_ids) or new_ids != set(fresh_ids) | old_ids:
        raise ValueError("upgrade migration ledger differs from fresh install")
    for key in ("groups", "credentials", "access_keys"):
        if before.get(key) != after.get(key):
            raise ValueError("upgrade changed fixture " + key)
    # 0026 adds the encoding column; the rows written before it must be preserved
    # byte for byte, in append order, and only carry a supported encoding.
    if before.get("chunks"):
        if not after["encoding_column"]:
            raise ValueError("debug capture chunk encoding column is missing")
        if not {chunk[6] for chunk in after["chunks"]} <= {"identity", "zstd"}:
            raise ValueError("upgrade wrote an unsupported debug capture chunk encoding")
        if [chunk[:6] for chunk in before["chunks"]] != [chunk[:6] for chunk in after["chunks"]]:
            raise ValueError("upgrade changed the fixture debug capture chunks")


def scenario(image, previous, work, name, version, fresh_ids=None):
    # The first image creates and populates its own schema. On fresh installs it
    # is the target; upgrades use the actual previously deployed image.
    (work / "auth.key").write_text("local-verification-auth", encoding="ascii")
    (work / "encryption.key").write_text("local-verification-encryption", encoding="ascii")
    try:
        start(previous, work, name)
        seed(name)
        check_api(name)
    finally:
        stop(name)
    seed_capture(work)
    before = inspect(work)
    if not before["chunks"]:
        raise ValueError("fixture debug capture chunks are missing from the old-image database")
    ids = before["migration_ids"] if fresh_ids is None else fresh_ids
    for _ in range(2):
        try:
            start(image, work, name)
            check_api(name)
            check_version(name, version)
            check_capture(name)
        finally:
            stop(name)
        after = inspect(work)
        check_database(before, after, ids)
    return after["migration_ids"]


def verify(image, previous, version):
    os.umask(0o077)
    rehearsal.check_docker_local()
    # Resolve tags once so both starts exercise exactly the images inspected.
    def image_id(value):
        return rehearsal.command(["docker", "image", "inspect", value, "--format", "{{.Id}}"],
                                 stdout=subprocess.PIPE, text=True).stdout.strip()
    image, previous = image_id(image), image_id(previous)
    # Docker bind mounts need an absolute path, and commas cannot be escaped.
    with tempfile.TemporaryDirectory(prefix="gpt-load-image-check-") as directory:
        root = Path(directory).resolve()
        if "," in str(root):
            raise ValueError("temporary directory must not contain comma")
        fresh, upgrade = root / "fresh", root / "upgrade"
        fresh.mkdir(mode=0o700)
        upgrade.mkdir(mode=0o700)
        name = "local-rehearsal-" + uuid.uuid4().hex
        ids = scenario(image, image, fresh, name, version)
        scenario(image, previous, upgrade, name, version, ids)
    print("Built image verified: fresh install, populated old-image upgrade, two target starts, API, decryption, ledger and SQLite integrity")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--previous-image", required=True)
    parser.add_argument("--expected-version", required=True,
                        help="version the built image must report on /health")
    args = parser.parse_args()
    verify(args.image, args.previous_image, args.expected_version)


if __name__ == "__main__":
    main()
