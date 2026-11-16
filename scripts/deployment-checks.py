#!/usr/bin/env python3
"""Optional operator checks; never deploys or prints API responses or command errors."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "vps-kl"]
PUBLIC = "https://gptl.tanyaleoallen.online/health"  # Matches deploy.sh.


def command(args, timeout=30):
    result = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, check=True)
    return result.stdout.decode("utf-8")


def read_json(args):
    return json.loads(command(args))


def run_checks(tasks):
    # Independent checks finish even when one fails. Never print exceptions: command
    # errors and API responses may contain credentials or other sensitive data.
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        futures = {name: pool.submit(task) for name, task in tasks.items()}
        ok = True
        for name, future in futures.items():
            try:
                result = future.result()
                print(f"{name}: {result}")
                if result.startswith("UNVERIFIED"):
                    ok = False
            except Exception:
                print(f"{name}: FAIL (inspect this check safely; details suppressed)")
                ok = False
    return ok


def check_git():
    branch = command(["git", "branch", "--show-current"]).strip()
    sha = command(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(command(["git", "status", "--porcelain=v1", "--untracked-files=all"]))
    remote = command(["git", "ls-remote", "--exit-code", "origin", "refs/heads/dev"], timeout=60).split()
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or len(remote) != 2 or not re.fullmatch(r"[0-9a-f]{40}", remote[0]) or remote[1] != "refs/heads/dev":
        raise ValueError("invalid git head")
    # Branch names can be arbitrary; do not echo them.
    return f"HEAD={sha[:8]} branch={'dev' if branch == 'dev' else 'other'} worktree={'dirty' if dirty else 'clean'}; remote dev {'matches' if remote[0] == sha else 'differs'}"


def check_host():
    # Read-only observations only. The deployment switch repeats its own guards.
    command(SSH + ["docker inspect gpt-load >/dev/null && curl -fsS --max-time 10 http://127.0.0.1:3001/health >/dev/null && df -P /opt/gpt-load >/dev/null"])
    return "container, loopback health and filesystem accessible; inspect capacity and mount separately"


def check_tests():
    command(["make", "test-deploy"], timeout=300)
    return "offline deployment regression suite passed (run change-specific tests separately)"


def check_health(args, version):
    health = read_json(args)
    if health.get("status") != "ok" or health.get("version") != version:
        raise ValueError("health or version mismatch")
    return f"ok version={version}"


def api(path, method="GET", credential_id=None):
    if not re.fullmatch(r"/[A-Za-z0-9_/?=&.-]+", path):
        raise ValueError("invalid API path")
    curl = ['curl', '-fsS', '--max-time', '10', '--header', '@-']
    if method == "POST" and type(credential_id) is int and credential_id > 0:
        curl += ['--header', f'X-Credential-ID: {credential_id}',
                 '--header', 'Content-Type: application/json', '--data', '{}']
    elif method != "GET":
        raise ValueError("invalid API method")
    curl.append('http://127.0.0.1:3001' + path)
    # printf is a shell builtin; neither docker nor curl receives the key in argv.
    header = 'key=$(cat /app/data/auth.key) && [ -n "$key" ] && printf "Authorization: Bearer %s\\n" "$key"'
    pipeline = 'docker exec gpt-load sh -c ' + shlex.quote(header) + ' | ' + shlex.join(curl)
    return read_json(SSH + ['bash -o pipefail -c ' + shlex.quote(pipeline)])


def envelope(value):
    if not isinstance(value, dict) or value.get("code") != 0 or "data" not in value:
        raise ValueError("admin API failed")
    return value["data"]


def group_ids():
    page, ids, seen, expected = 1, [], set(), None
    while True:
        data = envelope(api(f"/api/groups?page={page}&page_size=100&sort=created"))
        pagination = data['pagination']
        total, pages = pagination['total_items'], pagination['total_pages']
        if (type(total) is not int or total < 0 or type(pages) is not int or
                pages != (total + 99) // 100 or pages > 100000 or
                pagination['page'] != page or pagination['page_size'] != 100 or
                (expected is not None and expected != (total, pages)) or
                not isinstance(data['items'], list) or
                len(data['items']) != min(100, max(0, total - (page - 1) * 100))):
            raise ValueError('inconsistent group pagination')
        expected = total, pages
        for group in data["items"]:
            group_id = group["id"]
            if type(group_id) is not int or group_id <= 0 or group_id in seen:
                raise ValueError("invalid or duplicate group")
            seen.add(group_id)
            ids.append(group_id)
        if page >= pages:
            break
        page += 1
    if len(ids) != total:
        raise ValueError("incomplete group listing")
    return ids


def check_admin():
    envelope(api("/api/settings"))
    ids, decrypted = group_ids(), 0
    for group_id in ids:
        credential = envelope(api(f"/api/groups/{group_id}/credential"))["credential"]
        if credential is not None:
            cid = credential["credential_id"]
            if type(cid) is not int or cid <= 0:
                raise ValueError("invalid credential")
            revealed = envelope(api(f"/api/groups/{group_id}/credential/reveal", "POST", cid))
            if revealed.get("credential_id") != cid or not isinstance(revealed.get("credential"), dict) or not revealed["credential"]:
                raise ValueError("credential reveal failed")
            decrypted += 1
    if group_ids() != ids:
        raise ValueError("group listing changed during verification")
    count = len(ids)
    if not decrypted:
        return f"UNVERIFIED admin ok groups={count}; real credential decryption unverified (none available)"
    return f"admin ok groups={count} decrypted={decrypted}"


def check_container(version):
    info = command(SSH + ["docker inspect gpt-load --format '{{.Config.Image}} {{.State.Health.Status}} {{.RestartCount}}'"]).strip().split()
    if len(info) != 3 or info[0] != "gpt-load:" + version or info[1] != "healthy" or not info[2].isdigit():
        raise ValueError("container mismatch")
    return f"image={info[0]} health={info[1]} restarts={info[2]}"


def check_logs():
    # Print count, never log lines, which can contain request data.
    count = command(SSH + ["logs=$(docker logs --since 5m gpt-load 2>&1) || exit 1; printf '%s\n' \"$logs\" | grep -icE 'error|fatal|panic' || true"]).strip()
    if not count.isdigit():
        raise ValueError("log count unavailable")
    return f"last 5m error/fatal/panic matching lines={count}; inspect securely if needed"


def check_backup(path):
    if not re.fullmatch(r"/opt/gpt-load/pre-upgrade\.[A-Za-z0-9]+", path):
        raise ValueError("invalid backup path")
    command(SSH + [f"test -s {path}/data.tar && test -s {path}/docker-compose.yml"])
    return "archive and compose exist and are nonempty; restore not verified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("preflight")
    post = sub.add_parser("post")
    post.add_argument("--version", required=True, help="exact dev-<short sha> from deployment")
    post.add_argument("--backup", required=True, help="exact pre-upgrade path printed by this deployment")
    args = parser.parse_args()
    if args.action == "preflight":
        tasks = {"git": check_git, "host": check_host, "tests": check_tests}
    else:
        if not re.fullmatch(r"dev-[0-9a-f]{4,40}", args.version):
            parser.error("invalid version")
        tasks = {
            "container": lambda: check_container(args.version),
            "loopback": lambda: check_health(SSH + ["curl -fsS --max-time 10 http://127.0.0.1:3001/health"], args.version),
            "public": lambda: check_health(["curl", "-fsS", "--max-time", "10", PUBLIC], args.version),
            "admin/decryption": check_admin,
            "logs": check_logs,
            "backup": lambda: check_backup(args.backup),
        }
    return 0 if run_checks(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())
