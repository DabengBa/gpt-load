#!/usr/bin/env bash
# Fetch structured request logs and every administrator raw capture for one request.
# Raw files are written below tmp/ and are never printed to stdout.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/fetch-hostinger-request-log.sh <request-id> [options]

Options:
  --host <ssh-target>   SSH target, default vps-kl
  --container <name>    container name, default gpt-load
  --since <duration>    docker log window, default 24h (for example 24h)
  --tail <count>        maximum docker log lines, default 2000
  --output <directory>  output root, default tmp/hostinger-request-logs
  -h, --help            show this help
EOF
}

request_id=""
ssh_target="${HOSTINGER_SSH_TARGET:-vps-kl}"
container="${GPTLOAD_CONTAINER:-gpt-load}"
since="${HOSTINGER_LOG_SINCE:-24h}"
tail_lines="${HOSTINGER_LOG_TAIL:-2000}"
output_root="${HOSTINGER_LOG_OUTPUT_DIR:-tmp/hostinger-request-logs}"

while (($# > 0)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --host) (($# >= 2)) || { echo "missing --host value" >&2; exit 2; }; ssh_target=$2; shift 2 ;;
    --container) (($# >= 2)) || { echo "missing --container value" >&2; exit 2; }; container=$2; shift 2 ;;
    --since) (($# >= 2)) || { echo "missing --since value" >&2; exit 2; }; since=$2; shift 2 ;;
    --tail) (($# >= 2)) || { echo "missing --tail value" >&2; exit 2; }; tail_lines=$2; shift 2 ;;
    --output) (($# >= 2)) || { echo "missing --output value" >&2; }; output_root=$2; shift 2 ;;
    --*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *) [[ -z "$request_id" ]] || { echo "only one request-id is allowed" >&2; exit 2; }; request_id=$1; shift ;;
  esac
done

[[ "$request_id" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$ ]] || {
  echo "request-id must be a lowercase UUIDv4" >&2
  exit 2
}
[[ "$since" =~ ^[0-9]+[smhd]$ ]] || { echo "--since must look like 24h" >&2; exit 2; }
[[ "$tail_lines" =~ ^[1-9][0-9]*$ ]] || { echo "--tail must be a positive integer" >&2; exit 2; }
[[ "$container" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid container name" >&2; exit 2; }

output_dir="$output_root/$request_id"
mkdir -p "$output_dir/captures"
request_json="$output_dir/request-log.json"
captures_json="$output_dir/debug-captures.json"
container_log="$output_dir/container-log-matches.txt"
analysis="$output_dir/evidence-matrix.txt"

remote_get() {
  local path=$1
  local command="auth=\$(cat /app/data/auth.key) && wget -qO- --header=Authorization:\ Bearer\ \$auth http://127.0.0.1:3001${path}"
  ssh "$ssh_target" "sudo -n docker exec $container sh -c '$command'"
}

remote_get "/api/logs/$request_id" >"$request_json"
remote_get "/api/debug-captures?request_id=$request_id&limit=200" >"$captures_json"

python3 - "$captures_json" "$output_dir/capture-ids.txt" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    envelope = json.load(handle)
if envelope.get("code") != 0:
    raise SystemExit(f"debug capture API returned code {envelope.get('code')}")
items = (envelope.get("data") or {}).get("items") or []
if (envelope.get("data") or {}).get("more"):
    raise SystemExit("debug capture API returned more than 200 items; paginate before analysis")
ids = []
for item in items:
    capture_id = item.get("id")
    if isinstance(capture_id, str) and len(capture_id) == 32 and all(c in "0123456789abcdef" for c in capture_id):
        ids.append(capture_id)
with open(destination, "w", encoding="ascii") as handle:
    handle.write("\n".join(ids))
    if ids:
        handle.write("\n")
PY

while IFS= read -r capture_id; do
  [[ -n "$capture_id" ]] || continue
  remote_get "/api/debug-captures/$capture_id/download" >"$output_dir/captures/$capture_id.zip"
done <"$output_dir/capture-ids.txt"

ssh "$ssh_target" "sudo -n docker logs --since '$since' --tail '$tail_lines' $container 2>&1 | grep -F -- '$request_id' || true" >"$container_log"

set +e
python3 - "$request_json" "$captures_json" "$output_dir" "$analysis" <<'PY'
import json
import os
import sys
import zipfile

request_path, captures_path, output_dir, analysis_path = sys.argv[1:]
with open(request_path, encoding="utf-8") as handle:
    request_envelope = json.load(handle)
with open(captures_path, encoding="utf-8") as handle:
    capture_envelope = json.load(handle)
if request_envelope.get("code") != 0:
    raise SystemExit(f"request log API returned code {request_envelope.get('code')}")
if capture_envelope.get("code") != 0:
    raise SystemExit(f"debug capture API returned code {capture_envelope.get('code')}")

request_data = request_envelope.get("data") or {}
request_attempts = request_data.get("attempts") or []
captures = (capture_envelope.get("data") or {}).get("items") or []
rows = []
failures = []

for capture in captures:
    capture_id = capture.get("id")
    archive_path = os.path.join(output_dir, "captures", f"{capture_id}.zip")
    if not os.path.isfile(archive_path):
        failures.append(f"capture={capture_id} missing zip")
        continue
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for attempt in capture.get("attempts") or []:
            attempt_id = attempt.get("id")
            prefix = f"attempts/{attempt_id}/"
            metadata_name = prefix + "metadata.json"
            metadata = {}
            if metadata_name in names:
                with archive.open(metadata_name) as handle:
                    metadata = json.load(handle)
            events = metadata.get("events") or []
            terminations = [e for e in events if e.get("kind") == "response_termination"]
            no_response = any(e.get("outcome") == "no_response" for e in terminations)
            response_headers = prefix + "parts/response.headers"
            response_body = prefix + "parts/response.body"
            request_headers = prefix + "parts/request.headers"
            request_body = prefix + "parts/request.body"
            response_received = response_headers in names and archive.getinfo(response_headers).file_size > 0
            parts_ok = request_headers in names and request_body in names
            if response_received:
                parts_ok = parts_ok and response_body in names
            elif not no_response:
                parts_ok = False
            complete = bool(terminations) and parts_ok
            if not complete:
                failures.append(f"capture={capture_id} attempt={attempt_id} raw evidence incomplete")
            rows.append({
                "capture": capture_id,
                "sequence": attempt.get("sequence"),
                "attempt": attempt_id,
                "state": attempt.get("state"),
                "response_received": response_received,
                "termination": ",".join(e.get("outcome", "") for e in terminations) or "missing",
                "raw_complete": complete,
            })

for request_attempt in request_attempts:
    sequence = request_attempt.get("sequence")
    if not any(row["sequence"] == sequence for row in rows):
        failures.append(f"request attempt sequence={sequence} has no debug capture attempt")

lines = [
    f"request_id: {request_data.get('request_id')}",
    f"request_status: {request_data.get('status')}",
    f"request_attempt_count: {len(request_attempts)}",
    f"debug_capture_count: {len(captures)}",
    "raw_evidence_status: " + ("INCOMPLETE" if failures else "COMPLETE"),
    "",
    "attempt_evidence_matrix:",
    "capture\tsequence\tattempt\tstate\tresponse_received\ttermination\traw_complete",
]
for row in rows:
    lines.append("\t".join(str(row[key]) for key in ("capture", "sequence", "attempt", "state", "response_received", "termination", "raw_complete")))
if failures:
    lines += ["", "failures:"] + [f"- {failure}" for failure in failures]
with open(analysis_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print(f"raw_evidence_status: {'INCOMPLETE' if failures else 'COMPLETE'}")
print(f"request_log_json: {request_path}")
print(f"debug_captures_json: {captures_path}")
print(f"evidence_matrix: {analysis_path}")
if failures:
    raise SystemExit(3)
PY
status=$?
set -e
exit "$status"
