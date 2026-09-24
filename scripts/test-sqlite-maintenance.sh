#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
script=$repo_root/scripts/sqlite-maintenance.sh
work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  exit 1
}

assert_file() {
  [ -e "$1" ] || fail "expected file: $1"
}

assert_not_file() {
  [ ! -e "$1" ] || fail "file must not exist: $1"
}

assert_eq() {
  [ "$1" = "$2" ] || fail "expected '$1', got '$2'"
}

printf '1. syntax check\n'
bash -n "$script"

printf '2. successful temporary-directory maintenance and complete backup\n'
data="$work/data"
backup="$work/backups"
mkdir -p "$data" "$backup"
sqlite3 "$data/gpt-load.db" <<'SQL'
PRAGMA journal_mode=WAL;
CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT NOT NULL);
INSERT INTO records(value) VALUES ('fixture');
SQL
printf 'auth-fixture\n' >"$data/auth.key"
printf 'encryption-fixture\n' >"$data/encryption.key"
printf 'wal-fixture\n' >"$data/gpt-load.db-wal"
printf 'shm-fixture\n' >"$data/gpt-load.db-shm"
printf 'other-fixture\n' >"$data/runtime-state"

bash "$script" --data-dir "$data" --backup-dir "$backup" --stopped-confirmed --dry-run >"$work/dry-run.out"
assert_eq "$(find "$backup" -maxdepth 1 -type f | wc -l)" "0"
assert_eq "$(cat "$data/auth.key")" "auth-fixture"

bash "$script" --data-dir "$data" --backup-dir "$backup" --stopped-confirmed >"$work/success.out"
archive=$(find "$backup" -maxdepth 1 -type f -name '*.tar.gz' -print -quit)
assert_file "$archive"
for entry in ./gpt-load.db ./auth.key ./encryption.key ./gpt-load.db-wal ./gpt-load.db-shm ./runtime-state; do
  tar -tzf "$archive" | grep -Fxq "$entry" || fail "backup missing $entry"
done
assert_eq "$(sqlite3 "$data/gpt-load.db" 'PRAGMA integrity_check;')" "ok"
assert_eq "$(cat "$data/auth.key")" "auth-fixture"
assert_eq "$(cat "$data/encryption.key")" "encryption-fixture"
if grep -qE 'auth-fixture|encryption-fixture' "$work/success.out"; then
  fail "maintenance output leaked key material"
fi

printf '3. running-service refusal\n'
runtime="$work/fake-runtime"
mkdir -p "$runtime"
cat >"$runtime/docker" <<'DOCKER'
#!/usr/bin/env bash
if [ "${1:-}" = inspect ]; then
  printf 'true\n'
  exit 0
fi
exit 1
DOCKER
chmod +x "$runtime/docker"
service_data="$work/service-data"
mkdir -p "$service_data"
printf 'service-original\n' >"$service_data/auth.key"
if PATH="$runtime:$PATH" bash "$script" --data-dir "$service_data" --backup-dir "$work/service-backups" --service running-container; then
  fail "running service was accepted"
fi
assert_eq "$(cat "$service_data/auth.key")" "service-original"
assert_not_file "$work/service-backups"

printf '4. integrity failure refuses commit and startup\n'
bad_data="$work/bad-data"
bad_backup="$work/bad-backups"
mkdir -p "$bad_data" "$bad_backup"
printf 'not sqlite\n' >"$bad_data/gpt-load.db"
printf 'bad-auth\n' >"$bad_data/auth.key"
before=$(sha256sum "$bad_data/gpt-load.db" | cut -d' ' -f1)
marker="$work/must-not-start"
if bash "$script" --data-dir "$bad_data" --backup-dir "$bad_backup" --stopped-confirmed \
    --start-command "touch '$marker'" --health-url http://127.0.0.1:1/health \
    --verify-command "test -s '$bad_data/encryption.key'"; then
  fail "corrupt database was accepted"
fi
after=$(sha256sum "$bad_data/gpt-load.db" | cut -d' ' -f1)
assert_eq "$after" "$before"
assert_not_file "$marker"
assert_file "$(find "$bad_backup" -maxdepth 1 -type f -name '*.tar.gz' -print -quit)"

printf '5. busy checkpoint refuses commit and startup\n'
busy_data="$work/busy-data"
busy_backup="$work/busy-backups"
busy_bin="$work/busy-bin"
mkdir -p "$busy_data" "$busy_backup" "$busy_bin"
sqlite3 "$busy_data/gpt-load.db" "CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT NOT NULL); INSERT INTO records(value) VALUES ('busy-fixture');"
busy_before=$(sha256sum "$busy_data/gpt-load.db" | cut -d' ' -f1)
real_sqlite3=$(command -v sqlite3)
cat >"$busy_bin/sqlite3" <<SQLITE
#!/usr/bin/env bash
if [[ "\$*" == *"PRAGMA wal_checkpoint(TRUNCATE);"* ]]; then
  printf '1|1|0\n'
  exit 0
fi
exec "$real_sqlite3" "\$@"
SQLITE
chmod +x "$busy_bin/sqlite3"
busy_marker="$work/busy-must-not-start"
if PATH="$busy_bin:$PATH" bash "$script" --data-dir "$busy_data" --backup-dir "$busy_backup" --stopped-confirmed \
    --start-command "touch '$busy_marker'" --health-url http://127.0.0.1:1/health \
    --verify-command "true" >"$work/busy.out" 2>&1; then
  fail "busy checkpoint was accepted"
fi
assert_eq "$(sha256sum "$busy_data/gpt-load.db" | cut -d' ' -f1)" "$busy_before"
assert_not_file "$busy_marker"
assert_file "$(find "$busy_backup" -maxdepth 1 -type f -name '*.tar.gz' -print -quit)"
grep -Fq 'WAL checkpoint was busy' "$work/busy.out" || fail "busy checkpoint failure was not reported"

printf '6. VACUUM failure refuses commit and startup\n'
vacuum_data="$work/vacuum-data"
vacuum_backup="$work/vacuum-backups"
vacuum_bin="$work/vacuum-bin"
mkdir -p "$vacuum_data" "$vacuum_backup" "$vacuum_bin"
sqlite3 "$vacuum_data/gpt-load.db" "CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT NOT NULL); INSERT INTO records(value) VALUES ('vacuum-fixture');"
vacuum_before=$(sha256sum "$vacuum_data/gpt-load.db" | cut -d' ' -f1)
cat >"$vacuum_bin/sqlite3" <<SQLITE
#!/usr/bin/env bash
if [[ "\$*" == *"VACUUM;"* ]]; then
  printf 'simulated VACUUM failure\n' >&2
  exit 1
fi
exec "$real_sqlite3" "\$@"
SQLITE
chmod +x "$vacuum_bin/sqlite3"
vacuum_marker="$work/vacuum-must-not-start"
if PATH="$vacuum_bin:$PATH" bash "$script" --data-dir "$vacuum_data" --backup-dir "$vacuum_backup" --stopped-confirmed \
    --start-command "touch '$vacuum_marker'" --health-url http://127.0.0.1:1/health \
    --verify-command "true" >"$work/vacuum.out" 2>&1; then
  fail "VACUUM failure was accepted"
fi
assert_eq "$(sha256sum "$vacuum_data/gpt-load.db" | cut -d' ' -f1)" "$vacuum_before"
assert_not_file "$vacuum_marker"
assert_file "$(find "$vacuum_backup" -maxdepth 1 -type f -name '*.tar.gz' -print -quit)"
grep -Fq 'simulated VACUUM failure' "$work/vacuum.out" || fail "VACUUM failure was not injected"
grep -Fq 'VACUUM failed' "$work/vacuum.out" || fail "VACUUM failure was not reported"

printf '7. real volume name resolution, stopped container, post-start health and encrypted read\n'
volume_runtime="$work/volume-runtime"
mkdir -p "$volume_runtime"
cat >"$volume_runtime/docker" <<'DOCKER'
#!/usr/bin/env bash
case "$1" in
  inspect)
    case "$3" in
      *State.Running*) printf 'false\n' ;;
      *Mounts*Name*) printf 'project42_gpt-load-data\n' ;;
      *Mounts*Source*) printf '%s\n' "$FIXTURE_DATA" ;;
      *) exit 1 ;;
    esac ;;
  volume) printf '%s\n' "$FIXTURE_DATA" ;;
  ps) exit 0 ;;
  *) exit 1 ;;
esac
DOCKER
cat >"$volume_runtime/curl" <<'CURL'
#!/usr/bin/env bash
[ "${*: -1}" = 'http://127.0.0.1:9999/health' ]
CURL
chmod +x "$volume_runtime/docker" "$volume_runtime/curl"
export FIXTURE_DATA="$data"
start_marker="$work/start-marker"
verify_marker="$work/verify-marker"
PATH="$volume_runtime:$PATH" bash "$script" --service stopped-container --backup-dir "$backup" \
  --start-command "touch '$start_marker'" --health-url http://127.0.0.1:9999/health \
  --verify-command "sqlite3 '$data/gpt-load.db' 'SELECT value FROM records;' | grep -qx fixture && test -s '$data/encryption.key' && touch '$verify_marker'" \
  >"$work/volume.out"
assert_file "$start_marker"
assert_file "$verify_marker"
assert_eq "$(find "$backup" -maxdepth 1 -name '*.tar.gz' | wc -l)" "2"

printf 'sqlite maintenance tests: PASS\n'
