#!/usr/bin/env bash
# Offline maintenance of application-managed SQLite. Backups contain credentials.
set -euo pipefail
umask 077

usage() {
  printf 'Usage: %s (--data-dir DIR | --volume NAME | --service CONTAINER) --backup-dir DIR [--service CONTAINER | --stopped-confirmed] [--dry-run] [--start-command COMMAND --health-url URL --verify-command COMMAND]\n' "$0" >&2
  exit 2
}

fail() { printf 'sqlite maintenance: %s\n' "$*" >&2; exit 1; }

data_dir='' volume='' service='' backup_dir='' stopped_confirmed=false dry_run=false
start_command='' health_url='' verify_command=''
while (( $# )); do
  case "$1" in
    --data-dir|--volume|--service|--backup-dir|--start-command|--health-url|--verify-command)
      (( $# >= 2 )) && [ -n "$2" ] || usage
      case "$1" in
        --data-dir) data_dir=$2 ;;
        --volume) volume=$2 ;;
        --service) service=$2 ;;
        --backup-dir) backup_dir=$2 ;;
        --start-command) start_command=$2 ;;
        --health-url) health_url=$2 ;;
        --verify-command) verify_command=$2 ;;
      esac
      shift 2 ;;
    --stopped-confirmed) stopped_confirmed=true; shift ;;
    --dry-run) dry_run=true; shift ;;
    *) usage ;;
  esac
done
[ -n "$backup_dir" ] || usage
[ -z "$data_dir" ] || [ -z "$volume" ] || usage
[ -n "$data_dir$volume$service" ] || usage
[ -z "$verify_command" ] || [ -n "$start_command" ] || usage
if [ -n "$start_command" ]; then
  [ -n "$health_url" ] && [ -n "$verify_command" ] || usage
fi
[ -z "$health_url" ] || [ -n "$start_command" ] || usage
[ -n "$service" ] || [ "$stopped_confirmed" = true ] || fail 'local data requires --stopped-confirmed (operator must stop all writers first)'

if [ -n "$service" ]; then
  command -v docker >/dev/null || fail 'docker is required for --service'
  running=$(docker inspect --format '{{.State.Running}}' "$service") || fail 'cannot inspect service'
  [ "$running" = false ] || fail 'service is running; stop it before maintenance'
  service_volume=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/app/data"}}{{.Name}}{{end}}{{end}}' "$service") || fail 'cannot inspect service mount'
  service_source=$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/app/data"}}{{.Source}}{{end}}{{end}}' "$service") || fail 'cannot inspect service mount'
  [ -n "$service_source" ] || fail 'service has no data mount at /app/data'
  if [ -n "$volume" ]; then
    [ "$volume" = "$service_volume" ] || fail 'volume does not match service /app/data mount'
  elif [ -n "$data_dir" ]; then
    [ "$(cd -- "$data_dir" && pwd -P)" = "$service_source" ] || fail 'data directory does not match service /app/data mount'
  elif [ -n "$service_volume" ]; then
    volume=$service_volume
  else
    data_dir=$service_source
  fi
fi
if [ -n "$volume" ]; then
  command -v docker >/dev/null || fail 'docker is required for --volume'
  mountpoint=$(docker volume inspect --format '{{.Mountpoint}}' "$volume") || fail 'cannot resolve Docker volume'
  [ -n "$mountpoint" ] || fail 'Docker volume mountpoint is empty'
  [ -z "$(docker ps -q --filter "volume=$volume")" ] || fail 'a container using the volume is running'
  data_dir=$mountpoint
fi
[ -d "$data_dir" ] && [ ! -L "$data_dir" ] || fail 'data directory must be an existing non-symlink directory'
[ -f "$data_dir/gpt-load.db" ] && [ ! -L "$data_dir/gpt-load.db" ] || fail 'managed gpt-load.db is missing or a symlink'
[ -d "$backup_dir" ] && [ ! -L "$backup_dir" ] || fail 'backup directory must be an existing non-symlink directory'
data_dir=$(cd -- "$data_dir" && pwd -P)
backup_dir=$(cd -- "$backup_dir" && pwd -P)
case "$backup_dir/" in "$data_dir/"*) fail 'backup directory must be outside the data directory' ;; esac
case "$data_dir/" in "$backup_dir/"*) fail 'data directory must be outside the backup directory' ;; esac
command -v sqlite3 >/dev/null || fail 'sqlite3 CLI is required on the host'
command -v tar >/dev/null || fail 'tar is required on the host'

if [ "$dry_run" = true ]; then
  printf 'Dry run: stopped check passed; would back up whole data directory, then checkpoint, VACUUM and integrity_check. No writes performed.\n'
  exit 0
fi

# Never replace an older backup, even on failure; keep the original files in place.
archive=$(mktemp "$backup_dir/gpt-load-offline-XXXXXXXX.tar.gz") || fail 'cannot create backup'
if ! tar -C "$data_dir" -czf "$archive" . || ! tar -tzf "$archive" >/dev/null; then
  fail 'backup failed validation; partial backup retained; original untouched'
fi
printf 'Sensitive whole-directory backup saved: %s (restrict access and retain for recovery)\n' "$archive"

check_integrity() {
  local result
  result=$(sqlite3 -batch -bail "$data_dir/gpt-load.db" 'PRAGMA integrity_check;') || fail 'SQLite integrity_check failed; backup retained; do not start'
  [ "$result" = ok ] || fail 'SQLite integrity_check is not ok; backup retained; do not start'
}
check_integrity
checkpoint=$(sqlite3 -batch -bail "$data_dir/gpt-load.db" 'PRAGMA wal_checkpoint(TRUNCATE);') || fail 'WAL checkpoint failed; backup retained; do not start'
case "$checkpoint" in 0\|*) ;; *) fail 'WAL checkpoint was busy; backup retained; do not start' ;; esac
sqlite3 -batch -bail "$data_dir/gpt-load.db" 'VACUUM;' >/dev/null || fail 'VACUUM failed; backup retained; do not start'
check_integrity
printf 'Checkpoint, VACUUM and integrity_check: OK\n'

if [ -n "$start_command" ]; then
  bash -c "$start_command" || fail 'start command failed; inspect service before recovery'
  healthy=false
  for (( attempt=0; attempt<24; attempt++ )); do
    if curl --fail --silent --show-error --max-time 5 --output /dev/null "$health_url" 2>/dev/null; then healthy=true; break; fi
    sleep 5
  done
  [ "$healthy" = true ] || fail 'health failed after start; backup retained'
  bash -c "$verify_command" >/dev/null 2>&1 || fail 'encrypted configuration verification failed; backup retained'
  printf 'Post-start health and encrypted configuration verification: OK\n'
else
  printf 'Service remains stopped. After starting, verify /health and an authenticated encrypted setting read before returning traffic.\n'
fi
