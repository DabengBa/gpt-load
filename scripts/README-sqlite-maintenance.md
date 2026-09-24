# Offline managed SQLite maintenance

Requires host `bash`, `sqlite3`, `tar`, `curl`; Docker CLI for container/volume modes. Stop all writers first (`docker compose stop gpt-load`); this script **never stops the service itself**. Only use for the application-managed `${DATA_DIR}/gpt-load.db` (empty `DATABASE_DSN`). Backup directory must already exist, must be outside the data directory, and must have adequate free space; the resulting `.tar.gz` is sensitive and contains all keys, WAL/SHM, and runtime state. Secure it and keep it for recovery. The original data is never deleted, and existing archives are never overwritten. If maintenance fails after the backup, leave the service stopped and restore the entire archived data directory before attempting startup.

```bash
# Resolve the actual volume via the stopped container mount (not the logical Compose volume name).
bash scripts/sqlite-maintenance.sh --service gpt-load --backup-dir /secure/backups --dry-run
bash scripts/sqlite-maintenance.sh --service gpt-load --backup-dir /secure/backups

# Explicit actual Docker volume name (e.g. docker inspect gpt-load -> .Mounts[].Name).
bash scripts/sqlite-maintenance.sh --volume actual_project_gpt-load-data \
  --backup-dir /secure/backups --stopped-confirmed

# Non-Docker / isolated rehearsal: operator confirms there are no writers.
bash scripts/sqlite-maintenance.sh --data-dir /tmp/test-data \
  --backup-dir /tmp/test-backups --stopped-confirmed --dry-run
```

With `--start-command 'docker compose up -d gpt-load'`, also pass `--health-url http://127.0.0.1:3001/health` **and** `--verify-command '...'`. The verification command must perform an authenticated read of a real encrypted configuration item with the original key material (not merely check that the key file exists); a zero exit code confirms it succeeded. Build this command for your deployment without printing keys or embedding keys directly in the CLI/history. `--verify-command` stdout/stderr is suppressed; no start is attempted on backup, checkpoint, VACUUM or integrity failure. Without these three startup options the script leaves the service stopped for manual restart and verification. `--dry-run` performs stop/mount/path checks and no backup, SQLite query, or startup. `--data-dir` plus `--service` checks that the supplied directory equals the container's actual `/app/data` mount.

Run the isolated integration rehearsal with `bash scripts/test-sqlite-maintenance.sh` (test owns and removes its `mktemp` data directory; normally <5 seconds). A successful `--verify-command` is operator-supplied: the script cannot infer which row contains encrypted configuration or manufacture an authenticated test credential.
