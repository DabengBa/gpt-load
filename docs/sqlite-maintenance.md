# SQLite maintenance and recovery

This guide applies to the **application-managed SQLite** deployment: leave
`DATABASE_DSN` empty so the database is `${DATA_DIR}/gpt-load.db` (the official
Compose data directory is `/app/data`). A non-empty `DATABASE_DSN` is an
operator-managed external database according to the deployment configuration;
this guide does not require or announce a forced migration to an external
database. The offline script below is only for the application-managed data
directory.

The Chinese summary is included below so the English and Chinese README
entries point to one maintained procedure.

## Runtime retention and SQLite state

- Request-log retention defaults to **7 days**. An administrator can change it
  in the persisted management settings; the supported range is 1–365 days.
  The effective value and whether it comes from the default or a persisted
  system setting are reported by the authenticated `GET
  /api/sqlite-maintenance` endpoint.
- The retention runtime runs one sweep **immediately at startup**, then runs
  again every hour. The sweep deletes expired request logs in batches. It does
  not run `VACUUM` or a WAL checkpoint, so retention is not an online database
  compaction mechanism.
- `DELETE` makes SQLite pages reusable, but does not automatically shrink the
  database file or remove freelist pages from disk. Inspect `freelist_pages`
  in the status response and use the offline procedure when compaction is
  appropriate.
- File-backed, application-managed SQLite defaults to **WAL**. When
  `DATABASE_DSN` is empty, `SQLITE_JOURNAL_MODE` is supported as a managed-only
  override and accepts `wal`, `delete`, `truncate`, or `persist`. Startup parses
  and validates the configured value, then validates the actual journal mode;
  the authenticated status endpoint reports that actual value. For external
  SQLite, configure the journal mode through the DSN `_pragma` parameter
  instead; `SQLITE_JOURNAL_MODE` does not apply.
- The status endpoint is authenticated and read-only. Its response has the
  following shape (values vary by deployment):

  ```json
  {
    "data": {
      "retention_days": 7,
      "retention_source": "default",
      "sweep_interval_seconds": 3600,
      "sqlite": {
        "journal_mode": "wal",
        "freelist_pages": 0,
        "maintenance_mode": "offline_only",
        "error": ""
      }
    }
  }
  ```

  `maintenance_mode` being `offline_only` means the application status path
  only observes SQLite. It does not perform `VACUUM` or checkpoint while the
  application is running. Inspection failures are reported as the bounded
  `inspection_failed` error value without exposing a DSN or filesystem path.

## Backup sets and online snapshots

For a managed database, treat the database and key material as one recovery
set:

- `gpt-load.db`;
- `gpt-load.db-wal` and `gpt-load.db-shm` when present;
- `auth.key`;
- `encryption.key`; and
- the other files in `${DATA_DIR}` that belong to the deployment's runtime
  state.

The `encryption.key` is required to decrypt stored channel credentials. Losing
or replacing it makes existing encrypted credentials unavailable; never print
key contents or put them in shell history, logs, screenshots, or tickets.

The [deployment guide](deployment.md) also shows SQLite's online `.backup` command. That flow
copies a consistent **database snapshot while the service is online** and is
useful for investigation or a database-only snapshot. It does not archive the
whole data directory and therefore does not by itself preserve the keys,
WAL/SHM sidecars, or all runtime state. Keep the required keys with any copy
intended for recovery. For offline disaster recovery and compaction, stop all
writers and use the full-directory archive below instead.

## Offline maintenance command

The command is deliberately offline and **never stops the service itself**.
Stop the Compose service and all other writers first. The backup directory
must already exist, be outside the data directory, and have enough persistent
disk space. The archive is sensitive, is created with restricted permissions,
and is never overwritten by a later run.

The script accepts these parameters:

| Parameter | Meaning |
| --- | --- |
| `--data-dir DIR` | Existing managed data directory. Requires `--stopped-confirmed` unless a matching `--service` is also supplied. |
| `--volume NAME` | Actual Docker volume name. The script resolves its mount point and requires `--stopped-confirmed` unless a matching `--service` is also supplied. |
| `--service CONTAINER` | Stopped Docker container. The script reads its real `/app/data` mount and checks that the container is stopped. It can resolve the actual volume without `--stopped-confirmed`. |
| `--backup-dir DIR` | Existing directory outside the data directory where a new archive is created; required. |
| `--stopped-confirmed` | Explicit operator confirmation for local/data-dir or volume mode that all writers have stopped. |
| `--dry-run` | Performs target, mount, path, and stopped checks only; it does not create a backup, query SQLite, or start a service. |
| `--start-command COMMAND` | Optional command to start the service after successful maintenance. Must be supplied together with the two verification options below. |
| `--health-url URL` | Health URL polled after `--start-command` succeeds. |
| `--verify-command COMMAND` | Deployment-supplied authenticated read of a real encrypted configuration item. Its output is suppressed and it must succeed. |

The three startup options are all-or-nothing. Without them, the script leaves
the service stopped and the operator must check health and encrypted settings
manually before returning traffic. The verification command must prove that
the original key material can decrypt a real configuration value; checking
only that a key file exists is insufficient. Do not put credentials directly
in the command line.

Examples:

```bash
# Resolve /app/data from the stopped container, then preview the operation.
bash scripts/sqlite-maintenance.sh \
  --service gpt-load --backup-dir /secure/backups --dry-run

# Run offline maintenance and leave the service stopped for manual verification.
bash scripts/sqlite-maintenance.sh \
  --service gpt-load --backup-dir /secure/backups

# Use an explicitly resolved Docker volume without a container target.
bash scripts/sqlite-maintenance.sh \
  --volume actual_project_gpt-load-data \
  --backup-dir /secure/backups --stopped-confirmed

# Non-Docker rehearsal or native deployment.
bash scripts/sqlite-maintenance.sh \
  --data-dir /srv/gpt-load/data --backup-dir /srv/gpt-load-backups \
  --stopped-confirmed
```

The script first creates and validates a complete data-directory archive. It
then runs `integrity_check`, `PRAGMA wal_checkpoint(TRUNCATE)`, `VACUUM`, and a
second `integrity_check`. A busy checkpoint, failed `VACUUM`, failed integrity
check, or any other maintenance error exits non-zero, retains the archive and
original files, and refuses to start the service. Leave the service stopped;
restore the entire archive before attempting startup if the data needs to be
rolled back. A successful maintenance run is not a promise that a production
Docker daemon or encrypted configuration has been exercised by the repository
test suite.

With the optional startup trio, the script starts the supplied command, polls
the health URL, and then runs the authenticated encrypted-configuration
verification. A health success alone is not enough. If either post-start
check fails, treat the service as unverified and recover from the retained
full-directory archive as appropriate.

For the isolated regression rehearsal, run:

```bash
bash scripts/test-sqlite-maintenance.sh
```

That test uses temporary SQLite/filesystem fixtures and mocked Docker/curl
boundaries. It does not constitute live production Docker or real encrypted
settings evidence.

## 中文说明

本页适用于**应用托管的 SQLite**：保持 `DATABASE_DSN` 为空，数据库位于
`${DATA_DIR}/gpt-load.db`，官方 Compose 中对应容器内的 `/app/data`。非空
`DATABASE_DSN` 按现有部署语义属于运维方管理的外部数据库；本页不要求、也不
宣称必须迁移到外部数据库。下面的停机脚本只处理应用托管的数据目录。

### 运行时保留与状态

- 请求日志默认保留 **7 天**。管理员可以在持久化的管理设置中修改，支持范围
  为 1–365 天。认证后的 `GET /api/sqlite-maintenance` 会返回生效天数以及
  `default` 或 `system_setting` 来源。
- retention runtime **启动时立即清理一次**，之后每小时清理一次。清理按批次
  删除过期请求日志；不会在线执行 `VACUUM` 或 WAL checkpoint，因此清理不等于
  在线压缩数据库。
- `DELETE` 只会让 SQLite 页面可复用，不会自动缩小数据库文件或清除磁盘上的
  freelist pages。应通过状态中的 `freelist_pages` 观察，再按需安排停机维护。
- 应用托管的文件 SQLite 默认使用 **WAL**。当 `DATABASE_DSN` 为空时，支持
  使用 `SQLITE_JOURNAL_MODE` 覆盖，且仅适用于应用托管数据库；可选值为
  `wal`、`delete`、`truncate`、`persist`。启动时会解析并校验配置值，再校验
  实际 journal mode；认证状态 API 返回实际模式。外部 SQLite 应改用 DSN 的
  `_pragma` 参数配置，`SQLITE_JOURNAL_MODE` 对外部数据库不生效。
- API 是需要认证的只读入口，返回 `retention_days`、`retention_source`、
  `sweep_interval_seconds` 以及 `sqlite.journal_mode`、`freelist_pages`、
  `maintenance_mode`、`error`。其中 `maintenance_mode` 为 `offline_only`，表示
  应用只观察，不在运行期间执行 `VACUUM` 或 checkpoint。

### 备份集合与脚本

灾备应把 `gpt-load.db`、可能存在的 `gpt-load.db-wal`/`gpt-load.db-shm`、
`auth.key`、`encryption.key` 以及数据目录中的其他运行时文件作为一个整体保存。
`encryption.key` 丢失或替换后，已有加密渠道凭据无法解密；不要打印密钥内容。

`docs/deployment.md` 中的 SQLite `.backup` 是服务在线时的**数据库快照**，适合
在线排障或数据库副本；它不是包含密钥、WAL/SHM 和运行时状态的整目录灾备。需要
停机压缩或灾备时，停止所有写入者，使用
[`scripts/sqlite-maintenance.sh`](../scripts/sqlite-maintenance.sh) 创建整目录归档。

该脚本不会自行停止服务。先停服务，再根据实际部署使用
`--service CONTAINER`、`--volume NAME` 或 `--data-dir DIR`，并始终提供必需的
`--backup-dir DIR`。没有 `--service` 时，`--volume`/`--data-dir` 还需要
`--stopped-confirmed`；`--dry-run` 只做停机、挂载和路径检查，不备份、不查询 SQLite、
不启动服务。可选的 `--start-command` 必须和 `--health-url`、`--verify-command`
一起提供；启动后既要检查 health，也要用部署方提供的认证读取验证真实加密配置，
不能只检查密钥文件存在。

脚本会先完整归档并校验数据目录，再执行 integrity check、WAL checkpoint、
`VACUUM` 和第二次 integrity check。checkpoint busy、`VACUUM` 失败或完整性失败
都会非零退出、保留归档和原始文件，并拒绝启动；失败后保持停机，必要时先恢复
整目录归档。仓库的脚本测试使用临时 SQLite 和 mock Docker/curl 边界，不代表真实
生产 Docker 或真实加密配置已经验证。

```bash
bash scripts/sqlite-maintenance.sh --service gpt-load \
  --backup-dir /secure/backups --dry-run
bash scripts/sqlite-maintenance.sh --service gpt-load \
  --backup-dir /secure/backups
bash scripts/test-sqlite-maintenance.sh
```
