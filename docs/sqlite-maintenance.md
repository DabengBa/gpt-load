# SQLite maintenance and recovery

> Breaking / 破坏性变更：本次删除内置订阅/CPA 的版本无法原地打开旧库。旧库只能停机后用 `gpt-load convert-legacy-db` 离线转换，或改用全新数据库。拒绝来自**启动**而非转换：应用启动时拒绝含退役 migration `0003` 的旧库且不就地修改；转换器反而要求台账含 `0003`，因此 API-only 旧库可以正常转换，只有含退役订阅结构或丢失 `0003` 的库会被转换器拒绝。转换还需要持久磁盘上的额外临时空间（`TMPDIR` 至少容纳一份源库大小的快照），不要指向小容量 tmpfs。本页的常规更新/恢复流程不会转换旧库，也不会把它接入此构建；转换步骤见 [旧库离线转换](../.docs/tech/legacy-db-conversion.md)。保留旧实例和备份。本次不部署、不修改业务库；启动拒绝不保证 SQLite 文件/WAL 零副作用。详见 [README](../README_CN.md#快速开始)。

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

## Verifying a copy explicitly

Starting the application does not prove a database is free of foreign-key
violations: `PRAGMA foreign_key_check` only runs inside the transaction that
applies pending migrations, and a start with no pending migration just
revalidates the migration ledger and each applied migration's schema checks
(owned by [Deployment
Prerequisites](../.docs/tech/frozen-model-pricing.md#deployment-prerequisites)).
The offline maintenance script above is also limited to `integrity_check`,
checkpoint, and `VACUUM`.

Whenever a database copy must be proven consistent — a pre-upgrade rehearsal, a
restored archive, a converted legacy database, or a snapshot kept as evidence —
run both checks on that copy yourself, before starting any version on it:

```bash
test -s "$COPY_DIR/gpt-load.db" || { echo 'Missing or empty copy' >&2; exit 1; }
sqlite3 -readonly -batch "$COPY_DIR/gpt-load.db" 'SELECT count(*) FROM schema_migrations;' # must succeed
sqlite3 -readonly -batch "$COPY_DIR/gpt-load.db" 'PRAGMA integrity_check;'    # must print ok
sqlite3 -readonly -batch "$COPY_DIR/gpt-load.db" 'PRAGMA foreign_key_check;'  # must print nothing
```

`foreign_key_check` prints one row per violating child row (table, rowid,
parent table, constraint index); any output means the copy is damaged and must
not be started or promoted. Both statements are read-only, so they are safe
against an opened copy or a `?mode=ro` URI. Check the copy, never the live
volume, and record the command and its output with the copy.

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
- Debug Communication Capture 独立按固定 **24 小时**过期并清理 chunks。新版本
  将新写入 chunk 按行使用 zstd（仅压缩结果更小时），历史 chunk 保持原始字节；读取
  和 ZIP 下载会还原原始内容。删除过期 capture 只释放 SQLite 内部页面供复用，不保证
  数据库文件缩小；需要归还文件系统空间时按本文停机维护流程检查备份和余量后执行。
- 应用升级执行新的 chunk encoding schema migration 后，不支持把旧应用镜像直接回滚到
  已升级的数据库。回滚需按部署恢复流程恢复升级前完整数据目录和对应镜像。
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

### 显式校验副本

启动本身不能证明数据库没有外键违规。存在待执行迁移时，`PRAGMA foreign_key_check`
会跑在迁移事务内并让本次启动失败；但没有待执行迁移的启动只重新校验迁移台账和各
个已应用迁移的 schema 检查，故意不扫描既有数据行。上面的停机维护脚本也只执行
integrity check、checkpoint 和 `VACUUM`，不做外键检查。

凡是需要证明某个副本一致——升级前演练、恢复的归档、转换后的旧库、作为证据保留的
快照——都要在启动任何版本之前，自行对这个副本执行两项检查：

```bash
test -s "$COPY_DIR/gpt-load.db" || { echo '副本不存在或为空' >&2; exit 1; }
sqlite3 -readonly -batch "$COPY_DIR/gpt-load.db" 'SELECT count(*) FROM schema_migrations;' # 必须成功
sqlite3 -readonly -batch "$COPY_DIR/gpt-load.db" 'PRAGMA integrity_check;'    # 必须输出 ok
sqlite3 -readonly -batch "$COPY_DIR/gpt-load.db" 'PRAGMA foreign_key_check;'  # 必须没有任何输出
```

`foreign_key_check` 会为每条违规子行输出一行（表名、rowid、父表、约束序号），只要
有输出就说明副本已损坏，不得启动或切换使用。两条语句都是只读的，可以安全地作用于
已打开的副本或 `?mode=ro` URI。检查对象必须是副本而不是生产卷，并把命令和输出与该
副本一起留存。

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
