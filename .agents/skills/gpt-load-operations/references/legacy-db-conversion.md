# Legacy DB Conversion

仅支持最后一次带 `0003_remove_observation_fresh_until` 的 release 起的库；startup refusal 不等于 conversion refusal。

## 不可反推约束

- 只允许离线 `gpt-load convert-legacy-db --source --target` 或全新库；无原地迁移、无兼容层、无自动删除。
- source file 只读打开（SQLite `VACUUM INTO`，含 WAL）；target 必须不存在，no-replace 原子发布；失败不留 target。
- 保留 groups、加密凭据、access keys、quota、settings；不迁 request/usage/debug 数据。
- runbook：副本试转 → stop 服务并确认无 writer → 整套 data/auth/encryption/env/compose 备份 → 新目录/卷离线无网络转换 → 同 encryption key → spare port 启新实例 → 验证后切换。
- `TMPDIR` 需要 source-size 的持久空间：过程产生两个 DB-size 临时副本；tmpfs 会报 `disk I/O error` / `database or disk is full`，这不是 source 判定。
