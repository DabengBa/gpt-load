# SQLite Maintenance

## 不可反推约束

- 备份必须含 WAL（`VACUUM INTO` 或等价一致性快照）；验证副本要显式执行完整性检查，不以「备份成功」代替。
- `applyMigrationsLocked` 只在 pending migration 时跑 `PRAGMA foreign_key_check`；纯启动重放不扫描既有行。要证明副本无违反必须显式检查。
- 维护/恢复操作前确认无 writer；恢复后用新实例端口验证再切换。

## 边界

具体命令与脚本见 `scripts/`（sqlite maintenance 脚本随代码演进）；本文只保不可反推的约束。
