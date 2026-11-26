# Frozen Model Pricing Deployment

## 不可反推约束

- receipt 是 v7-only：不解释 v5/v6 receipt，历史不重定价/不转换；部署前需三项独立批准：current schema + exact active ledger（retired IDs 仍留 ledger）、旧 receipts 的外部归档/读取方案、旧 idempotency history/replay policy。
- 非原地升级、无兼容层、无自动删除；`applyMigrationsLocked` 只在 pending migration 时跑 `PRAGMA foreign_key_check`，副本验证须显式检查。
- 计算细节见 `web/scripts/receipt-v7.test.ts`、`web/scripts/receipt-formula.test.ts`（真值源）。

## 边界

receipt 算术与字段表不在本文；以测试与代码为准。
