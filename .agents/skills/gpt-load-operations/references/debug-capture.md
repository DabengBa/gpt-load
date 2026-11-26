# Debug Communication Capture

## 不可反推约束

- opt-in：Unix 环境 `DEBUG_CAPTURE_ENABLED=true` 才启用，默认关闭；probe/ModelTrace 同样只进 opt-in store。
- session 24 小时过期；启动时 + 每小时 best-effort cleanup，进程停止或反复失败会延迟清理。retention 不等于物理删除：异步清理，DB pages/backups/exported archives 不做安全擦除。
- 上限 64 sessions / 8 MiB body / 128 pending events，超限只让 capture 失败，不改数据面响应。
- health `debug_capture` 计数是未过期 DB 行；`running=false` 不代表 cleanup 停止；`last_sweep_at_ms` 是尝试时间非成功时间，`last_failure_at_ms` 是最近失败。
- 禁止新 session 不撤销未过期 session 的读权限。
- capture 从 Gateway data-plane `/v1`、`/v1beta` 认证前起始；可能含 Authorization/Cookie/API key/prompt/body，必须保护 AUTH_KEY、DB、backup、export，禁止不可信共享部署。
- evidence 边界：只能证明「captured bytes」，不能推断 TLS/socket framing；所有 observation/callback 排空后无 completion 记 `no_observation_source`，不能据 executor result 断言发生了 wire 交换。
- 空完成排错顺序：先 structured log 找 `error_code=upstream_empty_completion`，再 raw capture 按 attempt 配对。
