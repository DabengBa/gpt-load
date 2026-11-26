# Capture Runtime 与 ZIP 契约

## 运行时约束（不可反推）

- opt-in：Unix 环境 `DEBUG_CAPTURE_ENABLED=true` 才启用；默认关闭；probe/ModelTrace 同样只进 opt-in store。
- session 创建后 24 小时过期；启动时 + 每小时 best-effort cleanup（进程停止或反复失败会延迟）。「retained 24 hours」≠物理删除：异步清理，DB pages/backups/exported archives 不安全擦除。
- 上限 64 sessions / 8 MiB body / 128 pending events；超限只让 capture 失败，不改数据面响应。chunk zstd/identity、64 KiB body chunk、per-row `encoding` 列；compression migration 后不得把旧 binary 回滚到该 DB。
- health `debug_capture` 计数是未过期 DB 行非内存 slot；`running=false` 不等于 cleanup 停止；`last_sweep_at_ms` 是尝试非成功，`last_failure_at_ms` 是最近失败；禁新 session 不撤销未过期 session 读权限。
- capture boundary 从 Gateway data-plane `/v1`、`/v1beta` 认证前起；可能含 Authorization/Cookie/API key/prompt/body；必须保护 AUTH_KEY、DB、backup、export；禁止不可信共享部署。
- 空完成排错顺序：先 structured log 找 `error_code=upstream_empty_completion`，再 raw capture 按 attempt 配对。

## 管理 API（仅管理员）

```text
GET /api/debug-captures?request_id=<request-id>
GET /api/debug-captures/<capture-id>
GET /api/debug-captures/<capture-id>/download
```

ZIP 预期：

```text
manifest.json
attempts/<attempt-id>/metadata.json
attempts/<attempt-id>/parts/request.headers
attempts/<attempt-id>/parts/request.body
attempts/<attempt-id>/parts/response.headers
attempts/<attempt-id>/parts/response.body
```
