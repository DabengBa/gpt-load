---
description: "Classify upstream insufficient-balance failures as a distinct billing category with a 24h credential cooldown and a Home attention alert."
kind: technical
topic: failure-classification
code:
  paths:
    - internal/execution/contracts.go
    - internal/channel/failure_class.go
    - internal/health/execution_judge.go
    - internal/storage/migrations/0023_billing_failure_category.go
    - web/src/frontends/classic/features/home/attention.ts
---

# 余额不足失败分类与首页告警

## Review

第二评审来源：Claude Code CLI 不可用（proxy auth 401）且共享脚本 `invoke-claude-readonly-review.ps1` 缺失，本次仅主评审。

### 已确认发现并修复

- [x] Review: CPA provider 直写 hint 绕过 neutralFailureHint——antigravity `INSUFFICIENT_G1_CREDITS_BALANCE` 被归 rate_limited、grok 402 被归 candidate_unavailable、claude/codex 的 billing marker 会被泛化分支抢先 (severity: high; scope: 分类链路; evidence: internal/execution/cpa/{antigravity,claude,codex,grok}_provider.go; proof: 各 provider 增加 InsufficientBalanceSignal 前置分支 + 两个测试改判通过)
- [x] Review: `request_log_attempts` 表 CHECK 约束 `chk_request_log_attempt_failure_category` 不含 `billing`，billing attempt 写入会被拒 (severity: high; scope: 持久化; evidence: internal/storage/models/request.go + migrations/0006; proof: 新增迁移 0023 重建表并扩约束，测试覆盖保留数据/billing 可写/非法分类仍拒/幂等)
- [x] Review: `fixedErrorSummaries` 缺 `upstream_insufficient_balance`，无供应商摘要时日志落成通用 "Upstream request failed." (severity: medium; scope: 日志可观测; evidence: internal/gateway/request_log.go:884; proof: 目录补齐 + linux vet 编译通过)
- [x] Review: web-search 操作的效果剥离门未豁免 billing，会剥掉 24h 凭据冷却 (severity: medium; scope: 决策落地; evidence: internal/gateway/execution_boundary.go:133; proof: billing 加入与 auth/invalid_key 同级的账号态例外)
- [x] Review: 既有测试 `generic payment required switches candidate` 仍按旧预期断言 402→http_4xx_retry (severity: low; scope: 测试; evidence: execution_judge_v2_test.go; proof: 已改为断言 billing.insufficient_balance，health 包测试通过)

### 未解决证据缺口

- `internal/{storage,requestlog,control,gateway,execution/bifrost,execution/cpa}` 在本机 Windows 上无法执行测试（`securefile`/`catalog` 仅 linux 变体）。已用 `GOOS=linux go vet` 验证全部相关包（含测试文件）编译通过；运行时验证依赖 CI/Linux 环境。
- 前端无单元测试设施；proof 面为 `vue-tsc` type-check + eslint（均已通过）。

### 复核过的边界

- billing 判定先于 401/403/429/4xx 状态分支（channel、bifrost、cpa、judge 四层一致）；`quota_exceeded`/`resource_exhausted`/`usage_limit_reached` 仍归 rate_limited（有测试锚定）。
- 决策走 `decisionForExecutionCategory` 后仍经 `constrainOperationReplay`/`constrainCommittedDecision`，committed 响应安全约束不被绕过（有测试）。
- 冷却只延长不缩短：`SetCooldownWithChange` 语义保证 billing 的 24h 不会被后续限流冷却覆盖回短值。
- Home Attention 仅暴露 group+数量+最早重探时刻，不泄露凭据身份；`credential_status=cooldown` 是合法路由筛选值。
- `attentionTotal` 对 billing 按凭据数计数；行上限与汇总行为不变。

- [x] Review complete
