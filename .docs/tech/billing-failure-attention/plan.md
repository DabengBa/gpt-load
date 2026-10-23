---
description: "Classify upstream insufficient-balance failures as a distinct billing category, cool a credential only after consecutive billing failures using the published blacklist_release_seconds window, and surface it as a Home attention alert."
kind: technical
topic: failure-classification
code:
  paths:
    - internal/execution/contracts.go
    - internal/channel/failure_class.go
    - internal/health/execution_judge.go
    - internal/state/registry.go
    - internal/gateway/handler.go
    - internal/storage/migrations/0023_billing_failure_category.go
    - web/src/shared/domain/home/attention.ts
    - web/src/frontends/astryx/features/home/HomeAttention.tsx
---

# 余额不足失败分类与首页告警

> 当前范围更新：内置订阅及 CPA 已删除；下方旧评审中 CPA provider 的修复记录仅为历史，不代表当前依赖或执行路径。标准 API 的余额失败分类与首页告警保留。

## 当前实现与验证表面

本节按当前源码定位实现与可复用验证入口，不声明本次已执行这些测试。
下方旧评审中的通过记录、Windows 限制及 Vue 工具链均只属于当时环境，
不能作为当前 React/shared 实现的验证结论。

| 边界 | 当前实现 | 当前验证入口与状态 |
|---|---|---|
| 余额信号分类 | `internal/execution/contracts.go`、`internal/channel/failure_class.go` | `internal/execution/contracts_test.go`、`internal/channel/failure_class_test.go`；本次仅核对路径，未运行 |
| 冷却时长参数与效果落地 | `internal/health/execution_judge.go`（`DecisionContext.BillingCooldown`）、`internal/gateway/handler.go`（复用 `blacklistReleaseDeadline`） | `internal/health/execution_judge_billing_test.go`、`internal/gateway/billing_streak_test.go` |
| 连续两次门限与连续段状态 | `internal/state/registry.go`（`BillingStreak`、`RecordBillingFailure`、`ResetBillingStreak`）、`internal/gateway/handler.go`（`billingStreakThreshold`） | `internal/state/weights_test.go`、`internal/gateway/billing_streak_test.go` |
| committed 安全约束 | `internal/health/execution_judge.go`（`constrainCommittedDecision`） | `internal/health/execution_judge_billing_test.go` |
| billing 持久化约束 | `internal/storage/migrations/0023_billing_failure_category.go` | `internal/storage/migrations/0023_billing_failure_category_test.go`；本次仅核对路径，未运行 |
| 首页聚合与显示 | `web/src/shared/domain/home/attention.ts`、`web/src/frontends/astryx/features/home/HomeAttention.tsx` | 当前 `web/package.json` 的 `type-check`（`tsc`）及 `lint`（ESLint 等）脚本；本次未运行，不等同于 UI 行为证明 |

共享聚合按分组统计 billing 冷却凭据数量并取最早冷却结束时刻，React
`HomeAttention.tsx` 显示该聚合结果；当前实现不以 `vue-tsc` 为验证入口。

## 冷却门限与时长（当前实现）

- 门限（连续 2 次）：`health` 仍只给出「换候选 + 冷却凭据」的意图与截止时间；
  `gateway` 把这条 billing 冷却交给 registry 的 `RecordBillingFailure`，在**同一
  次加锁内**完成「计数 → 判断阈值 → 落冷却 → 清零」，因此计数与冷却原子，
  并发失败也只有一次真正落冷却。只有同一凭据的**连续**两次 billing 失败才冷却；
  首次 billing 只换候选保住请求，凭据继续参与调度。
- 打断：任何一次成功或其他结果都结束连续段。`applyGroupDecisionEffectForEntry`
  对所有非 billing 决策调用 `resetBillingStreakForResult`（覆盖 `EffectNone`
  的上游 4xx 与模型 scope 提前返回分支），成功走 `recordCredentialSuccess`；
  registry 的 `ClearFailure`/`Recover`/`IncrFailure`/`RecordFailureWithBlacklist`/
  已落地的冷却写入同样清零连续段。
- 触发后重置：冷却开始（`SetCooldownWithChange`/`SetCooldownWithChangeIfVersion`）
  或 `RecordBillingFailure` 落冷却时立即结束连续段，窗口结束后的下一次 billing
  从 1 重新计数。
- 时长：`judge` 用 `DecisionContext.BillingCooldown` 计算截止时间，`gateway` 用
  同一个 `handler.blacklistReleaseDuration()` 填充它——即系统熔断释放参数
  `blacklist_release_seconds`，与黑名单释放窗口完全一致，固定 24h 常量已删除。
  未注入时 `normalizeDecisionContext` 使用该参数的出厂默认值
  （`state.DefaultRuntimeSettings().BlacklistReleaseSeconds`），不退到限流冷却，
  保证 `Decision.Validate` 要求的「冷却必须带截止时间」成立。
- 并发安全：`RecordBillingFailure` 在计数前校验凭据版本（`expectedVersion`，
  0 表示不限），过期版本既不推进连续段也不写冷却；计数与冷却都在 registry
  锁内完成。
- 重试指令不变：billing 决策的 `Retry`/`RuleID`（`billing.insufficient_balance`）
  与 committed 安全约束（`constrainCommittedDecision`）行为保持不变。

## 历史 Review（原始评审记录）

以下保留既有评审原文，包括其 proof、环境限制与完成勾选；它们是历史记录，
不是本次复验结果或当前待办。原文未注明评审日期，不补造时间或历史成功证据。
原文中的文件行号及工具链名称也只用于定位当时证据。

第二评审来源：Claude Code CLI 不可用（proxy auth 401）且共享脚本 `invoke-claude-readonly-review.ps1` 缺失，本次仅主评审。

### 已确认发现并修复

- [x] Review: CPA provider 直写 hint 绕过 neutralFailureHint——antigravity `INSUFFICIENT_G1_CREDITS_BALANCE` 被归 rate_limited、grok 402 被归 candidate_unavailable、claude/codex 的 billing marker 会被泛化分支抢先 (severity: high; scope: 分类链路; evidence: internal/execution/cpa/{antigravity,claude,codex,grok}_provider.go; proof: 各 provider 增加 InsufficientBalanceSignal 前置分支 + 两个测试改判通过)
- [x] Review: `request_log_attempts` 表 CHECK 约束 `chk_request_log_attempt_failure_category` 不含 `billing`，billing attempt 写入会被拒 (severity: high; scope: 持久化; evidence: internal/storage/models/request.go + migrations/0006; proof: 新增迁移 0023 重建表并扩约束，测试覆盖保留数据/billing 可写/非法分类仍拒/幂等)
- [x] Review: `fixedErrorSummaries` 缺 `upstream_insufficient_balance`，无供应商摘要时日志落成通用 "Upstream request failed." (severity: medium; scope: 日志可观测; evidence: internal/gateway/request_log.go:884; proof: 目录补齐 + linux vet 编译通过)
- [x] Review: web-search 操作的效果剥离门未豁免 billing，会剥掉当时固定的 24h 凭据冷却 (severity: medium; scope: 决策落地; evidence: internal/gateway/execution_boundary.go:133; proof: billing 加入与 auth/invalid_key 同级的账号态例外)
- [x] Review: 既有测试 `generic payment required switches candidate` 仍按旧预期断言 402→http_4xx_retry (severity: low; scope: 测试; evidence: execution_judge_v2_test.go; proof: 已改为断言 billing.insufficient_balance，health 包测试通过)

### 未解决证据缺口

- `internal/{storage,requestlog,control,gateway,execution/bifrost,execution/cpa}` 在本机 Windows 上无法执行测试（`securefile`/`catalog` 仅 linux 变体）。已用 `GOOS=linux go vet` 验证全部相关包（含测试文件）编译通过；运行时验证依赖 CI/Linux 环境。
- 前端无单元测试设施；proof 面为 `vue-tsc` type-check + eslint（均已通过）。

### 复核过的边界

- billing 判定先于 401/403/429/4xx 状态分支（channel、bifrost、cpa、judge 四层一致）；`quota_exceeded`/`resource_exhausted`/`usage_limit_reached` 仍归 rate_limited（有测试锚定）。
- 决策走 `decisionForExecutionCategory` 后仍经 `constrainOperationReplay`/`constrainCommittedDecision`，committed 响应安全约束不被绕过（有测试）。
- 冷却只延长不缩短：`SetCooldownWithChange` 语义保证 billing 的冷却窗口不会被后续更短的限流冷却覆盖回短值（窗口长度已改为 `blacklist_release_seconds`）。
- Home Attention 仅暴露 group+数量+最早重探时刻，不泄露凭据身份；`credential_status=cooldown` 是合法路由筛选值。
- `attentionTotal` 对 billing 按凭据数计数；行上限与汇总行为不变。

- [x] Review complete
