# Main Ahead 提交评估

## 基线

- 更新时间：`2026-09-19T09:27:47+08:00`
- `dev` 评估基线：`origin/dev@d356e731`（PR #96 合并后）；当前评估工作树基于该提交。
- `origin/main`：`b517f713`（#691，`2026-09-18T19:33:36+08:00`）。
- 分析截止：已分析到 #691；后续只考虑 `b517f713` 的后继提交（预期从 #692 开始），不重复处理 #673–#691。
- 分叉统计：`origin/dev...origin/main = 284 70`；仅用于观察长期分叉，不作为待移植数量。
- 本轮已核对范围：#673 至 #691；其中 #679、#689 没有对应的 `origin/main` 提交，另含依赖更新 #640。
- 策略：不整体合并 `origin/main`，只按 `dev` 合同进行行为级移植。判定沿用一组一凭据、entry 级健康、全局重试预算、已提交流不可重放、续接/WS 身份隔离和迁移只追加。

## 上一轮取舍（本轮更新）

| 范围 | 上轮取舍 | 当前状态 |
|---|---|---|
| #602、#603、#605、#620、#622、#623 子集、#624 | 确定移植 | 已通过本地适配落地：复制兜底、零输出/协议修复、Gemini 生图、WS 参数兼容、UA、测活增强和 service tier。 |
| #630、#631、#632、#633、#634、#638、#639 | 跟进上游修复 | 已落地；#632 保留本地 WebSocket/额度接口，#634 保留 Linux/Docker 验收范围，均不再作为待办。 |
| #635 | 部分移植 | 显式 `prompt_cache_key`、亲和及独立 provider continuity 已在 `ec862c8f`、`cc9aec32` 落地；`affinity_kind` 及上游 `0014` 不直接移植。 |
| #587、#590、#591、#594、#596、#597、#599、#606、#608、#611、#614 | 放弃 | 保持放弃；原因是与本地凭据/调度/筛选/迁移合同冲突，或没有独立落地点。 |
| #621、#625 | 低价值挂起 | 保持挂起；仅是监控快捷时间 UI 调整，当前不引入新的时间筛选合同。 |
| #643、#644 | 不直接移植 | #643 的部分 4xx failover 已由 `f78ee8a0` 覆盖，继续以本地 ReplaySafety 矩阵为准；#644 仅为 CPA 测试时序修复，暂不处理。 |
| #646 | 下一项优先移植 | 已落地：`07f96aa9` 直接拣选上游提交，`8fe19338` 为本地合同适配（一凭据一组拓扑、4xx 用例裁剪）。保留本地 entry 健康和身份边界。 |
| #652 | 确定移植（并入 #646） | 已落地：`d8f2eca5` 拣选上游提交并解决两处冲突——`runWebsocketAttempt` 保留本地 `spec.Body` 形参；`websocket_retry_test.go` 按本地重试合同适配：bound 连接上 4xx/429 均可重试（`retryUpstreamClientStatus`），故 "explicit request rejection"（Effect none）与 "unknown replay safety"（Effect cooldown_credential）转入重连用例表，"unknown provider error"（无 status → RetryNone）转入边界用例表；`websocket_test.go` fork/cache 用例的错误事件去掉 `status` 以保持不可重放。`cooldown_model` 断言统一改 `cooldown_credential`。 |
| #653 | 确定移植 | 已落地：`8590f49b` 直接拣选上游提交，零冲突；恢复 `OpenAICompatible` 渠道工具降级转发、`bifrost/tool_compatibility.go` 白名单适配、压缩上下文与 tool history 丢失拒绝、count-tokens 工具约束、CPA `prepareConvertedFidelity` 白名单边界。 |
| #657 | 低价值挂起 | 仅上游 README 赞助位新增（Fluxion AI 行 + 图片）；本地 README 独立维护，如需赞助位同步再处理。 |
| #674（`dad1f050`） | 确定移植 | 已落地（本提交手工适配）：新增 `POST /v1/alpha/search` 与 `web_search` 操作全链路——方言早退校验（POST/非流式/非空 id）、Codex native 路由、CPA 状态码透传与读体失败元数据保留（不回填 #599 通用 header 块）、embedded 独立执行器、gateway 健康/quota/定价豁免、requestlog 聚合与直读排除、前端 operation 枚举与三语标签。本地差异适配：无 `BaseURL`/`ResolveCodexAPIEndpoints`，搜索目标固定 `defaultCodexBaseURL + /alpha/search`；上游 `usage_query_minute.go` 直读排除由本地 `withoutControlPlaneObservations` 集中覆盖；前端组件体系不同，仅补枚举与标签，未移植 Globe 图标。 |
| #673 | 确定移植 | 已落地：`c37ab19` 拣选上游网关协议 probe 实现，并按本地单一 probe contract/raw evidence 边界适配；仅多协议 gateway 使用显式声明的 OpenAI Responses/Anthropic/Gemini probe 路由，保留现有其他 provider 合同。全量 Go 测试通过。 |
| #688 | 已分析，暂不移植 | 仅调整 CI/Release 的 race 命令为 `go test -race -vet=off`，并将 SQLite 迁移测试并行化；属于验证效率优化，不改变运行时行为。 |
| #690 | 已分析，暂不移植 | 将 Release 的静态检查、race、CPA 和数据库合同验证迁回自托管 ARM64 runner，制品发布仍使用 GitHub-hosted runner；依赖本地 runner 基础设施，不直接改变产品行为。 |
| #691 | 已分析，待单独决定 | 现代首页账户与健康提醒增加渠道图标、分组 tooltip 和三语文案；需要按本地现代前端的账户/分组查询合同评估适配，不直接拣选上游提交。 |
| #656、#682 | 不适用 | 与本地架构不匹配，不移植。 |

## 当前动作

#673 已按本地合同移植并通过 PR #96 合入 `origin/dev`；#674 已按本地合同手工适配并通过 PR #85 合入 `origin/dev`。
#688、#690、#691 已完成分析记录，当前不自动移植；分析边界停在 `origin/main@b517f713`（#691）。后续只检查该提交之后的新 commits，不再回看 #673–#691。
