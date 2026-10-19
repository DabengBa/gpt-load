# Main Ahead 提交评估

## 基线与范围

- 评估日期：2026-10-16。已依次执行 `git fetch --prune origin`、`git fetch --prune upstream`。
- 维护仓库：`DabengBa/gpt-load`；canonical 范围以 `origin/main` 为准。`upstream/main`（`tbphp/gpt-load`）仅作额外参考。
- 本轮只做静态代码与提交差异评估，没有移植代码、运行测试或请求真实 Provider；下文“所需验证”是后续适配的验证要求，不是本轮测试结果。

| ref | commit | commit 时间 |
|---|---|---|
| `dev` / `origin/dev` | `f3448c425566111f81feeece7d98a29da0d3f0d1` | 2026-10-16T06:14:05+08:00 |
| `origin/main` | `84896129357f7a801329c6283ca91b995e7d8e22` | 2026-09-30T17:26:33+08:00 |
| `upstream/main` | `22eacc1b1987240924edc9faefdb83782c8449e4` | 2026-10-14T18:25:55+08:00 |

| 比较（左侧独有 / 右侧独有） | 统计 |
|---|---:|
| `dev...origin/main` | 476 / 137 |
| `dev...upstream/main` | 476 / 197 |
| `origin/main...upstream/main` | 0 / 60 |

共同祖先为 `0ddc41d8b718c0281b1ba2f5bdfe5b23622621ce`。`git cherry origin/dev origin/main` 有 10 个 patch 等价项；SHA 不等价仍可能已被本地重写覆盖。137 是整个分叉历史中的 main 独有提交数，不能视为 137 项缺失功能，也不能据此用 main 替代 dev。

### 已完成评估的截止点与本轮待处理范围

保留此前截止点，不用最新检查终点或本地 cherry-pick SHA 替代：

| ref | 上次截止点 | 本轮检查终点 | 范围提交数 |
|---|---|---|---:|
| `origin/main` | `1f615d83` | `84896129` | 43 |
| `upstream/main` | `93502ced` | `22eacc1b` | 101 |

upstream 范围中的 41 个提交同时在 canonical main，另外 60 个仅 upstream 可达。canonical 的 43 项还包含 `93502ced` 本身和它之前的 `dec6dc7f`，因此本轮去重后共 103 项。下文分两份清单，不把 upstream 参考项混入 canonical 待合并范围。

本轮已为 103 项逐一形成静态取舍；Adapt 中仍有需求或行为边界待验证，尚未完成实施闭环，因此保留上次截止点。2026-10-13 的四项协议专题验证已进入 dev（`aadf4460` 可达），本轮不重复执行。已有 URL、reasoning summary、工具输出和 Grok 缓存键的结论继续以 `.docs/tech/reasoning-replay.md` 及本地捕获测试为准。

## 分类与实际价值

1. **协议与网关正确性**：分组扩展参数、Chat/Responses reasoning、工具结果、Base URL 和 WS 恢复多数已有本地实现。新增关注点是流式空响应候选重试，以及 upstream 工具回放、中断传播等尚需场景验证的差异。
2. **资源保护与统计**：main 的三层 in-flight 并发限制确有能力差异，最有直接运维价值；RPM 趋势不同于已有 RPM 限流。dev 已有 Provider 首响应时间与输出速率，不重复移植该统计体系。
3. **渠道与模型**：七个兼容预设、OpenCode 和 Gemini embeddings 已有；Cline 上游模块、Mistral 原生 HTTP 是独立候选。Mistral 兼容预设存在不等于原生协议已支持，Cline 客户端图标也不等于 Cline Provider 已支持。
4. **审计与数据记录**：JEV、请求内容改写/可逆脱敏、Voice 不在当前采用能力中。dev 的 raw capture 已记录原始 body；按 Access Key 常态入日志属于另一需求，需先确认保存范围、容量和展示边界。
5. **前端**：Astryx 是唯一管理 UI（`.docs/tech/frontend-structure.md`）；Vue Classic/Modern 代码无直接落点。仅在有本地交互缺口时吸收行为，不恢复 Vue 或双前端。
6. **依赖、构建、CI、品牌**：按本轮分类归为 Do Not Port，含维护/安全价值的项仍记录版本差异；这不代表宣称当前依赖已安全或永远不更新。

| 来源 | Port | Adapt | Do Not Port |
|---|---:|---:|---:|
| canonical main（43） | 0 | 8 | 35 |
| upstream 独有参考（60） | 0 | 29 | 31 |

Port 表示可按原提交顺序直接移植；Adapt 表示只吸收行为、按 dev 合同重写；Do Not Port 包括已覆盖、无落点及未采用功能。当前没有足够证据支持直接 Port 的项。Adapt 是评估结论，带“待验证”的项须先证明缺口，不能直接当作实施任务。

难度仅分低/中/高；已覆盖项的低难度表示核对成本。价值描述针对当前项目的新增收益，与上游功能本身的价值分开。

### 优先取舍

| 优先级 | 来源与提交 | 本地收益与边界 |
|---|---|---|
| 1 | main `84896129` | 保护 Hostinger 单实例的 HTTP/流式/Responses WS 资源；接入现有生命周期，保留重试总预算。dev 的 `0023` 已用于计费失败分类，必须使用本地后续迁移序列；不引入已删除的 Live/JEV 路径。 |
| 1 | main `43a6b66f` | 本地已检测部分空响应，剩余是流式语义、提交前缓冲和候选切换；不能覆盖 first-provider retry、affinity、Provider feedback，也不能把 tool-only/reasoning-only 判空。 |
| 2 | main `c51bdd8f` | 补 Codex 接入配置中的模型发现字段，改动小；需要确认当前 Codex CLI 配置键及本地模型端点。 |
| 按使用需求 | main `54db81fd`、`2ef32f41` | Cline Provider 和 Mistral 原生 HTTP 可扩大实际接入能力；不引入 Voice/realtime 或 Vue。 |
| 参考候选 | upstream `c24243f7`、`78884001`、`9702cc09` | 数组通配、可配置错误规则、配额恢复触发冷却恢复；分别保留参数覆盖顺序/控制字段保护、错误归因/重试预算、冷却来源及版本保护。 |
| 参考候选 | upstream `36d1b976`、`02e1879e` | 客户端 IP 观测与集中代理资源管理；IP 日志不得改变现有直接 peer CIDR 认证规则，代理资源需复用当前 outboundproxy。 |
| 后置或先确认需求 | main RPM/启动优化、upstream 常态请求体记录/批测/UI | 先利用现有日志、capture、query 轮询和单组单凭据模型，确认具体缺口再增加持久化或界面。 |

## Canonical main：逐提交摘要（43 项）

按提交时间升序。文件范围由 `git diff-tree --name-only` 提取，列关键文件/目录及总文件数；时间、标题和 PR 编号取自 Git。每项本地证据均指 dev，不以 main 文件存在证明 dev 已实现。

### 1. `dec6dc7f` fix(web): 调整版本检查反馈位置 (#728)

- 时间：2026-09-22T11:30:55+08:00；影响层：前端；文件范围（3 个）：`web/src/frontends/modern`。
- 行为与影响：仅 modern(Vue) 前端：版本检查结果从内联状态文字改为 toast/注脚，checkForUpdate 返回 CheckState 供调用方提示。
- 本地对照：上游改动 web/src/frontends/modern/features/system/SystemStatus.vue、useSystemStatus.ts、SettingsSystemInfo.vue；本地 web/src/frontends 下仅存 astryx（ls web/src/frontends = astryx/），modern 树已删除。本地等价实现在 web/src/frontends/astryx/features/settings/SystemInfoSection.tsx:199 checkForUpdate 与 :316 显示 t('settings.system.latestVersion')，i18n 键存在于 web/src/shared/i18n/locales/en-US/settings.ts:241。无对应 UX 问题记录。
- 结论：**Do Not Port**；覆盖状态：已覆盖/无落点；难度：**低**；新增价值：新增价值无（astryx 已有等价反馈）。
- 冲突与保留边界：目标前端不存在，无法套用；astryx 已有自有检查流程。
- 所需验证：无。若未来对齐交互属独立 UX 决策，需 astryx 组件渲染断言。

### 2. `93502ced` fix(execution): drop reasoning when converting Responses to compatible chat (#730)

- 时间：2026-09-22T12:28:55+08:00；影响层：执行；文件范围（2 个）：`internal/execution/bifrost`。
- 行为与影响：Responses→OpenAI Compatible 转换时丢弃请求参数 reasoning（reasoning.effort/reasoning.max_tokens 序列化后的 reasoning_effort），避免兼容上游把它当大额 reasoning budget 返回 400。注意：这只影响单次请求的 reasoning effort 参数，不动历史消息里的 raw reasoning 内容（reasoning_text/reasoning_content 条目），两者是不同层面，不要混淆。
- 本地对照：上游 internal/execution/bifrost/conversion_fidelity.go 新增 dropCompatibleResponsesReasoning（Params.Reasoning=nil）+ responses_compat_wire_probe_test.go 4 例线上探针。本地 internal/execution/bifrost/conversion_fidelity.go:48 已在 finishConvertedPreparation 调用 dropCompatibleResponsesReasoning，:126-130 实现在位，:201-206 的 deepSeekConversionDisablesThinking 读取的是同一 Params.Reasoning 参数；raw reasoning 历史转换由 third_party/bifrost-core/schemas/mux.go:400-410 与 reasoning-replay.md 另行管辖。
- 结论：**Do Not Port**；覆盖状态：已覆盖；难度：**低**；新增价值：已合入，无新增价值。
- 冲突与保留边界：无；dev 版本还把兼容目标的 tool result 折叠逻辑接在同一调用点。
- 所需验证：已存在：internal/execution/bifrost/responses_compat_wire_probe_test.go。无需新增。

### 3. `619aaa9f` fix(storage): quote reserved system setting key (#734)

- 时间：2026-09-22T17:13:43+08:00；影响层：存储、状态、CI；文件范围（5 个）：`.github/scripts/ci-database-contract.sh`、`internal/state/loader`、`internal/storage/reserved_identifier_contract_test.go`。
- 行为与影响：GORM 裸字符串条件 WHERE key = ? 在 MySQL/PostgreSQL 命中保留列名 key，改为结构体条件 systemSettingKeyScope 让驱动自动加引号，外加保留字 AST 扫描契约与外部数据库集成测试。
- 本地对照：上游修改 internal/state/loader/loader.go 的 migrateLegacyAutoModel 并增加保留字与外部数据库测试。dev 已无该 legacy 迁移函数；本地为 SQLite 专用存储，MySQL 支持已在 cef04938 删除；迁移入口由 internal/storage/migration.go 和 migration_validation.go 管理。
- 结论：**Do Not Port**；覆盖状态：无落点；难度：**低**；新增价值：新增价值低（本地 SQLite 专用）。
- 冲突与保留边界：上游迁移体系与本地已分叉（本地无 migration_recovery.go），补丁无法直接套用。
- 所需验证：无。若未来引入外部数据库需重建保留字契约测试。

### 4. `7b57fa94` feat(guardrails): add experimental JEV content guardrails (#727)

- 时间：2026-09-22T18:09:37+08:00；影响层：网关、存储、控制面、状态、前端；文件范围（95 个）：`internal/control/experimental_settings.go`、`internal/control/group_options.go`、`internal/control/group_options_test.go`、`internal/control/request_audit.go`、`internal/control/request_audit_settings_test.go`、`internal/control/request_logs.go`、`internal/control/request_logs_audit_test.go` 等。
- 行为与影响：实验性 JEV 内容护栏：决策路由选项、请求审计（request_audit）结果挂到 request log、audit_status/audit_rule 查询过滤、AccessKey 审计脱敏、控制端 experimental_settings。
- 本地对照：上游新增 internal/control/experimental_settings.go、request_audit.go，internal/control/request_logs.go 增加 RequestAudit 字段与 audit_status 过滤，引入 gpt-load/internal/jev 包。本地检索 internal/control 无 experimental_settings.go、request_audit.go；rg 'jev' internal 无命中；rg 'audit_status|RequestAudit' internal 无命中；request log 查询白名单（internal/control/request_logs.go parseRequestLogQuery 同构位置）不含 audit 字段。协议面（protocol.Decisions）本地也未启用对应审计消费。
- 结论：**Do Not Port**；覆盖状态：排除功能，无落点；难度：**高**；新增价值：新增价值无（排除范围）。
- 冲突与保留边界：需要先引入整个 JEV 决策协议族才成立，单独移植无意义。
- 所需验证：无。

### 5. `c14e24d7` feat(gateway): Add configurable request redaction (#735)

- 时间：2026-09-22T21:49:02+08:00；影响层：网关、控制面、状态、前端；文件范围（38 个）：`internal/control/http_routes.go`、`internal/control/request_redaction.go`、`internal/control/request_redaction_test.go`、`internal/control/settings.go`、`internal/gateway/auto_model.go`、`internal/gateway/decisions_test.go`、`internal/gateway/handler.go` 等。
- 行为与影响：可配置请求脱敏：正则+替换规则存设置，state snapshot/loader 携带规则，网关在转发前改写请求（internal/requestredact/redact.go 330 行 + gateway 接线 + 两套前端设置页约 700 行）。
- 本地对照：上游新增 internal/requestredact/、internal/gateway/request_redaction.go，state/snapshot.go 与 loader.go 加 RequestRedaction 字段，web 两前端各加 RequestRedactionSection/Editor.vue。本地 rg 'requestredact|RequestRedaction' internal web/src 均无命中；internal/state/snapshot.go 无该字段；internal/control/settings.go 的设置解析不含 request_redaction。
- 结论：**Do Not Port**；覆盖状态：排除功能，无落点；难度：**高**；新增价值：新增价值无（排除范围）。
- 冲突与保留边界：上游后续 #749/#758 在此之上大改（+9700 行），单独移植 #735 会立即过时；且现有文档明确排除脱敏。
- 所需验证：无。

### 6. `38fe91f0` fix(codex): upgrade CPA to v7.3.15 and sync model discovery (#738)

- 时间：2026-09-23T09:26:02+08:00；影响层：网关、执行、依赖、依赖/执行；文件范围（16 个）：`THIRD_PARTY_NOTICES.md`、`go.mod`、`go.sum`、`internal/catalog/client_model_test.go`、`internal/catalog/codex_client_models.go`、`internal/catalog/codex_client_models_0.154.0.json`、`internal/catalog/codex_client_models_0.155.0.json` 等。
- 行为与影响：CPA SDK 7.3.6→7.3.15，Codex 模型目录快照 0.154.0→0.155.0，codex_headers.go CodexClientVersion 同步。
- 本地对照：上游改 go.mod CLIProxyAPI/v7 v7.3.6→v7.3.15、internal/catalog/codex_client_models_0.155.0.json、third_party/cpaembedded/embedded/codex_headers.go。本地 go.mod:16,96 已是 CLIProxyAPI/v8 v8.0.8 且 replace 指向本地 third_party/cpaembedded；third_party/cpaembedded/modelcatalog/client.go:10 ClientVersion="0.159.2"，目录为 codex_client_models_0.159.2.json，已越过上游 0.155.0 两个版本；third_party/cpaembedded/embedded/codex_headers.go:13 CodexClientVersion = modelcatalog.ClientVersion 由测试联动。无 internal/catalog/codex_client_models.go（本地目录结构已重构为 modelcatalog 包）。
- 结论：**Do Not Port**；覆盖状态：已被本地演进取代；难度：**低**；新增价值：新增价值无（本地已超前到 v8/0.159.2）。
- 冲突与保留边界：补丁基于 v7 + internal/catalog 旧结构，与本地 v8 + modelcatalog 包完全不兼容。
- 所需验证：无需。本地已有 codex_headers_test.go 联动校验。

### 7. `0e94d566` fix(storage): Reduce migration validation metadata queries (#740)

- 时间：2026-09-23T11:58:13+08:00；影响层：存储；文件范围（4 个）：`internal/storage/migration.go`、`internal/storage/migration_inspection.go`、`internal/storage/migration_recovery.go`、`internal/storage/migration_validation_queries_test.go`。
- 行为与影响：在启动时复用迁移元数据，减少逐迁移校验查询，并调整 DDL 后验证与恢复入口；用户收益主要是启动耗时。
- 本地对照：上游 internal/storage/migration.go +31/-10（applyMigrationsLocked 中 hadMigrationLedger 与 ValidateCurrent 循环改走 inspection）、新增 migration_inspection.go 318 行、migration_recovery.go 恢复路径同步、migration_validation_queries_test.go 159 行。本地 internal/storage/migration.go:219-221 仍是逐次 db.Migrator().HasTable(migrationLedgerTable/initialSchemaSentinelTable) + :234 起逐 applied 循环调 validator；本地无 migration_inspection.go，也无 migration_recovery.go（恢复路径文件不存在，说明本地迁移/恢复结构与上游已分叉）。本地生产为 SQLite 单库、迁移条目数有限，无任何启动耗时证据支持此优化是必要的（YAGNI）。
- 结论：**Adapt**；覆盖状态：待评估（无启动耗时证据，YAGNI 存疑）；难度：**低**；新增价值：新增价值低（无本地启动耗时证据前不必要）。
- 冲突与保留边界：本地为 SQLite 专用迁移链，编号和验证入口已分叉；先量化启动开销，再决定是否适配，不能按原提交直接套用。
- 所需验证：若实施：移植 migration_validation_queries_test.go 的查询次数断言 + 本地迁移回归（migration_test.go 扩展）；仅 SQLite，不需要外部数据库矩阵。

### 8. `631e5abf` fix(execution): Chat→Responses 的 reasoning item 补上 summary (#733)

- 时间：2026-09-23T12:50:47+08:00；影响层：执行；文件范围（2 个）：`internal/execution/bifrost`。
- 行为与影响：Grok CLI 拒绝缺 summary 的 reasoning item；Chat→Responses 转换产物始终带 summary 数组（有 reasoning 文本时用 summary_text），覆盖流式与非流式两条编码路径。
- 本地对照：上游改 internal/execution/bifrost/converted.go +53 与 responses_reasoning_summary_test.go +122。本地行为由 third_party/bifrost-core/schemas/mux.go:400-410 实现（"Non-nil: the Responses schema requires summary to be an array"，转换 item 恒填 summary），Owner 文档 .docs/tech/reasoning-replay.md『Chat Completions To Responses』一节已把该契约写成当前实现（"always contain a summary array… Unary conversion and streaming … use the same reasoning item shape"），验证在 internal/execution/bifrost/reasoning_replay_test.go（真实 HTTP 端点断言不丢 summary）。
- 结论：**Do Not Port**；覆盖状态：已覆盖；难度：**低**；新增价值：已合入，无新增价值。
- 冲突与保留边界：无；本地经 bifrost-core fork 实现，与上游 converted.go 补丁路径不同但行为等价。
- 所需验证：已存在：internal/execution/bifrost/reasoning_replay_test.go、third_party/bifrost-core/schemas/reasoningreplay_test.go。

### 9. `c51bdd8f` feat(codex): add model discovery configuration (#742)

- 时间：2026-09-23T14:30:07+08:00；影响层：前端；文件范围（2 个）：`web/src/frontends/classic`、`web/src/frontends/modern`。
- 行为与影响：Codex CLI 接入配置片段新增 model_catalog_url（base_url+/models）、supports_websockets=true、[features] api_key_model_discovery=true，provider name 改为 OpenAI，使 Codex CLI 自动发现网关模型列表。
- 本地对照：上游仅改 web/src/frontends/classic/features/home/gateway-clients.ts（+10）与 web/src/frontends/modern/features/home/gateway-config.ts（+6），纯 TOML 片段模板。本地两树均已删除（web/src/frontends 仅 astryx）；检索 web/src/frontends/astryx 与 web/src/shared 无 model_catalog_url/api_key_model_discovery/supports_websockets 命中，说明 astryx 的 Codex 配置生成尚未包含这些键。网关侧 /v1beta/models 与模型列表端点已存在（internal/gateway/router.go resolveModelListRoute 同构位置），后端无阻塞。
- 结论：**Adapt**；覆盖状态：待移植；难度：**低**；新增价值：新增价值中低（Codex CLI 自动发现模型）。
- 冲突与保留边界：需落到 astryx home 的 gateway 配置生成模块（React 版），并确认片段键与本地 /models 响应行为匹配。
- 所需验证：生成的 TOML 片段快照断言（astryx 对应组件/工具函数单测）。

### 10. `f2840466` fix(grok): 把网关缓存键写回 Responses 请求体 (#737)

- 时间：2026-09-23T15:32:57+08:00；影响层：依赖/执行；文件范围（2 个）：`third_party/cpaembedded/embedded`。
- 行为与影响：cli-chat-proxy 只按 prompt_cache_key 复用提示词缓存：清掉调用方会话字段后把网关 UUID 写回 Responses/Chat 请求体 prompt_cache_key，且不放进 ExecutionSessionMetadataKey（避免被当可信 reasoning replay 会话，下一轮把推理块插到用户输入前）。
- 本地对照：上游 third_party/cpaembedded/embedded/grok_executor.go +123/-22（scopeGrokExecutionRequest/applyGrokPromptCacheKey/grokCarriesPromptCacheKey/writeGrokPromptCacheKey）。本地同文件已含全部函数：grok_executor.go:238 writeGrokPromptCacheKey 的 sjson.SetBytes(raw,"prompt_cache_key",cacheKey)，:271 会话字段剥离列表；Owner 文档 .docs/tech/reasoning-replay.md『Grok Prompt Cache Identity』一节记录该行为；测试 third_party/cpaembedded/embedded/ahead_grok_cache_test.go 与 internal/execution/cpa/ahead_grok_scope_test.go 在位。
- 结论：**Do Not Port**；覆盖状态：已覆盖；难度：**低**；新增价值：已合入，无新增价值。
- 冲突与保留边界：无。
- 所需验证：已存在（见 evidence）。

### 11. `8f894c3e` fix(bifrost): fold converted tool results into a string for OpenAI Compatible (#744)

- 时间：2026-09-23T16:51:45+08:00；影响层：执行；文件范围（2 个）：`internal/execution/bifrost`。
- 行为与影响：Responses 允许 function_call_output 为内容块数组，Chat 类型 tool content 必须是字符串，严格上游报 'Input should be a valid string'。上游方案：折叠为字符串，文本保留，图片/文件/音频换成 CPA 同款占位标记。
- 本地对照：上游 internal/execution/bifrost/conversion_fidelity.go +89（flattenCompatibleToolOutputs）+ compatible_tool_output_test.go 194 行。本地 conversion_fidelity.go:49 已在 Compatible 目标调用 flattenCompatibleToolOutputs，:133-145 本地签名演化为返回 *execution.AttemptResult——含 reasoning_summary/reasoning/unknown 块时不是静默标记而是以 critical_semantic_loss 拒绝转换、保留原请求换路由（:143 注释），与 Owner 文档 .docs/tech/reasoning-replay.md『Responses To OpenAI-Compatible Chat Completions』'Results containing non-text blocks … are rejected before dispatch with critical_semantic_loss' 一致。测试 internal/execution/bifrost/ahead_tool_output_test.go 在位。
- 结论：**Do Not Port**；覆盖状态：已覆盖（本地更严格）；难度：**低**；新增价值：已合入，本地实现更严格，无新增价值。
- 冲突与保留边界：无；行为差异是有意为之的本地契约（拒绝优于静默丢内容），不要被上游补丁回退。
- 所需验证：已存在：ahead_tool_output_test.go、conversion_fidelity_test.go。

### 12. `dd8fa169` feat(rpm): add AccessKey and credential RPM statistics (#739)

- 时间：2026-09-24T09:11:49+08:00；影响层：网关、存储、控制面、前端；文件范围（52 个）：`internal/container/container.go`、`internal/control/access_key_collection.go`、`internal/control/access_key_collection_http.go`、`internal/control/access_key_collection_query.go`、`internal/control/credentials.go`、`internal/control/http_routes.go`、`internal/control/modern_credentials.go` 等。
- 行为与影响：按 AccessKey 与凭据维度的每分钟请求统计：独立 rpm_stats 表（迁移 0022）、internal/rpm 内存存储、requestlog checkpoint/worker/查询、control 聚合接口、modern 前端小时趋势/凭据卡片/Sparkline。
- 本地对照：上游 +1930 行：internal/rpm/store.go、internal/requestlog/rpm.go+rpm_checkpoint.go、internal/control/rpm.go、internal/storage/migrations/0022_rpm_stats.go、modern 前端 RPMTrend.vue 等。本地 rg 确认无 internal/rpm、internal/control/rpm.go、rpm 迁移；本地限流侧已有每请求 RPM 检查（internal/gateway/websocket_turn.go:166 limiter.Allow、internal/ratelimit/access_key_rpm 同构位置）但只做准入不落统计；request log 保留体系为自有实现（internal/requestlog/retention.go retentionBatchSize、credential_window_usage）。本地 requestlog 迁移序列需核对与 0022 的编号衔接。
- 结论：**Adapt**；覆盖状态：可选移植；难度：**高**；新增价值：新增价值中（多凭据 RPM 维度排障）。
- 冲突与保留边界：迁移号需对齐本地迁移序列；前端 1930 行需全部重写为 astryx（工作量主体）；access key/凭据集合接口需并入本地 shared/control resources。
- 所需验证：store 并发/过期测试（上游 rpm_test.go、store_bench_test.go 可参考）、checkpoint 聚合回归、control 查询接口测试；前端组件渲染断言。建议先只做后端。

### 13. `43a6b66f` feat(gateway): add empty response detection with candidate retry (#746)

- 时间：2026-09-24T10:26:54+08:00；影响层：网关、控制面、状态、前端；文件范围（40 个）：`README.md`、`README_CN.md`、`internal/control/group_detail.go`、`internal/control/settings.go`、`internal/dialect/stream_content.go`、`internal/dialect/stream_content_test.go`、`internal/dialect/stream_event.go` 等。
- 行为与影响：自然结束但零内容的流此前被判成功（清失败计数、写入 affinity），空响应目标被反复复用。上游新增：dialect 内容分类器、提交前预读缓冲（8 事件/5s/256KiB + 定时器兜底）、空响应换候选重试（不计冷却/黑名单/失败数）、empty_response_retry 设置（默认关、分组可覆盖）、豁免矩阵。
- 本地对照：本地已有（非流式、仅 Chat）：internal/gateway/execution_forward.go:935-939 定义 upstreamEmptyCompletionCode（'usage 证明生成发生但 assistant 消息无 payload'），:1119-1130 在 emptyChatCompletion 命中时注入 ExecutionError；internal/health/execution_judge.go:250-258 对该 code 判 FailureCategoryAmbiguous + RetryNone + EffectNone（rule upstream.empty_completion）；internal/gateway/request_log.go:853-856 isUpstreamDiagnosticErrorCode 把它记为失败请求但客户端仍收到原始响应；测试 execution_forward_test.go（e7ca 形状矩阵）、execution_judge_empty_completion_test.go。本地缺口：(1) 仅覆盖非流式 Chat completion，流式空流无检测（无 internal/dialect/stream_content.go、无 stream 内容分类器）；(2) 判定是 RetryNone——请求失败给客户端，而不是换候选重试；(3) 无预读缓冲与 empty_response_retry 设置（rg EmptyResponseRetry internal/state/runtime_settings.go 无命中）；(4) 无豁免矩阵与 affinity 写入抑制。上游新增 internal/dialect/stream_content.go 323 行、internal/gateway/empty_response.go 123 行、execution_forward.go/handler.go/stream_observation.go 接线 +96 行、health 分类 +4 行、设置 +20 行、测试约 1100 行。
- 结论：**Adapt**；覆盖状态：部分已有，缺口明确；难度：**高**；新增价值：新增价值高（流式空响应检测与换候选重试）。
- 冲突与保留边界：流生命周期是本地改动最密集区域：request affinity、first-provider retry（feat/first-provider-retry 分支已并入 dev 主线相关逻辑）、provider-feedback 契约（.docs/tech/provider-feedback.md）。预读缓冲必须与这些状态机共存；RetryNone 语义与上游『换候选重试』直接冲突，需要决策是保留本地失败语义只加流式检测，还是完整引入重试。前端设置页需重写为 astryx。
- 所需验证：豁免矩阵（prewarm/previous_response_id/conversation/WebSocket）、预读窗口计时器（上游第二轮修复的场景：上游静默后 header 被长期压住）、完整空 Responses 流（恰好 8 事件含终态）不再被事件上限截断、与 affinity/first-provider-retry 交互、health 分类回归；上游 empty_response_integration_test.go 390 行可参考移植。

### 14. `96ce3cd2` feat(redaction): add stateless reversible message protection (#749)

- 时间：2026-09-24T16:37:03+08:00；影响层：网关、控制面、前端、依赖；文件范围（55 个）：`go.mod`、`go.sum`、`internal/control/route_inspect_test.go`、`internal/gateway/auto_model.go`、`internal/gateway/execution_boundary.go`、`internal/gateway/execution_forward.go`、`internal/gateway/forward.go` 等。
- 行为与影响：可逆加密脱敏：AES-SIV 派生密钥生成 gld1_ token，请求加密、响应（unary/SSE/WebSocket）还原，覆盖流式缓冲、协议边界、工具结果，+6819 行。
- 本地对照：上游新增 internal/platform/encryption/redaction.go（RedactionCipher 接口）、internal/gateway/redaction_restore_*.go 约 1700 行、internal/requestredact 扩展 260 行、websocket_turn.go +233。本地 rg 'RedactionCipher|redaction_restore|gld1_' internal 无命中；internal/platform/encryption 与 internal/requestredact 不存在；websocket_turn.go 无相应接线。
- 结论：**Do Not Port**；覆盖状态：排除功能，无落点；难度：**高**；新增价值：新增价值无（排除范围）。
- 冲突与保留边界：排除范围；且与 #735/#758 构成自洽体系，无法摘取子集。
- 所需验证：无。

### 15. `36b40124` feat(release): add manual pre-release acceptance (#752)

- 时间：2026-09-24T17:40:25+08:00；影响层：前端；文件范围（20 个）：`.dockerignore`、`.editorconfig`、`.gitignore`、`CONTRIBUTING.md`、`Makefile`、`scripts/release-e2e/package.json`、`scripts/release-e2e/playwright.config.ts` 等。
- 行为与影响：本地发布验收工具：scripts/release*.py 约 1800 行（源准备、tag 提示、compose 冒烟、数据库矩阵、浏览器冒烟）、Playwright release 冒烟、Makefile release/release-simulate 目标。
- 本地对照：上游 scripts/release.py:29 硬编码 REPOSITORY="tbphp/gpt-load"，流程绑定上游 CI 与发布渠道。本地发布体系独立：.github/scripts/release-*.sh（compare-semver、docker-smoke、image-digest、native-smoke、promote-image-channels 等）、internal/releasecheck/checker.go、scripts/deploy.sh（Hostinger 流程，仓库指向 DabengBa/gpt-load）。本地 Makefile 无 release 目标。
- 结论：**Do Not Port**；覆盖状态：无落点；难度：**中**；新增价值：新增价值低（绑上游仓库流程）。
- 冲突与保留边界：工具链绑死上游仓库与发布流程；本地已有等价脚本族。
- 所需验证：无。

### 16. `0ec0d610` feat(gemini): add native Gemini embeddings protocol (#754)

- 时间：2026-09-25T08:43:27+08:00；影响层：网关、执行、控制面、渠道、前端；文件范围（45 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/channel/compiler.go`、`internal/channel/gemini_embeddings_module_test.go`、`internal/channel/gpt_load_module_test.go`、`internal/channel/modules` 等。
- 行为与影响：新增 gemini-embeddings 客户端协议（:embedContent/:batchEmbedContents），路由按动作后缀分流、专用 dialect、probe 校验（向量值必须为 JSON 数字）、string token 流式扫描凭据字面量、双前端协议标签。
- 本地对照：本地已实现：internal/protocol/protocol.go:14 GeminiEmbeddings 并入 Valid()/列表；internal/dialect/gemini_embeddings.go 存在；internal/gateway/router.go 同构位置含 geminiModelActionPath 分流（上游补丁的 resolveGeminiModelActionRoute 模式）；internal/execution/bifrost/gemini_native_embeddings_test.go（+322 行同源测试）与 gemini_embeddings_gateway_test.go 在位；web/src/shared/control/protocols.ts:31 已注册协议、shared/domain/monitor/log-format.ts:21 有展示名。#754 第二轮修复（probe 校验、scanNumbers）在 internal/execution/geminiembedding/convert.go 的 scanNumbers 可见。
- 结论：**Do Not Port**；覆盖状态：已覆盖；难度：**低**；新增价值：已合入，无新增价值。
- 冲突与保留边界：无。
- 所需验证：已存在（见 evidence）。

### 17. `0a01cdde` feat(gateway): add Codex live voice relay (#756)

- 时间：2026-09-25T14:04:21+08:00；影响层：网关、执行、调度、控制面、状态、订阅、渠道、前端、依赖、依赖/执行；文件范围（73 个）：`.env.example`、`go.mod`、`go.sum`、`internal/app/app.go`、`internal/channel/channel.go`、`internal/channel/codex_live_test.go`、`internal/channel/compiler.go` 等。
- 行为与影响：Codex 实时语音：WebRTC 媒体会话/中继、live 网关路由（protocol.CodexLive）、live_proxy/live_session/live_media 约 1500 行、CPA live 桥、.env 配置、双前端协议/inspector 支持，+4019 行。
- 本地对照：上游新增 internal/gateway/live.go/live_media.go/live_proxy.go/live_session.go、internal/execution/cpa/live.go、third_party/cpaembedded/embedded/codex_live.go、internal/platform/config/config.go live 配置、protocol.CodexLive。本地 rg 'CodexLive|liveMedia|LiveProxy' internal 无命中；internal/gateway 无 live*.go（仅 images_stream_live_test.go，与语音无关）；protocol.go 无 CodexLive；internal/execution/cpa 无 live.go；.env.example 无相关键。
- 结论：**Do Not Port**；覆盖状态：排除功能，无落点；难度：**高**；新增价值：新增价值无（Voice 排除）。
- 冲突与保留边界：排除范围（Voice）；协议面、配置面、依赖（go.mod +20 行）均不存在。
- 所需验证：无。

### 18. `7587c60f` feat(modern): improve access key console (#760)

- 时间：2026-09-25T14:58:56+08:00；影响层：前端；文件范围（6 个）：`web/src/frontends/modern`。
- 行为与影响：modern 非管理员视角：Home 状态栏在 cost_limit_status.allowed=false 时显式告警（stateKeyExhausted）、当前密钥掩码+复制按钮（AppCopyValue 解析明文）、非 admin 隐藏模型卡片管理专属列（来源分组/计价方法/catalog/价格状态筛选）。
- 本地对照：上游改 modern HomeStatusBar.vue/ModelCard.vue/ModelDetailPanel.vue/ModelsView.vue +123/-28 与 i18n。本地 modern 树已删除；astryx 对应物：web/src/frontends/astryx/features/home/CurrentAccessKeyCard.tsx:303-388 已展示 rpm_limit 与掩码语义，但检索 astryx home 无 cost_limit_status 驱动的『密钥额度已用尽』状态（无 stateKeyExhausted 键，web/src/shared/i18n 无该键）；models 视图的 admin/只读分支与上游行为差异未核对到等价实现。
- 结论：**Adapt**；覆盖状态：可选移植；难度：**中**；新增价值：新增价值中低（额度耗尽显式提示值得摘取）。
- 冲突与保留边界：React/astryx 重写；admin/non-admin 分支需映射到 astryx 权限模型；cost_limit_status 数据需确认本地 home API 已暴露。
- 所需验证：astryx CurrentAccessKeyCard 与 models 视图的渲染断言（额度耗尽态、只读列隐藏）。建议只摘取『额度耗尽显式提示』这一可感知改进。

### 19. `607fb916` feat(gemini): convert OpenAI Embeddings to Gemini embeddings (#759)

- 时间：2026-09-25T15:04:12+08:00；影响层：网关、执行、控制面、渠道；文件范围（18 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/channel/embeddings_module_test.go`、`internal/channel/modules`、`internal/channel/route_golden_test.go`、`internal/control/route_inspect_gemini_embeddings_test.go` 等。
- 行为与影响：Gemini 渠道承接 openai-embeddings 请求：文本输入转 :batchEmbedContents，dimensions→outputDimensionality，float/base64(float32 LE) 输出，恒回 OpenAI usage 对象（上游无 usage 不计价），向量数/类型不匹配拒绝，token ID 输入拒绝。
- 本地对照：本地已有完整转换器 internal/execution/geminiembedding/convert.go（ConvertRequest/ConvertResponse/encodeBase64Vector/scanNumbers，与上游 239 行版本同构，含 usage 证据与 ErrInvalidResponse 语义）及 convert_test.go（191 行）；executor 接线在 internal/execution/bifrost/passthrough.go、executor.go（convertedEmbeddings 分支）、capabilities.go（Gemini+OpenAIEmbeddings 转换路由）；路由声明 internal/channel/modules/gemini.go 含 OpenAIEmbeddings converted 路由；集成测试 internal/execution/bifrost/gemini_embeddings_conversion_test.go、internal/gateway/gemini_embeddings_gateway_test.go 在位。
- 结论：**Do Not Port**；覆盖状态：已覆盖；难度：**低**；新增价值：已合入，无新增价值。
- 冲突与保留边界：无。
- 所需验证：已存在（见 evidence）。

### 20. `c4a895ad` test(gateway): wait for live teardown before checking key rotation log (#761)

- 时间：2026-09-25T15:16:14+08:00；影响层：网关；文件范围（1 个）：`internal/gateway/live_integration_test.go`。
- 行为与影响：#756 的测试稳定性修复：TestCodexLiveRevokesActiveCallWhenAccessKeyRotates 把 2s 授权轮询与 WebRTC teardown 拆成两段等待（5s+10s），避免 race 模式下 CI 超时。
- 本地对照：上游仅改 internal/gateway/live_integration_test.go +8/-2。本地无该文件、无 live 集成测试（#756 未移植，rg TestCodexLive internal 无命中），测试对象不存在。
- 结论：**Do Not Port**；覆盖状态：无落点；难度：**低**；新增价值：新增价值无。
- 冲突与保留边界：完全依赖 #756。
- 所需验证：无。

### 21. `dd0cfcc5` fix(redaction): cover all model output and preserve signed content (#758)

- 时间：2026-09-25T21:40:15+08:00；影响层：网关；文件范围（28 个）：`internal/gateway/execution_forward.go`、`internal/gateway/handler.go`、`internal/gateway/redaction_boundaries_test.go`、`internal/gateway/redaction_contract_test.go`、`internal/gateway/redaction_delivery_contract_test.go`、`internal/gateway/redaction_observation_test.go`、`internal/gateway/redaction_response_integration_test.go` 等。
- 行为与影响：#749 大改版：还原所有模型输出（Chat refusal/reasoning、Responses reasoning、Gemini thought）、可逆加密覆盖请求历史、GLDSIG 签名记录（redaction_restore_signature.go 285 行 + requestredact/signature.go 349 行，按路径+哈希按位置回填签名块）、无 ciphertext 时跳过流缓冲，+2880/-269 行。
- 本地对照：上游新增 internal/gateway/redaction_restore_signature.go、internal/requestredact/signature.go，redaction_restore_stream.go +392。本地 rg 'GLDSIG|signatureMagic|redaction_restore' internal 无命中；依赖的 #735/#749 基础设施（RedactionCipher、requestredact 包）本地均不存在（见 #735/#749 条目）。
- 结论：**Do Not Port**；覆盖状态：排除功能，无落点；难度：**高**；新增价值：新增价值无（排除范围）。
- 冲突与保留边界：排除范围；无法在无基础体系时摘取。
- 所需验证：无。

### 22. `3abb8660` fix(settings): tighten redaction editor spacing (#764)

- 时间：2026-09-25T23:34:14+08:00；影响层：前端；文件范围（1 个）：`web/src/frontends/modern`。
- 行为与影响：modern RequestRedactionEditor.vue 的间距/图标尺寸微调（gap space-3→2、帮助图标 xs→xxs），+4/-3。
- 本地对照：上游仅改 web/src/frontends/modern/features/settings/RequestRedactionEditor.vue。本地 modern 树已删除；依赖 #749 的编辑器组件，本地无对应物（rg RequestRedactionEditor 无命中）。
- 结论：**Do Not Port**；覆盖状态：无落点；难度：**低**；新增价值：新增价值无。
- 冲突与保留边界：双重依赖不存在。
- 所需验证：无。

### 23. `03e1da48` feat(gateway): add voice modes and live call retries (#765)

- 时间：2026-09-26T10:14:31+08:00；影响层：网关、执行、调度、控制面、状态、前端；文件范围（51 个）：`.env.example`、`README.md`、`README_CN.md`、`README_JP.md`、`docker-compose.voice.yml`、`docker-compose.yml`、`internal/control/codex_live_settings_test.go` 等。
- 行为与影响：#756 扩展：codex_live_mode 设置（off/direct/relay，runtime_settings 分组覆盖）、live 建会话失败与 teardown 失败的按候选重试循环（retryAttemptLimit + ChargeReplay 凭据刷新）、docker-compose.voice.yml、README 三语、双前端设置/分组 UI，+1333/-261。
- 本地对照：上游 internal/gateway/live.go 重写 createCodexLive 为重试循环、internal/state/runtime_settings.go 加 SettingCodexLiveMode/CodexLiveMode。本地 runtime_settings.go 无 CodexLiveMode（rg 无命中）；internal/gateway/live.go 不存在（#756 未移植）；internal/execution/cpa/live.go、internal/scheduler 的 ChargeReplay 均无；docker-compose.voice.yml 不存在。
- 结论：**Do Not Port**；覆盖状态：排除功能，无落点；难度：**高**；新增价值：无。
- 冲突与保留边界：排除范围（Voice）；完全依赖 #756 的会话基础设施。
- 所需验证：无。

### 24. `48dad3de` fix(guardrails): Batch large JEV content reviews (#766)

- 时间：2026-09-26T10:23:10+08:00；影响层：网关、前端；文件范围（10 个）：`internal/gateway/request_audit.go`、`internal/gateway/request_audit_test.go`、`internal/requestaudit/audit.go`、`internal/requestaudit/audit_test.go`、`internal/requestaudit/batches.go`、`internal/requestaudit/batches_test.go`、`internal/requestaudit/chunks.go` 等。
- 行为与影响：#766：长内容JEV分批审查。
- 本地对照：基线dev=f3448c425566111f81feeece7d98a29da0d3f0d1；以下证据均指该Git引用；git cat-file -e dev:internal/requestaudit/audit.go失败；diff新增requestaudit/batches.go和chunks.go
- 结论：**Do Not Port**；覆盖状态：无落点；难度：**高**；新增价值：无；JEV已移除。
- 冲突与保留边界：dev无requestaudit，不能恢复已删除JEV。
- 所需验证：不移植；未运行测试

### 25. `8a4e482a` chore(deps): upgrade Bifrost and CPA SDKs (#768)

- 时间：2026-09-26T10:46:52+08:00；影响层：执行、依赖、依赖/执行；文件范围（10 个）：`THIRD_PARTY_NOTICES.md`、`go.mod`、`go.sum`、`internal/catalog/codex_client_models.go`、`internal/execution/bifrost`、`third_party/cpaembedded/README.md`、`third_party/cpaembedded/embedded` 等。
- 行为与影响：#768：升级Bifrost及CPA，并调整模型与转换测试。
- 本地对照：dev:go.mod为Bifrost v1.11.3，replace至third_party/bifrost-core；dev使用CLIProxyAPI/v8 v8.0.8；上游此提交使用CPA v7；dev:internal/catalog/codex_client_models.go不存在
- 结论：**Do Not Port**；覆盖状态：依赖更新且版本已分叉；难度：**高**；新增价值：无增量；依赖按要求排除。
- 冲突与保留边界：不能覆盖本地fork或倒灌旧SDK；依赖明确Do Not Port。
- 所需验证：不移植，无新增测试要求

### 26. `9a7b4826` fix(gateway): restore WebSocket error recovery (#769)

- 时间：2026-09-26T11:31:26+08:00；影响层：网关、执行；文件范围（11 个）：`internal/execution/cpa`、`internal/execution/wsnative`、`internal/gateway/reason.go`、`internal/gateway/websocket.go`、`internal/gateway/websocket_error_response_test.go`、`internal/gateway/websocket_errors_test.go`、`internal/gateway/websocket_retry_test.go` 等。
- 行为与影响：#769：WS握手错误分类、准备失败切换和重试耗尽保留上游拒绝。
- 本地对照：git diff 9a7b4826 dev -- internal/execution/wsnative/session.go无差异；dev:websocket_turn.go已有forwardAttempts、finishFailedAttempt、finishRejectedAttempt和requestClientReconnect；dev:internal/health/execution_judge.go已有握手HTTP拒绝处理；dev:internal/execution/cpa/websocket.go已改为CPA v8接口
- 结论：**Do Not Port**；覆盖状态：核心修复已覆盖；难度：**高**；新增价值：修复本身高价值；重复合并无增量。
- 冲突与保留边界：整提交会覆盖dev已演进的WS、健康反馈和功能删除。
- 所需验证：保留dev的websocket_errors_test.go、websocket_retry_test.go和health/websocket_handshake_test.go

### 27. `73b3122f` feat(channel): add seven compatible presets (#770)

- 时间：2026-09-26T11:59:16+08:00；影响层：执行、渠道、前端；文件范围（40 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/channel/builtins.go`、`internal/channel/channel.go`、`internal/channel/compatible_presets_test.go`、`internal/channel/embeddings_module_test.go` 等。
- 行为与影响：#770：Cerebras、Cohere、Hugging Face、Mistral、Nebius、Parasail和Wafer预设。
- 本地对照：dev 的 internal/channel/builtins.go 和 internal/channel/modules/ 已有七个预设的注册与定义；internal/channel/route_golden_test.go 列出对应渠道。
- 结论：**Do Not Port**；覆盖状态：7个兼容preset已覆盖；难度：**低**；新增价值：无增量。
- 冲突与保留边界：重复注册无收益；classic/modern资产路径不适用。
- 所需验证：沿用dev现有渠道和route golden测试

### 28. `7064ec70` feat(channel): add OpenCode Go and Zen presets (#771)

- 时间：2026-09-26T13:07:35+08:00；影响层：执行、渠道、前端；文件范围（14 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/channel/builtins.go`、`internal/channel/channel.go`、`internal/channel/final_provider_contract_test.go`、`internal/channel/modules` 等。
- 行为与影响：#771：增加OpenCode Go和Zen预设。
- 本地对照：dev:internal/channel/modules/opencode_go.go存在，地址为https://opencode.ai/zen/go；dev:internal/channel/modules/opencode_zen.go存在，地址为https://opencode.ai/zen；dev:internal/channel/builtins.go已注册两者
- 结论：**Do Not Port**；覆盖状态：OpenCode Go/Zen已覆盖；难度：**低**；新增价值：无增量。
- 冲突与保留边界：不重复添加preset，不恢复modern资产目录。
- 所需验证：沿用现有渠道注册和路由测试

### 29. `e39573c9` fix(ui): Simplify group setup and address guidance (#772)

- 时间：2026-09-26T13:25:17+08:00；影响层：前端；文件范围（13 个）：`web/src/frontends/modern`。
- 行为与影响：#772：简化modern分组创建和base URL提示。
- 本地对照：diff全在web/src/frontends/modern；dev:.docs/tech/frontend-structure.md规定Astryx是唯一UI；dev:web/package.json使用React 19.3.0和Astryx 0.6.2，无Vue；dev实际设置组件为astryx/features/groups/settings/GroupSettingsBaseForm.tsx
- 结论：**Do Not Port**；覆盖状态：Vue实现无落点；难度：**中**；新增价值：未证实有增量。
- 冲突与保留边界：不能把Vue组件移入Astryx；未证明dev存在对应缺陷。
- 所需验证：不移植，无新增测试要求

### 30. `8f4ac9a5` fix(ui): Consolidate request log pricing into cost (#773)

- 时间：2026-09-26T14:18:33+08:00；影响层：前端；文件范围（3 个）：`web/src/frontends/modern`。
- 行为与影响：#773：合并modern日志价格和费用展示。
- 本地对照：diff仅修改modern日志Vue组件及文案；dev日志组件为astryx/features/logs/LogsTable.tsx、LogDetailDrawer.tsx和PricingModeIndicator.tsx
- 结论：**Do Not Port**；覆盖状态：Vue日志组件无落点；难度：**中**；新增价值：未证实有增量。
- 冲突与保留边界：不能用main的旧UI变更证明dev费用展示有缺陷。
- 所需验证：不移植，无新增测试要求

### 31. `41339ae9` fix(gateway): Restore live logs and bound teardown (#774)

- 时间：2026-09-26T15:44:36+08:00；影响层：网关、执行、订阅、依赖/执行；文件范围（16 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/execution/cpa`、`internal/execution/live.go`、`internal/gateway/live.go`、`internal/gateway/live_control_test.go` 等。
- 行为与影响：#774：恢复live日志并约束会话teardown。
- 本地对照：git cat-file -e dev:internal/gateway/live.go失败；git cat-file -e dev:internal/gateway/live_session.go失败；dev:go.mod无github.com/pion依赖
- 结论：**Do Not Port**；覆盖状态：Voice/live已移除；难度：**高**；新增价值：无。
- 冲突与保留边界：dev 已无 gateway Live 路径，不为该修复恢复已删除的 Voice 功能。
- 所需验证：不移植，不恢复live测试

### 32. `96a2fa7c` fix(guardrails): Filter opaque content and allow review failures (#776)

- 时间：2026-09-26T16:58:44+08:00；影响层：网关、前端；文件范围（9 个）：`internal/gateway/handler.go`、`internal/gateway/request_audit.go`、`internal/gateway/request_audit_test.go`、`internal/gateway/websocket_turn.go`、`internal/requestaudit/audit_test.go`、`internal/requestaudit/content.go`、`internal/requestaudit/content_test.go` 等。
- 行为与影响：#776：过滤opaque内容并放行无法完成的审查。
- 本地对照：dev:internal/requestaudit/content.go和audit.go均不存在；diff修改requestaudit/content.go及gateway/request_audit.go
- 结论：**Do Not Port**；覆盖状态：无requestaudit落点；难度：**高**；新增价值：无。
- 冲突与保留边界：会重新引入JEV审查链。
- 所需验证：不移植，无新增测试要求

### 33. `54db81fd` feat(cline): Add API key channel support (#778)

- 时间：2026-09-26T20:42:13+08:00；影响层：网关、执行、渠道；文件范围（22 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/channel/builtins.go`、`internal/channel/channel.go`、`internal/channel/cline_module_test.go`、`internal/channel/final_provider_contract_test.go` 等。
- 行为与影响：#778：新增Cline channel及请求、流式响应和错误转换。
- 本地对照：dev 的 internal/channel/modules/cline.go、internal/execution/bifrost/cline.go 和 cline_stream.go 均不存在；现有 Cline 客户端图标不能证明 Cline 上游 Provider 已实现。
- 结论：**Adapt**；覆盖状态：dev无Cline上游模块；难度：**高**；新增价值：中；Cline API-key上游渠道有增量。
- 冲突与保留边界：新增上游渠道应适配当前Bifrost本地fork及参数保真，不能重做或覆盖已有Cline客户端接入。
- 所需验证：验证上游注册、请求路径、流式工具调用、错误保真和取消；验证已有Cline客户端标准协议路径不回归

### 34. `3a75b9fd` fix(guardrails): Limit reviews to one bounded JEV call (#779)

- 时间：2026-09-26T21:51:45+08:00；影响层：网关、控制面、前端；文件范围（26 个）：`internal/control/request_audit.go`、`internal/control/request_logs.go`、`internal/control/request_logs_audit_test.go`、`internal/gateway/request_audit.go`、`internal/gateway/request_audit_test.go`、`internal/requestaudit/audit.go`、`internal/requestaudit/batches.go` 等。
- 行为与影响：#779：改为单次有界JEV调用并调整审查日志。
- 本地对照：dev:internal/requestaudit/excerpt.go、payload.go和audit.go均不存在；diff新增excerpt.go/payload.go，删除batches.go/chunks.go
- 结论：**Do Not Port**；覆盖状态：无JEV落点；难度：**高**；新增价值：无。
- 冲突与保留边界：dev不是已采用单次审查，而是已移除整个requestaudit；撤回此前错误判断。
- 所需验证：不移植，无新增测试要求

### 35. `1d8232eb` test(ci): Stabilize flaky race tests (#781)

- 时间：2026-09-28T09:35:22+08:00；影响层：网关、执行；文件范围（2 个）：`internal/execution/bifrost`、`internal/gateway/live_lifecycle_test.go`。
- 行为与影响：#781：稳定Bifrost首帧超时及live race测试。
- 本地对照：dev:internal/execution/bifrost/executor_test.go已有release channel、250ms FirstByte、30s请求超时和5s测试截止时间；dev用HTTPObserver断言SDK已收到200响应头，比上游handler-side started更严格；dev:internal/gateway/live_lifecycle_test.go不存在
- 结论：**Do Not Port**；覆盖状态：有效测试修复已覆盖并加强；难度：**低**；新增价值：无增量。
- 冲突与保留边界：覆盖会弱化现有observer断言；不恢复live测试。
- 所需验证：保留当前TestConvertedStreamRequiresFirstClientFrame

### 36. `e31b3be9` fix(bifrost): Align provider base URL semantics across protocols (#786)

- 时间：2026-09-28T15:14:56+08:00；影响层：网关、执行；文件范围（12 个）：`internal/execution/bifrost`、`internal/gateway/websocket_error_response_test.go`、`internal/gateway/websocket_errors_test.go`、`internal/gateway/websocket_review_test.go`、`internal/gateway/websocket_settings_test.go`、`internal/gateway/websocket_test.go`。
- 行为与影响：#786：对齐images、Responses WS和Decisions的base URL语义。
- 本地对照：dev:internal/execution/bifrost/websocket.go已调用resolveTypedTargetURL(base, /v1/responses)；dev:internal/execution/bifrost/executor.go的images prefix条件已仅限ProviderOpenAICompatible；dev:internal/execution/bifrost/decisions.go不存在；dev:internal/protocol/protocol.go不含Decisions
- 结论：**Do Not Port**；覆盖状态：保留功能的修复已覆盖；难度：**中**；新增价值：无增量。
- 冲突与保留边界：不能恢复Decisions或覆盖dev当前URL实现。
- 所需验证：沿用dev现有images_test.go和websocket_test.go

### 37. `2ef32f41` feat(channel): add Mistral native routes (#780)

- 时间：2026-09-28T16:23:47+08:00；影响层：网关、执行、控制面、状态、渠道、前端；文件范围（43 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/channel/compatible_presets_test.go`、`internal/channel/compiler.go`、`internal/channel/modules`、`internal/channel/route_golden_test.go` 等。
- 行为与影响：#780：增加Mistral原生dialect、HTTP执行和realtime转录。
- 本地对照：dev:internal/channel/modules/mistral.go已有Chat、embeddings及Responses/Anthropic/Gemini转换路由；dev:internal/dialect/mistral.go不存在；dev:internal/execution/bifrost/mistral.go不存在；dev:internal/gateway/mistral_realtime.go不存在；dev:internal/protocol/protocol.go无Mistral协议
- 结论：**Adapt**；覆盖状态：已有Mistral preset，无Mistral原生协议；难度：**高**；新增价值：中；OCR/FIM/moderation等原生HTTP端点有增量。
- 冲突与保留边界：preset和参数保真不等于原生协议；仅适配所需HTTP能力，不恢复Voice/realtime、Pion或Vue。
- 所需验证：验证原生HTTP路由、模型提取、认证和请求响应保真；验证已有Mistral兼容路由不回归；验证账号绑定资源限制，不开放realtime

### 38. `d20eb619` chore(web): bump the web-minor-and-patch group in /web with 8 updates (#784)

- 时间：2026-09-28T16:59:09+08:00；影响层：前端；文件范围（2 个）：`web/package.json`、`web/pnpm-lock.yaml`。
- 行为与影响：#784：升级Vue生态及Node类型、lint和format工具。
- 本地对照：dev:web/package.json为React 19.3.0、react-query 5.103.1和lucide-react 1.47.0；dev无vue、vue-query、reka-ui和Lucide Vue
- 结论：**Do Not Port**；覆盖状态：依赖更新被排除；Vue依赖无落点；难度：**中**；新增价值：按要求不迁移依赖。
- 冲突与保留边界：不覆盖lockfile，不恢复Vue；共享开发工具更新也按要求排除。
- 所需验证：不移植，无新增测试要求

### 39. `d574c7e3` chore(deps): bump golang.org/x/net (#782)

- 时间：2026-09-28T17:05:55+08:00；影响层：依赖/执行；文件范围（2 个）：`third_party/cpaembedded/go.mod`、`third_party/cpaembedded/go.sum`。
- 行为与影响：#782：CPA子模块x/net 0.57.0升级至0.59.0。
- 本地对照：dev:third_party/cpaembedded/go.mod声明x/net v0.57.0；dev:go.mod顶层声明x/net v0.59.0；diff仅改CPA子模块go.mod/go.sum；子模块声明不能直接等同顶层构建选用版本
- 结论：**Do Not Port**；覆盖状态：依赖更新被排除；难度：**低**；新增价值：按要求不迁移依赖。
- 冲突与保留边界：无生产逻辑修复，依赖明确Do Not Port。
- 所需验证：不移植，无新增测试要求

### 40. `21929aef` chore(deps): bump the go-minor-and-patch group with 6 updates (#783)

- 时间：2026-09-28T17:20:10+08:00；影响层：依赖；文件范围（2 个）：`go.mod`、`go.sum`。
- 行为与影响：#783：升级Pion WebRTC、ICE、RTP、SDP和STUN。
- 本地对照：dev:go.mod直接和间接依赖均无github.com/pion；dev:internal/gateway/live.go不存在；diff仅修改go.mod/go.sum
- 结论：**Do Not Port**；覆盖状态：dev无Pion，Voice/WebRTC无落点；难度：**高**；新增价值：无；依赖按要求排除。
- 冲突与保留边界：会恢复已删除Voice/WebRTC依赖；撤回此前Pion存在的错误描述。
- 所需验证：不引入Pion，不恢复WebRTC测试

### 41. `bab12b28` fix(execution): 修复协议转换中的分组扩展参数丢失 (#788)

- 时间：2026-09-29T09:09:52+08:00；影响层：网关、执行；文件范围（15 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/execution/bifrost`、`internal/execution/contracts.go`、`internal/execution/contracts_test.go`、`internal/gateway/execution_forward.go` 等。
- 行为与影响：#788：保留协议转换中的分组扩展参数并保护控制字段。
- 本地对照：dev:internal/execution/bifrost/parameter_overrides.go已有convertedParameterOverrides和mergeConvertedParameterOverrides；dev的convertedRequestFields额外包含OpenAIEmbeddings；dev:internal/execution/bifrost/converted.go已调用合并逻辑；dev已有parameter_overrides_test.go、parameter_override_conversion_test.go和parameter_override_size_test.go
- 结论：**Do Not Port**；覆盖状态：参数保真已覆盖且dev有扩展；难度：**低**；新增价值：无增量。
- 冲突与保留边界：覆盖会丢失dev的embeddings等扩展。
- 所需验证：保留当前参数保真、转换和大小限制测试

### 42. `7ceb5976` chore(codex): upgrade CPA and align Codex 0.159.2 model support (#791)

- 时间：2026-09-30T09:47:56+08:00；影响层：网关、执行、订阅、依赖、依赖/执行；文件范围（66 个）：`THIRD_PARTY_NOTICES.md`、`codex_version_test.go`、`go.mod`、`go.sum`、`internal/catalog/client_model_test.go`、`internal/catalog/codex_client_models.go`、`internal/catalog/codex_client_models_0.155.0.json` 等。
- 行为与影响：#791：升级CPA并对齐Codex 0.159.2模型、headers、service tier、WS和live。
- 本地对照：dev 的 third_party/cpaembedded/modelcatalog/client.go 已声明 ClientVersion="0.159.2"；go.mod 使用 CPA v8.0.8，本地 headers、模型目录和 WS 接口已有后续实现，且无 Live 路径。
- 结论：**Do Not Port**；覆盖状态：模型目录已覆盖，SDK 版本已演进；难度：**高**；新增价值：按要求不迁移依赖。
- 冲突与保留边界：不整体覆盖当前CPA，不恢复静态客户端模型目录或Voice；撤回此前目录存在的错误证据。
- 所需验证：不移植，无新增测试要求

### 43. `84896129` feat(gateway): add data-plane concurrency limits (#792)

- 时间：2026-09-30T17:26:33+08:00；影响层：网关、存储、控制面、状态、前端；文件范围（88 个）：`README.md`、`README_CN.md`、`README_JP.md`、`internal/control/access_key_collection.go`、`internal/control/access_key_idempotency.go`、`internal/control/access_key_rotation.go`、`internal/control/access_key_update_idempotency.go` 等。
- 行为与影响：#792：增加全局/访问密钥/分组并发限制、继承覆盖配置及持久化和UI。
- 本地对照：dev树无internal/ratelimit/concurrency.go、gateway/concurrency.go和control/concurrency.go；dev:internal/state/runtime_settings.go未找到concurrency_limit；dev已有gateway/handler.go、websocket_turn.go、state/manager.go和storage/models/access_key.go；dev:.docs/tech/frontend-structure.md规定Astryx唯一UI；dev无requestaudit、gateway live、Cline模块或Mistral realtime，不能假设这些接入点存在
- 结论：**Adapt**；覆盖状态：dev尚无三层数据面并发限制；难度：**高**；新增价值：高；保护HTTP、流式和Responses WS容量。
- 冲突与保留边界：只适配现存HTTP/流式/Responses WS；排除JEV、auto-model、Voice和Mistral realtime；按dev迁移序列和Astryx API/UI实现，不照搬上游0023编号及Vue代码。
- 所需验证：验证三层上限、继承、0不限和幂等释放；验证HTTP/流式/WS拒绝、取消及重试无名额泄漏；验证存储配置、API和Astryx界面；相关计数器运行race检查

## Upstream 独有：额外参考（60 项）

范围：`git rev-list --reverse upstream/main ^93502ced ^origin/main`。以下不是 canonical main 的待合并提交；仅记录对本地的潜在价值，不改变维护仓库或目标分支。

### 1. `82036b36` feat(codex): unify model capabilities and effort updates (#793)

- 时间：2026-09-30T22:22:48+08:00；影响层：网关、执行、控制面、订阅、依赖/执行；文件范围（31 个）：`internal/catalog/client_model_test.go`、`internal/catalog/codex_client_models.go`、`internal/catalog/codex_client_models_0.159.2.json`、`internal/control/credential_mutations.go`、`internal/control/credential_observations.go`、`internal/control/credential_plan.go`、`internal/control/credential_plan_test.go` 等。
- 行为与影响：统一Codex模型能力表与effort更新，核心在third_party/cpaembedded内嵌结构
- 本地对照：dev cpaembedded/embedded 自有 codex_service_tier.go、codex_headers.go 等独立版本线；internal/reasoning 自有推理逻辑
- 结论：**Do Not Port**；覆盖状态：different；难度：**高**；新增价值：低。
- 冲突与保留边界：upstream 改 cpaembedded 内嵌数据结构，与 dev 分叉的第三方目录直接冲突
- 所需验证：若未来适配，分别在 CPA 子模块和根模块验证客户端能力与 reasoning 行为；本轮不运行。

### 2. `3d54cd6f` fix(codex): preserve legacy credential fingerprints (#794)

- 时间：2026-10-01T08:29:19+08:00；影响层：状态、订阅、依赖/执行；文件范围（6 个）：`internal/state/loader`、`internal/subscription/providers`、`third_party/cpaembedded/embedded`。
- 行为与影响：保留旧凭据指纹避免订阅重新登录
- 本地对照：dev 的 internal/storage/models/group.go 存在 Fingerprint/IdentityFingerprint 及唯一索引；internal/subscription/providers/codex/identity.go 和 identity_test.go 自有身份规范化。旧导入格式是否改变指纹仍需具体样本核对。
- 结论：**Adapt**；覆盖状态：部分覆盖，旧指纹边界待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：必须保留单凭据分组和现有身份唯一约束；只在旧凭据重导入确实产生身份漂移时适配，不引入兼容迁移。
- 所需验证：旧/新凭据样本的身份一致性、重复导入拒绝及刷新后指纹稳定性。

### 3. `2ebf2302` fix(docker): update pinned OpenSSL packages (#797)

- 时间：2026-10-01T09:27:38+08:00；影响层：构建/文档；文件范围（2 个）：`Dockerfile`、`internal/webui/workflow_test.go`。
- 行为与影响：Docker 镜像固定 OpenSSL 补丁版本
- 本地对照：dev Dockerfile 存在且 go.mod go 1.27.0；镜像基线与 upstream 不一定一致，需核对当前 pin
- 结论：**Do Not Port**；覆盖状态：待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：构建依赖 pin 更新，按本次分类不作为功能移植；是否需要独立更新取决于当前镜像基线。
- 所需验证：docker build 后镜像内 openssl version 核对

### 4. `6b79b261` feat(credentials): add API key file import (#796)

- 时间：2026-10-01T09:34:42+08:00；影响层：前端；文件范围（8 个）：`web/src/frontends/classic`、`web/src/frontends/modern`、`web/src/shared/api-key-file-import.ts`。
- 行为与影响：凭据支持从文件批量导入 API Key
- 本地对照：dev 有 web/src/frontends/astryx/features/import/CredentialTextarea.tsx、internal/control/credential_stages.go；单凭据约束由 internal/storage/migrations/0010_single_credential_per_group.go 实施。
- 结论：**Adapt**；覆盖状态：文本导入已有，文件格式差异待验证；难度：**中**；新增价值：中。
- 冲突与保留边界：只吸收确有需求的文件读取/解析；不能引入上游同组批量多凭据导入，保留单组单凭据合同。
- 所需验证：前端组件测试 + 导入操作 e2e

### 5. `5690326e` fix(web): simplify concurrency limit display (#798)

- 时间：2026-10-01T10:20:44+08:00；影响层：前端；文件范围（10 个）：`README.md`、`README_CN.md`、`README_JP.md`、`web/src/frontends/classic`、`web/src/frontends/modern`。
- 行为与影响：并发限制展示简化并精简文案
- 本地对照：dev 无并发限制功能，无展示对象，无可适配项
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：upstream 改 Vue 组件与 i18n locale 文件，dev 为 astryx React
- 所需验证：前端渲染快照

### 6. `ece3539e` fix(grok): update CLI client version for chat proxy (#799)

- 时间：2026-10-01T11:36:50+08:00；影响层：依赖/执行；文件范围（1 个）：`third_party/cpaembedded/embedded`。
- 行为与影响：Grok CLI 版本号更新
- 本地对照：dev third_party/cpaembedded/embedded/grok_models.go 自有版本维护，版本线随 dev 自行演进
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：第三方内嵌目录版本线不同
- 所需验证：无

### 7. `54214fb3` chore(cpa): upgrade SDK to v8.0.8 and verify Grok identity (#801)

- 时间：2026-10-01T12:10:56+08:00；影响层：依赖、依赖/执行；文件范围（11 个）：`THIRD_PARTY_NOTICES.md`、`go.mod`、`go.sum`、`internal/catalog/codex_client_models.go`、`third_party/cpaembedded/README.md`、`third_party/cpaembedded/embedded`、`third_party/cpaembedded/go.mod` 等。
- 行为与影响：CPA SDK 升级并校验 Grok 身份
- 本地对照：dev third_party/cpaembedded/go.mod go 1.26.6，SDK 版本线独立
- 结论：**Do Not Port**；覆盖状态：different；难度：**中**；新增价值：低。
- 冲突与保留边界：SDK 版本升级需整体跟随，不能单独摘补丁
- 所需验证：若独立升级，在 third_party/cpaembedded 子模块运行对应测试，并验证根模块 CPA 接口。

### 8. `dbac9c26` fix(logs): align request logs with final reasoning settings (#800)

- 时间：2026-10-01T13:49:54+08:00；影响层：网关、执行、订阅、依赖/执行；文件范围（25 个）：`internal/execution/cpa`、`internal/gateway/request_log.go`、`internal/gateway/websocket_reasoning_test.go`、`internal/gateway/websocket_turn.go`、`internal/subscription/providers`、`third_party/cpaembedded/embedded`。
- 行为与影响：请求日志按最终生效 reasoning 参数记录
- 本地对照：dev internal/requestlog/mapper.go + internal/reasoning 包自有记录链路；是否含『最终生效值回填日志』需对照验证
- 结论：**Adapt**；覆盖状态：partial；难度：**中**；新增价值：中。
- 冲突与保留边界：需按 dev mapper 字段结构适配
- 所需验证：go test ./internal/requestlog/...；对照日志样例

### 9. `7cc0df2e` fix(ui): reset log refresh and page scroll positions (#802)

- 时间：2026-10-01T15:04:45+08:00；影响层：前端；文件范围（8 个）：`web/src/frontends/modern`。
- 行为与影响：刷新后滚动位置复位
- 本地对照：astryx LogsTable.tsx/列表组件中未见滚动复位逻辑
- 结论：**Adapt**；覆盖状态：待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：Vue 实现改 AppListFrame/AppLayout，astryx 需按自身布局组件改
- 所需验证：前端交互测试

### 10. `25142f96` chore(deps): upgrade Bifrost SDK to v1.11.0 (#803)

- 时间：2026-10-01T15:51:17+08:00；影响层：依赖；文件范围（2 个）：`go.mod`、`go.sum`。
- 行为与影响：Bifrost SDK 升级
- 本地对照：dev go.mod: github.com/maximhq/bifrost/core v1.11.3（高于 upstream v1.11.0）且 replace 到 third_party/bifrost-core
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：低。
- 冲突与保留边界：无；dev 版本已更新
- 所需验证：go build ./...

### 11. `addd6987` fix(logs): Unify request latency and output speed (#804)

- 时间：2026-10-01T17:40:00+08:00；影响层：网关、存储、控制面、前端；文件范围（41 个）：`internal/control/request_log_output_timing_test.go`、`internal/control/request_logs.go`、`internal/dialect/output_timing.go`、`internal/dialect/output_timing_test.go`、`internal/gateway/execution_forward.go`、`internal/gateway/forward.go`、`internal/gateway/handler.go` 等。
- 行为与影响：统一首响应时延与输出速度口径
- 本地对照：.docs/tech/provider-feedback.md：first-response latency 与 output rate 分类（<10 t/s faulty、10-20 slow），迁移 0024_provider_feedback；internal/health/feedback.go ClassifyFeedback
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：低。
- 冲突与保留边界：dev 已有等价且更细的 provider-feedback 体系
- 所需验证：go test ./internal/health/... ./internal/requestlog/...

### 12. `5ddc867f` fix(bifrost): preserve compatible tool calls and reasoning (#806)

- 时间：2026-10-02T14:10:17+08:00；影响层：执行；文件范围（9 个）：`internal/execution/bifrost`。
- 行为与影响：compatible 协议保真工具调用与推理内容
- 本地对照：dev internal/execution/bifrost/converted.go(28.8K)+conversion_fidelity.go+ahead_tool_output_test.go 自有保真实现；覆盖面与 upstream 差异需对照用例
- 结论：**Adapt**；覆盖状态：partial；难度：**高**；新增价值：中。
- 冲突与保留边界：两套 converted 转换层结构不同，只能对照回归用例补缺
- 所需验证：go test ./internal/execution/bifrost/ -run Fidelity

### 13. `02e1879e` feat(proxy): Add centralized proxy management (#805)

- 时间：2026-10-02T17:41:06+08:00；影响层：存储、控制面、状态、订阅、前端；文件范围（74 个）：`internal/control/credential_mutations.go`、`internal/control/credential_probe.go`、`internal/control/discover.go`、`internal/control/discover_test.go`、`internal/control/group_create.go`、`internal/control/group_idempotency.go`、`internal/control/group_settings.go` 等。
- 行为与影响：集中式代理资源管理（独立 CRUD+引用计数+验证）
- 本地对照：dev 仅有分组级 internal/control/proxy_config.go（optionalField[outboundproxy.Config]）与 internal/outboundproxy/proxy.go；无 proxies 独立资源/路由/引用
- 结论：**Adapt**；覆盖状态：missing；难度：**高**；新增价值：高。
- 冲突与保留边界：upstream 改 74 文件含 control/scheduler/loader；需按 dev 的 proxy_config→outboundproxy 链路重做资源层与存储
- 所需验证：go test ./internal/control/... ./internal/outboundproxy/...；代理连通性 probe

### 14. `e966eb41` feat(credentials): Add aliases and mask subscription accounts (#809)

- 时间：2026-10-02T20:12:33+08:00；影响层：存储、控制面、状态、前端；文件范围（58 个）：`internal/control/credential_display.go`、`internal/control/credential_mutations.go`、`internal/control/credential_name_test.go`、`internal/control/credentials.go`、`internal/control/health.go`、`internal/control/home_subscription_accounts.go`、`internal/control/home_subscription_accounts_test.go` 等。
- 行为与影响：凭据别名+订阅账号邮箱打码
- 本地对照：dev internal/state/loader/loader.go:85 Alias/TestAlias 已回填；internal/control/credentials.go:79 EmailMask+maskCanonicalCredential 已有打码；订阅卡展示覆盖面待验证
- 结论：**Adapt**；覆盖状态：partial；难度：**中**；新增价值：中。
- 冲突与保留边界：展示层需落到 astryx SubscriptionAccountCard.tsx
- 所需验证：go test ./internal/state/... ./internal/control/...；前端快照

### 15. `4ee34470` fix(ui): improve narrow-screen layout of group detail and health pages (#810)

- 时间：2026-10-02T21:38:36+08:00；影响层：前端；文件范围（3 个）：`web/src/frontends/modern`。
- 行为与影响：窄屏下分组详情与健康页布局
- 本地对照：dev GroupDetailView.tsx 12.4K 自有布局；响应式断点行为待实测
- 结论：**Adapt**；覆盖状态：待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：Vue 组件改 astryx TSX
- 所需验证：窄屏视口快照

### 16. `3cb66939` fix(logs): Align request timing and average output speed (#811)

- 时间：2026-10-02T22:22:09+08:00；影响层：网关、执行、存储、控制面、订阅、前端、依赖/执行；文件范围（57 个）：`internal/control/request_log_first_response_test.go`、`internal/control/request_log_output_timing_test.go`、`internal/control/request_logs.go`、`internal/dialect/first_output.go`、`internal/dialect/first_output_test.go`、`internal/dialect/output_timing.go`、`internal/dialect/output_timing_test.go` 等。
- 行为与影响：对 #804 时延/速度口径的修正（first_output 替代 output_timing）
- 本地对照：同 #804：.docs/tech/provider-feedback.md first-response/output-rate 已是最终口径；internal/requestlog/feedback_test.go
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：低。
- 冲突与保留边界：dev 无 output_timing 历史包袱
- 所需验证：无

### 17. `36d1b976` feat(logs): Add configurable client IP tracking (#812)

- 时间：2026-10-02T22:56:29+08:00；影响层：网关、存储、控制面、前端；文件范围（65 个）：`.env.example`、`README.md`、`README_CN.md`、`README_JP.md`、`internal/app/app.go`、`internal/app/client_ip_test.go`、`internal/app/lifecycle_integration_test.go` 等。
- 行为与影响：可配置记录客户端 IP（含代理头信任配置）
- 本地对照：全库 grep 无 ClientIP/client_ip 记录逻辑；internal/gateway/auth_test.go 仅涉及转发头校验，不入日志
- 结论：**Adapt**；覆盖状态：missing；难度：**中**；新增价值：高。
- 冲突与保留边界：需按 dev requestlog/types.go+mapper.go+storage migration 加列；信任代理配置入口按 dev settings 体系
- 所需验证：go test ./internal/requestlog/... ./internal/gateway/...；X-Forwarded-For 场景

### 18. `2f0343f1` refactor(storage): reuse SQLite migration metadata (#814)

- 时间：2026-10-03T01:11:56+08:00；影响层：存储；文件范围（12 个）：`internal/storage/access_key_cost_limit_schema_test.go`、`internal/storage/client_ip_migration_test.go`、`internal/storage/concurrency_migration_test.go`、`internal/storage/credential_name_migration_test.go`、`internal/storage/db_test.go`、`internal/storage/migration_inspection.go`、`internal/storage/migration_validation_queries_test.go` 等。
- 行为与影响：复用 SQLite 迁移元数据的内部重构
- 本地对照：dev internal/storage 自有迁移体系（migrations/0024_provider_feedback.go 等），测试组织不同
- 结论：**Do Not Port**；覆盖状态：different；难度：**中**；新增价值：低。
- 冲突与保留边界：纯内部重构，随 upstream 迁移史强耦合
- 所需验证：go test ./internal/storage/...

### 19. `82115ac6` fix(logs): Improve text selection and filter click targets (#815)

- 时间：2026-10-03T01:41:02+08:00；影响层：前端；文件范围（3 个）：`web/src/frontends/modern`。
- 行为与影响：日志文本可选中、筛选点击区扩大
- 本地对照：astryx LogsTable.tsx 选中/点击行为待实测
- 结论：**Adapt**；覆盖状态：待验证；难度：**低**；新增价值：中。
- 冲突与保留边界：Vue LogTableCell 改 astryx 表格组件
- 所需验证：前端交互测试

### 20. `65c6509f` feat(catalog): add editable Codex model catalogs (#817)

- 时间：2026-10-04T08:51:41+08:00；影响层：网关、控制面、状态、前端；文件范围（58 个）：`internal/catalog/client_model.go`、`internal/catalog/client_model_catalog.go`、`internal/catalog/client_model_test.go`、`internal/catalog/codex_client_models.go`、`internal/clientcatalog/catalog.go`、`internal/clientcatalog/catalog_test.go`、`internal/control/client_catalog.go` 等。
- 行为与影响：Codex 客户端模型目录可编辑
- 本地对照：dev 的 third_party/cpaembedded/modelcatalog/client.go 嵌入 Codex 0.159.2 目录；internal/catalog/client.go、control/catalog_sync.go 负责公开目录，二者不是同一数据源。
- 结论：**Do Not Port**；覆盖状态：different；难度：**中**；新增价值：低。
- 冲突与保留边界：可编辑客户端能力目录并非当前需求；不要以 models.dev 同步替代 Codex 目录语义，也不引入第二套目录管理入口。
- 所需验证：go test ./internal/catalog/... ./internal/control/...

### 21. `3c106567` chore(deps): bump the go-minor-and-patch group with 3 updates (#819)

- 时间：2026-10-05T08:46:07+08:00；影响层：依赖；文件范围（2 个）：`go.mod`、`go.sum`。
- 行为与影响：Go 依赖例行升级
- 本地对照：dev go.mod 依赖树独立（bifrost v1.11.3、x/net v0.59.0）
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：依赖树不同不能抄 go.sum
- 所需验证：go mod tidy && go build

### 22. `3884dac0` chore(web): bump the web-minor-and-patch group in /web with 4 updates (#820)

- 时间：2026-10-05T08:55:24+08:00；影响层：前端；文件范围（2 个）：`web/package.json`、`web/pnpm-lock.yaml`。
- 行为与影响：web 前端依赖例行升级
- 本地对照：dev web 前端为 astryx React，与 upstream Vue 依赖树不同
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：package.json/pnpm-lock 不兼容
- 所需验证：pnpm install && pnpm build

### 23. `643a075f` chore(deps): bump github.com/andybalholm/brotli (#818)

- 时间：2026-10-05T09:04:08+08:00；影响层：依赖/执行；文件范围（2 个）：`third_party/cpaembedded/go.mod`、`third_party/cpaembedded/go.sum`。
- 行为与影响：brotli 安全升级
- 本地对照：dev go.mod 与 third_party/cpaembedded/go.mod 均 brotli v1.2.4，已含该补丁线
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：低。
- 冲突与保留边界：无
- 所需验证：go build ./...

### 24. `49d1bf65` feat(codex): Sync and display credit balances (#823)

- 时间：2026-10-05T09:09:20+08:00；影响层：执行、控制面、订阅、前端；文件范围（25 个）：`internal/control/classic_api_contract_test.go`、`internal/control/credential_credits_test.go`、`internal/control/credential_observations.go`、`internal/execution/cpa`、`internal/subscription/passive_credits_test.go`、`internal/subscription/passive_quota.go`、`internal/subscription/passive_quota_flush.go` 等。
- 行为与影响：Codex 订阅余额(credits)同步与展示（区别于 reset-credits）
- 本地对照：dev internal/subscription/providers/codex/codex.go:213 ObserveResetCredits 已有 reset-credit 观测；余额(balance)拉取与展示链路待验证
- 结论：**Adapt**；覆盖状态：partial；难度：**中**；新增价值：中。
- 冲突与保留边界：需按 dev observation/pipeline 扩展余额字段，astryx SubscriptionAccountCard.tsx 加展示
- 所需验证：go test ./internal/subscription/providers/codex/...；UI 快照

### 25. `d3257cf4` fix(web): Recover frontend startup failures (#824)

- 时间：2026-10-05T10:04:12+08:00；影响层：前端；文件范围（4 个）：`web/src/frontends/modern`、`web/src/main.ts`、`web/src/startup-recovery.css`。
- 行为与影响：前端启动失败自恢复兜底页
- 本地对照：dev web/src/frontends/astryx/main.tsx:84-100 已有 showStartupFailure 启动失败兜底
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：中。
- 冲突与保留边界：Vue bootstrap 改 React 入口
- 所需验证：模拟资源加载失败的前端测试

### 26. `b88b395c` feat(scheduler): add group priority and tiered retry fallback (#825)

- 时间：2026-10-05T18:25:25+08:00；影响层：网关、调度、存储、控制面、状态、前端；文件范围（58 个）：`internal/control/group_collection.go`、`internal/control/group_create.go`、`internal/control/group_idempotency.go`、`internal/control/group_priority_test.go`、`internal/control/group_settings.go`、`internal/control/group_update.go`、`internal/control/modern_groups.go` 等。
- 行为与影响：分组优先级+分层重试回退
- 本地对照：dev internal/control/group_write.go:37 Priority 字段+internal/scheduler/scheduler.go:76-89 tier 分层迭代已自有；fallback 四桶见 internal/gateway/attempts.go deferredAttempts。upstream 的『组级(非凭据级)优先级+跨组回退』语义差异待验证
- 结论：**Do Not Port**；覆盖状态：partial；难度：**高**；新增价值：低。
- 冲突与保留边界：调度器两套实现，直接移植会破坏 dev tier 语义
- 所需验证：go test ./internal/scheduler/... ./internal/gateway/...

### 27. `5c6ec0f2` fix(web): expose the modern credential test icon (#826)

- 时间：2026-10-05T19:46:03+08:00；影响层：前端；文件范围（1 个）：`web/src/frontends/modern`。
- 行为与影响：凭据测试入口图标可见性修复
- 本地对照：astryx 凭据卡操作区(CredentialCardActions 等价物)可见性待实测
- 结论：**Adapt**；覆盖状态：待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：Vue 组件改 astryx
- 所需验证：前端快照

### 28. `a5c691bb` fix(bifrost): merge compatible assistant tool turns (#828)

- 时间：2026-10-06T16:56:16+08:00；影响层：执行；文件范围（3 个）：`internal/execution/bifrost`。
- 行为与影响：compatible 多轮 assistant 工具消息合并
- 本地对照：dev internal/execution/bifrost/converted.go 自有多轮转换；是否合并连续 assistant tool turns 需对照用例验证
- 结论：**Adapt**；覆盖状态：partial；难度：**低**；新增价值：中。
- 冲突与保留边界：转换层结构不同，对照回归补缺
- 所需验证：go test ./internal/execution/bifrost/ -run Compatible

### 29. `b771ef0a` docs(readme): update Fluxion AI sponsorship to Sidrune AI (#832)

- 时间：2026-10-08T15:33:28+08:00；影响层：文档/工具；文件范围（5 个）：`README.md`、`README_CN.md`、`README_JP.md`、`screenshot/fluxionai-horizontal.png`、`screenshot/sidrune-ai.png`。
- 行为与影响：README 赞助商更换
- 本地对照：dev README 自有品牌维护
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：无。
- 冲突与保留边界：无
- 所需验证：无

### 30. `6c6380ad` fix(channel): declare native embeddings and rerank routes for SiliconFlow (#834)

- 时间：2026-10-08T19:13:34+08:00；影响层：渠道；文件范围（4 个）：`internal/channel/embeddings_module_test.go`、`internal/channel/modules`、`internal/channel/rerank_module_test.go`、`internal/channel/route_golden_test.go`。
- 行为与影响：SiliconFlow 声明原生 embeddings/rerank 路由
- 本地对照：dev:internal/channel/modules/siliconflow.go 仅声明 Chat、ListModels、Probe 与 Responses/Anthropic/Gemini 转换，没有 Embeddings/Rerank 路由。
- 结论：**Adapt**；覆盖状态：已确认缺少原生路由声明；难度：**低**；新增价值：中：使用 SiliconFlow 向量/重排序时有直接价值。。
- 冲突与保留边界：低
- 所需验证：go test ./internal/channel/ -run SiliconFlow；route golden

### 31. `935aff97` fix(cpa): refresh subscription model catalogs online (#835)

- 时间：2026-10-09T15:20:01+08:00；影响层：控制面、订阅、依赖/执行；文件范围（6 个）：`internal/container/container.go`、`internal/control/runtime.go`、`internal/control/runtime_test.go`、`internal/subscription/providers`、`third_party/cpaembedded/embedded`。
- 行为与影响：订阅(cpaembedded)模型目录在线热更新
- 本地对照：dev internal/control/catalog_sync.go 是 models.dev 目录 24h 同步，属公开模型目录；订阅侧(cpaembedded embedded)模型目录热更新无对应，不能以目录同步判定已覆盖
- 结论：**Adapt**；覆盖状态：different；难度：**中**；新增价值：低。
- 冲突与保留边界：upstream 更新器挂在 cpaembedded/modelcatalog；dev 该目录版本线不同
- 所需验证：在 third_party/cpaembedded 子模块验证目录热更新、失败保留和并发读取；补控制面发现测试。

### 32. `c71c34df` fix(codex): upgrade CPA and support response interrupts (#838)

- 时间：2026-10-09T20:22:47+08:00；影响层：网关、执行、订阅、依赖、依赖/执行；文件范围（21 个）：`go.mod`、`go.sum`、`internal/catalog/codex_client_models.go`、`internal/execution/cpa`、`internal/execution/websocket.go`、`internal/gateway/execution_forward.go`、`internal/gateway/websocket.go` 等。
- 行为与影响：客户端断开时中断上游响应
- 本地对照：dev internal/gateway/stream.go:293,338,472 streamWatchdog.interrupt+websocket_turn.go:736 idle 中断已自有；『WebSocket turn 级响应中断传播到 cpa』行为待验证
- 结论：**Adapt**；覆盖状态：partial；难度：**高**；新增价值：中。
- 冲突与保留边界：upstream 中断挂在 websocket_interrupt.go+cpaembedded 新接口，dev 需按自身 watchdog 链路适配
- 所需验证：go test ./internal/gateway/ -run Interrupt/Cancel；长流取消实测

### 33. `25ec9aa2` fix(build): upgrade Go to patched 1.27.2 (#839)

- 时间：2026-10-09T21:36:17+08:00；影响层：依赖；文件范围（3 个）：`Dockerfile`、`go.mod`、`internal/webui/workflow_test.go`。
- 行为与影响：Go 工具链补丁升级
- 本地对照：纯 Go 工具链版本升级，dev 自行随构建演进
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：低。
- 冲突与保留边界：低；go.mod 一行+CI
- 所需验证：go build+全量测试

### 34. `5a008796` fix(codex): upgrade CPA and ignore late interrupts (#840)

- 时间：2026-10-10T09:08:13+08:00；影响层：网关、依赖、依赖/执行；文件范围（9 个）：`go.mod`、`go.sum`、`internal/catalog/codex_client_models.go`、`internal/gateway/websocket_interrupt.go`、`internal/gateway/websocket_interrupt_test.go`、`third_party/cpaembedded/embedded`、`third_party/cpaembedded/go.mod` 等。
- 行为与影响：忽略迟到中断避免误判失败
- 本地对照：dev stream.go watchdog 中断后是否有 late-interrupt 丢弃逻辑待验证
- 结论：**Adapt**；覆盖状态：待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：随 #838 中断体系
- 所需验证：中断竞态测试

### 35. `3745fbab` feat(cli): add version and update commands (#842)

- 时间：2026-10-10T10:28:59+08:00；影响层：文档/工具；文件范围（10 个）：`internal/releasecheck/selection.go`、`internal/releasecheck/selection_test.go`、`main.go`、`main_test.go`、`update_cli.go`、`update_cli_test.go`、`update_replace_nonwindows.go` 等。
- 行为与影响：CLI 自更新（含 Windows 替换）
- 本地对照：dev main.go 无 update/version 子命令；internal/releasecheck 存在可选作更新检查基础
- 结论：**Adapt**；覆盖状态：missing；难度：**中**；新增价值：中。
- 冲突与保留边界：需按 dev releasecheck+部署方式（Hostinger 镜像部署）评估自更新是否适用容器场景
- 所需验证：go test main_test.go update_*_test.go

### 36. `78884001` feat(health): Add configurable upstream error rules (#830)

- 时间：2026-10-10T11:40:05+08:00；影响层：网关、控制面、状态、前端；文件范围（51 个）：`internal/control/error_rules_test.go`、`internal/control/group_detail.go`、`internal/control/group_settings.go`、`internal/control/settings.go`、`internal/gateway/custom_error_rules_test.go`、`internal/gateway/handler.go`、`internal/gateway/jev_decision.go` 等。
- 行为与影响：按分组/全局自定义上游错误码重试/冷却规则
- 本地对照：dev internal/health/execution_judge.go:721 retryableUpstreamStatus 为硬编码状态码表，无规则配置；state/runtime_settings.go 无 error_rules
- 结论：**Adapt**；覆盖状态：missing；难度：**高**；新增价值：高。
- 冲突与保留边界：涉及 control 设置面+gateway 判定+health judge 多层；需按 dev runtime_settings 体系设计配置注入
- 所需验证：go test ./internal/health/... ./internal/gateway/...；规则生效 e2e

### 37. `6cc3b889` refactor(ui): Unify modern form help and interactions (#843)

- 时间：2026-10-10T13:43:50+08:00；影响层：前端；文件范围（42 个）：`web/src/frontends/modern`、`web/src/shared/frontend`。
- 行为与影响：modern 表单帮助交互统一
- 本地对照：dev astryx 组件体系自有一致性规范，无 modern 组件对应物
- 结论：**Do Not Port**；覆盖状态：different；难度：**中**；新增价值：低。
- 冲突与保留边界：42 文件 Vue 组件重构不可迁移
- 所需验证：无

### 38. `cbea3dfc` fix(deps): upgrade x/net to patched v0.60.0 (#844)

- 时间：2026-10-10T14:14:36+08:00；影响层：依赖、依赖/执行；文件范围（4 个）：`go.mod`、`go.sum`、`third_party/cpaembedded/go.mod`、`third_party/cpaembedded/go.sum`。
- 行为与影响：x/net 安全补丁升级
- 本地对照：纯依赖升级(x/net)，dev 依赖树独立，自行评估升级即可
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：中。
- 冲突与保留边界：低；go.mod+go.sum
- 所需验证：go build+go test ./...

### 39. `c24243f7` feat(parameteroverride): support array wildcard field overrides (#847)

- 时间：2026-10-11T10:00:43+08:00；影响层：网关、执行；文件范围（19 个）：`internal/automodel/decision.go`、`internal/execution/bifrost`、`internal/execution/contracts.go`、`internal/execution/contracts_test.go`、`internal/gateway/auto_model.go`、`internal/gateway/auto_model_test.go`、`internal/gateway/execution_forward.go` 等。
- 行为与影响：参数覆盖支持数组通配（如 items[*].role）
- 本地对照：dev internal/parameteroverride/rules.go:380,383 明确『array element paths are not supported』；rules.go:184-192 仅有模型名 trailing 前缀通配（model wildcard≠array wildcard）
- 结论：**Adapt**；覆盖状态：missing；难度：**中**；新增价值：高。
- 冲突与保留边界：需扩展 JSON Pointer 求值器支持数组索引/通配，含 size 限制测试
- 所需验证：go test ./internal/parameteroverride/... ./internal/gateway/ -run ParameterOverride

### 40. `435f2683` test(gateway): stabilize Codex live race-test flakes (#854)

- 时间：2026-10-11T10:14:24+08:00；影响层：网关；文件范围（2 个）：`internal/gateway/live_integration_test.go`、`internal/gateway/websocket_test.go`。
- 行为与影响：修复 live race 测试抖动
- 本地对照：仅改 upstream 测试文件；dev 无 live 测试套件，无对应可改对象
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：upstream 测试文件结构不同，仅借思路
- 所需验证：go test -race ./internal/gateway/（重复跑验证稳定）

### 41. `52b6cc70` feat(credentials): add batch API key testing (#855)

- 时间：2026-10-11T11:26:08+08:00；影响层：前端；文件范围（16 个）：`web/src/frontends/classic`、`web/src/frontends/modern`、`web/src/shared/credential-test-batch.ts`、`web/src/shared/credential-test-messages.ts`。
- 行为与影响：凭据批量测试入口与结果汇总
- 本地对照：internal/control/credential_probe.go/credential_actions.go 中 grep 无 batch 聚合入口；astryx credentials 页无批量栏；单个 probe 链路已自有，批量编排是否存在待确认
- 结论：**Adapt**；覆盖状态：待验证；难度：**中**；新增价值：中：本地每组单凭据，多组批测有潜在价值；上游同组多密钥批测不直接适用。。
- 冲突与保留边界：按本地单组单凭据模型评估跨组批测编排，复用单次 probe 和已有并发限制；不恢复同组多凭据批量栏。
- 所需验证：go test ./internal/control/ -run Batch；前端 e2e

### 42. `d8c0bfd9` chore(deps): bump the go-minor-and-patch group with 5 updates (#858)

- 时间：2026-10-11T15:05:58+08:00；影响层：依赖；文件范围（2 个）：`go.mod`、`go.sum`。
- 行为与影响：Go 依赖例行升级
- 本地对照：dev go.mod 依赖树独立
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：不可抄
- 所需验证：go mod tidy

### 43. `ec8a16e9` chore(web): bump the web-minor-and-patch group in /web with 9 updates (#859)

- 时间：2026-10-11T15:11:46+08:00；影响层：前端；文件范围（2 个）：`web/package.json`、`web/pnpm-lock.yaml`。
- 行为与影响：web 依赖例行升级
- 本地对照：dev 前端依赖树不同（React vs Vue）
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：不可抄
- 所需验证：pnpm audit

### 44. `b27a86b7` chore(deps): bump github.com/tidwall/gjson (#856)

- 时间：2026-10-11T15:16:19+08:00；影响层：依赖/执行；文件范围（2 个）：`third_party/cpaembedded/go.mod`、`third_party/cpaembedded/go.sum`。
- 行为与影响：gjson 升级
- 本地对照：dev go.mod 与 cpaembedded go.mod 均 gjson v1.19.0，已在同补丁线
- 结论：**Do Not Port**；覆盖状态：covered；难度：**低**；新增价值：低。
- 冲突与保留边界：无
- 所需验证：go build

### 45. `bba39e7c` chore(ci): bump the github-actions-minor-and-patch group with 2 updates (#857)

- 时间：2026-10-11T15:28:23+08:00；影响层：CI；文件范围（2 个）：`.github/workflows/release.yml`、`internal/webui/workflow_test.go`。
- 行为与影响：GitHub Actions 版本升级
- 本地对照：dev CI workflow 自有维护
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：无。
- 冲突与保留边界：纯 CI 品牌
- 所需验证：无

### 46. `207a948c` feat(claude): unify subscription reset credits (#860)

- 时间：2026-10-11T15:34:47+08:00；影响层：控制面、订阅、渠道、前端、依赖/执行；文件范围（38 个）：`internal/channel/claude_module_test.go`、`internal/channel/modules`、`internal/control/credential_observations_test.go`、`internal/control/credential_reset_credits.go`、`internal/control/credential_reset_credits_claude_test.go`、`internal/control/credential_reset_credits_target_test.go`、`internal/control/server.go` 等。
- 行为与影响：Claude 订阅 reset-credits 统一探测与展示
- 本地对照：dev internal/control/credential_reset_credits.go(控制面通用)已有；third_party/cpaembedded/embedded/ 无 claude_reset_credits.go（dev claude 文件清单无此项），provider 端探测缺口待验证
- 结论：**Adapt**；覆盖状态：partial；难度：**中**；新增价值：中。
- 冲突与保留边界：cpaembedded claude 内部结构分叉，需按 dev bridge 接口适配
- 所需验证：go test ./internal/control/ -run ResetCredit；claude 实测

### 47. `52eede88` fix(deps): patch vulnerable frontend transitive dependencies (#861)

- 时间：2026-10-11T18:34:29+08:00；影响层：前端；文件范围（2 个）：`web/pnpm-lock.yaml`、`web/pnpm-workspace.yaml`。
- 行为与影响：前端传递依赖漏洞修补
- 本地对照：纯前端传递依赖修补，dev pnpm 依赖树不同，需自行 pnpm audit
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：中。
- 冲突与保留边界：lockfile 不可抄，需自行 pnpm audit
- 所需验证：pnpm audit && pnpm build

### 48. `1b309fe9` feat(cpa): unify client version maintenance (#865)

- 时间：2026-10-11T21:57:01+08:00；影响层：订阅、依赖/执行；文件范围（17 个）：`internal/subscription/providers`、`third_party/cpaembedded/embedded`。
- 行为与影响：各 provider 客户端版本集中到 client_versions.go
- 本地对照：dev third_party/cpaembedded/embedded/ 版本号分散于 codex_headers.go/grok_models.go/claude_account.go 等，无集中文件
- 结论：**Do Not Port**；覆盖状态：partial；难度：**低**；新增价值：中。
- 冲突与保留边界：纯版本维护结构重构；本地 Codex 版本已集中于 modelcatalog.ClientVersion，尚无其他版本常量漂移的失败证据，不为维护便利新增抽象。
- 所需验证：若未来确需重构，在 third_party/cpaembedded 子模块验证版本常量消费者及客户端头部。

### 49. `6ab829aa` docs(readme): add Modelflare principal partner showcase (#866)

- 时间：2026-10-12T11:09:24+08:00；影响层：文档/工具；文件范围（6 个）：`README.md`、`README_CN.md`、`README_JP.md`、`screenshot/modelflare-banner-en.png`、`screenshot/modelflare-banner-ja.png`、`screenshot/modelflare-banner-zh.png`。
- 行为与影响：README 赞助横幅
- 本地对照：dev README 自有维护
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：无。
- 冲突与保留边界：无
- 所需验证：无

### 50. `3d04d9f1` feat(channel): add partner tiers and Modelflare preset (#867)

- 时间：2026-10-12T11:44:43+08:00；影响层：执行、渠道、前端；文件范围（29 个）：`internal/channel/builtins.go`、`internal/channel/channel.go`、`internal/channel/compiler.go`、`internal/channel/compiler_test.go`、`internal/channel/final_provider_contract_test.go`、`internal/channel/modelflare_module_test.go`、`internal/channel/modules` 等。
- 行为与影响：Modelflare 合作渠道预设+渠道分层
- 本地对照：dev internal/channel/modules/ 与 builtins 中无 modelflare；该渠道为 upstream 赞助方，dev 用户无使用场景
- 结论：**Do Not Port**；覆盖状态：missing；难度：**中**；新增价值：低。
- 冲突与保留边界：若确需，仅借『渠道 partner tier』字段设计
- 所需验证：无

### 51. `d64c954f` feat(web): add shared auto refresh to logs and usage (#849)

- 时间：2026-10-12T12:43:27+08:00；影响层：前端；文件范围（17 个）：`web/src/frontends/classic`、`web/src/frontends/modern`、`web/src/shared/auto-refresh.ts`。
- 行为与影响：日志/用量页自动刷新
- 本地对照：dev 的 web/src/shared/control/resources/usage.ts:731 和 health.ts:384 已配置 refetchInterval 且后台不刷新；仍需核对 Astryx 日志页及共享刷新偏好的覆盖。
- 结论：**Adapt**；覆盖状态：用量定时刷新已有，共享偏好和日志行为待验证；难度：**低**；新增价值：中。
- 冲突与保留边界：复用现有 query 轮询和页面可见性逻辑，只补确有缺口的共享偏好，不另建刷新调度器。
- 所需验证：前端组件测试

### 52. `ab655129` feat(logs): record client request bodies per access key (#851)

- 时间：2026-10-12T13:06:31+08:00；影响层：网关、存储、控制面、状态、前端；文件范围（72 个）：`internal/control/access_key_collection.go`、`internal/control/access_key_idempotency.go`、`internal/control/access_key_rotation.go`、`internal/control/access_keys.go`、`internal/control/auth.go`、`internal/control/home.go`、`internal/control/request_body_recording_test.go` 等。
- 行为与影响：按 Access Key 开关常态记录请求体入日志
- 本地对照：dev 已有 raw capture 能力：internal/control/probe_capture.go:45,145 AppendRequestBody + internal/gateway/capture.go + internal/debugcapture/store.go:751 appendPart(PartBody)；但为调试捕获，非常态化 per-access-key 日志字段，二者不等价，常态化缺口待验证
- 结论：**Adapt**；覆盖状态：partial；难度：**高**；新增价值：中。
- 冲突与保留边界：需按 dev requestlog 表加列+脱敏+容量策略；开关挂 Access Key 模型
- 所需验证：go test ./internal/requestlog/... ./internal/gateway/；脱敏校验

### 53. `52a82ef1` feat(channels): add aggregation search and passthrough (#852)

- 时间：2026-10-12T13:37:56+08:00；影响层：执行、调度、渠道；文件范围（16 个）：`internal/channel/cliproxyapi_module_test.go`、`internal/channel/gpt_load_module_test.go`、`internal/channel/modules`、`internal/channel/newapi_module_test.go`、`internal/channel/route_golden_test.go`、`internal/execution/bifrost`、`internal/scheduler/channel_scheduler_test.go`。
- 行为与影响：聚合渠道(cliproxyapi/newapi/gpt-load)的 /v1/search 透传
- 本地对照：dev internal/channel/modules/{cliproxyapi,gpt_load,newapi}.go 存在；internal/execution/bifrost/passthrough.go 与 capabilities.go 中 grep 无 search 聚合声明
- 结论：**Adapt**；覆盖状态：待验证；难度：**中**；新增价值：中。
- 冲突与保留边界：需按 dev bifrost capabilities/route golden 体系扩展
- 所需验证：go test ./internal/channel/ ./internal/execution/bifrost/

### 54. `d129d2a6` docs(readme): replace chief sponsor banners (#869)

- 时间：2026-10-12T19:29:16+08:00；影响层：文档/工具；文件范围（3 个）：`screenshot/modelflare-banner-en.png`、`screenshot/modelflare-banner-ja.png`、`screenshot/modelflare-banner-zh.png`。
- 行为与影响：赞助横幅图片替换
- 本地对照：dev README 自有维护
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：无。
- 冲突与保留边界：无
- 所需验证：无

### 55. `3453b405` fix(web): show retained credential failure details (#870)

- 时间：2026-10-12T19:41:07+08:00；影响层：前端；文件范围（3 个）：`web/src/frontends/classic`、`web/src/frontends/modern`。
- 行为与影响：凭据失败详情保留展示
- 本地对照：astryx 凭据卡失败详情展示行为待实测（internal/credential_activity 有活动记录链路）
- 结论：**Adapt**；覆盖状态：待验证；难度：**低**；新增价值：中。
- 冲突与保留边界：Vue 组件改 astryx
- 所需验证：前端快照

### 56. `74e957c0` fix(logs): compact and unify modern log popovers (#871)

- 时间：2026-10-12T20:22:49+08:00；影响层：前端；文件范围（3 个）：`web/src/frontends/modern`。
- 行为与影响：modern 日志弹层精简统一
- 本地对照：astryx 日志弹层/浮层组件待实测对照
- 结论：**Adapt**；覆盖状态：待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：Vue 组件改 astryx
- 所需验证：前端快照

### 57. `2591d6e6` fix(subscription): filter CPA import fields by channel (#873)

- 时间：2026-10-13T08:59:07+08:00；影响层：控制面、订阅；文件范围（6 个）：`internal/control/credential_import_batch_test.go`、`internal/control/credential_stages_test.go`、`internal/subscription/providers`。
- 行为与影响：CPA 凭据导入字段按渠道过滤
- 本地对照：dev internal/subscription/providers/ 无 importfile 子包（providers 仅 antigravity/claude/codex/grok/observation）；导入走 internal/control/credential_stages.go，是否已按渠道过滤待验证
- 结论：**Adapt**；覆盖状态：导入流程已有，字段过滤差异待验证；难度：**中**；新增价值：低。
- 冲突与保留边界：upstream 引入 importfile/cpa.go；dev 需按 credential_stages 导入管线适配
- 所需验证：go test ./internal/control/ -run Import/Stages

### 58. `9702cc09` fix(subscription): restore cooldowns when quota recovers (#874)

- 时间：2026-10-13T13:56:25+08:00；影响层：网关、控制面、状态、订阅；文件范围（24 个）：`internal/control/credential_mutations.go`、`internal/control/credential_observations.go`、`internal/control/credential_probe.go`、`internal/control/model_cooldown_test.go`、`internal/control/quota_cooldown_recovery_test.go`、`internal/gateway/handler.go`、`internal/gateway/handler_test.go` 等。
- 行为与影响：配额恢复后自动解除冷却
- 本地对照：dev 的 internal/state/registry.go 已有 RecoverEntry、RecoverEntryIfVersionMatch，并明确保留 Provider cooldown deadlines；internal/subscription/credential_manager.go 和 credential_manager_recovery_test.go 维护恢复流程。
- 结论：**Adapt**；覆盖状态：已有恢复入口，配额恢复触发条件待验证；难度：**高**；新增价值：高。
- 冲突与保留边界：配额恢复只能解除配额来源的冷却，不能误清余额/认证/风控/性能故障状态；需先对照本地恢复测试证明缺口。
- 所需验证：go test ./internal/state/ ./internal/subscription/ ./internal/gateway/；配额恢复 e2e

### 59. `5cac4e1d` fix(logs): preserve upstream response model observations (#877)

- 时间：2026-10-14T07:04:45+08:00；影响层：网关、执行；文件范围（17 个）：`internal/execution/bifrost`、`internal/execution/cpa`、`internal/execution/responsemodel`、`internal/gateway/execution_forward.go`、`internal/gateway/execution_forward_test.go`、`internal/gateway/response_model_integration_test.go`、`internal/gateway/websocket_test.go` 等。
- 行为与影响：保留上游实际返回模型观察
- 本地对照：dev 有 internal/gateway/response_model_observation.go、internal/execution/responsealias，以及现有模型一致性字段；本轮未运行所有协议的截断/终止/别名回写场景。
- 结论：**Do Not Port**；覆盖状态：已有响应模型观测，边界差异待验证；难度：**低**；新增价值：低。
- 冲突与保留边界：保留客户端别名与真实上游模型分层；只在捕获证实遗漏时补修，不重复构建观测器。
- 所需验证：go test ./internal/gateway/ -run ResponseModel

### 60. `22eacc1b` chore(cpa): update SDK and client versions (#878)

- 时间：2026-10-14T18:25:55+08:00；影响层：执行、依赖、依赖/执行；文件范围（16 个）：`go.mod`、`go.sum`、`internal/catalog/codex_client_models.go`、`internal/execution/cpa`、`third_party/cpaembedded/embedded`、`third_party/cpaembedded/go.mod`、`third_party/cpaembedded/go.sum` 等。
- 行为与影响：CPA SDK 与客户端版本例行更新
- 本地对照：dev third_party/cpaembedded 版本线独立（go 1.26.6，自有 codex 目录 json）
- 结论：**Do Not Port**；覆盖状态：different；难度：**低**；新增价值：低。
- 冲突与保留边界：第三方版本线不可混
- 所需验证：若独立升级，在 third_party/cpaembedded 子模块及根模块分别验证 CPA 接口与 Provider 请求。

## 后续工作流

1. **确定上次截止点**：读取“已完成评估的截止点与本轮待处理范围”中的上次截止点，不把“最新同步检查”的 ref 当作已移植截止点；`origin/main` 是维护目标，`upstream/main` 只作额外参考。不要用本地 cherry-pick 后产生的新 SHA 替代上游截止点。
2. **刷新 refs**：依次执行 `git fetch --prune origin` 和 `git fetch --prune upstream`，然后重新记录三个 ref 的 commit、时间和分叉统计。
3. **发现新提交**：分别检查 `<上次 origin/main commit>..origin/main` 和 `<上次 upstream/main commit>..upstream/main`。将 canonical `origin` 新提交与 upstream 独有提交分开，避免把参考分支提交误当成待合并提交。
4. **逐提交建立摘要**：按提交时间顺序记录 SHA、PR 编号、标题、变更文件范围，并用一到三句话说明行为变化、影响层（后端、网关、调度、执行、前端、依赖或 CI）和用户可见影响。
5. **对照本地合同评估**：检查 `dev` 是否已有等价能力；区分“已覆盖”“可直接移植”“需要行为级适配”和“没有本地落点”。对需要适配的提交，明确冲突点、保留的本地不变量和需要补充的测试，不直接整体 cherry-pick。
6. **形成取舍**：使用三类结论：
   - `Port`：本地结构和语义兼容，可以按原提交顺序移植。
   - `Adapt`：只吸收目标行为，按本地合同重写；记录不能直接移植的原因。
   - `Do Not Port`：纯依赖、CI、品牌、未采用功能或没有独立本地落点。
7. **执行已批准变更**：用户明确要求后，按指定顺序 cherry-pick 或实现适配；冲突时保留 `dev` 的执行、调度、重试、凭据、WebSocket 和数据迁移边界，不新增兼容层或无依据的 fallback。
8. **验证并推进截止点**：运行受影响包的定向测试，再运行必要的全量测试、前端类型检查和 `git diff --check`。确认范围内的提交均已有完成结论后，推进相应截止点，并覆盖最新 ref、时间和分叉统计；只做检查或仍有待处理提交时，仅更新检查快照和专题文档链接，不在本文保留已完成提交的历史清单。
9. **保护工作树**：检查 `git status --short`，不修改与本轮分析无关的已有改动或未跟踪文件。

每次运行的最小摘要格式：

```text
<时间> <SHA>（#<PR>）<标题>：<行为摘要>；影响：<范围>；结论：Port / Adapt / Do Not Port。
```
