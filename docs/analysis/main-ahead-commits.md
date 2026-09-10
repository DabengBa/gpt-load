# Main Ahead 提交评估（#606–#619）

- 最新提交（评估基线）：`d0f75e97c6340d7793d9170f9ad1007c6bcc5350`（`Merge pull request #28 from DabengBa/plan/main-ahead-integration`）
- 生成时间：`2026-09-11T05:44:00+08:00`（本文件自身的提交哈希不写入本文，避免自引用）

## 基线

- 分析分支：`dev`
- 当前基线：`dev@d0f75e97c6340d7793d9170f9ad1007c6bcc5350`
- 对比目标：`upstream/main@abc6483d0ed081e305859d88eba41c81b6efd1d8`（`fix(codex): 升级 CPA 并固定上游版本声明 (#619)`）
- 本地 `main`、`origin/main`、`upstream/main` 三者已对齐（同为 `abc6483d`），`main` 是上游镜像。
- `git rev-list --left-right --count dev...upstream/main`：`91 24`
  - `dev` 独有 91 个提交
  - 上游独有 24 个提交
- 合并基点：`0ddc41d8b718c0281b1ba2f5bdfe5b23622621ce`
- `git merge-tree --write-tree dev upstream/main` 预演：**103 个冲突路径**
  - `web/src` 28、`internal/control` 21、`internal/state` 11、`internal/gateway` 9、`third_party` 6、`internal/scheduler` 6
- 评估范围：上游 `#606`–`#619` 共 10 个提交（`#605` 及更早的结论见「前置结论」，不在本文重开）。
- 上游仍有一个未合并的开放 PR 未计入本文范围：`#620` `fix(gateway): 兼容 WS 请求中的 stream 布尔参数`（分支 `tbphp/fix-responses-websocket-stream@76176c58`，base `main`）。它是 `#616` 的兼容性修复，`#616` 未落地前不可评估。

## 当前合同

L1/L2/L3 判断以当前 `dev` 合同为边界：

- 每个分组只有一个上游凭据；普通第二凭据创建、导入、连接必须原子拒绝。
- 权重、优先级、熔断只属于模型 Route Entry；不恢复分组权重或凭据权重。
- 调度中心负责入口级路由、候选排序和运行态；不恢复旧 scheduler、Registry 或前端调度 API。
- 请求重试预算是系统级单一来源（`snapshot.Settings.RetryCount`），分组 `retry_count` 已退役。
- `(group_id, entry_id)` 只用于后端定位和 mutation，不进入用户可见的调度文案或筛选维度。
- 监控筛选的权威查询状态是 `range` preset；`from_ms`/`to_ms` 查询合同（上游 `#594`）未引入。
- 迁移台账只允许追加，编号必须与注册表位置连续（见「迁移编号重排方案」）。
- 存储层只支持 SQLite 与 PostgreSQL；MySQL 支持已随 `cef04938` 移除，`gormmysql` 与
  `mysqlRequiresCheckDropSyntax0003` 已从 `internal/storage` 删除。
- 交付面只覆盖 Linux 与 Docker；Windows service、`securefile` Windows ACL、Windows 打包不在范围内。
- Protocol、operation、external model 和 target config 必须保持当前 URL、API、目标冻结和候选 fallback 语义。

## 迁移编号重排方案

### 冲突事实

`dev` 与 `upstream/main` 对 `0010`–`0012` 三号码段赋予完全不同的语义：

| 编号 | `dev` | `upstream/main` |
| --- | --- | --- |
| `0010` | `0010_single_credential_per_group` | `0010_model_cooldown` |
| `0011` | `0011_usage_latency` | `0011_custom_access_keys` |
| `0012` | `0012_debug_captures` | `0012_access_key_mask_prefix` |

`0001`–`0009` 号码一致，但文件内容不同：`dev` 在这 6 个文件里删掉了全部 MySQL 分支
（`gormmysql` 导入、`DROP CHECK` 语法选择、`ValidateRecoverable0006` 重命名为
`validateDecisionColumns0006`）。因此上游 `0010`–`0012` 不能按原文复制，也不能只改文件名。

### 代码级约束（`internal/storage/migration.go`）

1. `validateMigrationRegistry` 要求 ID 匹配 `^(\d{4})_[a-z0-9]+(?:_[a-z0-9]+)*$`，并且
   **ID 的数字部分必须等于注册表切片中的 1-based 位置**。号码不连续或位置错位直接启动失败。
2. `applyMigrationsLocked` 用 `ORDER BY id ASC` 读取 `schema_migrations`，并要求
   `entries[index].ID == applied[index]`。**已应用的迁移永远不能改号或改 ID 字符串**，
   否则现网实例下次启动即报 `unknown or non-contiguous migration`。
3. 结论：上游待移植的 3 个迁移只能**追加**到 `0013`、`0014`、`0015`，且必须保持上游的相对顺序。
4. 符号名冲突是真的编译错误：上游 `const ID0010` 与 `dev` 的 `const ID0010` 同名；
   上游 `Up0010`/`Validate0010`/`ValidateRecoverable0010` 与 `dev` 的 `Up0010`/`Validate0010` 同名。
   一号之差同时解决号码冲突和符号冲突。
5. 上游 `0010`/`0011` 依赖 `mysqlRequiresCheckDropSyntax0003` 与 `gormmysql`，两者在 `dev` 已不存在；
   不删这两个分支就是编译失败，不是风格问题。

### 重排映射

| 上游 | `dev` 目标 | 文件 | ID 常量 | 重命名符号 |
| --- | --- | --- | --- | --- |
| `0010_model_cooldown` | `0013` | `0013_model_cooldown.go` | `ID0013 = "0013_model_cooldown"` | `Up0010`→`Up0013`、`Validate0010`→`Validate0013`、`ValidateRecoverable0010`→`ValidateRecoverable0013`、`*0010` 私有常量全部改 `0013` |
| `0011_custom_access_keys` | `0014` | `0014_custom_access_keys.go` | `ID0014 = "0014_custom_access_keys"` | `Up0011`→`Up0014`、`Validate0011`→`Validate0014`、`ValidateRecoverable0011`→`ValidateRecoverable0014`、`*0011` 私有常量全部改 `0014` |
| `0012_access_key_mask_prefix` | `0015` | `0015_access_key_mask_prefix.go` | `ID0015 = "0015_access_key_mask_prefix"` | `Up0012`→`Up0015`、`Validate0012`→`Validate0015`、`ValidateRecoverable0012`→`ValidateRecoverable0015`、`*0012` 私有常量全部改 `0015` |

顺序不可调换：上游 `0012` 给 `access_keys` 加 `key_prefix` 列，上游 `0011` 重写同表的
`chk_access_key_suffix` 约束；两者同表，必须按上游顺序串行。
`0013_model_cooldown` 扩展的是 `request_log_attempts.effect` 约束与 `cooldown_until_ms` 列，
与 `dev` 的 `0010`/`0011`/`0012` 表集合（`credentials`、usage/延迟列、`debug_captures`）无交集，可独立先落。

### 依赖前提（已核对，成立）

- `dev` 的 `request_log_attempts` 自 `0006` 之后没有任何迁移再触碰它，SQLite DDL 与上游
  `0010` 的 `rebuildModelCooldownSQLite0010` 前置状态一致；`0006` 的
  `CONSTRAINT chk_request_log_attempt_effect CHECK (effect IN ('','none','cooldown_credential','record_credential_failure','skip_group'))`
  与上游要替换的 `old` 字符串逐字相同。
- `dev` 的 `access_keys.key_suffix` 仍是上游 `0011` 的前置形态（`char(4)` + 四位十六进制校验），
  `key_prefix` 列在 `dev` 不存在，`0014`/`0015` 的 `HasColumn` 幂等判断成立。
- 反例：所有 `dev` 侧前置条件只对 **SQLite/PostgreSQL** 成立，MySQL 路径必须整段删除而不是保留。

### 移植步骤

1. 复制上游三个文件并改名、改 `const IDxxxx`、改全部 `Up/Validate/ValidateRecoverable` 与私有常量后缀。
2. 删除 MySQL 分支：`gormmysql` 导入、`mysqlRequiresCheckDropSyntax0003` 调用、
   `DROP CHECK`/`information_schema.check_constraints` 查询，只保留 SQLite 与 PostgreSQL 路径。
   与 `dev` 现有 `0006`/`0009` 的处理方式保持一致（`dev` 已删除「可恢复的 MySQL DDL 中断」概念，
   因此 `ValidateRecoverable` 命名是否保留需按 `dev` 风格决定）。
3. 在 `internal/storage/migration.go` 的 `migrations` 切片末尾按 `0013`→`0014`→`0015` 追加三项。
4. 同步 `internal/storage/migration_test.go` 的期望 ID 列表（当前只列到 `ID0012`）。
5. 若同批移植 `#611` 的 `internal/storage/migration_sqlite.go`（上游新增，SQLite 关闭外键后重建
   被引用表），需先确认 `dev` 现有迁移链是否已在该路径上处理同一问题；该文件同样只支持 SQLite。
6. 决定性验证：现网实例的 `schema_migrations` 已有 `0001`–`0012` 共 12 行，重排后的注册表必须
   前 12 项 ID 逐字不变；用真实数据卷跑一次启动，确认只新增 3 行且不触发 `unknown or non-contiguous`。
7. 平台验证：`0013` 的 `rebuildModelCooldownSQLite0010` 与 `0014` 的 `rebuildAccessKeysSQLite0011`
   都是「读 `sqlite_master` 原文 → 字符串替换 → 重建表 → 恢复索引」，必须在 SQLite 上实跑一次
   （含索引恢复与外键校验），不能只靠 PostgreSQL 通过。

## L1 / L2 / L3 汇总

| 上游提交 | PR | 主题 | 级别 | 硬前置 |
| --- | --- | --- | --- | --- |
| `eecef9f4` | #614 | README 赞助展示 | **L1** | 无 |
| `fe4b6ac1` | #607 | 原生文本重排序协议 | **L2** | 无（重试分类需改接 `dev` 判定链） |
| `0c9d1888` | #617 | 请求身份头覆盖与会话兼容 | **L2** | 无（需补 `HeaderRules.ConfiguredNames`） |
| `fb18dc9e` | #606 | 精简模型冷却展示与管理接口 | L3 | `#599`（`dev` 不存在） |
| `d417b7de` | #608 | 统一日志与用量时间筛选交互 | L3 | `#594`（`dev` 现为 `range` preset） |
| `de6d2d44` | #611 | 自定义访问密钥与编辑生成交互 | L3 | 迁移 `0014`/`0015` |
| `7cbd2e67` | #609 | 响应状态续接路由 | L3 | 新状态合同 |
| `f091528b` | #616 | 原生 Responses WebSocket 与逐轮治理 | L3 | `#609` 的续接合同 |
| `abc6483d` | #619 | 升级 CPA 并固定上游版本声明 | L3 | `#616`（`internal/execution/cpa/websocket.go`） |
| `7d80a981` | #612 | Codex 独立 WebSocket Session 封装 | 已完成 | — |

## 前置结论（沿用，不重开）

- 专题 1（凭据全量操作与单凭据导入，`e888fe60`+`96d3e0d5`）、专题 2（入口公平调度，`9cb3f986`）、
  专题 3（模型错误重试与健康恢复，`e0bfa07e`+`d4699dd2`）、专题 4（自定义订阅上游，`6da82242`）、
  专题 5（Usage 时间窗口，`33fb54bf`+`175949e2`）保持「不纳入范围」。
- 专题 6（全局请求重试预算，`a4255546`）已随 `dev` 的 `3274e32c` 同步完成；保留两处本地差异：
  `retryAttemptLimit(retryCount) = max(retryCount, 1)`（尝试总次数，默认 5，上游为 `retryCount + 1`）
  与 `fallback.missing_evidence_retry` replay 许可（`internal/health/execution_judge.go`）。
- 专题 7（Codex WebSocket）的合同半边已落地（`docs/design/codex-websocket-session.md`），
  数据面半边未落地。

## L1

### `eecef9f4` `docs(readme): 添加 OfoxAI 赞助展示 (#614)`

上游内容：

- `README.md`、`README_CN.md`、`README_JP.md` 增加赞助位，新增 `screenshot/ofoxai.svg`。

当前价值与适配边界：

- 零代码，不触碰任何运行合同，是本次新增范围里唯一可独立 cherry-pick 的提交。
- 唯一适配点是插入位置：`dev` 的 README 有自建实例、发布与回滚等自有章节，赞助位需手工对齐，不要整文件覆盖。

## L2

### `fe4b6ac1` `feat(gateway): 接入原生文本重排序协议 (#607)`

上游内容：

- 新增 `internal/dialect/rerank.go`、`internal/dialect/rerank_usage.go`、`internal/execution/bifrost/rerank.go`。
- 在 channel 能力位、Bifrost executor/passthrough/runtime_manager、`internal/execution/contracts.go`、
  `internal/provideradapter`、`internal/gateway/{execution_forward,models,request_log}.go` 接入 rerank 协议。
- 57 个文件，+1138/-41，其中 20 个是测试。

当前价值与适配边界：

- 纯 additive：新增协议不影响单凭据、入口级调度、重试预算和路由条目合同，是 L2 而不是 L3 的原因。
- 必须改接 `dev` 的失败判定链：上游把 rerank 的错误分类接到 `internal/execution/model_cooldown.go`，
  该文件在 `dev` 不存在（属 `#599`）。分类应落到 `dev` 的 `internal/health/execution_judge.go`。
- `dev` 的 `internal/gateway/execution_forward.go`、`internal/channel/compiler.go` 已被模型路由条目改造，
  接入时要保留入口级 priority/breaker 与候选 fallback 语义。
- `dev` 无 `internal/execution/wsnative`，rerank 与 WS 无关，不引入。
- 验证：新增协议的转换正确性、usage 保留、不可转换时的拒绝路径，以及 404/400 不被误升级为可重试。

### `0c9d1888` `fix(codex): 修正请求身份头覆盖与会话兼容 (#617)`

上游内容：

- 新增 `third_party/cpaembedded/embedded/codex_headers.go`（含测试），把 Codex 身份头的「显式配置」与
  「默认注入」分开，锁定覆盖优先级。
- `internal/execution/contracts.go` 增加 `ConfiguredHeaders []string`（reference-backed，`json:"-"`），
  由 `input.Group.HeaderRules.ConfiguredNames()` 填充。
- `internal/control/{credential_probe,discover_executor}.go`、`internal/execution/cpa/{adapter,codex_provider,provider}.go`、
  `internal/gateway/execution_forward.go`、`internal/state/snapshot.go`、
  `internal/subscription/providers/codex/codex.go` 跟随该合同调整。

当前价值与适配边界：

- 主体是身份头覆盖顺序修复，diff 里只有 3 处 websocket 提及，与 `#616` 的数据面**弱耦合**，
  可以先于 `#616` 单独落地，作为独立的请求身份正确性修复。
- `dev` 已有 `state.HeaderRules`（`internal/control/group_detail.go`、`discover_executor.go`），
  但没有 `ConfiguredNames()`；需要按 `dev` 的 `HeaderRulesResponse`/resolved 形态补齐该查询方法。
- `dev` 的 `internal/execution/cpa/codex_provider.go` 存在且被 `#619` 继续修改，迁移时要保证
  与后续 `#619` 的 CPA 版本声明兼容，避免两次改同一处身份头逻辑。
- 验证：显式头覆盖默认头、显式置空表示移除、Codex 订阅分支与 API Key 分支的身份头一致。

## L3

### `fb18dc9e` `refactor(cooldown): 精简模型冷却展示与管理接口 (#606)`

上游内容：

- 30 文件，+207/-323，`internal/control/model_cooldown.go`、`runtime_observation.go`、
  `web/src/components/ui/ModelCooldownDetails.vue`、健康页与凭据页展示、三语文案。

硬前置：

- 修改 `internal/control/model_cooldown.go`，该文件由 `#599` 新增，`dev` 不存在。
- 属专题 3（模型错误重试与健康恢复），已判「不纳入范围」。单独移植 `#606` 无从落地。

### `d417b7de` `feat(monitor): 统一日志与用量的时间筛选交互 (#608)`

上游内容：

- 14 个前端文件，+666/-620：`AppDateTimeRangePicker.vue`、`LogsAdvancedFilterDrawer.vue`、
  `LogsFilterForm.vue`、`UsageTab.vue`、`log-filters.ts`、`usage-filters.ts`、`monitor-route.ts`、
  `web/src/lib/time.ts`、三语文案。

硬前置：

- 依赖 `#594` 的 `from_ms`/`to_ms` 查询合同（上游 `usage-filters.ts` 是 `preset` + `from_ms`/`to_ms` 双轨）。
- `dev` 现合同是 `range` preset（`UsageFilters['range']`、`defaultTimeRange`），后端也没有 `from_ms`/`to_ms` 查询参数。
- 属专题 5（Usage 时间窗口），已判「不纳入范围」。只移植前端会直接对不上后端查询。

### `de6d2d44` `feat(access-keys): 支持自定义密钥并统一编辑与生成交互 (#611)`

上游内容：

- 55 文件，+2065/-633，覆盖后端、存储、前端和三语文案。
- 新增迁移 `0011_custom_access_keys`、`0012_access_key_mask_prefix`，新增
  `internal/storage/migration_sqlite.go`。
- 新增 `internal/control/access_key_update_idempotency.go`，改 `access_keys.go`、`bootstrap.go`、
  `health.go`、`home.go`、幂等摘要与操作恢复、`internal/platform/errors`、i18n locales、
  `internal/state/{loader,snapshot}.go`、`internal/storage/models/access_key.go`。
- 前端新增 `AccessKeyCredentialField.vue`、`AccessKeySelect.vue`、`AccessKeyHandoff.vue`、
  `access-key-strength.ts` 等。

当前价值与适配边界：

- 必须与「迁移编号重排方案」一起做：`0011`→`0014`、`0012`→`0015`，并删除迁移里的 MySQL 分支。
- `dev` 的 `access_keys` 表形态与上游前置状态一致（四位十六进制尾号、无 `key_prefix`），
  所以约束重写和加列都能落；但 `dev` 的访问密钥管理页有自己的分发/交接流程，
  上游新增的 `AccessKeyHandoff.vue` 等组件不能整目录覆盖。
- 幂等与操作恢复路径（`access_key_update_idempotency.go`、`operation_recovery.go`）与 `dev` 的
  单凭据原子拒绝逻辑同域，需要逐处确认不会绕过写入前拒绝。
- 验证：自定义密钥创建/编辑/轮换、短密钥全遮罩、列表无需解密即可显示前缀、
  幂等重放、以及迁移在 SQLite 上的表重建与索引恢复。

### `7cbd2e67` `feat(responses): 实现响应状态续接路由 (#609)`

上游内容：

- 38 文件，+1226/-85。
- 新增 `internal/gateway/responses_continuation.go`、`internal/state/response_bindings.go`、
  `internal/app/runtime_checkpoint.go`。
- 改 `internal/dialect/{dialect,openai,openai_responses,anthropic,request_execution,request_fields,rerank}.go`、
  `internal/execution/{contracts,validation,bifrost/executor}.go`、
  `internal/gateway/{execution_forward,forward,handler,reason}.go`、
  `internal/parameteroverride/rules.go`、`internal/scheduler/inspect.go`、`internal/control/group_create.go`、
  `internal/container/container.go`。

当前价值与适配边界：

- 引入新的运行态合同（响应绑定 + 检查点），必须先定义再实现，不能按提交顺序直接叠加。
- `dev` 只在 `internal/dialect/request_execution.go` 里把 `previous_response_id` 当作「不可重放字段」识别
  （`hasMeaningfulField(root, "previous_response_id")`，测试在 `prompt_affinity_test.go`），
  没有绑定表、没有检查点；续接路由是全新维度。
- `#609` 修改 `internal/dialect/rerank.go`，即**依赖 `#607` 已落地**；顺序上 `#607` 必须先做。
- 检查点只能恢复续接运行态，不能改变 `dev` 的模型路由 API、入口级 breaker 或重试预算。
- 验证：跨请求续接命中/未命中、检查点恢复后绑定不丢失、候选切换后续接语义、以及续接与
  `DispatchMaybeSent`/replay safety 的一致性。

### `f091528b` `feat(gateway): 接入原生 Responses WebSocket 与逐轮治理 (#616)`

上游内容：

- 55 文件，+4719/-152，是本次范围里最大的提交。
- 新增 `internal/execution/websocket.go`、`internal/execution/wsnative/session.go`、
  `internal/execution/bifrost/websocket.go`、`internal/execution/cpa/websocket.go`、
  `internal/provideradapter/websocket.go`、`internal/gateway/websocket.go`、
  `internal/gateway/websocket_turn.go`。
- 改 `internal/channel/{channel,compiler}.go`（WS 能力位）、`internal/channel/modules/{codex,openai,gpt_load,sub2api,cliproxyapi,xai}.go`、
  `internal/channel/spec/definition.go`、`internal/control/{settings,group_detail}.go`、
  `internal/gateway/{execution_forward,handler}.go`、`internal/scheduler/{scheduler,inspect}.go`、
  `internal/state/{snapshot,runtime_settings}.go`、`internal/subscription/providers/codex/websocket.go`、
  `third_party/cpaembedded/embedded/codex_websocket.go`、设置页与分组设置的前端和三语文案。
- diff 中出现 656 处 websocket、18 处 `previous_response_id`，即逐轮治理是内联在 `websocket_turn.go` 里实现的。

重新设计边界：

- 这不是替换 HTTP executor，而是新增连续多轮 Responses 传输生命周期，必须作为整体设计。
- Session 必须使用调度器已选定的 credential，不自行选凭据、不绕过入口级 priority/breaker，也不恢复旧 CPA Manager。
- 发送前失败必须是 `DispatchNotSent` 并交给当前候选策略；发送后断线、半响应和关闭必须保留
  `DispatchMaybeSent` 与 `ReplaySafetyUnknown` 证据。
- 不得加入 HTTP fallback、业务请求重放、隐式跨 credential 迁移或旧 retry/fallback 行为。
- `internal/channel` 能力位与设置页分层开关要按 `dev` 的分组设置覆盖交互重接，不能覆盖调度中心 URL 状态。
- 已知缺口（`#612` 落地时记录）：CPA 在通知 lifecycle 之前用默认 logger 输出上游正文，
  数据面接入前必须在 `dev` 运行时关闭，不在 vendored 层注册 `init()` 全局 hook。
- 验证：连接复用、同一 session 多轮、代理传播、主动取消、上游关闭、超时、发送后错误、
  credential replacement 后的 session 隔离、分组停用/删除时关闭连接。

### `abc6483d` `fix(codex): 升级 CPA 并固定上游版本声明 (#619)`

上游内容：

- 19 文件：`go.mod`/`go.sum`、`third_party/cpaembedded/{README.md,go.mod,go.sum}`、
  vendored `embedded/{embedded,codex_headers,codex_websocket}.go`、
  `internal/execution/cpa/{codex_provider,websocket}.go`、`THIRD_PARTY_NOTICES.md`。

硬前置：

- 修改 `internal/execution/cpa/websocket.go`，该文件由 `#616` 新增，`dev` 不存在。
- 因此 `#619` 必须排在 `#616` 之后，且自身没有可独立落地的部分。
- `dev` 的 `third_party/cpaembedded/embedded/codex_websocket.go` 目前是 `#612` 那一版（随 PR #28 落地），
  升级时要注意 `#616` 与 `#619` 两次修改的先后，不能跳版本。

## 已完成

### `7d80a981` `feat(codex): 添加独立上游 WebSocket Session 封装 (#612)`

- 已随 PR #28（`dev@2994c96e`）落地：vendored `CodexWSSession` + `internal/subscription/providers/codex.WSSession`。
- 边界：不注册到 `provideradapter`/`executor`、不在网关请求路径上被调用、不加 UI/配置项/数据库字段。
- 合同：`docs/design/codex-websocket-session.md`。
- 后续 `#616` 会反过来修改 `internal/subscription/providers/codex/websocket.go` 与 vendored
  `codex_websocket.go`，接数据面时以 `#616` 的版本为基准。

## 执行顺序

1. `eecef9f4`（#614）：L1，独立 cherry-pick，随时可做。
2. `fe4b6ac1`（#607）：L2，纯新增协议，先把 rerank 的失败分类改接到 `dev` 的 execution_judge。
3. `0c9d1888`（#617）：L2，身份头覆盖修复，补 `HeaderRules.ConfiguredNames()`；与第 2 步无耦合，可并行。
4. 「迁移编号重排方案」先行落地为独立一步：`0013_model_cooldown` 单独落，验证现网 `schema_migrations`
   仍为连续前缀，再考虑 `0014`/`0015`。
5. `de6d2d44`（#611）：L3，依赖第 4 步的 `0014`/`0015`。
6. `7cbd2e67`（#609）→ `f091528b`（#616）→ `abc6483d`（#619）：L3 链，必须整体设计、按序落地。
   `#616` 之后才评估上游开放 PR `#620`（WS `stream` 布尔兼容）。

## 结论

- 上游 `main` 整体合并不成立：`merge-tree` 预演 103 个冲突路径，且 `0010`–`0012` 号码语义完全对撞。
- 新增 #606–#619 里 L1 只有 1 个、L2 有 2 个、L3 有 6 个、已完成 1 个。
- L2 两项（`#607` rerank、`#617` 身份头修复）不含迁移、不碰单凭据与入口调度合同，是当前性价比最高的移植目标。
- `#606`、`#608` 的硬前置分别落在已判「不纳入范围」的专题 3 和专题 5 上，本次不单独移植。
- `#611` 自带迁移，是唯一必须先完成「迁移编号重排方案」才能动的 L3。
- `#609` → `#616` → `#619` 是一条不可拆的依赖链，且 `#616` 是本次范围里体量与风险都最高的提交。
- 本文只做评估与方案，不对上述上游提交执行任何代码 cherry-pick。
