# Main Ahead 提交评估（#606–#619）

- 最新提交（评估基线）：`4474b214950da6c70eb00348df3e26edfb363152`（`fix(codex): 升级 CPA 并固定上游版本声明 (#619)`）
- 生成时间：`2026-09-11T06:40:00+08:00`（本文件自身的提交哈希不写入本文，避免自引用）

## 基线

- 分析分支：`feat/responses-continuation-websocket`（本轮把 `#609` → `#616` → `#619` 落在该分支，**尚未合入 `dev`**）
- 当前基线：`4474b214950da6c70eb00348df3e26edfb363152`
- 对比目标：`upstream/main@abc6483d0ed081e305859d88eba41c81b6efd1d8`（`fix(codex): 升级 CPA 并固定上游版本声明 (#619)`）
- 本地 `main`、`origin/main`、`upstream/main` 三者已对齐（同为 `abc6483d`），`main` 是上游镜像。
- `git rev-list --left-right --count HEAD...upstream/main`：`99 24`
  - 本分支独有 99 个提交（本轮合入前为 95 个，即 `dev@08bcba82`）
  - 上游独有 24 个提交
- 合并基点：`0ddc41d8b718c0281b1ba2f5bdfe5b23622621ce`
- `git merge-tree --write-tree HEAD upstream/main` 预演：**116 个冲突路径**（本轮合入前 108 个）
  - `web/src` 28、`internal/control` 21、`internal/gateway` 15、`internal/state` 11、
    `internal/execution` 9、`third_party/cpaembedded` 7、`internal/scheduler` 6
  - 比合入前多 8 条，原因是本轮引入的文件本来就是上游同名文件的整体重写
    （`internal/gateway/websocket*.go`、`internal/execution/wsnative/*`、`internal/state/response_bindings.go`），
    以及 `scheduler`/`snapshot`/`runtime_settings` 的**双侧改动**。这是适配移植的必然结果，不是回归。
- 评估范围：上游 `#606`–`#619` 共 10 个提交。到本轮为止，范围内只剩 `#606`、`#608` 两个挂起项，
  其余 8 个已结清（`#607`/`#617` 上一轮合入，`#609`/`#616`/`#619` 本轮合入，`#611`/`#614` 放弃）。
- 上游仍有一个未合并的开放 PR 未计入本文范围：`#620` `fix(gateway): 兼容 WS 请求中的 stream 布尔参数`
  （分支 `tbphp/fix-responses-websocket-stream@76176c58`，base `main`）。`#616` 已落地，可以开始评估。

## 当前合同

L1/L2/L3 判断以当前合同为边界：

- 每个分组只有一个上游凭据；普通第二凭据创建、导入、连接必须原子拒绝。
- 权重、优先级、熔断只属于模型 Route Entry；不恢复分组权重或凭据权重。
- 调度中心负责入口级路由、候选排序和运行态；不恢复旧 scheduler、Registry 或前端调度 API。
- 请求重试预算是系统级单一来源（`snapshot.Settings.RetryCount`），分组 `retry_count` 已退役。
- `(group_id, entry_id)` 只用于后端定位和 mutation，不进入用户可见的调度文案或筛选维度。
- 监控筛选的权威查询状态是 `range` preset；`from_ms`/`to_ms` 查询合同（上游 `#594`）未引入。
- 迁移台账只允许追加，编号必须与注册表位置连续（见「迁移编号重排方案」）；`dev` 的 `0010`–`0012`
  永久占住上游同名号码段。
- 存储层只支持 SQLite 与 PostgreSQL；MySQL 支持已随 `cef04938` 移除，`gormmysql` 与
  `mysqlRequiresCheckDropSyntax0003` 已从 `internal/storage` 删除。
- 交付面只覆盖 Linux 与 Docker；Windows service、`securefile` Windows ACL、Windows 打包不在范围内。
- Protocol、operation、external model 和 target config 必须保持当前 URL、API、目标冻结和候选 fallback 语义。
- 无按模型冷却运行态：不引入上游 `#599` 的 credential+model cooldown，也没有
  `internal/execution/model_cooldown.go` 这类只在该模型下成立的判定入口。Codex 的
  `usage_limit_reached` 因此在 bridge 里固定按**凭据范围**上报。
- `previous_response_id` 续接由运行态归属表管辖：归属按 access key 隔离，命中后候选收窄到原凭据身份
  （`GroupID` + `IdentityGeneration` 都要匹配），未知或跨 access key 的 ID 在派发前拒绝；
  归属可在正常关闭时随运行态检查点恢复，检查点只恢复归属，不复制路由、权限与健康判断。
- Responses WebSocket 是**同一端口**上的 `GET /v1/responses` 升级：准入走 channel 声明的
  `ResponsesWebsocket` 能力位，逐轮重新做权限、额度与路由判定；没有 HTTP fallback，没有业务请求重放，
  没有隐式跨凭据迁移。开关是 `responses_websocket_enabled`（系统 + 分组，分组覆盖系统，默认开启）。
- Codex 身份头固定：`User-Agent` 与 `Version` 不能被下游或分组规则覆盖/清空，`Version` 固定为
  `codexClientVersion`；`Originator` 仍按显式规则（这条合同由上轮 `#617` 的「显式规则优先」被本轮
  `#619` 取代，见下）。
- CPA 固定 `v7.2.157`（根模块与 `third_party/cpaembedded` 两处 `go.mod` 一致）。

## 本轮已合入

### `7cbd2e67` `feat(responses): 实现响应状态续接路由 (#609)` → 本地 `c58d0297`

上游内容：38 文件，+1226/-85。

- 新增 `internal/gateway/responses_continuation.go`（响应归属观察者）、`internal/state/response_bindings.go`
  （有界内存归属索引 + 检查点捕获/恢复）。
- 把 `previous_response_id` 从「不可重放字段」升级为独立路由要求：新增
  `execution.ResponsesStorePreferenceRequireStored`、`AttemptSpec.ResponsesStorePreference` 校验、
  `dialect.RequestMetadata.PreviousResponseID`、`request_fields.go` 的重复字段拒绝。
- 响应下发前登记归属（HTTP 单次响应在 `OnResponse` 回调，SSE 在 `response.created` 之后、交付之前）。
- 参数覆盖禁止改写 `previous_response_id`（管理面保存时拒绝，运行期也校验）。

本地适配（37 文件，+1235/-84）：

1. 删除上游对 `internal/app/scheduling_checkpoint_test.go` 的改动：该文件由未移植的 `#591` 新增，
   `dev` 不存在；检查点文档只加 `Responses` 字段，不引入 `Scheduling`。
2. `dev` 的 `scheduler.Query` 没有 `AllowedCredentialRefs`（该字段来自 `#591` 的加权轮询）。
   绑定命中时用 `query.AllowedCredentialIDs` 收窄到单一凭据，并在 `Handler.Handle` 里用
   `registry.CredentialRef` 显式比对 `GroupID`/`IdentityGeneration`；身份不符时给出空候选集，
   与上游「找不到候选」的拒绝语义一致（503），不做静默换身份续接。
3. 成功路径保留 `dev` 的 `recordCredentialSuccess(selection.CredentialID, …)` 与 `recordEntrySuccess(…)`，
   只给 `recordAffinitySuccess` 加上 `originalMetadata.PreviousResponseID == ""` 守卫。
4. 转发路径保留 `dev` 的 `capture.beginForward` / `normalizeAndFinishForwardOwned`，
   `input.OnResponse` 冲突登记块插在该调用之后。
5. 测试按 `dev` 合同重写：夹具保持「一分组一凭据」（两个分组各一个凭据），随机值用
   `useAffinityRandomValues` 固定；模型冷却用例改用 Route Entry 冷却
   （`SetEntryCooldownForEntry`/`RecoverEntryForEntry`）；凭据禁用改用 `SetBlacklisted`；
   删除依赖 `#591` 调度序号的断言。
6. README 三语采用上游新增的续接说明段（去掉「Codex WS 未接入」的旧句子）。
7. 本地补证 `98371143`：检查点旧文件缺 `Responses` 字段与未来文件多未知字段都必须被容忍。

证据：`go build ./...` 通过；`go test ./... -count=1` 全绿；`gofmt -l internal` 无输出。

### `f091528b` `feat(gateway): 接入原生 Responses WebSocket 与逐轮治理 (#616)` → 本地 `1b90e572`

上游内容：55 文件，+4719/-152。

- 新增执行层 WS 传输：`internal/execution/{websocket,provideradapter/websocket}.go`、
  `internal/execution/wsnative/session.go`、`internal/execution/bifrost/websocket.go`、
  `internal/execution/cpa/websocket.go`。
- 新增网关数据面：`internal/gateway/websocket.go`（连接、预算、能力与设置判定）、
  `websocket_turn.go`（逐轮治理）。
- channel `ResponsesWebsocket` 能力位、`responses_websocket_enabled` 分层开关、前端分组设置与三语文案。

本地适配（60 文件，+4979/-193）：

1. `scheduler.Query` 同样没有 `AllowedCredentialRefs`：WS 逐轮候选用 `AllowedCredentialIDs` 收窄，
   再用 `registry.CredentialRef` 比对身份；不符按 `configuration_changed` 拒绝并关连接。
2. 没有 `Iterator.ChargeReplay`（同属 `#591`）：刷新重放直接复用已选定的 selection，不经过
   `iterator.Next()`，因此不需要重放名额记账；`scheduler.New` 补上 `dev` 的 `handler.newRandom()`。
3. `state.CredentialRef` 没有 `EncryptedProxy`/`ProxyFingerprint`（凭据级代理已从运行态移除）：
   刷新分支只比对身份，代理与请求头变化仍由绑定哈希检查拦住。
4. 记账改用 `dev` 的签名：`recordCredentialSuccess(ref.ID)`、`recordEntrySuccess`、
   `applyGroupDecisionEffectForEntry(selection.EntryID)`。
5. `internal/execution/cpa/websocket.go`：`validateSpec` 在 `dev` 只返回 2 值；
   `WSSessionOptions` 没有 `BaseURL`（`dev` 的 Codex 固定官方端点）；环境代理按合同**拒绝**
   （不再把 `FromEnvironment` 解析成具体 URL）；`DispatchState` 直接用 `execution.DispatchState`
   字符串，不引入上游的 `WSNotSent`/`WSMaybeSent`。
6. 关闭 `#612` 遗留的日志缺口（`dev` 合同第 6 节）：新增 `codex.NewLogHook()`，在 `runtime.go` 里
   与 `redact.NewHook` 相邻且**先于**通用脱敏注册，移除 CPA 在通知 lifecycle 之前写出的断连正文；
   **不在** vendored 包用 `init()` 注册全局 hook（上游放在 `codex_websocket_log.go` 的 `init()` 里）。
7. vendored README 不重新加入上游的 Codex WebSocket Session 段：`dev` 的该合同由上一轮的 PR #28
   收敛到 `docs/design/codex-websocket-session.md`，本轮按数据面接入把该文档重写（含本地差异、
   日志缺口关闭、错误映射与证据清单）。README 三语里指向 vendored 段落的链接改指该文档。
8. 测试按 `dev` 合同重写：夹具一律「一分组一凭据」；WS 断言用固定随机值；
   `websocket_review_test.go` 的模型冷却用例改为 `dev` 的既有语义（已下发的额度错误停在
   `safety.committed`，既不冷却凭据也不冷却 Route Entry），并删除依赖 `#591` 调度记账的断言；
   channel 能力位用例对固定官方端点的 Codex/Grok 用空 params。
9. 未引入上游对 `internal/execution/model_cooldown.go`、`internal/control/model_cooldown.go`
   与 `internal/scheduler/fair.go` 的改动（`#599`/`#591`，`dev` 不存在）。

证据：两个 module `go build ./...` 通过；`go vet` 无新增告警；根模块 `go test ./... -count=1` 全绿；
`third_party/cpaembedded` 的 `go test ./...` 通过；web 的 `type-check`、`lint`、`format` 三项通过；
`go mod tidy` 无变化；`gofmt -l internal` 无输出。

### `abc6483d` `fix(codex): 升级 CPA 并固定上游版本声明 (#619)` → 本地 `4474b214`

上游内容：19 文件，+410/-89。

- CPA `v7.2.151` → `v7.2.157`；根模块与 `third_party/cpaembedded` 的 `go.mod`/`go.sum`、
  `THIRD_PARTY_NOTICES.md` 同步。
- **身份头合同变更**：`User-Agent` 与 `Version` 改为固定，下游与分组规则不能覆盖、清空或删除；
  `Version` 固定到 `codexClientVersion`（`0.153.3`）；`Originator` 仍按显式规则（含空值与删除）；
  模型与账号观测请求用同一版本。**这一条取代了上一轮 `#617` 的「身份头覆盖合同」**。
- 两个 Codex executor 显式打开 CPA 的 `ModelLevelCooling`。
- 新增 `model_at_capacity`/`model_is_at_capacity` 分类与「可重试 `server_error`」证据；
  普通 `server_error` 不再获得安全重放证据；WS 的容量错误码不触发额度冷却。
- 新增 `internal/execution/cpa/codex_compatibility_test.go`。

本地适配（20 文件，+422/-94）：

1. `third_party/cpaembedded/go.mod` 只改 CPA 版本，不引入上游新增的直接 `logrus` 依赖
   （`dev` 的 vendored 包没有直接引用它），`logrus` 继续留在 indirect 段。
2. vendored README 不重新加入 Codex WebSocket Session 段（同 `#616`）；`ModelLevelCooling`
   段落按 `dev` 现实改写：`dev` 没有模型级冷却运行态，bridge 仍按凭据范围上报 `usage_limit_reached`。
3. `internal/execution/cpa/codex_provider.go`：上游把 `usage_limit_reached` 标为
   `ErrorScopeModel`，`dev` 保持 `ErrorScopeCredential`（否则该错误在 `dev` 无处生效，
   额度耗尽会变成不可处置），并补注释说明改这里必须同时改冷却模型。
4. `codex_compatibility_test.go`：两个额度用例的期望改为 `dev` 的既有判定
   （`scope=credential`、`effect=cooldown_credential`），容量与 `server_error` 用例保持上游期望。
5. vendored `codex_headers_test.go`：保留 `dev` 的转发 round tripper 夹具（`dev` 没有 `BaseURL`
   通道），只采用上游新的期望值与新增的 image 2.5 用例；测试改名为
   `TestCodexHTTPFixedIdentityOnImagesAndWire`。
6. `internal/execution/cpa/websocket_test.go` 的 `DispatchState` 断言改用 `execution.DispatchState`
   字符串；`codex/websocket_log_test.go` 按 pinned CPA 157 的真实日志形状
   （`session=… auth=… url=… reason=… err=…`）收窄用例，证明只剥离 `err=` 之后的正文。

证据：两个 module `go build ./...` 通过；`go vet` 无新增告警；根模块 `go test ./... -count=1` 全绿；
`third_party/cpaembedded` 的 `go test ./...` 通过；web 的 `type-check`、`lint`、`format` 通过；
`go list -m github.com/router-for-me/CLIProxyAPI/v7` 在根模块与 vendored 模块同为 `v7.2.157`；
`gofmt -l internal runtime.go third_party/cpaembedded/embedded` 无输出。

## 上一轮已合入

### `fe4b6ac1` `feat(gateway): 接入原生文本重排序协议 (#607)` → `dev@23ea2bd1`

- 按上游原文 cherry-pick，落库 56 个文件。
- 丢弃上游对 `internal/execution/model_cooldown.go` 的改动（`#599` 的死代码，`dev` 不存在）。
- 失败判定沿用 `dev` 的 `internal/health/execution_judge.go`，没有引入按模型冷却分类。

### `0c9d1888` `fix(codex): 修正请求身份头覆盖与会话兼容 (#617)` → `dev@fe080bf9`

- 把 Codex 身份头的「显式配置」与「默认注入」分开，`execution.AttemptSpec` 增加
  `ConfiguredHeaders []string`，由 `input.Group.HeaderRules.ConfiguredNames()` 填充。
- vendored README 只补入新增的身份头段落，**不**把上游的 Codex WebSocket Session 段搬回来：
  该契约收敛到 `docs/design/codex-websocket-session.md`。
- `embedded/codex_headers_test.go` 用转发到测试上游的 round tripper 取代 `request.BaseURL`
  （`dev` 无该通道），保留真实 `http.Transport`。
- **已被本轮 `#619` 取代**：身份头从「显式规则优先」改为「`User-Agent`/`Version` 固定、
  只有 `Originator` 仍可显式配置」。`ConfiguredHeaders` 字段本身仍然存在并被 HTTP/WS 两条路径使用。

## 本轮放弃

### `eecef9f4` `docs(readme): 添加 OfoxAI 赞助展示 (#614)`

- 与功能、可靠性、运维合同无关，且需要手工对齐 `dev` 自己的自建实例与发布章节。
  零代码不等于零维护，赞助位会随每次 README 冲突反复出现。

### `de6d2d44` `feat(access-keys): 支持自定义密钥并统一编辑与生成交互 (#611)`

- 这是范围内唯一必须动迁移编号和 `access_keys` 校验约束的提交：引入后要同时承担
  `0014`/`0015` 两次表重建、幂等与操作恢复路径、以及脱敏前缀合同的长期维护。
- `dev` 的访问密钥页已有自己的分发/交接流程，上游组件不能整目录覆盖。

## 迁移编号重排方案

> 现状：`#611` 已放弃，当前没有需要新迁移的上游提交，因此下面的映射**没有待执行的消费者**。
> 保留本节是因为 `dev` 的 `0010`–`0012` 永久占住上游同名号码段：任何未来要从上游移植迁移的工作
> 都必须从这里开始，不能直接复制上游文件，也不能重排已应用的编号。

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
与 `dev` 的 `0010`/`0011`/`0012` 表集合无交集，可独立先落。

### 依赖前提（已核对，成立）

- `dev` 的 `request_log_attempts` 自 `0006` 之后没有任何迁移再触碰它；`0006` 的
  `CONSTRAINT chk_request_log_attempt_effect CHECK (effect IN ('','none','cooldown_credential','record_credential_failure','skip_group'))`
  与上游 `0010` 要替换的 `old` 字符串逐字相同。
- `dev` 的 `access_keys.key_suffix` 仍是上游 `0011` 的前置形态（`char(4)` + 四位十六进制校验），
  `key_prefix` 列在 `dev` 不存在，`0014`/`0015` 的 `HasColumn` 幂等判断成立。
- 反例：所有 `dev` 侧前置条件只对 **SQLite/PostgreSQL** 成立，MySQL 路径必须整段删除而不是保留。

### 移植步骤

1. 复制上游三个文件并改名、改 `const IDxxxx`、改全部 `Up/Validate/ValidateRecoverable` 与私有常量后缀。
2. 删除 MySQL 分支：`gormmysql` 导入、`mysqlRequiresCheckDropSyntax0003` 调用、
   `DROP CHECK`/`information_schema.check_constraints` 查询，只保留 SQLite 与 PostgreSQL 路径。
3. 在 `internal/storage/migration.go` 的 `migrations` 切片末尾按 `0013`→`0014`→`0015` 追加三项。
4. 同步 `internal/storage/migration_test.go` 的期望 ID 列表。
5. 若同批移植 `#611` 的 `internal/storage/migration_sqlite.go`，需先确认 `dev` 现有迁移链是否已在该
   路径上处理同一问题；`#611` 已放弃，该文件当前不在范围内。
6. 决定性验证：现网实例的 `schema_migrations` 已有 `0001`–`0012` 共 12 行，重排后的注册表必须
   前 12 项 ID 逐字不变；用真实数据卷跑一次启动，确认只新增 3 行且不触发 `unknown or non-contiguous`。
7. 平台验证：`0013` 的 `rebuildModelCooldownSQLite0010` 与 `0014` 的 `rebuildAccessKeysSQLite0011`
   都是「读 `sqlite_master` 原文 → 字符串替换 → 重建表 → 恢复索引」，必须在 SQLite 上实跑一次
   （含索引恢复与外键校验），不能只靠 PostgreSQL 通过。

## L1 / L2 / L3 汇总

| 上游提交 | PR | 主题 | 级别 | 状态 / 硬前置 |
| --- | --- | --- | --- | --- |
| `fe4b6ac1` | #607 | 原生文本重排序协议 | 已合入 | `dev@23ea2bd1` |
| `0c9d1888` | #617 | 请求身份头覆盖与会话兼容 | 已合入 | `dev@fe080bf9`；身份头部分被 `#619` 取代 |
| `eecef9f4` | #614 | README 赞助展示 | 放弃 | — |
| `de6d2d44` | #611 | 自定义访问密钥与编辑生成交互 | 放弃 | — |
| `7cbd2e67` | #609 | 响应状态续接路由 | 已合入 | `c58d0297` |
| `f091528b` | #616 | 原生 Responses WebSocket 与逐轮治理 | 已合入 | `1b90e572`（依赖 `#609`） |
| `abc6483d` | #619 | 升级 CPA 并固定上游版本声明 | 已合入 | `4474b214`（依赖 `#616`） |
| `fb18dc9e` | #606 | 精简模型冷却展示与管理接口 | L3 | `#599`（`dev` 不存在） |
| `d417b7de` | #608 | 统一日志与用量时间筛选交互 | L3 | `#594`（`dev` 现为 `range` preset） |
| `7d80a981` | #612 | Codex 独立 WebSocket Session 封装 | 已完成 | — |

## 前置结论（沿用，不重开）

- 专题 1（凭据全量操作与单凭据导入，`e888fe60`+`96d3e0d5`）、专题 2（入口公平调度，`9cb3f986`）、
  专题 3（模型错误重试与健康恢复，`e0bfa07e`+`d4699dd2`）、专题 4（自定义订阅上游，`6da82242`）、
  专题 5（Usage 时间窗口，`33fb54bf`+`175949e2`）保持「不纳入范围」。
- 专题 6（全局请求重试预算，`a4255546`）已随 `dev` 的 `3274e32c` 同步完成；保留两处本地差异：
  `retryAttemptLimit(retryCount) = max(retryCount, 1)`（尝试总次数，默认 5，上游为 `retryCount + 1`）
  与 `fallback.missing_evidence_retry` replay 许可（`internal/health/execution_judge.go`）。
- 专题 7（Codex WebSocket）已全部落地：合同在 `docs/design/codex-websocket-session.md`，
  数据面在本轮的 `#616`，CPA 与身份头在 `#619`。

## L3（剩余）

### `fb18dc9e` `refactor(cooldown): 精简模型冷却展示与管理接口 (#606)`

- 30 文件，+207/-323，`internal/control/model_cooldown.go`、`runtime_observation.go`、
  `web/src/components/ui/ModelCooldownDetails.vue`、健康页与凭据页展示、三语文案。
- 硬前置：修改 `internal/control/model_cooldown.go`，该文件由 `#599` 新增，`dev` 不存在；
  属专题 3，已判「不纳入范围」。单独移植 `#606` 无从落地。

### `d417b7de` `feat(monitor): 统一日志与用量的时间筛选交互 (#608)`

- 14 个前端文件，+666/-620。
- 硬前置：依赖 `#594` 的 `from_ms`/`to_ms` 查询合同（上游 `usage-filters.ts` 是
  `preset` + `from_ms`/`to_ms` 双轨）；`dev` 现合同是 `range` preset，后端也没有这两个查询参数。
  属专题 5，已判「不纳入范围」。只移植前端会直接对不上后端查询。

## 已完成（PR #28）

### `7d80a981` `feat(codex): 添加独立上游 WebSocket Session 封装 (#612)` → `dev@2994c96e`

- vendored `CodexWSSession` + `internal/subscription/providers/codex.WSSession`。
- 当时的边界（「不注册到数据面、不在网关请求路径上被调用」）已由本轮 `#616` 取代；
  `docs/design/codex-websocket-session.md` 已按数据面接入重写。

## 执行顺序

1. ~~`fe4b6ac1`（#607）~~：已合入 `dev@23ea2bd1`。
2. ~~`0c9d1888`（#617）~~：已合入 `dev@fe080bf9`；身份头部分由 `#619` 取代。
3. ~~`#611`（迁移编号重排）~~：已放弃，`0013`–`0015` 暂无消费者。
4. ~~`7cbd2e67`（#609）~~：已合入 `c58d0297`。
5. ~~`f091528b`（#616）~~：已合入 `1b90e572`；~~`abc6483d`（#619）~~：已合入 `4474b214`。
6. 下一步候选：评估上游开放 PR `#620`（WS `stream` 布尔兼容）。
7. `fb18dc9e`（#606）与 `d417b7de`（#608）继续挂起：硬前置落在已判「不纳入范围」的专题 3 和专题 5 上。

## 结论

- 上游 `main` 整体合并不成立：`merge-tree` 预演 116 个冲突路径，且 `0010`–`0012` 号码语义完全对撞。
- 到本轮为止，范围内 10 个提交已结清 8 个：`#607`/`#617` 上一轮合入，`#609`/`#616`/`#619` 本轮合入，
  `#611`/`#614` 放弃；只剩 `#606`、`#608` 两个挂起项，硬前置都在不纳入范围的专题里。
- 本轮把 Codex WebSocket 从「只有一个未被调用的显式句柄」推进到完整数据面：合同、执行层、
  网关逐轮治理、能力位与分层开关、CPA 版本与身份头全部落地，并关闭了 `#612` 记录的日志缺口。
- `#619` 覆盖了 `#617` 的身份头语义：Codex 的 `User-Agent`/`Version` 现在由固定版本声明统一，
  不接受下游或分组规则改写；`Originator` 仍按显式规则。这是上游的最终状态，不是本地取舍。
- 本地与上游的实质差异集中在四处，且都有代码注释或设计文档兜底：
  续接绑定用凭据 ID + 身份比对而不是 `AllowedCredentialRefs`；WS 刷新重放不记账；
  `usage_limit_reached` 按凭据范围上报；模型级冷却运行态整体不存在。
- 「迁移编号重排方案」保留为常驻约束：`dev` 的 `0010`–`0012` 永久占住上游同名号码段，
  未来任何上游迁移移植都必须从 `0013` 起追加。
