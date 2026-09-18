# 设计方案:模型路由条目(Model Route Entries)

> **文档地位:本特性开发的最高依据。**
> 实现过程中的任何技术决策与本文件冲突时,以本文件为准;需要偏离时,必须先修订本文件。
> 配套实施清单见 `docs/design/model-route-entries-checklist.md`。

- 分支:`feature/model-route-entries`
- 状态:设计定稿,未开始实现
- 日期:2026-09-05

---

## 1. 背景与需求

### 1.1 需求陈述

客户端请求某个**对外模型名**(如 `A`),网关按**优先级或比例**将流量分配到多个
`(密钥, 上游模型)` 目标,包括:

- **同一把密钥**下的多个上游模型(如 K1 上的 A、B、C);
- **不同的密钥**上的其他上游模型(如 K2 的 B、K3 的 D)。

### 1.2 验收基准场景(全篇贯穿,验收以本表为准)

| 分组 | 密钥 | 对外模型 | 转发到上游模型 | 条目权重 | 优先级 | 组权重 | 期望长期占比 |
|---|---|---|---|---|---|---|---|
| 分组1(openai) | K1 | A | A | 30 | 1 | 60 | ≈18% |
| 分组1(openai) | K1 | A | B | 50 | 1 | 60 | ≈30% |
| 分组1(openai) | K1 | A | C | 20 | 2(兜底) | 60 | 仅兜底层 |
| 分组2(claude) | K2 | A | B | 默认 | 1 | 30 | ≈30% |
| 分组3(gemini) | K3 | A | D | 默认 | 1 | 10 | ≈10% |

语义:优先级 1 层内按组合权重加权随机,总占比 = 组权重 × 条目权重 × 密钥权重(归一化);
P1 层全部候选耗尽或失败后,才降级到 P2 兜底层。

### 1.3 现状结论(设计输入,已核实)

已具备:

1. **模型别名**:分组模型条目 `ModelConfig{ID(上游名), Alias(对外名)}`
   (`internal/state/snapshot.go:61`),请求按外部名命中后改写为上游 ID。
2. **跨分组多目标**:同一外部模型可挂多个分组,编译索引
   `(协议, 操作, 外部模型) → []RouteTarget{GroupID, UpstreamModelID}`(snapshot.go:354)。
3. **两级权重**:生效权重 = 组权重 × 密钥权重,加权随机
   (`internal/scheduler/inspect.go:244` `effectiveWeight`)。
4. **失败重试链**:`Iterator.Next()` 逐候选尝试,支持整组跳过
   (`internal/scheduler/scheduler.go`)。
5. **路由巡检接口**:`internal/control/route_inspect.go`,按协议+外部模型+访问密钥
   返回候选分组、上游模型、密钥可用性与原因码。

缺口(即本设计要补齐的):

- **G1** 同一分组内,一个对外模型只能映射到一个上游模型
  (alias 与条目 1:1;且调度器候选池 `targetsByMode` 以 GroupID 为键
  (scheduler.go:216),同组多目标会互相覆盖)。
- **G2** 没有"模型条目"级权重,无法表达组内 A→A/B/C 的比例分发。
- **G3** 没有"优先级"语义,无法表达"先用 X,不可用才降级到 Y"的分层。

### 1.4 方案选型记录

- **方案一(本设计,已选定)**:分组模型条目升级为"多映射 + 权重/优先级"。
  顺着现有 alias、多分组候选、两级权重的结构自然生长,不引入新的顶层实体。
- 方案二(否决):全局路由规则表实体。能钉死"密钥↔模型"关系,但破坏
  "密钥归属分组、分组承载策略"的层级,存储/快照/调度/API/UI 全动,
  一致性校验负担重。其中"密钥级模型权限"列为二期可选增量(见 §9)。
- 方案三(否决):纯配置 workaround(同一把 key 复制进多个分组)。
  仅作为上线前的临时验证手段,不作为产品能力。

---

## 2. 核心概念与术语

| 术语 | 代码/存储 | 含义 |
|---|---|---|
| 对外模型名 External Model | 条目 `alias`(留空则取 `id`) | 客户端请求体里的 `model` 值 |
| 上游模型 Upstream Model | 条目 `id` | 实际转发给上游的模型名(请求体改写目标) |
| **路由条目 Route Entry** | 分组 `Models` 数组的一个元素 | 本设计的核心单元:`(上游模型, 对外名, 权重, 优先级)` |
| 条目权重 entry weight | 条目 `weight` | 同组同对外名的条目之间的分流比例 |
| 优先级 priority | 条目 `priority` | 分层:数字越小越优先,低优先级层仅在高优先级层耗尽后启用 |
| 组合权重 combined weight | 运行时计算 | `组权重 × 条目权重 × 密钥权重`(三者任一 ≤ 0 则该三元组不参与) |

不变式(实现必须保证):

- **I1** 密钥仍归属唯一分组,模型权限仍是分组级:组内任意可用密钥可服务组内任意条目;
- **I2** 未配置新字段的存量数据,行为与现状完全一致(兼容性验收线);
- **I3** 对外名留空的条目,对外名 = 上游模型名(现有语义不变);
- **I4** 优先级先于权重:不同优先级层之间永远先小数字层,层内才比较权重;
- **I5** 条目级熔断是运行态(内存),不落库,重启清零(与现有密钥熔断同规则)。

---

## 3. 数据模型(存储层)

分组 `Models` 为 JSON 列(`internal/storage/models/group.go`,无 schema 迁移),
条目在现有 `{id, alias}` 基础上扩展:

```jsonc
// 分组1 的 Models(验收基准场景)
[
  { "id": "A", "alias": "A", "weight": 30, "priority": 1 },
  { "id": "B", "alias": "A", "weight": 50, "priority": 1 },
  { "id": "C", "alias": "A", "weight": 20, "priority": 2 },
  { "id": "gpt-4o-mini", "alias": "" }   // 对外名=上游名,单条目,默认权重
]
```

字段语义与默认值:

| 字段 | 类型 | 缺省 | 约束 |
|---|---|---|---|
| `weight` | `*int`(nil 表示未设置) | nil ⇒ 按 1 参与 | ≥ 0;`0` = 保留条目但不参与分流;负数拒绝;上限沿用现有手动权重上限(`state.MaxWeight`) |
| `priority` | `*int` | nil ⇒ 1 | ≥ 1;数字越小越优先;0 或负数拒绝 |

**校验规则**(管理面保存与快照编译两处都执行,编译处为最终防线):

- V1 同一分组内 `(对外名, 上游模型)` 组合唯一;
- V2 同一分组内,某对外名下所有条目权重之和必须 > 0(否则该对外模型完全不可路由,拒绝保存);
- V3 `id` 非空(现有规则保持);
- V4 `alias_enabled` 语义不变:alias 留空 = 对外名取 `id`。

---

## 4. 配置编译层(快照)

`internal/state/snapshot.go`:

1. `RouteTarget` 扩展字段:

```go
type RouteTarget struct {
    GroupID         uint
    UpstreamModelID string
    Mode            channel.RouteMode
    ResolvedTarget  channel.ResolvedTarget
    EntryWeight     int   // 条目权重,nil 归一为 1;0 保留但不参与
    Priority        int   // 条目优先级,nil 归一为 1
}
```

2. `appendExecutionTargets`(snapshot.go:302)按条目逐条 append:同一分组的同一
   对外名产生**多个** RouteTarget(每条目一个)。
3. 索引排序 `sortExecutionRouteIndex` 调整为 `(Priority, GroupID, UpstreamModelID)`
   升序,保证快照确定性。
4. 编译期执行 V1/V2 校验,违规返回编译错误(现有"模型配置是分组进入数据面调度的
   统一门槛"原则不变,snapshot.go:315)。
5. `NoModelRouteKey` 的资源类操作(Responses retrieve/delete 等)不涉及条目,
   行为不变。

---

## 5. 调度器

`internal/scheduler/`:

### 5.1 候选单元从 `(分组, 密钥)` 扩展为 `(分组, 条目, 密钥)` 三元组

- `candidatePool` 的 `targetsByMode` 从 `map[RouteMode]map[uint]candidateTarget`
  (以 GroupID 为键)改为支持同分组多 target 的结构(如键为
  `(GroupID, UpstreamModelID)` 或分组下 target 切片),`groupIDsByMode` 同步去重。
- 加权随机在**三元组**上进行,组合权重 =
  `组权重 × 条目权重 × 密钥权重`,替代现 `effectiveWeight`(inspect.go:244)
  的两因子乘积;任一因子 ≤ 0 时该三元组剔除。
- 数学性质:某上游模型的边缘占比与该组密钥数量无关,严格正比于条目权重 × 组权重
  (乘积结构保证)。

### 5.2 优先级分层调度

- 无合格亲和绑定时,`Iterator` 构建后按 `Priority` 分层,`Next()` 在**当前最高可用优先级层**
  内加权随机;合格绑定的跨层首试优先适用 §5.3。
- 层内候选耗尽(全部 tried / 冷却 / 拉黑)→ 降到下一优先级层;所有层耗尽 →
  返回现有 `ErrExhausted` 语义,配 `staticReason` 原因码(新增:条目权重为 0、
  条目熔断等,见 §6.3)。
- 未设置 priority 的存量条目全部落在 P1 层,与现状行为一致(I2)。

### 5.3 重试与去重

- `tried` 集合从"密钥维度"扩展为 `(密钥, 上游模型)` 维度:同一密钥换一个
  条目(上游模型)允许再试,同一 `(密钥, 上游模型)` 失败后不再重试。
- `SkipGroup` 整组跳过语义不变。
- 会话亲和(`PreferredCredentialID`)是合格候选中的**首试绑定**:它不扩大候选集合,
  但在所有通过资格过滤的优先级、路由模式和响应存储 bucket 中先尝试绑定目标。
  - 首试仍须通过 `evaluateTargets`、access-key filter、group enable/affinity enable、
    route requirement、entry weight、credential identity、cooldown/blacklist 等全部硬资格过滤;
    在所有合格 priority tier、route mode 和 regular/store-downgraded bucket 中查找绑定凭据。
  - 亲和目标被资格过滤排除时保留 durable row,当前请求可由其他合格候选正常服务,
    但其成功不能覆盖旧绑定。过滤、准备阶段错误、本地执行或下游失败不算 provider failure。
  - 对仍合格的绑定目标,只有实际 provider attempt 发生可重试的上游失败后,
    才按既有重试/重放规则 fallback;fallback 成功后才可迁移到实际成功的
    `(GroupID, CredentialID, IdentityGeneration)`。同目标成功可刷新绑定,无绑定时成功可建立绑定。
  - gateway 在单进程 singleton Handler 内用同 key 条纹锁协调 Cache CAS 与 durable upsert;
    不承诺跨进程写入排序。绑定粒度沿用 `PreferredCredentialID`,不增加 affinity EntryID。
  - 持久权威是现有 `system_settings` 的 `_internal.affinity.binding.<raw-hmac>` 行,
    值只含目标 ID 与身份代际,没有 TTL,不新增 schema migration。热缓存 miss、TTL 到期、
    LRU/capacity 淘汰或进程重启后由 gateway read-through 恢复,合格绑定仍为 `hit`;
    恢复需要同一数据库与键派生材料,不依赖停机 checkpoint。
  - 无关 group/entry weight、catalog/Models.dev 更新和 snapshot revision 变化不删除绑定。
    revision-only 不清热缓存,capacity/TTL 变化至多清热缓存;关闭分组亲和也保留 durable row。
  - `cache_miss` 是适用查找路径未找到绑定,生产启用路径包括 durable lookup;
    `hit` 包括 durable 恢复。`cache_unavailable` 同时涵盖本地 cache/config/key 不可用及
    store lookup/decode error;后者在 provider dispatch 前 fail-closed,不普通 fallback。
    all-disabled 候选跳过 store lookup,继续普通调度;memory-only 测试保留本地边界。
    provider 成功后的 upsert error 保留已交付响应与热绑定,记录 `affinity_binding_persist_failed`,
    不触发新 attempt;该失败写入不保证缓存丢失或重启后的恢复。
  - 这是与 `previous_response_id` **continuation ownership** 独立的另一种机制:
    续接请求把候选收窄到唯一归属凭据(`AllowedCredentialIDs` 单元素),亲和始终保留
    完整候选集合兜底。两者不得合并。
- 跨候选故障转移的前提是「尚未向客户端释放任何内容」,强制 buffered 的三个生成流
  (OpenAI Chat Completions、OpenAI Responses `create`、Anthropic Messages) 由
  `ResponsesReplayEligible` 判定请求是否引用了上游状态;Gemini 与 OpenAI Images 是
  实时例外,不进入该重放窗口,也不具备 buffered retry:
  - 引用上游状态的字段(`previous_response_id`、`conversation`、`prompt.id`、
    `input`/`tools` 中的 provider resource 引用)阻断重放;
  - 缓存与呈现提示(`prompt_cache_key`、`prompt_cache_retention`、
    `prompt_cache_options`、`reasoning`、`service_tier`)不引用上游状态,
    保持可重放——换候选只损失一次缓存命中。
- 组合后的完整链条:**第 1 轮建会话 → 第 2 轮亲和命中;第 3 轮绑定目标实际失败且
  未释放内容 → 换候选成功 → 亲和指针迁移;第 4 轮亲和命中新目标。**

---

## 6. 运行态(内存,不落库)

`internal/state/registry.go` 及失败处理链路:

### 6.1 状态分维度

- **密钥维度**(现有,保持):401/配额等凭据级失败 → 密钥冷却/拉黑,组内所有条目
  共享该密钥的可用性。
- **条目维度**(新增):模型级错误(404 model not found、渠道明确返回的
  无权限/无该模型)→ `(GroupID, UpstreamModelID)` 条目进入冷却/拉黑;
  该密钥的其他条目、其他分组的同名上游模型不受影响。

### 6.2 参数与恢复

- 阈值沿用分组 `BlacklistThreshold` 与现有冷却时长配置;条目维度与密钥维度
  各自独立计数。
- 运行态字段仅存在于内存 registry(I5),管理面可读(用于巡检展示),不可写;
  恢复路径与现有密钥熔断一致(冷却到期自动恢复 + 手动恢复接口沿用)。

### 6.3 原因码扩展

`scheduler/inspect.go` 新增:`entry_blacklisted`、`entry_cooldown`、
`entry_weight_zero`、`tier_demoted`(降级层在响应中标注)等,命名遵循现有
`ReasonCode` 风格。

---

## 7. 请求转发与改写

- 请求体 `model` 改写继续消费 `Selection.UpstreamModelID`,机制不变;
  本设计只改变"UpstreamModelID 的候选来源"。
- 会话亲和按 §5.3 的 durable binding、跨合格 bucket 首试和失败后迁移规则执行;
  Responses passthrough(NoModelRouteKey)、流式改写等既有链路继续由回归测试覆盖。

---

## 8. 管理面与可观测

### 8.1 管理 API(`internal/control/`)

- `group_models.go` 更新接口的条目 DTO 增加 `weight`、`priority`(指针语义,
  接受旧报文缺省字段);校验执行 §3 V1–V4。
- 幂等、审计等现有机制不动。

### 8.2 路由巡检(`internal/control/route_inspect.go`)

- 响应从"分组 → 密钥"两级扩展为"分组 → **条目行** → 密钥":
  每条目行含 `upstream_model`、`entry_weight`、`priority`、
  `effective_share`(该条目在当前查询下归一化后的**预期占比**,含兜底标注)、
  条目可用性与原因码;条目行下再列密钥可用性(现有结构)。
- `effective_share` 计算:在"当前快照 + 当前访问密钥 + 当前可用性"约束下,
  按第 5.1 节组合权重对 P1 层归一化;兜底层单独标注,不并入 P1 占比。

### 8.3 模型列表(`/v1/models` 与目录)

- 对外模型名跨条目、跨分组**去重**后输出(`home.go`、`group_options.go` 等
  以 alias 取名的路径统一收口为一个 helper)。

### 8.4 请求日志与用量

- 请求日志同时记录**外部模型名**与**上游模型名**(呈现为 `A → B`);
- 用量/计价继续按上游模型名匹配模型价格表(键为上游名,天然兼容)。

### 8.5 前端(`web/src/`)

**模型 Tab(`features/groups/models/GroupModelsTab.vue` + `ModelAliasEditor`)升级为路由表:**

- 表格新增列:权重(数字输入 + 组内占比实时预览)、优先级(默认 1,≥2 标"兜底"徽标);
- 按**对外名**分组折叠展示,组头显示条目占比分布条;
- 行内实时校验:重复 `(对外名, 上游模型)`、该对外名权重合计为 0、权重为 0 的
  条目置灰显示"已停用分流";
- "从上游同步"(`GroupModelSyncDialog`)行为不变,新条目默认
  `alias 空、weight 缺省、priority 缺省`(即完全保持现状行为);
- 分组列表页"模型数"列语义调整为 `对外模型数 / 路由条目数`。

**路由巡检页(`features/monitor/InspectorTab.vue`)**:按 8.2 的条目行结构展示,
含占比与原因码。

**密钥 Tab、访问密钥**:交互不变(密钥导入/测试/熔断恢复、`Filters.Groups`
分组过滤)。

**i18n**:zh-CN 全量新增词条;en / ja 同步翻译。

---

## 9. 明确不做(本期边界)

1. **严格短周期比例**(如每 10 次精确 3:5:2):本期为加权随机,仅保证长期期望;
   平滑加权轮询(SWRR)列为后续增强。
2. **跨分组优先级**:组间本期仍是权重;若将来需要"分组 1 整体不可用才走分组 2",
   在组级加 priority 字段,复用 §5.2 分层机制。
3. **密钥级模型权限**(钉死某把密钥只走某些上游模型):本期不做;作为二期把
   模型列表下放到密钥级 allowlist 的增量,本设计的数据结构不阻塞它。
4. 条目维度配置不下放到访问密钥;访问密钥仅有分组过滤(现状)。

---

## 10. 兼容性承诺(验收线)

- C1 旧版本保存的模型配置(无 weight/priority 字段)经升级后,路由、巡检、
  `/v1/models`、日志行为与升级前逐项一致;
- C2 旧版管理 API 报文(缺新字段)可继续保存成功,等价于 weight/priority 缺省;
- C3 单条目模型(无别名或 1:1 别名)的路由决策路径与现状一致,现有 golden
  路由测试(`internal/channel/route_golden_test.go` 等)除新增字段断言外不改写;
- C4 快照 revision、热更新机制不变;配置保存后无需重启即生效。

---

## 11. 验收方式

以 §1.2 基准场景搭建三个分组(可用 mock 渠道),逐项验证:

1. **比例**:P1 层四条目按 ≈18/30/30/10 长期分布(大样本统计,容差 ±3%);
2. **优先级**:对分组1 的 A、B 上游注入持续失败 → 流量滑入 C(P2),其余条目
   占比按 P1 剩余权重重新归一;恢复后回到原比例;
3. **条目熔断**:对上游 B 注入模型级错误,达到阈值后巡检页该条目显示
   `entry_cooldown`,其他条目不受影响;
4. **密钥熔断**:对 K1 注入凭据级错误,分组1 全部条目不可用,流量滑向分组2/3;
5. **兼容**:删除全部 weight/priority 字段后重复 1–4(应与升级前行为一致);
6. **巡检一致性**:巡检页显示的占比与实际流量分布一致;原因码与注入的故障类型
   一一对应;
7. **日志**:`A → B` 双模型名落日志;用量按上游模型计价正确。
8. **亲和 + 故障转移链**:同一会话前缀连续四轮请求——第 1 轮落到分组 A;第 2 轮
   亲和命中 A;第 3 轮 A 实际 attempt 失败且未释放内容 → 换到分组 B 并成功;
   第 4 轮亲和命中 B。同时验证目标未通过资格过滤时正常候选可服务,但不删除或覆盖旧 durable row。
9. **恢复与无关发布**:覆盖 TTL/LRU/capacity 热缓存丢失、同数据库重启后的 read-through hit,
   以及无关 entry weight、catalog-like snapshot republish 后仍首试原目标。
   现有 catalog-like 回归仅复现本地 Models.dev 风格 revision 发布,
   未调用真实 Models.dev 网络客户端或 `control/catalog_sync` 生产同步链路。
