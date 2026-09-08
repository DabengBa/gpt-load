# 设计方案:模型调度中心面板与候选级熔断参数

- 分支:`feature/model-route-entries`
- 状态:**设计草案 v2(已吸收评审修订),待确认后作为下一轮交付的最高依据**
- 日期:2026-09-06
- 前置:构建于《模型路由条目》特性之上(`docs/design/model-route-entries.md`);吸收 AIRelayHub 两条经验(跨供应商同别名调度中心化、候选级熔断参数显式化)
- v1 → v2 修订对照:见文末附录 A(评审 8 项 → 修订章节映射)

---

## 1. 需求与目标

### 1.1 需求陈述

同一对外模型名分布在多个分组时,权重/优先级分散在各分组的"模型与别名"Tab,无法在
一个视图里查看与调整跨分组候选的相对占比、运行态与熔断;条目级熔断沿用分组级参数,
所有条目一刀切。

### 1.2 目标

- **G1 中心调度面板**:按对外模型名聚合跨分组全部候选条目,同一视图内查看/编辑每条
  候选的权重、优先级、熔断参数、实时占比与运行态,支持一键恢复。
- **G2 候选级熔断参数显式化**:每个**路由条目**(以 §2 的 `entry_id` 为身份)可独立
  配置拉黑阈值与模型级冷却时长;未配置时行为与现状逐字节一致。
- **G3 全链路一致**:面板、路由检查页、真实流量三者共享同一占比口径与运行态。

### 1.3 明确不做

1. 不引入中心路由表实体/投影对账:条目所有权仍在分组(存储为分组 Models JSON 列),
   中心面板是"聚合视图 + 服务端字段级编辑"。
2. 不新增组级优先级:组间仍只按权重;条目级优先级保持"同一对外名下跨分组统一分层"
   的现状语义。
3. 不做模型名模式匹配;不做 SWRR;不做密钥级模型权限。
4. 条目熔断参数仅作用于**模型级失败链路**(条目冷却/计数/拉黑);凭据级失败链路
   (401 等,冷却/拉黑整把密钥)仍走分组/系统级参数。

---

## 2. 路由条目身份(基础定义,先行解决)

### 2.1 问题

现行运行态键为 `(GroupID, UpstreamModelID)`。但 V1 只约束 `(对外名, 上游模型)` 唯一,
**同一分组内同一上游模型可以不同 alias 出现多次**:

```json
[ { "id": "provider.model1", "alias": "A" },
  { "id": "provider.model1", "alias": "B" } ]
```

此时 (GroupID, UpstreamModelID) 无法区分 A/B:A 的失败会连带冷却 B,两者无法各自配置
熔断阈值,PATCH 亦无法指明更新对象。**这是身份定义冲突,必须在任何熔断/面板能力之前
解决。**

### 2.2 决策:引入稳定 `entry_id`(采纳评审推荐)

1. **存储**:每条目新增可选字段 `entry_id`(字符串)。由服务端在**任何创建路径**
   (手动添加、上游同步、分组创建/导入)生成,并在**读取路径懒回填**(GET 分组模型时
   为缺失 entry_id 的条目分配并持久化);格式 `e` + 12 位 hex,分组内唯一(V7)。
2. **快照编译容忍缺失**:为不破坏 C1(任意一次配置发布都是全局快照,不能因某个从未
   打开过的分组缺少 entry_id 而失败),编译期对缺失 entry_id 的条目使用**确定性派生
   身份** `derived:<external_model>#<upstream_model>` 参与本进程调度;派生身份不写出
   快照(管理 API 中 entry_id 为空即"未回填")。
3. **运行态键变更**:`EntryRuntimeKey` 由 `(GroupID, UpstreamModelID)` 改为
   `(GroupID, EntryID)`。运行态仅内存(I5),重启清零,无迁移问题。效果:
   - 同组同上游不同 alias(A/B)获得**独立**的冷却/计数/拉黑状态;
   - alias 改名:entry_id 不变(运行态保留);
   - 修改上游模型(`id`):该行保留 entry_id(身份是条目而非上游名),运行态不重置;
   - 未回填条目使用派生身份,alias 改名会重置运行态(一次性,回填后消失)。
4. **传播**:RouteTarget / Selection / 路由检查响应行 / 调度面板行均携带 `entry_id`;
   PATCH 与恢复请求以 `(group_id, entry_id)` 定位条目。
5. **重试去重保持不变**:调度器 `tried` 仍为 `(credentialID, upstreamModelID)`——
   同一上游模型 + 同一密钥就是同一个重试目标,与 alias 无关。这是**有意保留**的语义:
   重试维度 ≠ 健康维度。

### 2.3 校验

- **V7**:entry_id 若存在,必须匹配 `^e[0-9a-f]{12}$` 且分组内唯一;创建路径未提供时
  由服务端生成;懒回填与 V7 在 `internal/control/group_models.go` 与
  `internal/state/external_model.go`(校验器)两处生效。

---

## 3. 数据模型(存储层,无 schema 迁移)

### 3.1 条目扩展

```jsonc
{
  "entry_id": "e3f9c2a1b7d4",
  "id": "gemini-flash",
  "alias": "A",
  "weight": 20,
  "priority": 2,
  "circuit_breaker": {              // 整个对象可选
    "blacklist_threshold": 5,       // 可选;显式配置后该条目模型级失败开始计数
    "cooldown_seconds": 300         // 可选;覆盖判定器默认 1h;0 = 不冷却仅计数
  }
}
```

### 3.2 字段语义与继承

| 参数 | 条目显式配置 | 缺省(未配置) |
|---|---|---|
| `blacklist_threshold` | 模型级失败计数,达阈值拉黑(手动恢复) | **不计数、不拉黑**(现状;不继承分组 BlacklistThreshold,保 C1) |
| `cooldown_seconds` | 条目冷却 = 该值;`0` = 不冷却、仅计数 | 判定器默认 1h(现状) |

两个字段**彼此独立**:可只配阈值、只配冷却,或都配(响应三元组见 §5.3)。

### 3.3 校验(扩展 V1–V4)

- **V5** `blacklist_threshold` 为 nil 或 ≥1 整数;
- **V6** `cooldown_seconds` 为 nil 或 ≥0 整数;
- **V7** entry_id(§2.3);
- 管理面保存与快照编译两处执行;编译期校验失败仍拒绝发布。

### 3.4 涉及文件

`internal/control/group_models.go`(groupModelEntry 解码 + GET 懒回填 entry_id +
V5–V7)、`internal/control/group_update.go` / `group_create.go` / `group_write.go`
(请求/响应 DTO 贯通,见 §5.4)、`internal/state/external_model.go`(校验器)、
`internal/state/snapshot.go`(ModelConfig/RouteTarget/GroupView 增加 entry_id 与
`ModelBreakerByEntry map[uint]map[string]*EntryCircuitBreaker`)、
`internal/state/loader/loader.go`、`internal/storage/models/group.go`(无需迁移)。

---

## 4. 运行时(失败处理链接入)

### 4.1 计数/冷却触发规则(独立于 Decision.Effect 定义)

**评审修正(阻断项 1)**:不再从 `Effect` 推导熔断动作(判定器对典型 404 产出的是
`EffectCooldownCredential` 而非 `EffectRecordCredentialFailure`,按 Effect 推导会导致
"阈值配置了却永远不拉黑")。改为按**失败特征**定义:

> **模型级失败(计入条目熔断)**:最终 `Decision` 满足
> `Category == FailureCategoryModelUnavailable` 且 `Scope == ErrorScopeModel`
> 且 `RuleID != "safety.replay_unknown"`。
>
> 涵盖判定器全部上游模型失败分支:`model.unavailable`(EffectCooldownCredential)、
> `images.model_unavailable` / `embeddings.model_unavailable`(重放约束操作,
> EffectNone)、`candidate.unavailable`(供应商显式候选不可用,EffectNone)。
> 唯一排除 `safety.replay_unknown`(重放安全性未知,是否到达上游存在歧义:不计入、
> 也不冷却——维持现状)。下游取消、请求级、凭据级、主机级等其余分类天然不满足条件。

**生效方式**(handler 侧,判定器契约不动):

```
applyGroupDecisionEffectForEntry(scope=model 分支):
  breaker := selection.Group.ModelBreakerByEntry[groupID][entryID]
  counted := 满足上述模型级失败定义
  if counted:
    1. 计数/拉黑:if breaker?.BlacklistThreshold != nil:
         count := IncrEntryFailureForModel(groupID, entryID)
         if count >= *threshold: SetEntryBlacklistedForModel(groupID, entryID)
    2. 冷却:if breaker?.CooldownSeconds != nil:
         if *secs > 0: SetEntryCooldownForModel(groupID, entryID, now + *secs)
         if *secs == 0: 不调用 SetEntryCooldown(评审修正 6:避免遗留非零
            until 被巡检回显;历史遗留冷却自然到期,不主动清除)
       else(未配置冷却):
         沿用 decision 自带冷却(现状:model.unavailable → 1h;
         candidate.unavailable / images 等本无冷却,维持无冷却)
  if !counted: 现状行为逐字节不变(C1)
```

注:显式配置熔断参数后,`candidate.unavailable` 与 images/embeddings 分支**开始**
计数与冷却——这是有意的语义统一("配置了熔断的条目,所有上游模型级失败一视同仁");
未配置条目不受影响。

### 4.2 成功路径补全

`recordCredentialSuccess` 处对称补 `ClearEntryFailureForModel(groupID, entryID)`
(选中候选条目在非流式 2xx 与流式 CleanEOF 两个成功点清计数),与凭据侧
`ClearFailure` 对称。

### 4.3 运行态键与文件

- `EntryRuntimeKey` → `(GroupID, EntryID)`(registry.go,相关签名同步调整);
- `internal/gateway/handler.go` + `model_entry_breaker_test.go`(new):覆盖 §4.1
  全分支 + §4.2 + C1 一致性;
- 巡检响应 `entry_cooldown_until_ms` 仅在 `state == cooldown`(未来时间)时返回
  (评审修正 6:消除"可用但回显过去冷却时间"的歧义)。

---

## 5. 管理 API

新增 `internal/control/model_route_schedule.go`(+测试),管理员鉴权;复用
`InspectRoute` 的协议/操作解析与巡检引擎(含 `InspectWithEntryRuntime`)。

### 5.1 概览

```
GET /api/model-route/schedule
→ { items: [ { external_model, candidate_count, group_count,
              has_fallback, cooled_candidates, blacklisted_candidates } ] }
```

来源:当前快照 `ExecutionRouteCatalog` 跨分组聚合,对外名口径与 `/v1/models` 去重
一致。

### 5.2 详情(评审修正 4:补齐协议/操作上下文)

```
GET /api/model-route/schedule/detail
    ?external_model=A&protocol=openai-completions&operation=chat_completion
    &access_key_id=10
→ { external_model, protocol, operation, route_strategy, access_key: {...},
    groups: [ { group_id, group_name, channel_id, group_weight,
      entries: [ { entry_id, model_id, alias, weight, priority,
        circuit_breaker: { configured, effective, sources },   // §5.3
        runtime: { state, cooldown_until_ms|null, failure_count },
        included, routable, reason_code, effective_share,     // §6 口径
        credentials: [...] } ] } ] }
```

- `protocol` 必填;`operation` 可选,缺省按该协议默认操作解析(与路由检查页同逻辑);
  同一 alias 在不同协议/操作下候选不同,详情**按协议/操作上下文返回真实运行候选**,
  不做无上下文的"配置聚合"。
- 实现直接复用 `InspectRoute` 内核(entry runtime 注入),在行上补 `entry_id` 与
  `circuit_breaker`(读快照 `ModelBreakerByEntry`)。

### 5.3 熔断参数响应三元组(评审修正 7)

每条目返回:

```json
"circuit_breaker": {
  "configured": { "blacklist_threshold": 2, "cooldown_seconds": null },
  "effective":  { "blacklist_threshold": 2, "cooldown_seconds": 3600 },
  "sources":    { "blacklist_threshold": "entry", "cooldown_seconds": "default" }
}
```

`configured` 为条目显式配置(null=未配置);`effective` 为解析后实际生效值;
`sources` 逐参数标注 `entry | default`。可完整表达"只配阈值、冷却继承"等部分覆盖。

### 5.4 分组模型 API 的完整 round-trip(评审修正 3,阻断项)

`UpdateGroupModels` 是全量替换,若不处理,面板写入的 `circuit_breaker` 会被分组页
一次普通保存**静默擦除**。因此:

1. `GroupModel`(写请求)、`groupModelRequestWire`、`GroupModelResponse`(读响应)、
   `groupModelEntry`(存储解码)全部贯通 `entry_id` + `circuit_breaker`;
2. 请求 DTO 采用**三态字段**(§5.5):缺省=不变、`null`=清除、数值=设置;分组页
   前端草稿(model-draft)**原样携带并回传** `entry_id` 与 `circuit_breaker`
   (即使分组页本期不提供熔断编辑 UI,也不得丢字段);
3. 响应 DTO 回显 `configured/effective/sources`(分组页可只展示 configured 原样值);
4. 回归验收(§8.8):面板配置熔断 → 分组页仅改权重保存 → 熔断配置逐字节保留;
5. 创建/同步/导入路径:新条目由服务端生成 entry_id;同步条目 circuit_breaker 为空。

### 5.5 批量更新(评审修正 8:事务式提交;评审修正 7:三态字段)

```
PATCH /api/model-route/schedule
{ snapshot_revision, updates: [ { group_id, entry_id,
    weight?: <三态>, priority?: <三态>,
    circuit_breaker?: { blacklist_threshold?: <三态>, cooldown_seconds?: <三态> }
                      | null } ] }
→ { snapshot_revision_new, detail: <5.2 响应> }
```

- **三态**语义:字段缺席 = 不变;`null` = 清除覆盖(回退继承);数值 = 设置。
  `circuit_breaker: null` = 清除该条目全部熔断覆盖;`circuit_breaker: {}` = 熔断
  不变。
- **事务式**:①载入涉及分组的当前 Models;②服务端逐条做字段级合并(按
  `(group_id, entry_id)` 定位,不存在即该条失败);③全部条目通过 V1–V7 后,
  **单个数据库事务**内更新所有受影响分组行并恰好发布**一次**快照;任一校验失败则
  整体拒绝,**不产生任何写入**。跨组相对权重不会出现"半套生效"的中间态。
- **乐观并发**:请求携带 `snapshot_revision`,与服务端当前不一致时拒绝(409),
  前端刷新后重试——避免两个管理员相互覆盖。
- 服务端字段级合并意味着面板**不提交完整 Models 列表**,消除客户端读改写竞态
  (评审修正 3 的配套要求)。

### 5.6 恢复

```
POST /api/model-route/schedule/recover
{ group_id, entry_id }
→ registry.RecoverEntryForModel(三清);不影响凭据运行态。
```

---

## 6. 占比口径(评审修正 5:当前生效层归一化)

`applyEffectiveShares` 的归一化口径从"固定 P1"改为**当前生效层**:

1. `activeTier := min{ p | 该层为 included+routable 且层内可用凭据组合权重 > 0 }`;
2. 仅 `activeTier` 内的行输出 `effective_share = 组合权重 / 层内总组合权重`;
3. 其余层(含兜底层)输出 0 并保留 `fallback` 标注;无任何层有可用权重时全 0。

该口径与调度器"层序升序、取首个非空桶"的真实流量选择一致:当 P1 全部冷却时,
P2 显示归一化到 100%,而非固定 0。这是对**既有路由检查响应语义的有意变更**(行为
变更,非兼容破坏),同步更新既有巡检断言并新增"P1 全冷却 → P2 = 100%"用例。
调度面板、路由检查页与真实流量共享该口径(G3)。

---

## 6A. 前端(中心调度面板)

### 6A.1 定位:路由检查页升级为"调度面板"

不新建独立页面。现有**监控 → 路由检查**骨架(访问密钥/协议选择、条目行、占比、
凭据明细)原地升级为读写;另在其上新增"调度面板"入口:

1. **索引视图**(§5.1):对外模型名列表(候选数/分组数/异常计数),点击进详情;
2. **详情视图**(§5.2):分组分节 + 候选行编辑:
   - 权重(占比实时预览)、优先级、熔断参数(占位符显示 effective 默认值;留空 =
     不变;显式清空按钮 = 清除覆盖回继承);
   - 行内校验(0–100、≥1、≥0)与**事务式保存条**(整体成功/失败,409 时提示刷新);
   - 运行态徽标(冷却中/已拉黑/连续失败 N)+ 恢复按钮(§5.6);
   - 协议/操作选择沿用路由检查页形态(operation 按协议给默认值,可切换);
3. 路由检查页保持只读巡检语义;两页共享资源层
   (`route-inspection.ts` + 新增 `model-route-schedule.ts`)。

### 6A.2 涉及文件

`web/src/app/resources/model-route-schedule.ts`(new)、
`web/src/features/monitor/`(调度面板组件,new;`InspectorTab.vue` 仅加入口与共享
工具)、分组模型页 `model-draft.ts` 贯通 `entry_id/circuit_breaker` 原样回传(§5.4)、
i18n 三语全量;验证 `pnpm build` + `pnpm lint` + 浏览器走查(含 §8.8 回归)。

---

## 7. 兼容性承诺

- **C1** 未配置 `circuit_breaker` 的条目运行态行为与现状逐字节一致(不计数;冷却走
  判定默认);`safety.replay_unknown` 维持不计不冷;
- **C2** 旧管理报文(缺 entry_id/circuit_breaker)保存成功(GET 懒回填 entry_id;
  缺失熔断字段 = 不变);
- **C3** 单条目/无别名路径不变;
- **C4** 快照 revision/热更新不变;面板保存后即时生效;
- **C5(新增,评审修正 2)** 同组同上游不同 alias 的条目运行态彼此独立;alias 改名
  不重置运行态(entry_id 身份);未回填条目的派生身份在 alias 改名后会重置运行态
  (一次性,回填后消失);
- 回滚 = 丢弃代码;存量 JSON 中新增字段被旧版本读取时忽略(旧解码容忍未知键)。

---

## 8. 验收方式

以基准场景(分组1 A/B/C + 分组2/3)为准:

1. **聚合**:面板详情展示 5 条候选,分组/条目权重与配置一致;每行携带 entry_id;
2. **占比与降级口径**:正常态占比与路由检查、真实流量一致(±3pp);将分组1 全部
   P1 条目注入模型级失败后,详情中 C(P2)行 `effective_share` 归一化为 1.0
   (§6 行为变更的验证);
3. **跨分组编辑**:面板改分组2 权重 100→50,保存(单事务)后巡检占比即时重归一化;
4. **A/B 同上游身份隔离**:同组配置 `provider.model1` 两条目 alias A/B,A 配阈值
   2、B 不配;连续两次 A 的模型级失败 → A 拉黑,B 状态不变(冷却/计数均独立);
5. **显式冷却**:`cooldown_seconds=60` + 一次模型级失败 → 该条目 `entry_cooldown`
   ≈60s,同组其他条目不受影响;
6. **零冷却**:`cooldown_seconds=0` + 模型级失败 → 无 `entry_cooldown` 回显、计数
   +1;达阈值仍拉黑;
7. **replay_unknown 不计**:`safety.replay_unknown` 决策不计数、不冷却;
8. **round-trip 保持**:面板配置熔断 → 分组页仅改权重/别名保存 → 熔断配置逐字节
   保留(阻断项 3 回归);
9. **PATCH 语义**:`null`/缺席/数值三态与 `circuit_breaker: null` 清除均按 §5.5;
   校验失败时整体拒绝且零写入;`snapshot_revision` 不符返回 409;
10. **兼容**:删除全部新字段重复 1–9(与升级前一致);旧报文保存成功;
11. 全量 `go test ./...` 无新增失败(对照基线);前端 build/lint 通过。

---

## 9. 实施步骤建议(每步独立可评审)

| 步骤 | 内容 | 依赖 |
|---|---|---|
| S1 | 身份与存储:entry_id 生成/懒回填/V7、breaker 字段全链路(JSON、DTO 三态、V5–V7、ModelConfig/RouteTarget/GroupView 索引)+ 单测 | — |
| S2 | 运行时:EntryRuntimeKey 改造、§4.1 计数/冷却规则、§4.2 成功清零、巡检冷却回显收紧 + 单测(C1 一致性用例必含) | S1 |
| S3 | 占比口径:§6 当前生效层归一化(含既有断言更新)+ 单测 | 独立,可并行 |
| S4 | 管理 API:§5.2 详情(protocol/operation)、§5.5 事务式三态 PATCH、§5.6 恢复 + 集成测试 | S1–S3 |
| S5 | 前端:§6A 调度面板 + 分组页 round-trip 贯通 + 三语 i18n + build/lint | S4 |
| S6 | 全量回归 + §8 验收走查 + 部署(沿用 fork 构建 → compose 切换流程) | S5 |

---

## 附录 A:评审修订对照(v1 → v2)

| 评审问题 | 类别 | 修订位置 |
|---|---|---|
| 1. 阈值按 Effect 推导对 404 不生效 | 阻断 | §4.1:熔断触发改为独立失败特征定义(Category+Scope,排除 replay_unknown),不再借用 Effect |
| 2. (GroupID, UpstreamModelID) 身份歧义(A/B 同上游) | 阻断 | §2:引入服务端生成 entry_id(懒回填+派生身份),运行态/接口/快照索引全部切换;tried 去重保持 upstream 维度(有意) |
| 3. 分组页全量保存静默擦除熔断配置 | 阻断 | §5.4:请求/响应/存储解码全链路贯通 + 三态保留;§5.5 面板改为服务端字段级合并,不提交完整列表;§8.8 回归 |
| 4. 详情缺 protocol/operation | 重要 | §5.2:必填 protocol + 可选 operation,复用 InspectRoute 内核,返回真实运行候选 |
| 5. 占比固定 P1 归一化,降级时失真 | 重要 | §6:当前生效层(最小有可用权重层)归一化;P1 全冷却时 P2=100%;巡检语义有意变更 |
| 6. cooldown_seconds=0 语义未闭合 | 重要 | §4.1:0 时不调用 SetEntryCooldown;巡检仅在未来冷却时回显 until;§8.6 验收 |
| 7. 部分熔断覆盖/三态不可表达 | 重要 | §5.3:configured/effective/sources 三元组;§5.5:字段级三态 PATCH(null=清除、缺席=不变) |
| 8. 跨分组批量非原子 | 重要 | §5.5:预校验全部 → 单事务更新 + 单次快照发布,失败零写入;snapshot_revision 乐观并发 |
