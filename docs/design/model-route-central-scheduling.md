# 设计方案:模型调度中心面板与候选级熔断参数

- 分支:`feature/model-route-entries`
- 状态:**设计草案 v1,待确认后作为下一轮交付的最高依据**
- 日期:2026-09-06
- 前置:本方案构建于已交付的《模型路由条目》特性之上(`docs/design/model-route-entries.md`),吸收 AIRelayHub 的两条已验证经验:①跨供应商同别名的调度参数中心化管理;②候选级熔断参数显式化
- 配套实施清单:交付启动后按本方案另立 checklist

---

## 1. 需求与目标

### 1.1 需求陈述

当前模型路由条目的权重/优先级分散在各分组的"模型与别名"Tab 中。当同一个对外模型名
分布在多个分组时,运营者需要逐个分组打开页面调整,无法在一个视图里看到同一别名的全部
候选(跨分组)的相对占比、运行态和熔断情况,也无法统一调整。

同时,条目级熔断完全沿用分组级参数(BlacklistThreshold/冷却时长),所有条目一刀切:
高频付费渠道和廉价兜底渠道的熔断阈值、冷却时间没有区分。

### 1.2 目标

- **G1 中心调度面板**:提供一个跨分组的"模型调度面板",按对外模型名聚合全部候选条目
  (跨分组),支持在同一视图内查看与编辑每条候选的权重、优先级、实时占比预览、运行态
  (冷却/拉黑/连续失败),并支持一键恢复。
- **G2 候选级熔断参数显式化**:每个路由条目可独立配置熔断参数(拉黑阈值、模型级冷却
  时长);未配置时保持现状(继承分组/系统默认),存量行为不变。
- **G3 全链路一致**:面板写入的数据经同一套 V1–V4 校验与快照编译生效;巡检页
  (路由检查)与调度面板共享同一份运行态与占比口径。

### 1.3 明确不做(边界)

1. **不改变数据归属**:条目的所有权仍在分组(存储仍是分组 Models JSON 列)。中心面板
   是"聚合视图 + 跨分组批量编辑",不引入中心路由表实体,不做规则投影(AIRelayHub 的
   投影/对账复杂度对本项目无必要——我们的条目天然归属分组,无跨实体漂移问题)。
2. **不新增组级优先级**:组与组之间仍只按权重竞争。条目级优先级保持现状语义
   (同一对外名的全部条目跨分组统一分层)。原设计 §9 "跨分组优先级(组级 priority
   字段)"仍不做。
3. 不做模型名的模式匹配(exact/prefix/regex)——入口仍是精确对外名。
4. 不做 SWRR、密钥级模型权限(维持原 §9 边界)。
5. 条目熔断参数仅覆盖**模型级失败链路**(条目冷却与条目拉黑);凭据级失败链路
   (401 等,冷却/拉黑整把密钥)仍走分组/系统级参数,不下放条目。

---

## 2. 数据模型(存储层,无 schema 迁移)

### 2.1 条目扩展

分组 `Models` JSON 数组的元素在 `{id, alias, weight, priority}` 基础上扩展:

```jsonc
{
  "id": "gemini-flash",          // 上游模型
  "alias": "A",                  // 对外名
  "weight": 20,
  "priority": 2,
  "circuit_breaker": {           // 全部可选;缺省 = 继承(见 §2.2)
    "blacklist_threshold": 5,    // 连续模型级失败达到该值 → 条目拉黑(手动恢复)
    "cooldown_seconds": 300      // 模型级失败的条目冷却时长(覆盖判定器默认 1h)
  }
}
```

| 字段 | 类型 | 缺省 | 约束 |
|---|---|---|---|
| `circuit_breaker.blacklist_threshold` | `*int` | 缺省 = **不计数、不拉黑**(现状) | ≥1;显式配置后该条目的模型级失败开始计数 |
| `circuit_breaker.cooldown_seconds` | `*int` | 缺省 = 判定器默认(1h) | ≥0;0 表示模型级失败不冷却(仅计数) |

### 2.2 继承与默认值解析(优先级:条目 > 分组 > 系统)

| 参数 | 条目显式配置 | 条目缺省时 |
|---|---|---|
| 拉黑阈值 | 条目值 | **不启用计数**(保持现状;不继承分组 BlacklistThreshold,避免存量分组行为变化——C1) |
| 模型级冷却时长 | 条目值 | 判定器默认 1h(现状) |

> 设计取舍说明:AIRelayHub 的熔断参数是"目标级显式必填",我们选择"可选+继承现状"。
> 原因:gpt-load 的兼容承诺 C1(存量行为不变)优先级更高;显式配置是增量能力。

### 2.3 校验规则(V1–V4 之上扩展)

- **V5** `blacklist_threshold` 为 nil 或 ≥1 的整数;
- **V6** `cooldown_seconds` 为 nil 或 ≥0 的整数;
- 违反 V5/V6 与 V1–V4 同样在管理面保存与快照编译两处拒绝。

### 2.4 涉及文件

- `internal/storage/models/group.go` — 无改动(JSON 透传);
- `internal/control/group_models.go` — `groupModelEntry` 解码扩展 + 保存路径 V5/V6;
- `internal/control/group_update.go` / `group_create.go` — V5/V6 贯通;
- `internal/state/external_model.go` — 校验器扩展(管理面与编译共用);
- `internal/state/snapshot.go` — `ModelConfig` 增加 `CircuitBreaker *EntryCircuitBreaker`;
  `GroupView` 增加只读索引 `ModelBreakerByUpstream map[string]*EntryCircuitBreaker`
  (编译期构建,供 handler O(1) 查询;nil 值字段不建条目);
- `internal/state/loader/loader.go` — 存储行读写贯通。

---

## 3. 运行时(失败处理链接入)

### 3.1 模型级失败链路(现状 → 目标)

现状:`judgeUpstreamResult` 对模型级错误产出 `EffectCooldownCredential + 1h`
(execution_judge.go `model.unavailable` 硬编码),handler 的
`applyGroupDecisionEffectForEntry` 将其写入条目冷却;条目失败计数/拉黑路径在生产中
不可达(判定器从不为模型级错误产出 `EffectRecordCredentialFailure`)。

目标(全部在 handler 侧完成,**不改判定器契约**):

```
applyGroupDecisionEffectForEntry(scope=model 分支):
  breaker := selection.Group.ModelBreakerByUpstream[upstreamModelID]   // §2.4 索引
  1. 冷却:if breaker != nil && breaker.CooldownSeconds != nil:
         until = attemptNow + *breaker.CooldownSeconds        // 显式覆盖判定器默认
     (nil → 沿用 decision.CooldownUntil,即 1h)
     SetEntryCooldownForModel(groupID, upstreamModelID, until)
  2. 计数/拉黑:if breaker != nil && breaker.BlacklistThreshold != nil:
         count := IncrEntryFailureForModel(groupID, upstreamModelID)
         if count >= *breaker.BlacklistThreshold:
             SetEntryBlacklistedForModel(groupID, upstreamModelID)
             记录条目拉黑事件日志
     (nil → 不计数,保持现状)
```

### 3.2 成功路径补全(现状缺口)

`recordCredentialSuccess` 目前只清凭据失败计数。补:同请求成功的候选条目调用
`ClearEntryFailureForModel(groupID, upstreamModelID)`,使已计数未达阈值的条目在恢复后
归零(与凭据侧 `ClearFailure` 对称)。非流式 2xx 与流式 CleanEOF 两个成功点都要覆盖。

### 3.3 涉及文件

- `internal/gateway/handler.go` — §3.1/§3.2;
- `internal/gateway/handler_test.go` / 新增 `internal/gateway/model_entry_breaker_test.go` —
  覆盖:显式冷却覆盖判定默认;阈值触达拉黑;计数未达阈值被成功清零;未配置条目行为与
  现状逐字节一致(C1);`Responses NoModelRouteKey` 不受影响。
- 不改 `internal/health/*`(判定器契约与单测不动)。

---

## 4. 管理 API(中心调度面板后端)

新增文件 `internal/control/model_route_schedule.go`(+测试),全部走既有管理员鉴权。

### 4.1 概览:对外模型名索引

```
GET /api/model-route/schedule
→ { items: [ { external_model, candidate_count, group_count,
              has_fallback, disabled_candidates, cooled_candidates } ] }
```

数据源:当前快照 `ExecutionRouteCatalog` 跨分组聚合(与 `/v1/models` 去重口径一致,
复用 `state.ExternalModelName` helper)。

### 4.2 详情:单个对外模型的候选列表

```
GET /api/model-route/schedule/detail?external_model=A&access_key_id=10
→ { external_model, route_strategy, groups: [
     { group_id, group_name, channel_id, group_weight,
       entries: [ { model_id, alias, weight, priority,
                    circuit_breaker: {blacklist_threshold, cooldown_seconds} | null,
                    breaker_source: "entry" | "default",
                    runtime: { state: available|cooldown|blacklisted,
                               cooldown_until_ms, failure_count },
                    included, routable, reason_code,
                    effective_share, credentials: [...] } ] } ] }
```

实现:复用调度器 `InspectWithEntryRuntime`(巡检同源,占比口径 §一致),
`circuit_breaker` 从快照 `GroupView` 读配置,`runtime` 从 registry
`EntryRuntimeSnapshot` 读运行态。`access_key_id` 必填(占比预览的归一化口径依赖
访问密钥的分组过滤),与路由检查页保持一致。

### 4.3 批量写:跨分组条目更新

```
PATCH /api/model-route/schedule
{ updates: [ { group_id, model_id,
               weight?, priority?,
               circuit_breaker?: {blacklist_threshold?, cooldown_seconds?} | null } ],
  access_key_id }          // 回显占比用
→ { results: [ { group_id, model_id, ok, error? } ], detail: <4.2 响应> }
```

- 逐条走该分组的既有更新路径(重写分组 Models JSON → V1–V6 校验 → 发布快照),
  复用 `UpdateGroupModels` 内核;**跨分组非原子**:按序应用,逐条返回结果,部分失败
  不回滚已成功项(前端明示每行状态;响应携带最新 detail 供刷新)。
- 频控:同一分组 1s 内多次 PATCH 合并去抖(前端节流即可,后端不加锁)。

### 4.4 运行态恢复

```
POST /api/model-route/schedule/recover
{ group_id, model_id }
→ 条目运行态三清(blacklist/failure_count/cooldown),
  即 registry.RecoverEntryForModel;不影响凭据运行态。
```

### 4.5 涉及文件

- `internal/control/model_route_schedule.go`(new)+ `model_route_schedule_test.go`(new);
- 路由注册挂到既有 control Server;响应投影复用 `route_inspect.go` 的行结构
  (新增 `circuit_breaker`/`runtime` 字段,旧字段不动——路由检查页不受影响)。

---

## 5. 前端(中心调度面板)

### 5.1 定位:路由检查页升级为"调度面板"

不新建页面。现有 **监控 → 路由检查** 已具备"选访问密钥 + 对外模型 → 候选条目行
(占比/层/原因码/凭据明细)"的全部骨架,原地升级为读写:

1. **新增调度入口页**:路由检查 Tab 旁新增"调度面板"二级页(或 Tab 内顶部切换),
   首屏为 §4.1 的对外模型名索引表(候选数/分组数/异常数),点击进入详情。
2. **详情 = §4.2 响应的编辑视图**:
   - 分组分节展示候选条目(同分组多条目沿用现有折叠交互);
   - 每行:权重(数字输入+占比实时预览)、优先级、**熔断参数(阈值/冷却秒,占位符
     显示继承默认值,留空=继承)**;
   - 行内校验(0–100、≥1、≥0)与批量保存条(逐行成功/失败状态);
   - 运行态徽标(冷却中/已拉黑/连续失败 N)+ **恢复按钮**(§4.4,仅运行态异常时可用);
   - 顶部访问密钥选择器沿用(占比口径)。
3. 路由检查页保持只读巡检语义不变(验收走查 §11.6 仍以它为准);调度面板与它共享
   资源层与类型(`route-inspection.ts` 扩展,新增 `model-route-schedule.ts`)。

### 5.2 涉及文件

- `web/src/app/resources/model-route-schedule.ts`(new,API 资源与投影);
- `web/src/features/monitor/`(调度面板组件,新建文件;`InspectorTab.vue` 仅加入口);
- i18n:zh-CN / en-US / ja-JP 三语全量词条;
- 验证:`pnpm build`(type-check)+ `pnpm lint` + 浏览器人工走查(参照 008 的实测清单)。

---

## 6. 兼容性承诺

- **C1** 未配置 `circuit_breaker` 的存量条目:运行态行为与现状逐字节一致(不计数、
  判定器默认冷却);
- **C2** 旧管理 API 报文(缺新字段)保存成功;
- **C3** 单条目/无别名条目路径不变;
- **C4** 快照 revision 与热更新机制不变;面板保存后无需重启即生效;
- 中心面板只读写既有分组数据,不迁移、不新增存储实体;回滚 = 丢弃代码,存量 JSON 中
  多出的 `circuit_breaker` 字段被旧版本读取时忽略(旧解码容忍未知键,已验证)。

---

## 7. 验收方式

以 §1.2 基准场景(分组1 A/B/C + 分组2/3)为准:

1. **聚合正确性**:调度面板详情对基准场景展示 5 条候选,分组权重/条目权重与配置一致;
2. **占比一致**:面板占比预览与路由检查页、真实流量分布三者一致(±3pp);
3. **跨分组编辑**:在面板把分组2 条目权重 100→50,保存后巡检占比即时重归一化,
   无需触碰分组页面;
4. **显式冷却**:为某条目配 `cooldown_seconds=60`,注入模型级 404,巡检页该条目
   `entry_cooldown` 且冷却截止 ≈ 60s(其他条目不受影响);
5. **阈值拉黑与恢复**:配 `blacklist_threshold=2`,连续两次模型级失败 → 条目
   `entry_blacklisted` 且调度流量绕行;面板点"恢复"后重新参与;
6. **成功清零**:计数 1 次后一次成功请求 → 计数归零;
7. **兼容回归**:删除全部新字段重复 1–6(行为与升级前一致);旧报文保存成功;
8. 全量 `go test ./...` 无新增失败(对照基线);前端 build/lint 通过。

---

## 8. 实施步骤建议(每步独立可评审)

| 步骤 | 内容 | 依赖 |
|---|---|---|
| S1 | 存储与校验:§2 全链路(JSON 解码/编码、V5/V6、ModelConfig/GroupView 索引、loader)+ 单测 | — |
| S2 | 运行时:§3 失败/成功链接入 + 单测 | S1 |
| S3 | 管理 API:§4 三端点 + 集成测试(含跨分组批量与部分失败) | S1 |
| S4 | 前端:§5 调度面板 + 三语 i18n + build/lint | S3 |
| S5 | 全量回归 + §7 验收走查 + 部署(沿用本次 fork 构建 → compose 切换流程) | S2–S4 |
