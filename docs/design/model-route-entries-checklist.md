# 实现 Checklist:模型路由条目(Model Route Entries)

> 依据:`docs/design/model-route-entries.md`(最高开发依据,下称"设计文档")。
> 勾选规则:实现 + 对应测试通过 + 符合设计文档不变式(I1–I5)后方可勾选。
> 每完成一个工作流(§)建议独立提交,提交信息遵循仓库 conventional commits 风格。

- 分支:`feature/model-route-entries`
- 状态:未开始

---

## §0 前置准备

- [ ] 通读设计文档 §2 不变式与 §10 兼容性承诺,列出个人理解疑点并对齐
- [ ] 跑通基线:`go test ./...` 与 `web` 构建,记录基线结果(后续回归对照)
- [ ] 梳理模型条目在代码中的全部触点(grep `ModelConfig`、`Alias`、
      `UpstreamModelID`、`externalModelName`),产出触点清单贴在 PR 描述

## §1 存储与领域模型层

- [ ] 条目 DTO 扩展:`weight`/`priority` 以指针语义加入
      - `internal/control/group_models.go`(管理面 DTO,`:21` 附近)
      - `internal/state/snapshot.go` `ModelConfig`(`:61`)
      - 序列化字段名:`weight`、`priority`;缺省 = nil
- [ ] 字段约束实现:weight ≥ 0(0=停用分流)、priority ≥ 1;负数拒绝;
      上限对齐 `state.MaxWeight`
- [ ] 校验规则 V1–V4(设计文档 §3):
      - V1 组内 `(对外名, 上游模型)` 唯一
      - V2 同对外名条目权重之和 > 0
      - V3 `id` 非空(保持现状)
      - V4 对外名取值规则:alias 非空用 alias,否则用 id(收口为单一 helper,
        供快照/巡检/目录/前端复用)
- [ ] JSON 兼容测试:旧报文(无新字段)解析后 weight=1、priority=1;
      新旧报文互转 round-trip
- [ ] 无 schema 迁移确认:`Group.Models` 为 JSON 列,老库直读不报错

## §2 快照编译层(`internal/state/snapshot.go`)

- [ ] `RouteTarget` 增加 `EntryWeight`、`Priority`(设计文档 §4)
- [ ] `appendExecutionTargets`(`:302`)按条目逐条 append,同组同对外名产生多 target
- [ ] `sortExecutionRouteIndex` 排序键改为 `(Priority, GroupID, UpstreamModelID)`
- [ ] 编译期执行 V1/V2 校验(最终防线),违规返回带分组/模型名的编译错误
- [ ] `NoModelRouteKey` 资源类操作路径回归不变
- [ ] 单测:多映射索引、排序确定性、编译校验错误信息

## §3 调度器(`internal/scheduler/`)

- [ ] `candidatePool.targetsByMode` 改为支持同分组多 target(键
      `(GroupID, UpstreamModelID)` 或分组内 target 切片);`groupIDsByMode` 去重
- [ ] 组合权重:`组权重 × 条目权重 × 密钥权重`,任一 ≤ 0 剔除
      (替换 `inspect.go:244` `effectiveWeight` 两因子逻辑,保留旧函数供兼容路径
      或标注废弃)
- [ ] 优先级分层:`Iterator` 构建时按 `Priority` 分层,`Next()` 仅在当前最高
      可用层内加权随机;层耗尽降级;全耗尽 → `ErrExhausted` + `staticReason`
- [ ] `tried` 集合改为 `(密钥, 上游模型)` 维度(设计文档 §5.3)
- [ ] `SkipGroup` 整组跳过语义不变(整组 = 跳过该组所有条目)
- [ ] 会话亲和:优先密钥命中逻辑回归;亲和密钥不在当前层时的降级路径
- [ ] 单测:
      - 比例正确性:基准场景大样本统计 ≈18/30/30/10(±3%)
      - 分层:P1 耗尽后进入 P2;P2 也可用时不与 P1 混选
      - 边缘占比与组内密钥数量无关
      - `(密钥, 上游模型)` 去重:同密钥换条目可重试

## §4 运行态(`internal/state/registry.go` + 失败处理链路)

- [ ] 失败分类:凭据级(401/配额等)vs 模型级(404/无该模型/无权限)
      ——先盘点现有错误映射点,补充分类判定(映射表建议放 channel/dialect 层)
- [ ] 条目维度熔断:键 `(GroupID, UpstreamModelID)`,冷却/拉黑/恢复机制
      与密钥维度同参数(`BlacklistThreshold`、冷却时长)
- [ ] 条目熔断不落库(内存 registry),重启清零(I5)
- [ ] 管理面只读暴露条目运行态(供巡检)
- [ ] 单测/集成:模型级错误只影响该条目;凭据级错误影响该密钥全部条目

## §5 转发与改写(`internal/gateway/`)

- [ ] 确认 `model` 改写全部走 `Selection.UpstreamModelID`,列出涉及的
      operation 清单(chat/images/embeddings/responses 等)并补回归断言
- [ ] 流式/非流式、Responses passthrough 回归
- [ ] 失败重试路径:换三元组(密钥或条目)后请求体 model 正确重写

## §6 管理 API(`internal/control/`)

- [ ] `group_models.go` 保存链路:新字段解析 + V1–V4 校验 + 错误信息友好
- [ ] 旧报文缺字段保存成功(C2),幂等/审计机制不受影响
- [ ] strict_json 校验兼容(未知字段策略不因新字段收紧而改变)
- [ ] API 测试:新增字段保存/读取/校验失败用例;存量配置读取回归

## §7 路由巡检(`internal/control/route_inspect.go`)

- [ ] 响应结构:分组 → 条目行 → 密钥三级;条目行含 `upstream_model`、
      `entry_weight`、`priority`、`effective_share`、可用性 + 原因码
- [ ] `effective_share` 计算:P1 层按组合权重归一化;兜底层(P≥2)单独标注,
      不并入 P1 占比
- [ ] 新原因码接入(设计文档 §6.3):`entry_blacklisted`、`entry_cooldown`、
      `entry_weight_zero`、`tier_demoted`
- [ ] 巡检响应测试:基准场景输入 → 占比与设计文档 §1.2 表一致

## §8 模型列表与目录

- [ ] 对外名去重 helper 收口;`/v1/models`、home 统计、分组模型选项
      (`internal/control/home.go`、`group_options.go` 等)统一改用
- [ ] 回归:单条目模型列表与现状一致(C1)

## §9 观测(`internal/requestlog/`、usage 链路)

- [ ] 请求日志同时落"外部模型名 + 上游模型名";确认落库字段与查询 API 输出
- [ ] 用量/计价按上游模型名匹配价格表(确认无回归)
- [ ] 按外部模型名的调用量聚合(满足"模型 A 的流量都去了哪"的运营问题)

## §10 前端(`web/src/`)

- [ ] `features/groups/models/GroupModelsTab.vue`:
      - 表格新增权重、优先级列(行内编辑,风格对齐现有 ModelAliasEditor)
      - 权重输入旁实时组内占比预览;优先级 ≥2 显示"兜底"徽标
      - 按对外名分组折叠展示;组头占比分布条
      - 行内校验:重复 `(对外名, 上游模型)`、权重合计为 0、权重 0 置灰
        "已停用分流"
- [ ] `features/models/ModelAliasEditor.vue`:列扩展 + 校验回调
- [ ] `features/groups/models/GroupModelSyncDialog.vue`:同步新条目默认值
      (alias 空、weight/priority 缺省),diff 逻辑(model-diff.ts)兼容新字段
- [ ] `features/monitor/InspectorTab.vue` + `app/resources/route-inspection.ts`:
      条目行结构、占比列、原因码展示
- [ ] 分组列表页"模型数"列改为 `对外模型数 / 条目数`
- [ ] i18n:`zh-CN` 全量词条 + `en` / `ja` 同步(核对 `web/src/i18n/locales/`)
- [ ] 前端构建 + 既有组件测试通过

## §11 测试与回归(横切)

- [ ] golden 路由测试更新:`internal/channel/route_golden_test.go`——单条目
      断言不变(C3),新增多条目用例
- [ ] 集成测试:基准场景(设计文档 §1.2)端到端:
      - 比例分布、优先级降级、条目熔断、密钥熔断、恢复回归
- [ ] 兼容回归:全库删除 weight/priority 字段的数据跑既有测试套件(C1)
- [ ] `go test ./...` 全绿;`go vet` / lint 通过;前端 lint/build 通过

## §12 验收与收尾

- [ ] 按设计文档 §11 验收方式 1–7 逐项人工走查并记录证据(截图/日志)
- [ ] README / 功能文档补充(如仓库有对应章节;无则最小化,不在本期扩写)
- [ ] PR 描述:需求 → 设计要点 → 触点清单 → 测试证据 → 兼容性说明(C1–C4)

---

## 依赖顺序与并行建议

```
§1 → §2 → §3 → §4 → §5      (核心链路,串行)
        §2 → §6/§7/§8        (管理面,可与 §3 后半并行)
              §9、§10        (观测与前端,依赖 §7 响应定稿)
§11 贯穿;§12 最后
```

## 风险提示(实现时重点盯)

1. `targetsByMode` 键结构变更是本设计唯一的结构性破坏点,存量调度测试全部过一遍;
2. 失败分类(凭据级 vs 模型级)依赖各渠道错误语义,盘点时注意 dialect 层已有的
   错误归类,避免重复造轮子;
3. 巡检 `effective_share` 与实际加权随机分布的口径必须同源(同一组合权重函数),
   否则会出现"巡检显示与实际流量不符"的验收事故;
4. alias 收口 helper 改动面广(快照/目录/巡检/日志),先收口再动编译逻辑,
   避免两套对外名推导并存。
