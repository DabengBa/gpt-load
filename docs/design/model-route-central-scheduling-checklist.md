# 实施清单:模型调度中心面板与候选级熔断参数

- 最高依据:`docs/design/model-route-central-scheduling.md`(v2,bf7e2a9)
- 本清单按其 §9 拆解为 6 个审查单元;每个审查单元 = 一次"实现 → 独立评审 → 主 Agent 提交"闭环
- 通用边界(每个审查单元违反任一即评审失效):
  - 只允许写该单元【写集合】列出的文件;读集合之外可自由阅读,读集合内禁止修改
  - 禁止改 `.github/`、CI 配置、设计文档语义、既有基线已知失败测试
  - 禁止提交 git commit(评审通过后由主 Agent 统一提交)
  - 全量测试对照基线 `results/baseline-after-crlf-fix.log.names`(13 项环境失败),仅新增失败计入回归
  - 前端验证:`pnpm build`(type-check)+ `pnpm lint`(引擎警告照旧披露)

## S1 · entry_id 身份与熔断字段存储/校验(设计 §2/§3)

- 写集合:`internal/control/group_models.go`、`internal/control/group_write.go`、
  `internal/control/group_update.go`、`internal/control/group_create.go`、
  `internal/state/external_model.go`、`internal/state/external_model_test.go`、
  `internal/state/snapshot.go`、`internal/state/snapshot_test.go`、
  `internal/state/loader/loader.go`、`internal/state/loader/loader_test.go`、
  `internal/gateway/model_entry_breaker_test.go`(new,占位编译通过即可,S2 填充)
- 完成标准:
  1. `entry_id`:创建路径服务端生成(`e`+12hex)、GET 懒回填并持久化、V7 校验
     (格式+分组内唯一);请求 DTO 三态保留(缺省=不变,null/值按 §5.5 语义在 S4 生效,
     本步先保证 round-trip 不丢字段);
  2. `circuit_breaker.blacklist_threshold/cooldown_seconds`:JSON/DTO/存储解码全链路;
     V5(≥1)/V6(≥0)校验;`ModelConfig.CircuitBreaker` +
     `GroupView.ModelBreakerByEntry` 编译索引;
  3. `RouteTarget.EntryID` 贯通编译;派生身份 `derived:<external>#<upstream>` 仅在
     缺失时用于编译期调度键(不写出快照);
  4. C1/C2 回归:旧 JSON(无新字段)行为不变;旧报文保存成功;
  5. 单测:V5–V7 各分支、懒回填幂等、索引构建、A/B 同上游两条目独立 entry_id。
- Proof:`go build ./... && go test ./internal/control/ ./internal/state/... -count=1`

## S2 · 运行时:计数/冷却规则接线(设计 §4)

- 写集合:`internal/state/registry.go`、`internal/state/registry_entry_key_test.go`(new)、
  `internal/gateway/handler.go`、`internal/gateway/handler_test.go`、
  `internal/gateway/model_entry_breaker_test.go`
- 完成标准:
  1. `EntryRuntimeKey` → `(GroupID, EntryID)`;相关 API 签名同步;内存态无迁移;
  2. §4.1 计数规则独立于 Effect:最终 Decision 满足
     `Category==ModelUnavailable && Scope==model && RuleID != "safety.replay_unknown"`
     时,若条目配置阈值 → `IncrEntryFailure` + 达阈值 `SetEntryBlacklisted`;
  3. 冷却:配置 `cooldown_seconds>0` → 覆盖判定默认;`=0` → 不调用
     SetEntryCooldown;未配置 → 沿用 decision 自带冷却(candidate.unavailable 等
     EffectNone 分支维持无冷却);
  4. 成功路径 `ClearEntryFailureForModel`(非流式 2xx + 流式 CleanEOF);
  5. C1:未配置条目与升级前行为逐字节一致(含 safety.replay_unknown 不计不冷);
  6. 单测:A/B 同上游隔离、阈值拉黑、零冷却、成功清零、replay_unknown 排除。
- Proof:`go build ./... && go test ./internal/state/... ./internal/gateway/... ./internal/scheduler/ -count=1`

## S3 · 占比口径:当前生效层归一化(设计 §6)

- 写集合:`internal/scheduler/inspect.go`、`internal/scheduler/inspect_test.go`、
  `internal/control/route_inspect_test.go`
- 完成标准:
  1. `applyEffectiveShares` 改为:activeTier = min{p | 层内 included+routable 且
     可用凭据组合权重 > 0};仅该层归一化,其余层 0 + fallback 标注;
  2. 语义变更披露:正常态结果与旧口径一致;P1 全冷却时 P2 归一化到 100%;
  3. 巡检 `entry_cooldown_until_ms` 仅在未来冷却时返回;
  4. 新增用例:P1 全冷却 → P2 share=1.0;正常态基准场景断言保持 1800/3000/3000/1000
     over 8800(±1e-9)。
- Proof:`go test ./internal/scheduler/ ./internal/control/ -count=1`

## S4 · 管理 API(设计 §5)

- 写集合:`internal/control/model_route_schedule.go`(new)、
  `internal/control/model_route_schedule_test.go`(new)、
  `internal/control/route_inspect.go`(若共享投影需微调)、
  `internal/control/server.go`(路由注册)
- 完成标准:
  1. `GET /api/model-route/schedule`(对外名索引,快照聚合,口径同 /v1/models 去重);
  2. `GET /api/model-route/schedule/detail`(protocol 必填 + operation 可选,复用
     InspectRoute 内核;行含 entry_id/circuit_breaker 三元组/runtime/effective_share);
  3. `PATCH /api/model-route/schedule`:三态字段(缺席/`null`/数值;`circuit_breaker:
     null`=清全部);服务端按 `(group_id, entry_id)` 字段级合并;**单事务**更新全部
     受影响分组 + 恰好发布一次快照;任一校验失败零写入;`snapshot_revision` 乐观并发
     (409);逐条结果 + 最新 detail 回显;
  4. `POST /api/model-route/schedule/recover`:`RecoverEntryForModel` 三清;
  5. 集成测试:三态各分支、跨组事务回滚、409、未配置条目 C1、非法 entry_id 404。
- Proof:`go test ./internal/control/ -count=1`(全绿)

## S5 · 前端调度面板(设计 §6A)

- 写集合:`web/src/app/resources/model-route-schedule.ts`(new)、
  `web/src/app/resources/route-inspection.ts`(共享类型扩展)、
  `web/src/features/monitor/SchedulePanel*.vue`(new)、`web/src/features/monitor/MonitorView*.vue`
  (入口)、`web/src/features/models/model-draft.ts`、
  `web/src/features/groups/models/GroupModelsTab.vue`(round-trip 原样回传)、
  `web/src/features/groups/models/model-diff.ts`、
  `web/src/i18n/locales/{zh-CN,en-US,ja-JP}/{monitor,group}.ts`
- 完成标准:
  1. 索引视图(对外模型名/候选数/异常计数)+ 详情视图(分组分节、行编辑);
  2. 权重/优先级/熔断参数编辑;熔断占位符显示 effective 默认,显式清空 = 清除覆盖;
  3. 事务式保存条(整体成功/失败 + 409 刷新提示);运行态徽标 + 恢复按钮;
  4. 协议/操作选择(默认操作可切换);
  5. 分组模型页 round-trip:entry_id/circuit_breaker 原样携带回传,不丢字段;
  6. i18n 三语无缺失;浏览器走查(参照上一轮实测清单)。
- Proof:`pnpm build && pnpm lint`;浏览器截图走查记录写入结果文件

## S6 · 全量回归与验收部署

- 写集合:`docs/design/model-route-central-scheduling-acceptance.md`(验收走查记录)、
  必要时各包 acceptance 测试文件(new)
- 完成标准:
  1. `go test ./...` 对照基线无新增失败;前端 build/lint 通过;
  2. 设计 §8 验收 1–11 逐项走查(自动覆盖/人工项),记录证据;
  3. 无 schema 迁移文件新增;
  4. 部署:服务器从 fork 构建新镜像 → compose 切换 → 健康检查与线上验证
     (沿用本次流程,备份先行)。
- Proof:验收记录文档 + 部署命令与输出
