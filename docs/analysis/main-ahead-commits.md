# Main Ahead L3 专题分析

## 基线

- 分析分支：`plan/main-ahead-integration`
- 当前基线：`dev@ab89c3869ae3fa602b547640b31730ee889bbe33`
- 对比目标：`origin/main@0c9d188892aa4e998f7ec3fcea0f3219da25afcb`
- `git rev-list --left-right --count origin/dev...origin/main`：`81 22`
  - `dev` 独有 81 个提交
  - `main` 独有 22 个提交
- 合并基点：`0ddc41d8b718c0281b1ba2f5bdfe5b23622621ce`
- 本文件只保留需要重新设计、拆分移植或单独验证的 L3 内容；已合入 dev 的 L1/L2 选择性移植和不纳入范围不在本文展开。

## 当前合同

L3 方案必须以当前 `dev` 合同为边界：

- 每个分组只有一个上游凭据；普通第二凭据创建、导入、连接必须原子拒绝。
- 权重、优先级、熔断只属于模型 Route Entry；不恢复分组权重或凭据权重。
- 调度中心负责入口级路由、候选排序和运行态；不恢复旧 scheduler、Registry 或前端调度 API。
- 分组级 retry 配置仍属于当前高级配置；全局预算方案不能直接删除该配置。
- `(group_id, entry_id)` 只用于后端定位和 mutation，不进入用户可见的调度文案或筛选维度。
- Protocol、operation、external model 和 target config 必须保持当前 URL、API、目标冻结和候选 fallback 语义。
- 迁移必须避开当前迁移编号和状态恢复合同；不添加兼容字段、旧 API fallback 或批量多凭据迁移层。

## L3 专题

### 1. 凭据全量操作与单凭据导入

涉及提交：

- `e888fe60` `feat(credentials): 统一全量操作并支持五千条密钥导入 (#587)`
- `96d3e0d5` `feat(subscription): 支持多格式订阅凭据导入 (#596)`

上游内容：

- 增加全量恢复、全量下载、API Key 文本导出和大批量订阅凭据导入。
- 增加多格式 importfile 解析、provider importer、批量暂存 API 和导入报告。
- 增加重复身份、格式错误、大小、超时和授权失败分类。

重新设计边界：

- 导出、恢复、输入大小校验和多格式解析具有独立价值，可以拆分移植。
- 批量 endpoint 和导入 UI 必须收敛为单个最终凭据，不能按 main 的多凭据模型写入多个同组账号。
- 必须复用当前重复身份只读检查、写入前拒绝、同身份 replacement 和 `reauthorization_required`/`outcome_unknown` 替换边界。
- 导入失败必须在持久化前返回，不能产生部分成功或绕过单凭据唯一约束的中间状态。

### 2. 入口公平调度与连续分配限制

涉及提交：

- `9cb3f986` `feat(scheduler): 实现全局加权轮询与连续分配限制 (#591)`

上游内容：

- 引入全局加权公平调度、连续分配限制、SchedulingState、检查点和恢复校准。
- 替换随机候选选择，并处理亲和、回放和分组生命周期。

重新设计边界：

- 只研究入口级公平分配，不移植 main 的分组权重和凭据权重。
- 输入维度应是当前 Route Entry 的 weight、priority、breaker 和候选健康事实。
- 必须保持单凭据分组、入口级 retry deduplication、cooldown、blacklist、affinity 和 `DispatchMaybeSent` 语义。
- 检查点只能恢复调度运行态，不能恢复已删除的旧权重字段或改变模型路由 API。
- 需要先定义连续分配限制与入口 priority、breaker、preferred credential 的优先级，再实现公平算法。

### 3. 模型错误重试与健康恢复

涉及提交：

- `e0bfa07e` `feat(scheduler): 统一凭据权重并调整模型错误重试 (#597)`
- `d4699dd2` `feat(health): 支持凭据按模型冷却与统一恢复 (#599)`

上游内容：

- 调整模型不可用错误分类、冷却和重试语义。
- 增加凭据加模型维度的冷却运行态、检查点、请求日志字段、恢复 API 和监控展示。
- 删除或统一自动凭据权重。

重新设计边界：

- 凭据权重部分不适用于当前模型；权重只能留在 Route Entry。
- 必须明确 credential+model cooldown 与入口级 breaker 的权威关系，避免两个运行态同时决定候选可用性。
- 模型冷却不能让同一凭据的其他模型被错误隔离，也不能绕过入口 priority 或 group enabled 状态。
- 错误分类必须区分请求级、模型级、凭据级和上游主机级失败，并与 retry deduplication 和 `DispatchMaybeSent` 一致。
- 恢复 API 要保持当前 health observation/reset cache invalidation 合同；迁移编号和启动恢复必须单独验证。

### 4. 自定义订阅上游与目标冻结

涉及提交：

- `6da82242` `feat(subscription): support custom upstream URLs (#579)`

上游内容：

- 允许订阅渠道配置自定义上游地址。
- 对涉及凭据的订阅地址强制 HTTPS。
- 让模型发现、额度重置、健康状态和网关请求使用冻结后的 target config。

重新设计边界：

- 保留 target freeze 和 HTTPS 校验的思想，按当前 Group、单凭据、proxy 和生命周期合同重新接入。
- target config 必须在候选准备阶段冻结，并在 discovery、health、reset、probe 和实际 execution 间保持一致。
- 不新增生产 `BaseURL` 兼容字段，不恢复 main 的旧 channel endpoint contract。
- 自定义地址与 managed proxy、credential identity、provider binding 和 route mode 必须一起校验。
- 目标配置改变时必须有明确的 cache invalidation 和运行态重建边界，不能让旧 runtime 使用新目标或反之。

### 5. Usage 时间窗口与监控上下文

涉及提交：

- `33fb54bf` `fix(monitor): 修正最近一小时用量与成本统计 (#588)`
- `175949e2` `feat(usage): 支持自定义时间统计并统一时间筛选 (#594)`

这两个提交必须作为一个专题处理，不能按历史顺序直接 cherry-pick。

上游内容：

- 将最近一小时从单个小时桶改为 12 个 5 分钟桶，并补齐缺失空桶。
- 用精确 `from_ms`/`to_ms` 替换固定快捷范围作为查询核心，支持非整点窗口和动态桶宽。
- 统一用量、日志、AccessKey 之间的时间筛选、URL 状态和跳转上下文。

重新设计边界：

- 后端查询、request log 聚合、cost 估算、前端资源投影、图表桶宽和三语文案必须使用同一个时间范围合同。
- 保留当前 Monitor URL state、Scheduling Center 上下文、日志跳转和缓存失效规则。
- `preset` 只能是输入来源；保存后的 `from_ms`/`to_ms` 是可复现的权威查询状态。
- 时间范围跨越数据保留边界、空桶、时区和分页时，必须保持排序、总数和成本汇总一致。
- 需要分别验证最近一小时、非整点自定义窗口、跨天窗口、无数据窗口和 AccessKey 过滤，不把图表显示正确当作后端合同完成。

### 6. 全局请求重试预算

涉及提交：

- `a4255546` `feat(gateway): 统一全局请求重试预算 (#598)`

上游内容：

- 将分组级额外重试次数改为系统级全局重试预算。
- 让一次请求跨多个分组时使用统一预算。
- 从分组设置和资源投影中移除 retry count。

重新设计边界：

- 当前 dev 明确保留分组 retry 配置，不能直接删除配置字段或旧 UI 高级配置。
- 需要先定义 group retry 与全局预算的关系：预算是上限、补充额度还是候选链共享额度。
- 预算必须覆盖 native/conversion fallback、stream retry、credential refresh 和 `DispatchMaybeSent` 禁止重试场景。
- 重试计数必须在一次逻辑请求内全局递减，不能因跨 group、协议转换或 stream/unary 分支重新初始化。
- 运行日志、health effect、usage attempt sequence 和前端 settings 投影必须继续保持当前合同。

### 7. Codex 独立 WebSocket Session

涉及提交：

- `7d80a981` `feat(codex): 添加独立上游 WebSocket Session 封装 (#612)`

这是当前最新 upstream-only 范围中唯一新增的 L3 专题。

上游内容：

- 为 CPA 增加独立 Codex WebSocket Session facade。
- 支持 `NewWSSession`、`ExecuteTurn`、连接复用、取消、代理和生命周期管理。
- 增加连接日志、session 测试和禁止业务请求重放的边界覆盖。

重新设计边界：

- 该能力不是简单替换 HTTP executor，而是新增连续多轮 Responses 传输生命周期。
- Session 必须使用调度器已经选定的 credential，不自行选择凭据、不绕过入口级 priority/breaker，也不恢复旧 CPA Manager。
- `previous_response_id`、连接复用、关闭、取消、代理和 credential identity 必须与当前 attempt/session contract 对齐。
- 发送前失败必须是 `DispatchNotSent` 并允许当前候选策略处理；发送后断线、半响应和关闭必须保留 `DispatchMaybeSent` 与 `ReplaySafetyUnknown` 证据。
- 不得加入 HTTP fallback、业务请求重放、隐式跨 credential 迁移或旧 retry/fallback 行为。
- 需要覆盖连接复用、同一 session 多轮、代理传播、主动取消、上游关闭、超时、发送后错误和 credential replacement 后的 session 隔离。

## L3 执行顺序

1. 先定义 Codex WebSocket Session 的 dispatch、replay safety、credential identity 和 proxy 生命周期合同，并补最小编译/行为证据。
2. 再处理 Usage 专题，先冻结时间范围与 URL/resource 查询合同，再实现后端聚合和前端状态。
3. 并行前先完成 scheduler/health 的权威状态设计，避免连续分配、模型冷却和入口 breaker 形成多个真相源。
4. 最后按当前单凭据写入边界拆分凭据导入、target freeze 和 retry budget 能力。

## 结论

- 当前 `dev` 已吸收此前选择性移植的 L1/L2 能力；本文件不再把这些历史内容当作待合并范围。
- `origin/main` 仍不能整体合并：L3 领域合同与当前单凭据、入口级调度、迁移和 URL 状态存在结构差异。
- 当前下一项上游工作是 `7d80a981` Codex WebSocket Session；完成合同和证据后，再决定是否建立独立实现提交。
- 本次只更新基线和 L3 分析，不对上述 upstream commit 做代码 cherry-pick。
