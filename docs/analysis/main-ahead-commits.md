# Main Ahead 检查进度

> 本文档只用于跟踪上游提交检查进度，不记录详细分析、架构判断或采用建议。详细结论由检查时口头输出；需要实施时再建立对应设计/任务文档。

## 最新检查

- 检查日期：2026-10-25
- 维护仓库：`DabengBa/gpt-load`
- canonical 目标：`origin/main`
- `upstream/main`：仅作额外参考，不作为维护目标
- 本轮仅做静态检查，未移植代码、未运行真实 Provider 请求
- 本轮对照 `dev` 的 `6708c162` 完成 12 个新增提交分类：Port 0、Adapt 5、Do Not Port 7；分类完成不代表已实施
- 定向验证：3 个包、10 项已有行为测试通过，仅验证渠道移除和提交前后错误边界，不代表新增恢复行为已实现

## Upstream 新增提交检查边界

每次检查 upstream 时，只检查“上次记录的 upstream 截止点”之后、当前 `upstream/main` 之前新增的提交：

```text
上次 upstream 截止点..upstream/main
```

本轮范围因此是：

```text
b8cadb40..7ba66c3f
```

已在更早检查周期完成分类的提交，不重复作为本轮新增内容分析。只有在新提交依赖它，或当前代码明确推翻旧结论时，才在新增提交条目中简短标注历史结论变化。详细分析和采用建议只针对本轮新增提交在口头输出，不写入本文档。

| ref | commit | commit 时间 |
|---|---|---|
| `dev` / `origin/dev` | `6708c162ae421190b52c958e93522430b01e0272` | 2026-10-24T13:13:33+08:00 |
| `origin/main` | `84896129357f7a801329c6283ca91b995e7d8e22` | 2026-09-30T17:26:33+08:00 |
| `upstream/main` | `7ba66c3f7a6315a08b89bdf29f8da92c1aaede1b` | 2026-10-24T21:05:46+08:00 |

`upstream/main` 已执行远端刷新；canonical main 本轮未重新检查，沿用上轮检查点。

## 范围进度

| 来源 | 上次截止点 | 当前检查点 | 范围提交数 | 独有提交数 | Port | Adapt | Do Not Port |
|---|---|---|---:|---:|---:|---:|---:|
| canonical `origin/main` | `1f615d83` | `84896129` | 43 | 43 | 0 | 8 | 35 |
| 参考 `upstream/main`（累计） | `93502ced` | `7ba66c3f` | 119 | 78 | 0 | 35 | 43 |
| 参考 `upstream/main`（本轮） | `b8cadb40` | `7ba66c3f` | 12 | 12 | 0 | 5 | 7 |

说明：upstream 累计范围中的 41 个提交已同时进入上述 canonical 范围，78 个为相对该 canonical 检查点的 upstream 独有参考提交。累计分类保留历史检查结果，不表示本轮重新核实了旧提交；当前没有确认可以原样 Port 的提交。

## 已检查提交

### Canonical main

- 检查范围：`1f615d83..84896129`
- 状态：43 项已完成静态分类
- Adapt：8 项
- Do Not Port：35 项
- 重点待办：`84896129` 并发限制维度差异、`43a6b66f` 流式空响应、`c51bdd8f` Codex 模型发现

### Upstream 独有

- 累计检查范围：`93502ced..7ba66c3f`
- 状态：78 项已完成静态分类（历史 66 项，本轮新增 12 项）
- Adapt：35 项
- Do Not Port：43 项
- 本轮检查范围：`b8cadb40..7ba66c3f`

| commit | 提交主题 | 状态 |
|---|---|---|
| `5e772799` | 移除过期 Sidrune AI 广告（#891） | Do Not Port |
| `d5e44e6f` | modern 移动端列表滚动及代理工作流（#893） | Do Not Port |
| `41c82ac0` | SDK 更新及失败请求已计费用量保留（#895） | Adapt |
| `dd33ae39` | 订阅积分余额向上取整（#899） | Do Not Port |
| `2a94750b` | Antigravity 配额窗口标签消歧（#907） | Do Not Port |
| `d9453eb7` | advisor 迭代计费及模型访问控制（#904） | Adapt |
| `81aae47b` | 内嵌 Codex CPA 输出前 overload 重试（#905） | Do Not Port |
| `5b7a5ee3` | 按可用内存限制请求体大小（#903） | Adapt |
| `0e4b5bb5` | modern 配额历史阶梯线（#909） | Do Not Port |
| `b4eda1cf` | 凭据切换后的 Responses 上下文恢复（#879） | Adapt |
| `197d2092` | 内嵌 Codex CPA 流前导后的历史拒绝恢复（#910） | Do Not Port |
| `7ba66c3f` | WebSocket Responses 上下文拒绝恢复（#911） | Adapt |

历史结论边界：本地于 2026-10-21 移除了内嵌 CPA 与订阅渠道，本轮相关提交按当前移除状态分类；旧轮 CPA/订阅条目不自动转为当前实施待办，也未在本轮逐项重评。

## 下一步进度

- [ ] 确认“直接做”项目的具体实施顺序
- [ ] 为本轮 5 项 Adapt 确认实施范围；WebSocket 历史恢复依赖 HTTP 共用恢复逻辑先落地
- [ ] 为每个直接实施项目建立独立代码任务和验证范围
- [ ] 对需要设计的项目先确定本地状态机、迁移和 API 合同
- [ ] 实施后运行受影响包测试、前端检查和 `git diff --check`
- [x] 完成本轮 12 项静态分类，推进 upstream 检查截止点至 `7ba66c3f`（不是实施进度）

## 检查规则

1. `origin/main` 是维护目标，`upstream/main` 只作参考。
2. 不按 SHA 是否相同判断功能是否缺失，必须对照 `dev` 的实际代码和测试。
3. `Port` 表示结构和语义兼容；`Adapt` 表示只吸收行为并按本地合同重写；`Do Not Port` 表示已有覆盖、无落点、排除功能或不具备当前价值。
4. 本文档只记录进度和状态，不记录详细设计、证据展开或实施方案。

## 固定工作流

1. 读取本节和“范围进度”中的上次 upstream 截止点。
2. 执行 `git fetch --prune upstream`，记录当前 `upstream/main`。
3. 使用 `git log --reverse --oneline <上次截止点>..upstream/main` 发现本轮新增提交。
4. 只对本轮新增提交逐项检查：提交目的、变更范围、当前 `dev` 落点和 Port/Adapt/Do Not Port 状态。
5. 如果旧提交结论被新代码推翻，只记录为历史结论变更，不重新输出整段旧分析。
6. 更新“最新检查”“范围进度”“已检查提交”和下一步进度；不把详细设计写入本文档。
7. 只有完成本轮新增提交的检查后，才推进 upstream 截止点。
8. 检查结束执行 `git diff --check`，并保护工作区内与本次检查无关的已有修改。
