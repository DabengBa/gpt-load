# Main Ahead 提交评估

## 基线

### 最新同步检查

- 记录时间：`2026-10-13T11:53:43+08:00`。
- 本次已刷新 `origin/main`、`origin/dev`，未刷新 `upstream`，未切换或同步本地分支。
- 最新检查的 canonical main 提交：`84896129357f7a801329c6283ca91b995e7d8e22`（#792，数据面并发限制）。本次检查没有移植代码；检查终点不代表已合入或完成全部取舍。
- 当前 ref：

  | ref | commit | 时间 |
  |---|---|---|
  | `dev` / `origin/dev` | `7de57d41` | `2026-10-12T18:57:24+08:00` |
  | `origin/main` | `84896129` | `2026-09-30T17:26:33+08:00` |
  | `upstream/main`（上次快照，本次未刷新） | `65c6509f` | `2026-10-04T08:51:41+08:00` |

- 分叉统计使用 `git rev-list --left-right --count A...B`，左列是 `A` 独有提交数，右列是 `B` 独有提交数：

  | 比较 | 统计 |
  |---|---:|
  | `dev...origin/main` | `458 137` |

- 137 是整个分叉历史中 main 独有的提交数，不是 137 个未考察提交，也不是 137 项缺失功能。`git cherry origin/dev origin/main` 中 10 项已有 patch 等价实现；其余也可能被 dev 的重写实现覆盖。
- 2026-10-05 已记录检查到同一个 main 终点；本次 main 检查终点没有新增提交。相对下表保留的评估截止点 `1f615d83`，累计待归档范围仍为 43 个提交。

### 已完成评估的截止点与本轮待处理范围

上次记录时间为 `2026-09-22T14:43:20+08:00`。以下截止点暂不推进，避免下一轮漏掉尚未决定或移植的提交：

| ref | 上次截止点 | 本轮检查终点 | 新增提交数 |
|---|---|---|---:|
| `origin/main` | `1f615d83` | `84896129` | 43 |
| `upstream/main` | `93502ced` | `65c6509f` | 61 |

- 上游本轮新增的 61 个提交中，41 个已进入 `origin/main`，20 个仅在 `upstream/main`；`origin/main` 的另外 2 个新增提交已包含在上次上游截止点内。
- 上述数量是 Git 提交范围统计，不代表本地缺失能力的数量；需检查本地等价实现。
- 上次提及的 `upstream-update-analysis-2026-10-05.md` 当前不在仓库中，不能把该链接作为已完成专题评估的证据。本次代码对照结论直接记录在下节。

### 2026-10-13 有价值候选的进一步归类

范围以 canonical `origin/main` 为准；排除 MySQL、JEV/guardrails（用户所述 jav 按 JEV 理解）。历史候选只用于核对既有能力，不重复计作本轮新增。以下为静态代码对照，未运行测试、未执行真实 Provider 请求或 raw capture。

难度：低＝独立改动；中＝按 dev 合同调整；高＝跨存储、控制面、网关生命周期和 Astryx 界面适配。已覆盖项目的难度仅表示核对成本，不表示还需实现。

#### 已有实现，不重复移植

| 提交 | 当前证据 | 结论与实际价值 |
|---|---|---|
| `bab12b28` 参数保真 | `internal/execution/bifrost/parameter_overrides.go`、`parameter_overrides_test.go`；`converted.go` 和 `executor.go` 调用 `mergeConvertedParameterOverrides` | 已有扩展参数合回转换请求的实现，避免 SDK 丢字段；低，暂不新增工作 |
| `0ec0d610`、`607fb916` Gemini embeddings | `internal/dialect/gemini_embeddings.go`、`internal/execution/geminiembedding/convert.go`、`gemini_native_embeddings_test.go`、`gemini_embeddings_conversion_test.go` | 原生单条/批量和 OpenAI 转换均已有；低，不重复移植 |
| `fe4b6ac1` rerank（历史候选） | `internal/dialect/rerank.go`、`internal/execution/bifrost/rerank.go`、`rerank_test.go`；dev `23ea2bd1` | 已有原生重排序，不再列为缺失能力；低 |
| `2ef32f41`、`54db81fd`、`7064ec70`、`73b3122f` 渠道 | `internal/channel/builtins.go`、`internal/channel/modules/`、渠道合同测试 | 已有 Mistral、Cline、OpenCode 和兼容预设；不能仅凭 main SHA 未合入重复增加渠道；低 |
| `7ceb5976` Codex 目录 | `third_party/cpaembedded/modelcatalog/client.go` 声明 `ClientVersion = "0.159.2"`，嵌入对应 JSON | 0.159.2 目录已有；仅凭该提交标题不能要求升级整个 SDK；低 |
| `8688dc80`、`87fabd34` WS/UA（历史候选） | `internal/gateway/websocket_turn.go`、`internal/execution/bifrost/user_agent.go` | 已有 stream 字段规范化与有效 UA 保留逻辑；低 |
| `42b4a338` 历史资源引用（历史候选） | `internal/control/request_logs.go`、`internal/requestlog/query.go` | 当前日志已处理失效资源历史引用；应保留本地逻辑，避免重复移植；低 |

上述“已有实现”指核心行为有代码落点，不等于所有上游边界案例都已通过运行验证。主要相关行为已集中进入 dev 的 `fc17a659` Provider 能力更新，不能以 patch 不等价推断缺功能。

#### 优先验证行为差异，再决定适配

| 提交 | 难度 | 当前情况、价值与下一份证据 |
|---|---|---|
| `e31b3be9` Base URL | 中 | 上游实际修改 OpenAI image、WebSocket、OpenRouter Decisions 的 URL 行为。dev 有自定义 URL 和不同执行架构；没有 Decisions 文件不代表全部协议有缺陷。先检查 image/WS 对 host、`/v1`、tenant path 的最终地址，仅修失败路径；价值是避免重复版本路径或丢租户前缀 |
| `8f894c3e` 工具输出字符串化 | 中 | dev 有 `conversion_fidelity.go` 和较新的 vendored Bifrost。比较 Responses 的内容块数组工具输出转 Chat 后 `content` 类型；若严格兼容上游仍拒绝数组，再适配；价值是保持工具调用连续性 |
| `631e5abf` reasoning summary | 中 | dev 已有 reasoning/replay 转换处理；缺少上游同名函数不是缺口证据。检查 Chat→Responses 的 reasoning item 是否含合法 `summary` 数组；价值是兼容严格 Responses 客户端 |
| `f2840466` Grok cache key | 中 | dev `grok_executor.go` 已处理会话字段、亲和体系也不同。检查最终 body/header 的缓存标识是否稳定，不能为了沿用上游字段破坏当前会话身份合同；价值是 Grok 连续对话与缓存命中 |
| `43a6b66f` 空响应候选重试 | 高 | dev `internal/gateway/execution_forward.go` 已有 `upstream_empty_completion` 识别，dev `1c0c3737` 已修空成功响应。剩余要比较“检测”与“候选切换”策略，覆盖 tool-only、reasoning-only、图片和已发送流；不能按文件名不存在判缺功能 |

#### 确有独立能力差异，但按实际需求取舍

| 提交 | 难度 | 实际价值、适配边界与建议 |
|---|---|---|
| `84896129` 数据面并发限制 | 高 | dev 的 RPM 限流和 probe 并发上限不等于数据面 in-flight 限制。对 Hostinger 单实例防止长流/WS 累积有价值；优先候选。需核实服务器实际负载，按 HTTP、流式、WS 生命周期设计获取/释放槽位，覆盖取消、超时、断连，不能整体 cherry-pick |
| `dd8fa169` RPM 趋势统计 | 高 | dev 已有 `internal/ratelimit/access_key_rpm.go`，但速率限制不等于历史趋势。运维容量分析有价值，优先级低于正确性与并发保护；先确定按客户端请求还是上游 attempt 计数、保留期和是否可复用 request logs，再决定持久化与 Astryx 展示 |
| `2cdf97b8` 日志总数分页（历史候选） | 中 | dev 当前 cursor 分页不等于上游 total/page 分页；未发现日志列表总数契约。只有需要页码跳转或总数时才适配，权衡大量日志 COUNT 成本，不能把已有 cursor 判为完整覆盖 |

#### 已有较完整流程，仅接受具体缺口

| 提交 | 难度 | 当前情况与取舍 |
|---|---|---|
| `96d3e0d5` 多格式导入（历史候选） | 中 | dev 已有 credential staging、multipart/JSON 导入及单凭据合同。需用具体未支持的格式样本证明缺口；不引入上游批量导入来破坏每组单凭据约束 |
| `6da82242` 订阅自定义 URL（历史候选） | 中 | dev 已有 target/base URL 保存、规范化和发现路径。仅核对 refresh、quota、discovery、execute 是否一致使用 target；没有故障样本不重建 |
| `c51bdd8f` Codex 发现配置 | 中 | dev 已有 Codex discovery 和 Astryx ModelDiscoveryDrawer；仅比较具体配置项及发现结果，不整体引回 main catalog/前端结构 |

本轮排序：先验证协议正确性候选（URL、工具输出、summary、Grok cache），再按负载需求评估数据面并发上限；RPM 趋势和总数分页放后。语音、可逆脱敏、Vue modern/classic、JEV 和 MySQL 不纳入本轮建议。以上归类不授权代码移植，也不推进尚待归档的评估截止点。

### 四项协议边界开发验证（2026-10-13）

- Run worktree：`fix/verify-ahead-protocols`，基线 `7de57d41`。用户确认后使用 orchestrated-delivery，四个单元共享该 worktree。
- URL（`e31b3be9`）：现有图片与 WS 的本地上游捕获测试通过，版本路径、租户前缀和认证/查询边界已有覆盖；不移植。
- Reasoning summary（`631e5abf`）：现有 unary/stream 和严格 Responses 回放测试通过。dev 保留 raw reasoning 并使用合法 summary 数组，不能照搬旧实现把 raw reasoning 改为 summary；不移植。
- 工具输出（`8f894c3e`）：捕获证明转换后 Chat tool content 仍为数组。按当前合同适配普通文本字符串化，非文本结果通过既有 `critical_semantic_loss` 在发送前拒绝候选；不采用上游 marker 丢内容方案。最终契约见 `.docs/tech/reasoning-replay.md`。
- Grok cache（`f2840466`）：两轮 unary/stream 捕获证明 trusted session metadata 导致前轮 encrypted reasoning 自动注入。修复为 OpenAI/Responses body 的稳定缓存键，不再使用该 metadata；保留 credential/model/attempt 隔离。原有缓存键稳定性并非缺陷。
- 验证为本地模拟上游与真实转换流程，不代表全部线上 Provider 已接受。运行包在 `.tmp/verify-ahead-protocols_20261013T120000+0800/`，不提交运行包。main 全部 43 项仍未完成取舍，评估截止点不推进。

### 工作流

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
