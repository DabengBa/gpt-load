# Main Ahead 提交评估

## 基线

### refs 与分叉

- 记录时间：`2026-09-22T14:43:20+08:00`
- 当前 ref：

  | ref | commit | 时间 |
  |---|---|---|
  | `dev` | `59a21b7d` | `2026-09-22T14:09:20+08:00` |
  | `origin/main` | `1f615d83` | `2026-09-21T22:34:12+08:00` |
  | `upstream/main` | `93502ced` | `2026-09-22T12:28:55+08:00` |

- 分叉统计使用 `git rev-list --left-right --count A...B`，左列是 `A` 独有提交数，右列是 `B` 独有提交数：

  | 比较 | 统计 |
  |---|---:|
  | `dev...origin/main` | `304 94` |
  | `dev...upstream/main` | `304 96` |
  | `origin/main...upstream/main` | `0 2` |

### 工作流

1. **确定上次截止点**：读取本节记录的 `origin/main` 和 `upstream/main` commit；`origin/main` 是维护目标，`upstream/main` 只作额外参考。不要用本地 cherry-pick 后产生的新 SHA 替代上游截止点。
2. **刷新 refs**：依次执行 `git fetch --prune origin` 和 `git fetch --prune upstream`，然后重新记录三个 ref 的 commit、时间和分叉统计。
3. **发现新提交**：分别检查 `<上次 origin/main commit>..origin/main` 和 `<上次 upstream/main commit>..upstream/main`。将 canonical `origin` 新提交与 upstream 独有提交分开，避免把参考分支提交误当成待合并提交。
4. **逐提交建立摘要**：按提交时间顺序记录 SHA、PR 编号、标题、变更文件范围，并用一到三句话说明行为变化、影响层（后端、网关、调度、执行、前端、依赖或 CI）和用户可见影响。
5. **对照本地合同评估**：检查 `dev` 是否已有等价能力；区分“已覆盖”“可直接移植”“需要行为级适配”和“没有本地落点”。对需要适配的提交，明确冲突点、保留的本地不变量和需要补充的测试，不直接整体 cherry-pick。
6. **形成取舍**：使用三类结论：
   - `Port`：本地结构和语义兼容，可以按原提交顺序移植。
   - `Adapt`：只吸收目标行为，按本地合同重写；记录不能直接移植的原因。
   - `Do Not Port`：纯依赖、CI、品牌、未采用功能或没有独立本地落点。
7. **执行已批准变更**：用户明确要求后，按指定顺序 cherry-pick 或实现适配；冲突时保留 `dev` 的执行、调度、重试、凭据、WebSocket 和数据迁移边界，不新增兼容层或无依据的 fallback。
8. **验证并推进基线**：运行受影响包的定向测试，再运行必要的全量测试、前端类型检查和 `git diff --check`。确认结果后，用最新 `dev`、`origin/main`、`upstream/main` 的 commit、时间和分叉统计覆盖本节；不在本文保留已完成提交的历史清单。
9. **保护工作树**：检查 `git status --short`，不修改与本轮分析无关的已有改动或未跟踪文件。

每次运行的最小摘要格式：

```text
<时间> <SHA>（#<PR>）<标题>：<行为摘要>；影响：<范围>；结论：Port / Adapt / Do Not Port。
```
