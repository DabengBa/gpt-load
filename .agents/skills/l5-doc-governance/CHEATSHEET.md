# Wrap-Up Cheatsheet

收尾时逐项确认；`Pass Standard` 不满足就按 `Not-pass Handling` 处理，不追加解释性文档。

| 检查 | Pass Standard | Not-pass Handling |
| :--- | :--- | :--- |
| 排除矩阵 | 文档中没有目录树、API 表、Schema、UI 规格、依赖版本、`TODO.md` / `CHANGELOG.md` / 进度 | 删除该段；改由 `rg`/LSP、Linter+AST、OpenAPI/类型、Design Tokens、manifest+lockfile、任务图与 Git 历史承担 |
| 可预测性门禁 | 每条沉淀都答不出「若仅看合入后的代码、类型和单元测试，未来无记忆 Agent 能否反推出此结论？」 | 能反推 → 丢弃；不能反推 → 按 `references/distillation-rubric.md` 路由 |
| 单一 owner | 写入前查过 `existing_knowledge_index` / `existing_owner_files`，同一事实只有一个落点 | 更新既有条目；已编码进 Linter/类型的 INV 在同一提交删除 |
| 临时/永久隔离 | `.tmp/` 与 `**/.tmp/` 被忽略，`git ls-files` 无 `.tmp/` 文件 | `git rm -r --cached <path>/.tmp` 并补 `.gitignore` |
| Token 预算 | 根 `AGENTS.md` ≤120 行 / ≤1500 估算 tokens；`SKILL.md` ≤500 行 | 流程移入 `references/`/`scripts/`，或改成静态检查报错 |
| 悬空引用（含映射后 owner） | `check_doc_budgets.py` 无 `Dangling` / `Orphaned` 报错 | 查 git log 更新 `affects_symbols`/`paths`；失效则 `status: superseded` 或删除 |
| 脚本输出 | 只把 stdout / 退出码带入上下文 | 不读脚本源码，不整目录加载 owner 文档 |
| 既有 owner 体系 | 写入目标仓库既有 owner，并复用其既有校验入口 | 不新建平行 `.knowledge/` 树，不叠加第二套门禁 |
