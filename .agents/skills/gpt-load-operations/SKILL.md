---
name: gpt-load-operations
description: "Run GPT-Load operational procedures: Hostinger release, legacy database conversion, SQLite maintenance, debug capture analysis, and frozen-model-pricing deployment. Use when deploying, upgrading, converting, or diagnosing the self-hosted instance."
---

# GPT-Load Operations

## 触发

- 发布或回滚 Hostinger 自建实例（`scripts/deploy.sh`）。
- 旧版数据库离线转换（`gpt-load convert-legacy-db`）。
- SQLite 备份、校验或恢复。
- 采集或分析 debug communication capture。
- 部署 frozen-model-pricing v7 receipt 变更。

## 步骤

1. 确认目标：发布走 `scripts/deploy.sh [branch]`（默认 `dev`）；转换走离线命令；诊断先查 structured request log，再按需开 raw capture。
2. 读对应 reference 再动手：`references/release.md`、`references/legacy-db-conversion.md`、`references/sqlite-maintenance.md`、`references/debug-capture.md`、`references/frozen-pricing-deployment.md`。
3. 按 reference 的前置检查逐步执行；任一前置不满足就停下，不跳步。
4. 切换后核验：镜像版本必须是 `branch-shortsha`，本地与公网健康端点都通过，行为与本次变更相关项复验。
5. 发布后不做自动回滚（数据库可能已迁移）；回退只能显式版本回退并重新核验。

## 边界

- 镜像一律本地构建、SSH 上传；服务器只 load 镜像和跑 Compose，禁止在服务器构建或调用 BuildKit。
- 单实例发布是非零停机窗口；切换后无自动回滚。pre-switch 备份不可被 rehearsal 替代。
- 旧库转换只允许离线执行：无原地迁移、无兼容层、无自动删除；source 只读。
- raw capture 默认关闭（`DEBUG_CAPTURE_ENABLED=true` 才启用）；capture 可能含 Authorization/Cookie/API key/prompt/body，禁止不可信共享。
- frozen pricing 是 v7-only：不解释 v5/v6 receipt，不自动重定价历史数据。
