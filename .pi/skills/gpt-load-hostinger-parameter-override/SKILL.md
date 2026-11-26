---
name: gpt-load-hostinger-parameter-override
description: "在 Hostinger 上为 GPT-Load 分组定位并修正 Provider 参数冲突；重点处理客户端协议、客户端模型别名、参数覆盖顺序与 request log/raw capture 验证。"
---

# GPT-Load Hostinger 参数覆盖

本 skill 只适用于当前项目 `DabengBa/gpt-load`，用于在已授权的 Hostinger 实例上定位并修正单个 Provider 分组的请求参数冲突。

## 触发条件

- 直接在 Hostinger 上修改 GPT-Load 分组配置；
- 修复 Provider 报错，例如 `reasoning_effort` 与 `thinking_budget` 同时设置；
- 根据 request ID 反查实际客户端协议、客户端模型名、上游模型名和最终上游请求；
- 验证参数覆盖是否命中并真正改变了上游请求。

## 不可妥协的安全边界

1. 只使用用户已授权的 SSH 目标（本项目自建实例默认 `vps-kl`，先读本机 SSH 配置确认，不猜服务器）。
2. 绝不打印/写入 skill 或报告：SSH 私钥、`AUTH_KEY`、上游 API Key、`Authorization`、Cookie、完整 prompt、完整 raw request/response body；只输出脱敏后的 ID、名称、channel、模型名、Base URL、状态码、字段存在性、长度和 hash。
3. 读取与修改分组配置均通过 GPT-Load 管理 API；不直接改 SQLite、volume 或容器内配置文件，不为查询把在线数据库复制到 `/tmp`/tmpfs；确需备份按 `.agents/skills/gpt-load-operations/references/sqlite-maintenance.md` 检查持久磁盘空间并用 SQLite 在线备份，失败时清理半成品。
4. 参数配置成功后不需重启容器。除非用户明确要求，不部署、不重启、不切镜像、不改代码。
5. 写入前先读目标分组完整 `overrides` 并保留无关配置；必须整体替换时确认无并发修改并保留旧值以便回滚。

## 标准执行流程

1. 确认工作区和 SSH 入口（`pwd`/`git branch`/`git status` + `rg` SSH config；不输出私钥内容）。发布/运维上下文见 `.agents/skills/gpt-load-operations/SKILL.md`。
2. 只读确认远端健康与分组：`docker ps`、`curl http://127.0.0.1:3001/health`；管理令牌只留远端 shell 变量；未找到明确目标时不按名称猜测写入。
3. 读目标分组模型和设置（`GET /api/groups/<group_id>`、`/models`、`/settings`），只保留非敏感字段；写入阶段用宿主机 `curl -X PUT`（BusyBox `wget` 不支持 `--method=PUT`）。
4. 提供 request ID 时先用 `gpt-load-request-log-analysis` skill 确认顶层 `protocol`、`client_model`、`group_id`/`channel_id`/`upstream_model`、`route_mode`/`upstream_protocol`、Provider 错误与状态码；不凭截图或 Provider 名称写规则。
5. 用 raw capture 判断字段形状（part 清单、hash/长度、目标字段存在性与终止事件；`failed`/`incomplete`/缺 part 一律报 `raw evidence incomplete`）。
6. 生成最小规则（字段语义见 `references/parameter-override-rules.md`）：只保留一个供应商接受的推理控制字段，绝不同时 set 互斥字段。
7. 写入分组设置：保存旧 `overrides`，`PUT /api/groups/<group_id>/settings` 只改目标分组，已有无关 overrides 必须合并保留；响应只打印 `code`、分组 ID/名称和脱敏 overrides。
8. 立即读取并验证：`code == 0`、协议/模型匹配预期、旧规则被替换而非并存、remove path 与 set 字段准确、`/health` 仍 healthy。
9. 用新 request ID 做行为验证：协议/模型一致、上游 raw request 只含一个推理参数、Provider 不再报互斥错误、attempt/retry/fallback 正常；验证 raw logging 时所有 attempt 都要有完整 parts 与终止事件。

## 回滚

新请求仍失败或非预期时：不叠加规则；读当前设置与写入前旧值对比；同一 `PUT` 恢复旧 `overrides`；重新 GET 设置并检查 `/health`；用新 request ID 验证回滚。本 skill 不自动回滚、不重启、不部署。
