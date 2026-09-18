# 自建实例部署（gptl.tanyaleoallen.cloud）

本文只描述本仓库维护者自建实例的发布流程；上游通用部署（`ghcr.io/tbphp/gpt-load:2`、原生二进制）见 `README_CN.md` 的「部署与数据」。

## 拓扑

- 反代：Caddy 容器 `gptl-proxy`（配置 `/opt/gptl-proxy/Caddyfile`），`https://gptl.tanyaleoallen.cloud:1443` → `127.0.0.1:3001`。
- 应用：Compose 项目 `/opt/gpt-load`，容器 `gpt-load`，镜像为自建 `gpt-load:<分支>-<短sha>`，数据在具名卷 `gpt-load_gpt-load-data`（含 `gpt-load.db`、`auth.key`、`encryption.key`）。
- 源码与构建：`/opt/gpt-load-src`，是 `origin` 的克隆，只作为构建源；每次发布由脚本 `git reset --hard origin/<分支>`。
- 该容器已用 `com.centurylinklabs.watchtower.enable: "false"` 关闭自动更新，镜像只通过下面的脚本切换。

## 发布

```bash
ssh vps-kl /opt/gpt-load-src/scripts/deploy.sh            # 默认发布 dev
ssh vps-kl /opt/gpt-load-src/scripts/deploy.sh <分支>      # 发布其它分支，用于上线前验证
```

入口就是仓库里的 `scripts/deploy.sh`，服务器直接执行构建源中的这一份，所以发布逻辑只有一个版本，不存在需要同步的服务器副本。脚本按顺序执行：`git fetch` 目标分支 → `git reset --hard origin/<分支>` → `docker build --build-arg VERSION=<分支>-<短sha> -t gpt-load:<分支>-<短sha>` → 备份当前 `docker-compose.yml` → 改 `image` 标签与来源注释 → `docker compose up -d` → 轮询容器健康并打印 `/health`。健康未通过时脚本以非零退出，按下面的回滚步骤处理。

脚本会把 `/opt/gpt-load-src` reset 到目标提交，因此开跑前先把自己复制到 `/tmp` 再重新执行，避免 bash 边读边执行一个刚被覆盖的脚本。分支名里的 `/` 在镜像标签中写成 `-`。

## Raw 通信证据

在 Unix 运行时，debug capture 是可选的，默认关闭；设置 `DEBUG_CAPTURE_ENABLED=true` 后，capture store 才记录 Gateway 及支持的 Provider observer 实际观察到的请求/响应。raw capture 与 `request_logs` 分离，固定保留 4 小时，只能通过管理面管理员身份访问：

```text
GET /api/debug-captures?request_id=<request-id>
GET /api/debug-captures/<capture-id>/download
```

`request_logs` 用于状态、重试、错误分类和计费结果，不能替代 Provider 原始 Body 或 SSE。可在发布后使用 `scripts/fetch-hostinger-request-log.sh <request-id>` 下载结构化日志、全部 capture ZIP 和脱敏 attempt evidence matrix。脚本只把原始文件写入本地 `tmp/`，不会打印认证信息或 raw 内容。

## 空完成诊断

`upstream_empty_completion` 只表示非流式 OpenAI Chat Completions 的一个保守诊断：上游返回 2xx，响应已正常完成，JSON 是完整对象且包含至少一个 choice；每个 choice 都是 assistant message，message 没有可消费 payload，`finish_reason` 缺失或为空/`stop`，并且 usage 已完整、output tokens 大于 0。命中后 request log 记录 `status=error` 和 `error_code=upstream_empty_completion`，但客户端仍收到上游原始的 2xx status（通常为 200）、headers 和 body。

它不同于 `upstream_content_filter`：后者由明确的 `finish_reason=content_filter` 表示供应商因内容过滤终止；该 finish reason 不会被当作空完成。空完成只记录诊断，不轮询、不重试、不熔断、不冷却：attempt 的 `failure_category=ambiguous`、`RetryNone`、`EffectNone`，并使用规则 `upstream.empty_completion`。

发布后可使用管理面管理员凭据，通过现有 request-log API 按 attempt 错误码查询，不需要访问数据库：

```text
GET /api/logs?error_code=upstream_empty_completion
```

取得 request ID 后，使用该 ID 获取结构化日志及关联 capture：

```bash
scripts/fetch-hostinger-request-log.sh <request-id>
```

脚本调用现有的 `GET /api/logs/<request-id>`、`GET /api/debug-captures?request_id=<request-id>` 和 capture download API，并把结果写入本地 `tmp/hostinger-request-logs/<request-id>/`。raw capture 需要管理员身份；发布后验证时应以 request ID 关联 request log 与每个 attempt 的 capture，并检查 evidence matrix 是否完整。本文描述验证步骤，不表示某个版本已经部署。

## 验证

```bash
curl -s https://gptl.tanyaleoallen.cloud/health          # 期望 {"status":"ok","version":"dev-<短sha>"}
docker inspect gpt-load --format '{{.Config.Image}} {{.State.Health.Status}} {{.RestartCount}}'
docker logs --since 5m gpt-load 2>&1 | grep -icE 'error|fatal|panic'
```

## 回滚

```bash
ssh vps-kl
cd /opt/gpt-load
ls docker-compose.yml.bak-*                              # 每次发布都会留一份
cp docker-compose.yml.bak-<时间戳>-<旧分支>-<旧短sha> docker-compose.yml
docker compose up -d                                      # 必须在 /opt/gpt-load 下执行，保持项目名 gpt-load
```

旧镜像不会自动清理，`docker images | grep gpt-load` 可直接看到待回滚的版本。回滚只切镜像标签，`gpt-load_gpt-load-data` 卷保持不动。

## 注意

- 持久化数据与 `encryption.key` 必须一起备份，缺失密钥会导致已有渠道凭据无法解密。
- `docker-compose.yml.bak-*` 是回滚点，确认新版本稳定前不要删除。
- 发布后先看 `docker logs` 的错误计数和首屏 `/health`，再观察真实请求日志中的失败重试情况。
