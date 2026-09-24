# 自建实例部署（gptl.tanyaleoallen.cloud）

本文只描述本仓库维护者自建实例的发布流程；上游通用部署（`ghcr.io/tbphp/gpt-load:2`、原生二进制）见 `README_CN.md` 的「部署与数据」。SQLite retention、停机维护和恢复流程见 [`docs/sqlite-maintenance.md`](sqlite-maintenance.md)。

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

入口就是仓库里的 `scripts/deploy.sh`，服务器直接执行构建源中的这一份，所以发布逻辑只有一个版本，不存在需要同步的服务器副本。脚本按顺序执行：`git fetch` 目标分支 → `git reset --hard origin/<分支>` → `docker build --build-arg VERSION=<分支>-<短sha> -t gpt-load:<分支>-<短sha>` → 备份当前 `docker-compose.yml` → 改 `image` 标签与来源注释 → `docker compose up -d` → 轮询容器健康并打印 `/health`。健康未通过时脚本以非零退出；先按下文「健康检查异常排障」区分应用故障与宿主机执行环境故障，确认需要回退镜像后再回滚。

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

## 健康检查异常排障

`unhealthy` 仅表示容器健康检查失败，不等于应用不可用，也不能单凭一次 `/health` 成功断言历史上没有中断。发布脚本报告异常时先核对镜像、启动时间、重启次数，以及本机与对外 `/health`；再读 Docker 的 healthcheck 和 daemon 报错：

```bash
ssh vps-kl 'docker inspect gpt-load --format "image={{.Config.Image}} started={{.State.StartedAt}} restarts={{.RestartCount}} health={{.State.Health.Status}} check={{json .Config.Healthcheck}}"'
ssh vps-kl 'curl -fsS --max-time 10 http://127.0.0.1:3001/health'
curl -fsS --max-time 10 https://gptl.tanyaleoallen.cloud:1443/health
ssh vps-kl 'df -h /tmp; findmnt /tmp; journalctl -u docker --since "15 minutes ago" --no-pager | grep -E "Health check|no space left on device" | tail -20'
```

2026-09-23 的一次误报中，上一轮临时排查把在线 SQLite 数据库复制为宿主机 `/tmp/gl.db`，写入报 `No space left on device`；宿主机 `/tmp` 是 3.9G 的 tmpfs，而数据库超过该容量。daemon 记录 `Health check ... OCI runtime exec failed: write /tmp/runc-process...: no space left on device`。清理该临时副本后健康检查恢复，未因此回滚。`docker inspect` 只保留最近的健康检查日志；调查历史失败时以当时采集的证据和 daemon 日志为准。不要把此误报当成所有 `unhealthy` 的通用原因。

**读分组配置用管理员 API**（`GET /api/groups` 或 `GET /api/groups/<id>`），不要为了查询而 `cp` 在线数据库到 `/tmp`、容器的 `/tmp` 或其他 tmpfs。管理 API 的响应可能包含敏感信息，只输出必要的脱敏字段，不记录令牌或完整响应。

确需备份数据库时，先确认目标是**持久磁盘**、空间足够且备份文件受限访问，再用 SQLite 在线备份，不要直接复制运行中的数据库文件。以下命令在宿主机执行，不进入容器，也不修改源库；空间预检按当前 DB 大小的两倍预留，备份失败自动清理半成品（磁盘空间仍可能被其它进程并发占用）：

```bash
ssh vps-kl 'bash -se' <<'REMOTE'
umask 077
src=/var/lib/docker/volumes/gpt-load_gpt-load-data/_data/gpt-load.db
root=/opt/gpt-load
[ "$(findmnt -n -o FSTYPE -T "$root")" != tmpfs ] || { echo '备份目标不能是 tmpfs' >&2; exit 1; }
size=$(stat -c %s "$src")
avail=$(df -B1 --output=avail "$root" | tail -n 1)
[ "$avail" -gt "$((size * 2))" ] || { echo '持久磁盘空间不足' >&2; exit 1; }
dir=$(mktemp -d "$root/backup.XXXXXXXX")
trap 'rm -f "$dir/gpt-load.db"; rmdir "$dir"' EXIT
sqlite3 -readonly "$src" ".backup '$dir/gpt-load.db'"
[ -s "$dir/gpt-load.db" ] || { echo '备份为空' >&2; exit 1; }
trap - EXIT
printf '备份完成: %s/gpt-load.db\n' "$dir"
REMOTE
```

备份数据还需保管同卷的 `encryption.key`（以及恢复所需的 `auth.key`）；不要把密钥内容打印到终端或日志。备份成功不代表恢复已验证。当前实例 `/opt/gpt-load` 与数据卷同在宿主机磁盘，因此此处备份只适合临时排障；灾备必须转存到独立存储。空间不足或备份失败时停止，不要退回用 `cp` 或改放 `/tmp`。

上面的 SQLite `.backup` 是服务在线时生成的**数据库文件快照**，适合在线排障或取得一致的数据库副本；它不是离线灾备的替代品。离线维护或灾备请停止所有写入者，并使用 [`scripts/sqlite-maintenance.sh`](../scripts/sqlite-maintenance.sh) 归档整个数据目录，使 `gpt-load.db`、`-wal`、`-shm`、`auth.key`、`encryption.key` 及其他运行时文件保持同一恢复集合。两种流程都不表示已经完成真实生产 Docker 或加密恢复验证；验证仍需按维护文档执行部署相关的 health 和认证配置读取。

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
