# 自建实例部署（gptl.tanyaleoallen.cloud）

本文只描述本仓库维护者自建实例的发布流程；上游通用部署（`ghcr.io/tbphp/gpt-load:2`、原生二进制）见 `README_CN.md` 的「部署与数据」。SQLite retention、停机维护和恢复流程见 [`docs/sqlite-maintenance.md`](sqlite-maintenance.md)。

## SSH 目标

**本文及发布脚本中的 `ssh vps-kl` 连接的是 Hostinger 服务器，不是本地构建机或其他 VPS。**
当前 SSH 别名解析为 `root@72.62.76.115:22`，主机名为 `srv1138005`，使用本机
`~/.ssh/hostinger_root_ed25519`。部署前用 `ssh -G vps-kl` 核对解析，再用
`ssh -o BatchMode=yes vps-kl 'hostname'` 确认主机身份。连接信息变化时同步更新本文与 SSH
配置；不要把私钥复制进仓库、提交、打印或上传为发布产物。

## 拓扑

- 公网 443（主入口）：`https://gptl.tanyaleoallen.cloud` 由 GoDoxy `godoxy-app` 终止 TLS，再按 `config/vhosts.yml` 以 HTTP 反代到 `127.0.0.1:3001`：

  ```yaml
  gptl.tanyaleoallen.cloud:
    scheme: http
    host: 127.0.0.1
    port: 3001
    response_header_timeout: 2h
  ```

  443 的证书由 GoDoxy autocert（provider `hostinger`）签发，落在具名卷 `godoxy_godoxy-certs`（`/app/certs/gptl.tanyaleoallen.cloud.crt`）。`config.yml` 的 entrypoint 中间件对本域名生效：响应带 `Referrer-Policy`、`Strict-Transport-Security`、`X-Content-Type-Options`、`X-Frame-Options` 四个安全头，`http://` 请求在 80 端口得到 **308** 跳转到 https（切换前是 404）。443 响应**没有** `via: 1.1 Caddy`，并广告 `alt-svc: h3=":443"; ma=2592000`。
- 公网 1443（保留的回滚入口）：Caddy 容器 `gptl-proxy`（配置 `/opt/gptl-proxy/Caddyfile`），`https://gptl.tanyaleoallen.cloud:1443` → `127.0.0.1:3001`，`protocols h1 h2` + `header >Alt-Svc clear`，证书仍来自 `/acme` 的 Caddy 侧 ACME 目录。1443 只服务 HTTP/1.1 与 HTTP/2，**不在 443 路径上**。
- 应用：Compose 项目 `/opt/gpt-load`，容器 `gpt-load`，镜像为自建 `gpt-load:<分支>-<短sha>`，数据在具名卷 `gpt-load_gpt-load-data`（含 `gpt-load.db`、`auth.key`、`encryption.key`）。
- 源码与构建：在本地仓库执行发布脚本，本机 Docker/BuildKit 完成构建。服务器只加载镜像和运行容器，`/opt/gpt-load-src` 不再参与发布。
- 该容器已用 `com.centurylinklabs.watchtower.enable: "false"` 关闭自动更新，镜像只通过下面的脚本切换。
- Caddy **暂不删除**：Alt-Svc 排空与移除条件见 [`docs/godoxy-ingress.md`](godoxy-ingress.md) 的 6.4/6.6，本文不重复。发布健康 URL 用 `https://gptl.tanyaleoallen.cloud/health`（不带端口）。

## ingress 观测（443 路径看 GoDoxy，不是 Caddy）

443 路径现在完全在 GoDoxy 内部，**Caddy 的 JSON access log 里看不到任何 443 流量**，用它判断 443 问题会得到空结论。观测入口有两个：

```bash
# 1) GoDoxy 容器 stdout：TLS 握手错误、代理错误、404、http2 preface 错误、日志轮转提示
ssh vps-kl 'docker logs --since 1h godoxy-app 2>&1 | tail -50'
#    典型行：http: TLS handshake error from <ip>:<port>: local error: tls: bad record MAC
#            http proxy error error="..." url=gptl.tanyaleoallen.cloud/v1/responses
#            not found: <host>  /  http2: server: error reading preface from client

# 2) 访问日志文件（combined 格式，保留 30 天）
#    config.yml 里 access_log.stdout: false，因此这些行不在 docker logs 中
#    /app/logs 是 bind mount，宿主机上直接读即可
ssh vps-kl 'tail -n 50 /www/server/panel/data/compose/godoxy/logs/entrypoint.log'
ssh vps-kl 'ls -t /www/server/panel/data/compose/godoxy/logs/ | head'   # 轮转后的历史文件
```

- GoDoxy 按小时轮转 `entrypoint.log` 为 `entrypoint.log.<时间戳>`；刚发生轮转时当前文件可能为空，此时读最新的轮转文件。
- 镜像内没有 shell（`docker exec godoxy-app sh ...` 会报 `executable file not found in $PATH`）。需要把文件取回本机时用 `docker cp` 或直接读宿主机 bind 目录，不要假设容器内可执行命令。
- 该访问日志含路径、query 与客户端 IP，按现有日志权限约定处理，不要贴进工单或日志检索。
- 业务成败仍以 `GET /api/logs` 与 request log 为准（见下文「Raw 通信证据」）；ingress 日志只能回答「请求有没有到达 GoDoxy、状态码和耗时是多少」。

## 发布

**移除可配置价格倍率的版本仍需检查数据格式边界。** 发布前必须单独决定并完成
schema / migration ledger、旧 receipt 与
idempotency 历史三项准备，详见
[冻结模型计价与部署边界](../.docs/tech/frozen-model-pricing.md)。应用只接受 v7 receipt，
不提供旧数据迁移、兼容层或自动清理；下面的通用发布流程不替代这些准备，也不授权
删除历史数据。累计费用与已配置限额默认保留，不默认重置限额或重算历史费用。
已明确退役的 `0009_price_multipliers` 由 `removedMigrationIDs` 排除出有效迁移链，
不需要删除或重写历史台账行；其他未知 ID 或缺失有效迁移仍拒绝启动。

```bash
scripts/deploy.sh dev --prepare-only               # 在线构建、上传，不切换生产容器
scripts/deploy.sh dev --activate-only <完整提交SHA> # 演练后，一次完成停写、最终备份和镜像切换
```

入口是本地仓库里的 `scripts/deploy.sh`。本机需要 Git、Docker Buildx、gzip、curl 和可连接 `vps-kl` 的 SSH；Docker 必须使用本机 Unix socket。BuildKit 和构建缓存留在本机，服务器不执行 Git 拉取、源码编译或镜像构建。

1. **固定版本与授权范围。** 核对 Hostinger SSH 目标、现有镜像、health、数据卷和持久磁盘空间。完成相关测试，只提交本次文件并推送到 `DabengBa/gpt-load` 的目标分支（默认 `dev`），不要夹带其他工作区改动。记录完整提交 SHA、目标标签、旧镜像和需要的修复 SQL；没有格式问题就不修数据。
2. **在线准备镜像。** 在本机运行 `--prepare-only`。脚本 fetch 目标分支，以 `git archive` 导出已推送提交，按服务器架构本地构建，再通过 `docker save | gzip -1 | ssh vps-kl 'docker load'` 上传。生产容器保持运行。记录脚本输出的提交与镜像标签；Hostinger 只加载镜像，不构建。
3. **在线快照演练。** 按下节取得一致 SQLite 快照和原密钥，在独立目录、隔离网络、无生产数据卷的目标镜像中演练启动、迁移和再次启动。检查台账、数据格式、认证管理 API 和真实加密凭据读取。演练失败就停止发布，保留生产服务；禁止把演练库覆盖回生产，快照后的新写入不能丢失。
4. **必要 SQL 的部署边界。** 常规发布不执行手工 SQL，不删除迁移台账，不转换旧 receipt、不重算费用。确有格式问题时先在副本验证已审阅 SQL，另行制定停写、备份、事务、验证和恢复的连续执行步骤；不能用普通发布脚本代替该操作。
5. **连续切换。** 旧容器保持运行，本机执行 `--activate-only <完整提交SHA>`。脚本在停机前确认远端提交未变化、目标镜像存在、生产挂载一致及备份空间充足，然后在同一个 Hostinger 脚本内依次停止旧容器、确认卷无运行中的容器写入者、完整备份并检查归档、切换 Compose 和启动目标镜像。必须事先确认无宿主机或其他外部写入者。禁止在另一次命令中预先停止服务；脚本会拒绝从停止状态开始。
6. **失败恢复与验证。** 切换前备份失败时，脚本自动重启源库未修改的旧容器，执行者须核对内外 health；切换开始后数据库可能已被新版修改，不自动回滚镜像或恢复数据，按「回滚」判断兼容性。脚本忽略 SSH HUP，避免普通连接断开中止已开始的远端步骤，但主机故障、强制杀进程仍需要人工恢复。成功后核对镜像和 health 的目标版本、容器健康和重启次数、认证管理 API、真实凭据解密读取及相关业务行为。

当前单实例停机会取消在途 HTTP/流式请求，不能承诺零停机。整个停写至启动期间不得插入人工确认、额外工具调用、代码修改或耗时在线准备。发布不执行 `VACUUM`。`scripts/deploy.sh [分支]` 同样执行连续切换，但不代替预先快照演练；`--prepare-only` 不停止生产。备份与源库同盘，只是本次恢复点，不能代替独立存储灾备。脚本不清理旧镜像或备份。

### 快照演练约束

- SQLite `.backup` 方法见「健康检查异常排障」。目录必须位于持久磁盘，权限为 `0700`，文件为 `0600`，空间还要覆盖工作副本和最终备份；不要使用宿主机 `/tmp` 的 tmpfs。
- 原快照保持不变，另建工作副本，复制原 `auth.key`、`encryption.key`。若原容器通过环境变量提供密钥，安全复用相同变量，不能生成新密钥；环境文件同样受限访问且不进入 Git 或日志。
- 演练容器使用独立名称、`--network none`、独立数据目录，不绑定生产端口，不挂载生产卷；保持目标镜像的运行 UID/GID 与副本权限匹配。禁用 Models.dev 自动同步，不允许探测、刷新或请求真实 Provider。通过 `docker exec` 访问容器内 health 和认证只读管理 API，完整响应与令牌不打印。
- 记录目标版本、有效台账、前后关键业务计数，以及两次启动与加密凭据读取结果。health 成功不等于数据兼容；retention 启动清理只影响副本，检查计数时按已配置保留期解释。
- `0009` 曾被人工删除的库，可以仅在**工作副本**重新插入该历史 ID，验证目标版本允许它保留并重复启动；这不是生产数据修复步骤。
- 演练结束停止并移除演练容器，清理含密钥的临时环境文件。原快照与最终备份按受限权限保留；切换前仍须确认磁盘空间。演练只是旧状态验证，不能替代停写后的最终备份和前置条件复查。

### 2026-10-03 发布中断记录

此次中断的直接原因是执行者将停机备份和镜像切换拆为两次命令：第一条命令于
03:29:29 UTC 停止旧服务，03:29:53 UTC 完成备份后返回，但没有继续执行激活命令。
旧容器于 03:29:39 UTC 退出，新镜像未在生产启动，Compose 始终指向 `dev-700d11a1`。
直到 04:44:38 UTC 重启旧容器，公网从 502 恢复 200；服务中断约 75 分钟。
退出日志同时显示 debug capture/request log 关闭超时，这是退出码 1 的原因，
不是新版本迁移失败的证据。此次未执行生产修复 SQL、未切换新版本，也未恢复旧备份。

修正：停机、最终备份和目标镜像启动由发布脚本连续执行；切换前失败自动启动旧容器。
不再提供独立停止服务的最终备份发布示例，避免照抄后把服务留在停止状态。
已发布的 `405f271f` 包含退役迁移跳过逻辑和回归测试，不删除历史台账行。

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
curl -fsS https://gptl.tanyaleoallen.cloud/health    # 期望 {"status":"ok","version":"dev-<短sha>"}
docker inspect gpt-load --format '{{.Config.Image}} {{.State.Health.Status}} {{.RestartCount}}'
docker logs --since 5m gpt-load 2>&1 | grep -icE 'error|fatal|panic'
```

对外健康检查走 443（无端口）。`https://gptl.tanyaleoallen.cloud:1443/health` 只在需要单独确认保留的回滚入口 Caddy 时才用。

## 健康检查异常排障

`unhealthy` 仅表示容器健康检查失败，不等于应用不可用，也不能单凭一次 `/health` 成功断言历史上没有中断。发布脚本报告异常时先核对镜像、启动时间、重启次数，以及本机与对外 `/health`；再读 Docker 的 healthcheck 和 daemon 报错：

```bash
ssh vps-kl 'docker inspect gpt-load --format "image={{.Config.Image}} started={{.State.StartedAt}} restarts={{.RestartCount}} health={{.State.Health.Status}} check={{json .Config.Healthcheck}}"'
ssh vps-kl 'curl -fsS --max-time 10 http://127.0.0.1:3001/health'
curl -fsS --max-time 10 https://gptl.tanyaleoallen.cloud/health
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

先判断数据库是否已变化。只有旧版本可以读取当前 schema 和数据时，才可只回滚镜像。
迁移事务失败时检查其已回滚；迁移成功而新应用失败时不能默认旧镜像兼容。
如需恢复数据库，先停止所有写入者，将当前状态单独归档，再恢复最终备份的整个数据目录及原密钥，
不要混用新旧 WAL/SHM；同时恢复匹配的旧 Compose。恢复备份会丢弃备份后写入，须明确批准该数据损失后执行。
下列命令仅适用于已确认可以只回滚镜像的情况：

```bash
ssh vps-kl
cd /opt/gpt-load
ls docker-compose.yml.bak-*                              # 每次发布都会留一份
cp docker-compose.yml.bak-<时间戳>-<旧分支>-<旧短sha> docker-compose.yml
docker compose -p gpt-load up -d --no-build --pull never   # 保持项目名与原数据卷
```

旧镜像不会自动清理，`docker images | grep gpt-load` 可直接看到待回滚的版本。上述镜像回滚不恢复 `gpt-load_gpt-load-data` 卷；恢复后仍须验证版本、内外 health 和真实加密配置读取。

## 注意

- 持久化数据与 `encryption.key` 必须一起备份，缺失密钥会导致已有渠道凭据无法解密。
- `docker-compose.yml.bak-*` 是回滚点，确认新版本稳定前不要删除。
- 发布后先看 `docker logs` 的错误计数和首屏 `/health`，再观察真实请求日志中的失败重试情况。
