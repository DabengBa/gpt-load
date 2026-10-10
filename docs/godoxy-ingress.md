# GPT-Load ingress：GoDoxy 直连方案（已切换）

本文描述把 `gptl.tanyaleoallen.cloud` 从「GoDoxy TCP → Caddy → GPT-Load」改为「GoDoxy 直接终止 TLS 并反代 GPT-Load」的**架构、配置片段、切换与回滚步骤**。

**状态：生产已直连，Caddy 已于 2026-10-10 移除。** 公网 443 由 GoDoxy 终止 TLS 并直接反代 GPT-Load；`gptl-proxy` 容器及 `/opt/gptl-proxy` 启动配置已删除，TCP/UDP 1443 均无监听。配置与旧证书备份仍保留，详见 6.5/6.6。仓库保留配置片段（[`deploy/godoxy/`](../deploy/godoxy/)）与隔离验证探针（[`scripts/godoxy-ingress-probe.py`](../scripts/godoxy-ingress-probe.py)）。发布流程与 ingress 观测入口见 [`docs/deployment.md`](deployment.md)。

**切换执行时间线（UTC，服务器时区即 UTC）**：

| 时刻 (UTC) | 事件 | 证据 |
|---|---|---|
| 2026-09-28 23:06:57 | 切换后 443 所用证书的有效期起点（不等同于配置变更或路由切换时间） | 443 证书 `notBefore=Sep 28 23:06:57 2026 GMT`，issuer `Let's Encrypt YE1` |
| 2026-09-29 00:05:08 | §3.2 整目录备份 | 宿主机 `/www/server/panel/data/compose/godoxy/config.pre-gptl-20260929T000508Z` |
| 00:05–00:07 | §6.3 原地改写 `vhosts.yml` 条目，watcher 热重载 | `godoxy-app` `started=2026-09-19T20:01:44Z restarts=0`，切换全程未重启容器 |
| 00:07:28–00:07:38 | §6.4 Alt-Svc 排空：`gptl-proxy` 重启套用 `protocols h1 h2` + `Alt-Svc: clear` | `gptl-proxy` `started=2026-09-29T00:07:38Z`；Caddy 日志 `serving initial configuration` 后出现 `gptl-ingress-cutover-probe/1.0` 的 `/health` |
| 00:08:25 | 首条真实业务请求经新路径成功 | `entrypoint.log`：`POST /v1/responses` 200（客户端 IP 不入库） |

切换前最后一个 404 出现在 2026-09-28 14:28:06 +0800（`GET /health` → 404，路由当时仍是 TCP 透传），与此时间线一致。上述为 9 月切换历史；10 月 10 日 Caddy 移除记录见 6.6。

## 1. 切换前拓扑（历史）

```text
公网 :443 ── GoDoxy(godoxy-app, host network) ── SNI 匹配 gptl.* → TCP 原样转发 ──► 127.0.0.1:1443
                                                                        Caddy(gptl-proxy) 在此终止 TLS
                                                                              │
                                                                              ▼
                                                              GPT-Load 127.0.0.1:3001（仅回环发布）
```

- GoDoxy 侧的入口是一条 **stream route**，不是 HTTP route：`config/vhosts.yml` 中
  `gptl.tanyaleoallen.cloud: {scheme: tcp, host: 127.0.0.1, port: 443:1443}`。
  端口语法是 `[监听端口:]目标端口`（GoDoxy `internal/route/port.go`），即监听 443、转发到 1443。
  GoDoxy 的共享 HTTPS 监听器按 SNI 分流：命中的 stream route 走原始 TCP 透传，未命中的走 GoDoxy 自身 TLS 终止（`internal/entrypoint/sni_passthrough.go`）。
- Caddy：容器 `gptl-proxy`（`caddy:2-alpine`，host network），配置 `/opt/gptl-proxy/Caddyfile`，证书/私钥来自只读挂载的 `/acme`；`admin off`、`auto_https off`，只做 `reverse_proxy 127.0.0.1:3001`。
- 应用：Compose 项目 `/opt/gpt-load`，容器 `gpt-load` 只发布 `127.0.0.1:3001`，见 [`docs/deployment.md`](deployment.md)。

实测（2026-09-28，证据（本地 `../.tmp/godoxy-ingress_20260928/units/U004/evidence/public-probe.txt`））：

| 请求 | 结果 |
|---|---|
| `https://gptl.tanyaleoallen.cloud/health`（443） | 200，`via: 1.1 Caddy`，`alt-svc: h3=":1443"; ma=2592000` |
| `https://gptl.tanyaleoallen.cloud:1443/health` | 200，同样 `via: 1.1 Caddy` |
| `http://gptl.tanyaleoallen.cloud/health`（80） | **404**（GoDoxy 未匹配到 HTTP 路由，未重定向） |
| 443 与 1443 的证书 | 同一张 `CN=gptl.tanyaleoallen.cloud`，`issuer Let's Encrypt YE1`，`notAfter Dec 2 2026`（证据（本地 `../.tmp/godoxy-ingress_20260928/units/U004/evidence/tls-cert-probe.txt`）） |

证书所有权：切换前证书由 Caddy 侧的外部 ACME 流程管理（`/acme/gptl.tanyaleoallen.cloud_ecc/`，只读挂载），GoDoxy 的 autocert 里没有这个域名。切换后由 GoDoxy autocert（provider `hostinger`，DNS-01）管理；Caddy 容器已删除，`/acme` 旧证书仅保留用于恢复，不参与当前入口。

## 2. 切换后拓扑（当前实际）

```text
公网 :443 ── GoDoxy 共享 HTTPS 监听器 ── 无 stream 命中 → GoDoxy TLS 终止（autocert 证书）
                                              │
                                              ▼
                        entrypoint 全局 middleware → HTTP route gptl.* → 127.0.0.1:3001
公网 :80  ── GoDoxy HTTP entrypoint ── 同一 HTTP route ── RedirectHTTP 中间件 → 308 https
```

Caddy 与 `:1443` 已退出运行态。以下为切换当日实测（2026-09-29），其中 1443 的可用性和证书仅是历史结果，不代表当前端点可用：

| 请求 | 结果 |
|---|---|
| `https://gptl.tanyaleoallen.cloud/health`（443） | 200，**无** `via: 1.1 Caddy`，`alt-svc: h3=":443"; ma=2592000`，含 4 个安全头 |
| `https://gptl.tanyaleoallen.cloud:1443/health` | 200，`via: 1.1 Caddy`，`alt-svc: clear` |
| `http://gptl.tanyaleoallen.cloud/health`（80） | **308** → `https://gptl.tanyaleoallen.cloud/health` |
| 443 证书 | `CN=gptl.tanyaleoallen.cloud`，`issuer Let's Encrypt YE1`，`notBefore=Sep 28 23:06:57 2026 GMT`，`notAfter=Dec 27 23:06:56 2026 GMT` |
| 1443 证书 | 仍是 Caddy 侧旧证书，`notAfter=Dec 2 2026` |

## 3. 配置片段与注册

片段文件（由 U001 交付，内容为唯一权威）：

- [`deploy/godoxy/gpt-load-route.yml`](../deploy/godoxy/gpt-load-route.yml)

  ```yaml
  gptl.tanyaleoallen.cloud:
    scheme: http
    host: 127.0.0.1
    port: 3001
    response_header_timeout: 2h
  ```

- [`deploy/godoxy/autocert-extra.yml`](../deploy/godoxy/autocert-extra.yml)

  ```yaml
  autocert:
    extra:
      - domains:
          - gptl.tanyaleoallen.cloud
        cert_path: /app/certs/gptl.tanyaleoallen.cloud.crt
        key_path: /app/certs/gptl.tanyaleoallen.cloud.key
  ```

### 3.1 路由片段怎么注册

- 路由片段由 `config/config.yml` 的 `providers.include` 注册（`internal/config/types/config.go`：`Providers.Files` → `include`），条目是**相对于 `config/` 目录的文件名**（`internal/route/provider/file.go`：`path.Join(common.ConfigBasePath, filename)`）。当前值：`providers.include: ['hub.yml', 'vhosts.yml']`（证据（本地 `../.tmp/godoxy-ingress_20260928/units/U004/evidence/server-config-facts.txt`））。
- 文件变更会被目录 watcher 捕获并热重载（`internal/route/provider/file.go` → `watcher.NewConfigFileWatcher`；`config/config.yml` 自身由 `internal/config/events.go:52`（`WatchChanges`）与 `:67-91`（`OnConfigChange`，rename/delete 分支在 78/81）监听并 `Reload()`）。**重命名或删除 `config.yml` 不会触发重载**，所以一律做原地编辑，不要用「改名再改回」的方式。
- **同一域名只能有一个定义。** `gptl.tanyaleoallen.cloud` 已经存在于 `vhosts.yml`；如果再从 `include` 引入一个同名 key，第二条会在入口注册时报 `route already exists: from provider … and …`（`internal/route/route.go:624-646`）。
- 因此本域名的切换只有两种可选形式，二选一，且**必须与移除旧 TCP 条目在同一次重载中完成**：
  1. **推荐：原地改写 `vhosts.yml` 中的 `gptl.tanyaleoallen.cloud` 条目**，内容取 `deploy/godoxy/gpt-load-route.yml`（单文件、单次写入、单次重载，不存在中间态）。
  2. 备选：把 `gpt-load-route.yml` 复制为 `config/gpt-load.yml` 并在 `providers.include` 追加 `gpt-load.yml`，**同时**从 `vhosts.yml` 删除该 key。两次写入之间会短暂出现「重复 key」或「无路由」的中间态，写入前先把两个文件准备好，顺序为：先追加 include，再删 vhosts 条目；一旦日志出现 `route already exists`，立即回滚 include。

### 3.2 autocert 合并（保留既有条目与凭据）

`autocert.extra` 是**追加式**列表：把片段中的那一项追加到现有 `extra` 之后，**不要**重写 `provider`、`email`、`domains`、`options`，也不要删改已有的 5 个 extra（hub/freshrss/st/sttest/sy）。`options` 内是 Hostinger API 凭据，任何验证命令都不要打印它。

合并依据（GoDoxy `internal/autocert/config.go`）：

- `MergeExtraConfig`（`:304-344`）让 extra 继承主配置的 provider/email/options/resolvers，只覆盖 `domains`、`cert_path`、`key_path`；**extra 与主配置共用同一个 ACME key**（源码注释 `Using same ACME key as main provider`）。
- `validate`（`:109-124`）要求 `cert_path`、`key_path` 全局唯一；片段使用 `/app/certs/gptl.tanyaleoallen.cloud.{crt,key}`，与现有 6 组路径不冲突（U001 已核对）。既有 extra 用相对路径 `certs/<name>.crt`，相对路径按容器工作目录解析到同一持久卷 `/app/certs`，两种写法等价，但同名必须唯一。
- extra 同样需要 `domains` 与 `email`；片段省略 `email` 正是靠继承主配置。

操作顺序（切换时执行；**已于 2026-09-29 00:05:08 UTC 执行**，产出备份 `/www/server/panel/data/compose/godoxy/config.pre-gptl-20260929T000508Z`）：

```bash
ssh vps-kl 'config=/www/server/panel/data/compose/godoxy/config; stamp=$(date -u +%Y%m%dT%H%M%SZ); backup="${config}.pre-gptl-${stamp}"; cp -a "$config" "$backup" && test -f "$backup/config.yml" && test -f "$backup/vhosts.yml" && printf "backup=%s\n" "$backup"'
# 用编辑器在 autocert.extra 列表末尾追加片段中的那一项（保持其余行原样，避免重排丢失凭据/注释）
```

这是整个 GoDoxy `config/` 目录的恢复点，包含 `config.yml`、`vhosts.yml` 和被 include 的配置文件；其中含有凭据，保持原有权限，只留在服务器上，不要下载、提交或贴入日志。记录命令输出的备份路径。回滚时恢复所有本次改动过的文件，而不是只删新加的 autocert 项；文件应原地覆盖，避免替换/重命名整个目录或删除 `config.yml`（其 watcher 对删除和重命名不会触发 reload）。若采用 include 备选方案，也须移除本次新增且备份中不存在的 route 文件。

合并后的只读校验（不打印 `options`）：

```bash
ssh vps-kl 'python3 - <<"PY"
import yaml
c = yaml.safe_load(open("/www/server/panel/data/compose/godoxy/config/config.yml"))
ac = c["autocert"]
extras = ac.get("extra") or []
assert ac["provider"] == "hostinger", ac["provider"]
assert ac["domains"] == ["godoxy.tanyaleoallen.cloud"], ac["domains"]
assert len(extras) == 6, len(extras)
last = extras[-1]
assert last["domains"] == ["gptl.tanyaleoallen.cloud"], last
paths = [e.get("cert_path") for e in extras] + [e.get("key_path") for e in extras]
assert len(paths) == len(set(paths)), "路径重复"
print("autocert ok:", len(extras), "extras;", last["cert_path"])
PY'
```

## 4. `response_header_timeout: 2h` 的依据与边界

GoDoxy HTTP transport 默认 `ResponseHeaderTimeout` 为 60s，路由上正数的 `response_header_timeout` 覆盖该字段（`internal/types/http_config.go`、`internal/routeimpl/reverse_proxy.go`）。2h 不是随手取的值，而是按当前生产设置推出来的：

- U001 线上只读读取到：全局 `request_timeout=600s`、`first_byte_timeout=600s`、`stream_idle_timeout=300s`、系统 `retry_count=10`，54 个分组的有效 `request_timeout` 全部 600s（U001 result（本地 `../.tmp/godoxy-ingress_20260928/units/U001/result.md`））。
- `retry_count` 是**总尝试预算**（含跨分组/跨渠道切换），不是「首次之后再试 10 次」。非缓冲（非流式）路径只有单次 attempt 超时、没有整体 deadline，因此最坏 `10 × 600s = 100 分钟` 才会返回响应头；2h = 100 分钟 + 20 分钟余量。
- 缓冲流式路径把第一个选中分组的 `request_timeout` 冻结为整条请求的 deadline（当前 600s）；GoDoxy 收到响应头即 flush，之后响应头计时不再限制流式正文（`goutils/http/reverseproxy/reverse_proxy.go`）。
- 网关循环没有显式 sleep/backoff，`Retry-After` 只影响冷却资格，不会让当前请求等待。

**边界（必须记录，不能当成通用保证）**：`request_timeout` 的合法上限是 Go duration 范围，系统 `retry_count` 也没有小的产品级上限。任一被调大到超出 2h 预算，GPT-Load 就可能晚于 ingress 的响应头预算；那时应同步上调该路由超时，而不是默认它仍然安全。调参属于应用侧配置，需重新评估本文档的推导。

隔离验证（U003，同一延迟后端、并行发起）：默认 60s 的对照路由在 **60.120s** 返回 502，`response_header_timeout: 2h` 的目标路由在 **65.118s** 返回 200，客户端截止 75s（证据（本地 `../.tmp/godoxy-ingress_20260928/units/U003/result.md`））。

## 5. 全局 middleware 与访问日志的影响

生产 `config.yml` 的 `entrypoint`（证据（本地 `../.tmp/godoxy-ingress_20260928/units/U004/evidence/server-config-facts.txt`））：

- `middlewares: [ModifyResponse(设置 Referrer-Policy、Strict-Transport-Security、X-Content-Type-Options、X-Frame-Options), RedirectHTTP]`
- `access_log: {format: combined, path: /app/logs/entrypoint.log, stdout: false, keep: 30 days}`

**切换前这些都不作用于本域名**（GPT-Load 流量以原始 TCP 从 443 透传给 Caddy，GoDoxy 不解析 HTTP）。切换后已全部生效，以下 5 条的实际观察结果：

1. 响应多出上述 4 个安全头 —— **已确认**（443 的 200 响应带 `Referrer-Policy: same-origin`、`Strict-Transport-Security: max-age=63072000; includeSubDomains; preload`、`X-Content-Type-Options: nosniff`、`X-Frame-Options: SAMEORIGIN`）。
2. `http://` 请求从 404 变为 **308** 跳转到 https（`internal/net/gphttp/middleware/redirect_http.go`，`http.StatusPermanentRedirect`）—— **已确认**。
3. `via: 1.1 Caddy` 头消失，`alt-svc: h3=":1443"` 不再出现；443 现在广告 `alt-svc: h3=":443"; ma=2592000` —— **已确认**。
4. 每条请求写入 `/app/logs/entrypoint.log`（combined 格式，保留 30 天，按小时轮转）—— **已确认**。该文件经 bind mount 落在宿主机 `/www/server/panel/data/compose/godoxy/logs/`；`stdout: false` 意味着**这些行不在 `docker logs godoxy-app` 里**。日志含路径、query 与客户端 IP，按现有访问日志的权限约定处理，不要随手贴到工单/日志检索里。观测入口见 [`docs/deployment.md`](deployment.md) 的「ingress 观测」一节。
5. 中间件链对流式响应的正文改写受 `canBufferAndModifyResponseBody` 限制（`internal/net/gphttp/middleware/middleware.go`）。已观察到 HTTP 200 与 `request_logs` 中 `stream=true` 的成功记录，但状态码、响应大小和请求完成记录**不足以证明客户端在 EOF 前收到增量事件、SSE 未被整体缓冲或取消传播**。这些项目仍待按 6.4 清单完成真实生产验收。

## 6. 切换、验证与回滚（Caddy 已移除）

### 6.1 前置条件

- 生产域名 cutover 已于 2026-09-29 执行；用户于 2026-10-10 明确授权提前停用，在确认调用无影响后授权删除 Caddy，见 6.6。
- 按 3.2 在服务器上备份整个 GoDoxy `config/` 目录；确认备份路径可读、权限仍受保护，并准备好按 6.5 原地恢复全部改动文件。单个文件备份不足以代表共享入口的已知良好配置。
- 确认 `/app/certs` 对应的持久卷 `godoxy_godoxy-certs` 可写，Hostinger DNS 凭据仍有效。证书检查从宿主机通过 Docker named-volume 的实际 mountpoint 读取，不假定应用镜像含 shell 或 OpenSSL。
- GoDoxy 是多个 HTTPS vhost 共用的入口。ACME 准备阶段也可能影响其他 GoDoxy TLS 服务；恢复 Caddy 只能提供 GPT-Load 旁路，不能替代其他域名的检查。

### 6.2 第一步：先出证书，再动路由（已执行）

**执行结果**：切换后 443 使用 GoDoxy autocert 管理的证书（`notBefore=Sep 28 23:06:57 2026 GMT`，`notAfter=Dec 27 2026`）。证书有效期起点不证明 autocert 配置变更的执行时刻；00:05:08 UTC 的整目录备份仍不含本域名的 autocert 配置，且保留旧 TCP 路由。下述风险分析保留作为方法与教训。

顺序不能颠倒，但不能把 ACME 错误等同于 reload 被拒绝：`internal/config/state.go:138-159` 把 autocert 归为 optional component，初始化错误记为 `IssueDegraded`；`IssueDegraded.IsFailure()` 为 false（`internal/config/types/lifecycle.go:37-40`）。`RuntimeManager.transition` 只因 rejecting/failure issue 等原因拒绝候选；否则会停止旧 runtime 并 commit 新候选（`internal/config/runtime_manager.go:183-226`）。源码注释明确 activation failure 不会恢复旧配置（`:108-115`）。`state.initAutoCert` 只有在证书获取成功后才设置 provider（`internal/config/state.go:568-576`），而 HTTPS server 只在证书可用时创建（`goutils/server/server.go:90-105`）。因此 DNS-01/ACME 失败可能以 **committed + degraded** 结束，而非保留旧 runtime；其他 GoDoxy TLS vhost 也可能受影响。旧 GPT-Load TCP→Caddy route 尚未改动，并不能证明生产整体未受影响。

1. 按 3.2 完成整目录备份，再合并 autocert extra，原地保存 `config.yml`。
2. 等待 watcher 完成 reload（约 500ms 合并窗口，见 `internal/config/events.go`），检查**最新一次**结果。只有 `config committed: healthy` 且无 degraded/failed issue 才继续；`config committed: degraded`、`failed` 或 `rejected` 都要停下。特别是 committed-degraded 时，立即按 6.5 恢复整个已知良好的配置并验证，不要假设旧 runtime 仍在运行。即使结果为 healthy，也要逐一检查当前所有 GoDoxy TLS vhost 的 HTTPS 健康与证书，不能只检查 GPT-Load。
3. 只有 reload healthy 且既有 TLS vhost 检查通过，才检查新证书文件是否落地：

```bash
ssh vps-kl 'cert_dir=$(docker volume inspect --format "{{.Mountpoint}}" godoxy_godoxy-certs) && ls -l "$cert_dir/gptl.tanyaleoallen.cloud.crt" && openssl x509 -in "$cert_dir/gptl.tanyaleoallen.cloud.crt" -noout -subject -issuer -dates'
```

命令在 Docker 宿主机解析 `godoxy_godoxy-certs` 的实际 mountpoint，`openssl` 只读取并输出公开证书信息；不读取或输出私钥。若证书不存在、issuer/有效期不符、reload 非 healthy，或已有 TLS vhost 异常，**不要切路由**；恢复整个配置快照并按 6.5 验证运行态。单纯保持 GPT-Load 的旧 TCP 条目不足以保证其余 GoDoxy HTTPS 服务无影响。

> 本次开发只用**测试 CA**在隔离命名空间证明了「证书校验 + SNI/主机名校验」这条链路（U003），**不能**替代生产 ACME 签发/续期证据；上面这段是生产必须自己证明的部分。

### 6.3 第二步：切路由（已执行）

按 3.1 **原地改写 `vhosts.yml` 条目**（推荐形式）执行，保存后热重载，`godoxy-app` 全程未重启（`started=2026-09-19T20:01:44Z restarts=0`）。

### 6.4 第三步：验证（公网、只读）

下列 1–5 项在 2026-09-29 切换后均已复核通过（443 期望值见本文第 2 节的实测表）；命令保留为可重复执行的验收清单。

```bash
# 1) 发布健康 URL：公网 443，无需端口
curl -fsS --max-time 10 https://gptl.tanyaleoallen.cloud/health
# 期望 {"status":"ok","version":"dev-<短sha>"}

# 2) 证书由 GoDoxy 提供、域名匹配
echo | openssl s_client -servername gptl.tanyaleoallen.cloud -connect gptl.tanyaleoallen.cloud:443 2>/dev/null \
  | openssl x509 -noout -subject -issuer -dates

# 3) 中间件与头变化
curl -sS -D - -o /dev/null https://gptl.tanyaleoallen.cloud/health | grep -iE 'via|alt-svc|x-frame-options|strict-transport-security'
# 期望：不再有 via: 1.1 Caddy 与 alt-svc h3=":1443"，安全头出现

# 4) HTTP 从 404 变 308
curl -sS -o /dev/null -D - http://gptl.tanyaleoallen.cloud/health | head -1   # HTTP/1.1 308 Permanent Redirect

# 5) 本机回环与容器健康（不经过 ingress）
ssh vps-kl 'curl -fsS --max-time 10 http://127.0.0.1:3001/health'
ssh vps-kl 'docker inspect gpt-load --format "{{.Config.Image}} {{.State.Health.Status}} {{.RestartCount}}"'
```

**主入口健康检查已统一为 443**：部署文档与发布脚本使用无端口的 `https://gptl.tanyaleoallen.cloud/health`，回环探测保持 `http://127.0.0.1:3001/health`。1443 已关闭，以下排空配置与探测命令仅记录历史，不应作为当前验证步骤执行。

**Alt-Svc / 1443 排空（历史方案，已提前结束保留）**：切换前 Caddy 发布 `h3=":1443"; ma=2592000`。2026-09-29 已禁用 H3 并返回 `Alt-Svc: clear`；原定 30 天保留窗口未等满，用户于 2026-10-10 授权提前停用和删除。以下是当时的配置，并非现网运行配置：

```caddyfile
{
  admin off
  auto_https off
  servers {
    protocols h1 h2
  }
}

gptl.tanyaleoallen.cloud:1443 {
  tls /acme/gptl.tanyaleoallen.cloud_ecc/fullchain.cer /acme/gptl.tanyaleoallen.cloud_ecc/gptl.tanyaleoallen.cloud.key
  header >Alt-Svc clear
  reverse_proxy 127.0.0.1:3001
}
```

`protocols h1 h2` 禁用此 Caddy HTTP server 的 HTTP/3；`header >Alt-Svc clear` 在响应写出时设置 RFC 7838 的 `Alt-Svc: clear`，清除收到它的客户端为该 origin 缓存的 alternatives。**已缓存 `h3=":1443"` 的客户端不能通过已禁用的 HTTP/3 收到 clear**；它们需要回退到 origin 的可用 HTTP/1.1 或 HTTP/2 连接，期间可能出现失败尝试或延迟。路由仍指向 Caddy 时，该回退响应可携带 clear；切到 GoDoxy 后，不能假定其响应含 clear，也不能保证所有客户端已收到清除头，因此仍需以下缓存等待与流量观测。直接请求 `https://域名:1443` 的 origin 与默认 443 不同，该探测只核验旧端点响应头，不能证明默认 443 origin 的客户端缓存已清除。当前 Caddy 配置为 `admin off`，不能用 `caddy reload`；保存后先执行 `ssh vps-kl 'docker exec gptl-proxy caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile'`，再经单独授权执行 `ssh vps-kl 'docker restart gptl-proxy'` 使配置生效。重启会短暂中断仍经 Caddy 的 GPT-Load 流量。确认容器恢复后、且路由仍指向 Caddy 时，从公网分别探测默认 443 和 `:1443`：

```bash
for url in https://gptl.tanyaleoallen.cloud/health https://gptl.tanyaleoallen.cloud:1443/health; do
  printf '%s: ' "$url"
  curl -sS -D - -o /dev/null "$url" | grep -i '^alt-svc:'
done
```

切换后，1443 的 HTTP/1.1、HTTP/2 响应须返回 `Alt-Svc: clear`；443 可广告 `h3=":443"`，两处均不得再广告 `h3=":1443"`，确认后记录 UTC 时刻。**30 天排空期从 443 与 1443 最后一次可能发出正向 `h3=":1443"` 广告的时刻起算**，若之后再次观察到该广告则从最后一次重新计时。保持 Caddy 的 TCP 1443（HTTP/1.1、HTTP/2）可用，作为过渡与回滚入口；此时 UDP/HTTP/3 1443 已停用，不能把保留 TCP 服务表述为旧 H3 alternative 仍可用。

**原排空计划（历史）**：停止正向广告已于 2026-09-29 00:07:40 UTC 确认，原最早评估时间为 2026-10-29 00:07:40 UTC。以下是保留期的判断条件，不代表这些条件在提前删除前已全部满足：

- 1443 只返回 `alt-svc: clear`（2026-09-29 实测），不再有 `h3=":1443"`；
- 443 由 GoDoxy 应答，`alt-svc: h3=":443"; ma=2592000`，**没有**任何 `:1443` 正向广告；
- `ma=2592000` 是 30 天的新连接缓存有效期；RFC 7838 说明到期限制的是建立新连接，已经存在的 alternative connection 仍可能继续使用。因此，等满 30 天本身不证明没有活跃客户端。
- 排空期间使用能区分真实客户端与运维探测的 1443 请求/连接遥测。Caddyfile 已配置 JSON access log（`output stdout`），但**切换后 443 流量不再进入 Caddy**，该日志只剩 1443 与运维探测，不能单独用来证明客户端缺席。没有可信遥测时不得宣称已排空，继续保留 Caddy。
- 移除前要求完整 30 天无正向广告、无未结束的 1443 连接且观测窗口内无真实客户端请求；手工 `curl ...:1443/health` 只证明端点可用，**不能**证明没有其他客户端访问。

**真实 AI 流式/非流式验收边界**：新路径已有成功流式记录，用户在停用 Caddy 后确认调用无影响；这不替代 SSE 增量接收、未缓冲、取消传播及逐 attempt raw capture 的完整验收。此次移除没有新增这些项目的验证证据，不宣称全部通过。

### 6.5 恢复旧入口（需先重建 Caddy）

Caddy 已不存在，不能直接运行 `docker start gptl-proxy` 或只恢复 GoDoxy 的旧 TCP 路由。必须先取得新的恢复授权，将 `/opt/gptl-proxy-retired-20261010T065651Z` 中的配置恢复到原路径，核对 Compose、镜像及 `/acme` 证书仍有效，启动 Caddy 并验证 1443。保持 H1/H2 与 `Alt-Svc: clear`，不要重新广告旧 H3 端口。若证书过期，必须先解决证书问题。

只有 Caddy 旁路通过后才能考虑恢复旧 GoDoxy 路由。9 月整目录快照早于后续项目更新，不能盲目覆盖当前共享入口配置；先备份现网并审核其他域名的差异。以下为原切换的恢复方法，不是可直接照抄的当前回滚指令：

回滚要恢复此次改动过的**全部 GoDoxy 配置文件**，而非只删除 autocert extra。暂停其他配置编辑后，使用 3.2 记录的整目录快照 `/www/server/panel/data/compose/godoxy/config.pre-gptl-20260929T000508Z`；先用 `diff -qr "$backup" "$config"` 列出快照与当前目录的差异（只列路径，不打印配置内容），据此恢复所有被改动的旧文件，并仅删除此次新增且快照中没有的文件。本流程至少原地恢复 `vhosts.yml` 与 `config.yml`；若使用过 include 备选，也删除本次新增 route 文件。先恢复 route/provider 文件，最后恢复 `config.yml` 以触发完整 reload；不要 rename/delete `config.yml` 或替换整个配置目录：

```bash
ssh vps-kl 'backup=/www/server/panel/data/compose/godoxy/config.pre-gptl-<UTC时间戳>; config=/www/server/panel/data/compose/godoxy/config; cp -p "$backup/vhosts.yml" "$config/vhosts.yml" && cp -p "$backup/config.yml" "$config/config.yml"'
```

恢复配置文件只会触发新的候选 reload，不会自动恢复旧 runtime。核实 `config committed: healthy`，确认全部现有 HTTPS vhost、证书及已重建的 Caddy 路径正常后，才把恢复视为完成；若仍 degraded/failed，停止进一步切换并按生产事故流程处理。不要删除 GoDoxy 证书或重新启用旧 H3 正向广告。当前仅保留 Caddy 配置备份和 `/acme` 旧证书，并无运行中的 Caddy 容器。

### 6.6 Caddy 移除记录（2026-10-10）

原计划要求以下条件全部满足后再移除；本次用户明确授权提前结束保留，不能据此宣称原排空与验收已全部完成：

1. 6.4 的 1–5 全部通过；
2. 真实流式/非流式 AI 请求验收通过（6.4 末尾，含取消传播）；
3. Caddy 停止旧 H3 广告并返回 `Alt-Svc: clear` 后至少保留 30 天（不保证全部客户端收到 clear）：原确认时刻为 **2026-09-29 00:07:40 UTC**，最早评估时间为 **2026-10-29 00:07:40 UTC**；
4. 可信流量/连接遥测显示排空期内无真实客户端请求且无存续的 1443 连接。手工健康检查不计作客户端缺席证据；没有可信遥测时不得移除；
5. 主入口健康 URL（`docs/deployment.md` 与 `scripts/deploy.sh`）已改为无端口 443 形式，回环探测不受影响，并验证过一次发布。—— **URL 已修正；完整发布验证待执行**。

实际执行：

1. 2026-10-10 检查时，GoDoxy 直连 443 正常，1443 无已建立连接。Caddy 容器于 10 月 6 日重新创建，其保留日志中无访问记录；该证据不覆盖完整 30 天窗口。
2. 按用户授权停止 `gptl-proxy`，确认 TCP/UDP 1443 无监听、443 health 为 200、HTTP 为 308；用户随后确认调用无影响并授权删除。
3. 06:56:51 UTC 备份到 `/opt/gptl-proxy-retired-20261010T065651Z` 后删除容器及 `/opt/gptl-proxy` 启动配置。`/acme` 旧证书、GoDoxy 配置及 GPT-Load 数据未动。
4. 删除后 GPT-Load `dev-b6f03372` 为 `healthy`、重启次数 0，443 health 正常、HTTP 308，1443 无监听。

旧 `:1443` 地址不再可用；正常 443 调用不依赖 Caddy。提前移除不证明所有客户端的旧 Alt-Svc 缓存已清除，后续若出现旧端口依赖需按 6.5 单独评估恢复。

## 7. 隔离验证结论与不成立的结论

U003 在 Hostinger 上用**同一个生产镜像**（`sha256:3edf8b84…`，revision `78be6afb`）在隔离命名空间做了完整验证（U003 result（本地 `../.tmp/godoxy-ingress_20260928/units/U003/result.md`））：

- 首轮运行通过功能断言，但无法排除 Docker resolver 的上游转发，因此**保留该顾虑**；
- 第二轮在启动前架设命名空间级 IPv4/IPv6 防火墙并做 53 端口抓包：pcap **0 包**，DNS 与非回环出站计数全 0，run-owned resolver 只指向命名空间回环；该轮为其捕获窗口提供了可接受的独立证据，**不能消除首轮是否发生上游 DNS 转发的历史不确定性**；
- 默认 60s 超时对照失败 / `2h` 成功、TLS 测试 CA 与主机名校验、SSE 事件先于 EOF、取消后后端观察到断连、非 2xx 状态/头/正文透传、单次 UDP 往返均通过；真实 `127.0.0.1:3001/health` 200（补充项，未经过隔离入口）；
- 全部使用 run-owned 合成后端，**没有任何 AI 供应商请求**；测试 CA 只证明 TLS 校验，不证明生产 ACME 签发/续期；生产容器/路由/配置前后未变。隔离验证的结论**不因生产已切换而改变**。

## 8. 证据索引

下列 `.tmp/` 文件为本次工作树保留的本地运行记录，不随 Git 提交发布；克隆仓库后不会包含这些文件。可共享的验证结论见第 7 节，探针与测试脚本随仓库提供。

生产切换（2026-09-29 00:05–00:08 UTC）的证据取自服务器现网，可随时复核：

```bash
# 整目录备份
ssh vps-kl 'ls -ld /www/server/panel/data/compose/godoxy/config.pre-gptl-*'
# 当前容器状态；Caddy 已删除，不能再 inspect gptl-proxy
ssh vps-kl 'docker inspect godoxy-app --format "{{.Name}} started={{.State.StartedAt}} restarts={{.RestartCount}}"'
# 443 证书由 GoDoxy autocert 管理；1443 已关闭
echo | openssl s_client -servername gptl.tanyaleoallen.cloud -connect gptl.tanyaleoallen.cloud:443 2>/dev/null \
  | openssl x509 -noout -subject -issuer -dates
# 切换前最后一个 404 与切换后首个 200
ssh vps-kl 'grep -h "gptl.tanyaleoallen.cloud" /www/server/panel/data/compose/godoxy/logs/entrypoint.log.* | grep -E " 404 |POST /v1" | head -2'
```

- 本单元证据：`../.tmp/godoxy-ingress_20260928/units/U004/evidence/`（本地 `../.tmp/godoxy-ingress_20260928/units/U004/evidence/`）
  - `public-probe.txt`：443/1443/80 实测头与正文
  - `tls-cert-probe.txt`：443/1443 证书 subject/issuer/有效期
  - `server-config-facts.txt`：生产 `config.yml` 的 include、entrypoint middleware/access_log、autocert 概要与 `vhosts.yml`（未输出任何凭据）
  - `workspace-state.txt`：本工作树与主工作树待合并改动
- 上游单元：U001 超时与片段（本地 `../.tmp/godoxy-ingress_20260928/units/U001/result.md`）、U002 探针（本地 `../.tmp/godoxy-ingress_20260928/units/U002/result.md`）、U003 隔离运行（本地 `../.tmp/godoxy-ingress_20260928/units/U003/result.md`）
- GoDoxy 源码（revision `78be6afb`，本地 `/tmp/godoxy-inspect-78be6afb`）：`internal/route/port.go`、`internal/route/route.go`、`internal/route/provider/file.go`、`internal/config/types/config.go`、`internal/config/events.go`、`internal/config/state.go`、`internal/types/http_config.go`、`internal/routeimpl/reverse_proxy.go`、`internal/autocert/config.go`、`internal/autocert/provider.go`、`internal/entrypoint/sni_passthrough.go`、`internal/net/gphttp/middleware/redirect_http.go`、`internal/net/gphttp/middleware/middleware.go`、`goutils/http/reverseproxy/reverse_proxy.go`、`goutils/server/server.go`
