# GPT-Load ingress：GoDoxy 直连方案（目标态，未切换）

本文描述把 `gptl.tanyaleoallen.cloud` 从「GoDoxy TCP → Caddy → GPT-Load」改为「GoDoxy 直接终止 TLS 并反代 GPT-Load」的**目标架构、配置片段、切换与回滚步骤**。

**状态：生产尚未切换。** 截至本文写作时，公网流量仍走 Caddy；仓库里只有配置片段（[`deploy/godoxy/`](../deploy/godoxy/)）、隔离验证探针（[`scripts/godoxy-ingress-probe.py`](../scripts/godoxy-ingress-probe.py)）与隔离运行证据。发布流程本身见 [`docs/deployment.md`](deployment.md)。

## 1. 当前实际拓扑

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

证书所有权：当前证书由 Caddy 侧的外部 ACME 流程管理（`/acme/gptl.tanyaleoallen.cloud_ecc/`，只读挂载），GoDoxy 的 autocert 里**没有**这个域名。切换后证书改由 GoDoxy autocert（provider `hostinger`，DNS-01）签发与续期，`/acme` 与 Caddy 容器随之失去作用。

## 2. 目标拓扑（未部署）

```text
公网 :443 ── GoDoxy 共享 HTTPS 监听器 ── 无 stream 命中 → GoDoxy TLS 终止（autocert 证书）
                                              │
                                              ▼
                        entrypoint 全局 middleware → HTTP route gptl.* → 127.0.0.1:3001
公网 :80  ── GoDoxy HTTP entrypoint ── 同一 HTTP route ── RedirectHTTP 中间件 → 308 https
```

Caddy 与 `:1443` 对本域名不再参与；**只有在第 6 节全部验证通过后**才考虑移除 `gptl-proxy`。

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

操作顺序（切换时执行，本文档阶段**未执行**）：

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

**当前这些都不会作用于本域名**：GPT-Load 的流量以原始 TCP 从 443 透传给 Caddy，GoDoxy 不解析 HTTP。切换后会新增以下可观察变化，验收时要逐条核对：

1. 响应会多出上述 4 个安全头（当前 443 的 200 响应一个安全头都没有；80 端口 Go 自带的 404 只有 `X-Content-Type-Options`）。
2. `http://` 请求从 **404** 变为 **308** 跳转到 https（`internal/net/gphttp/middleware/redirect_http.go`，`http.StatusPermanentRedirect`）。
3. `via: 1.1 Caddy` 头消失，`alt-svc: h3=":1443"` 不再出现。
4. 每条请求写入 `/app/logs/entrypoint.log`（combined 格式，保留 30 天），日志量上升，且其中包含路径与 query；按现有访问日志的权限约定处理，不要随手贴到工单/日志检索里。
5. 中间件链对流式响应的正文改写受 `canBufferAndModifyResponseBody` 限制（`internal/net/gphttp/middleware/middleware.go`），SSE 需要实测确认不被缓冲（见第 6 步验收）。

## 6. 未来的切换、验证与回滚（需另行授权，本文档阶段未执行）

### 6.1 前置条件

- 生产域名 cutover、Caddy 移除**不在本次开发/隔离验证范围内**，执行前需明确授权。
- 按 3.2 在服务器上备份整个 GoDoxy `config/` 目录；确认备份路径可读、权限仍受保护，并准备好按 6.5 原地恢复全部改动文件。单个文件备份不足以代表共享入口的已知良好配置。
- 确认 `/app/certs` 对应的持久卷 `godoxy_godoxy-certs` 可写，Hostinger DNS 凭据仍有效。证书检查从宿主机通过 Docker named-volume 的实际 mountpoint 读取，不假定应用镜像含 shell 或 OpenSSL。
- GoDoxy 是多个 HTTPS vhost 共用的入口。ACME 准备阶段也可能影响其他 GoDoxy TLS 服务；Caddy 保留只提供 GPT-Load 的旁路，不等于其他生产 HTTPS 服务不受影响。

### 6.2 第一步：先出证书，再动路由

顺序不能颠倒，但不能把 ACME 错误等同于 reload 被拒绝：`internal/config/state.go:138-159` 把 autocert 归为 optional component，初始化错误记为 `IssueDegraded`；`IssueDegraded.IsFailure()` 为 false（`internal/config/types/lifecycle.go:37-40`）。`RuntimeManager.transition` 只因 rejecting/failure issue 等原因拒绝候选；否则会停止旧 runtime 并 commit 新候选（`internal/config/runtime_manager.go:183-226`）。源码注释明确 activation failure 不会恢复旧配置（`:108-115`）。`state.initAutoCert` 只有在证书获取成功后才设置 provider（`internal/config/state.go:568-576`），而 HTTPS server 只在证书可用时创建（`goutils/server/server.go:90-105`）。因此 DNS-01/ACME 失败可能以 **committed + degraded** 结束，而非保留旧 runtime；其他 GoDoxy TLS vhost 也可能受影响。旧 GPT-Load TCP→Caddy route 尚未改动，并不能证明生产整体未受影响。

1. 按 3.2 完成整目录备份，再合并 autocert extra，原地保存 `config.yml`。
2. 等待 watcher 完成 reload（约 500ms 合并窗口，见 `internal/config/events.go`），检查**最新一次**结果。只有 `config committed: healthy` 且无 degraded/failed issue 才继续；`config committed: degraded`、`failed` 或 `rejected` 都要停下。特别是 committed-degraded 时，立即按 6.5 恢复整个已知良好的配置并验证，不要假设旧 runtime 仍在运行。即使结果为 healthy，也要逐一检查当前所有 GoDoxy TLS vhost 的 HTTPS 健康与证书，不能只检查 GPT-Load。
3. 只有 reload healthy 且既有 TLS vhost 检查通过，才检查新证书文件是否落地：

```bash
ssh vps-kl 'cert_dir=$(docker volume inspect --format "{{.Mountpoint}}" godoxy_godoxy-certs) && ls -l "$cert_dir/gptl.tanyaleoallen.cloud.crt" && openssl x509 -in "$cert_dir/gptl.tanyaleoallen.cloud.crt" -noout -subject -issuer -dates'
```

命令在 Docker 宿主机解析 `godoxy_godoxy-certs` 的实际 mountpoint，`openssl` 只读取并输出公开证书信息；不读取或输出私钥。若证书不存在、issuer/有效期不符、reload 非 healthy，或已有 TLS vhost 异常，**不要切路由**；恢复整个配置快照并按 6.5 验证运行态。单纯保持 GPT-Load 的旧 TCP 条目不足以保证其余 GoDoxy HTTPS 服务无影响。

> 本次开发只用**测试 CA**在隔离命名空间证明了「证书校验 + SNI/主机名校验」这条链路（U003），**不能**替代生产 ACME 签发/续期证据；上面这段是生产必须自己证明的部分。

### 6.3 第二步：切路由

按 3.1 二选一执行（推荐原地改写 `vhosts.yml` 条目），保存后热重载。

### 6.4 第三步：验证（公网、只读）

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

**对外健康 URL 的 `:1443` 必须清零**：目前带端口的对外健康 URL 分布在 `docs/deployment.md`（「健康检查异常排障」里的 `https://gptl.tanyaleoallen.cloud:1443/health`，`grep -n 1443`）与主工作树未提交的 `scripts/deploy.sh`（第 70 行 `:1443/health`）中；本工作树基线的 `scripts/deploy.sh` 只探回环 `http://127.0.0.1:3001/health`（`scripts/deploy.sh:47`），不受切换影响。`https://gptl.tanyaleoallen.cloud/health`（默认 443）在切换前后都可用，所以集成时把所有对外 `:1443` 健康 URL 改回无端口形式，回环探测保持不变。主工作树还有未提交的「本地构建 + SSH 上传」改动落在 `docs/deployment.md` 与 `scripts/deploy.sh` 上，合并时按其现有改动一并处理，不要覆盖它们。

**Alt-Svc / 1443 排空（另行授权）**：当前 Caddy 经 443 和 1443 返回 `alt-svc: h3=":1443"; ma=2592000`。必须先阻止 Caddy 在旧 1443 服务上继续发布正向广告，并在仍由 Caddy 处理的 HTTP/1.1、HTTP/2 响应中返回明确清除头。切换前或切换时，在 Caddyfile 的现有全局块加入 `servers` 设置，并在 `gptl` site 内延后覆盖响应头；下面是增量示意，不是本次已部署配置：

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

两处都必须只返回 `Alt-Svc: clear`、不能返回 `h3=":1443"`，才记录 UTC 确认时刻。之后再切 GoDoxy 路由，并确认公网 443 不重新出现正向 `:1443` 广告。**30 天排空期从 443 与 1443 最后一次可能发出正向 `h3=":1443"` 广告的时刻起算**，若之后再次观察到该广告则从最后一次重新计时。保持 Caddy 的 TCP 1443（HTTP/1.1、HTTP/2）可用，作为过渡与回滚入口；此时 UDP/HTTP/3 1443 已停用，不能把保留 TCP 服务表述为旧 H3 alternative 仍可用。

`ma=2592000` 是 30 天的新连接缓存有效期；RFC 7838 说明到期限制的是建立新连接，已经存在的 alternative connection 仍可能继续使用。因此，等满 30 天本身不证明没有活跃客户端。排空期间使用能区分真实客户端与运维探测的 1443 请求/连接遥测；目前 Caddyfile 未配置 access log，若没有可信遥测，应在排空前另行授权并先建立合适的观测。移除前要求完整 30 天无正向广告、无未结束的 1443 连接且观测窗口内无真实客户端请求；手工 `curl ...:1443/health` 只证明端点可用，**不能**证明没有其他客户端访问。没有可信的客户端流量/连接证据时，不得宣称已排空，继续保留 Caddy。

**真实 AI 流式/非流式验收（延后）**：至少各跑一次真实认证的流式（SSE 增量、取消传播）与非流式请求，核对 request log、attempt 与 raw capture，确认 4 个安全头、SSE 不被缓冲、取消能让后端断连。**这属于生产切换验收，本次开发与隔离验证阶段刻意未执行**（会产生计费的 AI 请求）。

### 6.5 回滚

回滚要恢复此次改动过的**全部 GoDoxy 配置文件**，而非只删除 autocert extra。暂停其他配置编辑后，使用 3.2 记录的整目录快照；先用 `diff -qr "$backup" "$config"` 列出快照与当前目录的差异（只列路径，不打印配置内容），据此恢复所有被改动的旧文件，并仅删除此次新增且快照中没有的文件。本流程至少原地恢复 `vhosts.yml` 与 `config.yml`；若使用过 include 备选，也删除本次新增 route 文件。先恢复 route/provider 文件，最后恢复 `config.yml` 以触发完整 reload；不要 rename/delete `config.yml` 或替换整个配置目录：

```bash
ssh vps-kl 'backup=/www/server/panel/data/compose/godoxy/config.pre-gptl-<UTC时间戳>; config=/www/server/panel/data/compose/godoxy/config; cp -p "$backup/vhosts.yml" "$config/vhosts.yml" && cp -p "$backup/config.yml" "$config/config.yml"'
```

恢复配置文件只会触发一个新的候选 reload，不会自动恢复旧 runtime。等待并核实结果为 `config committed: healthy`，确认全部现有 HTTPS vhost、证书和 GPT-Load 旧 TCP→Caddy 路径恢复后，才把回滚视为完成；如果 reload 仍为 degraded/failed 或 TLS 服务未恢复，维持 Caddy、停止进一步配置切换并按生产事故流程处理。Caddy 容器和 `/acme` 证书应持续保留；6.4 的排空步骤会有意修改 Caddy 的协议/响应头配置并重启，故不能称整个切换期间保持不动。GoDoxy 回滚可继续使用这个提供 HTTP/1.1、HTTP/2 与 `Alt-Svc: clear` 的 Caddy；不要为回滚重新启用 H3 正向广告，否则必须重新计算排空窗口。已签发的新证书可以留在 GoDoxy volume 中，不要为了回滚删除证书文件。

### 6.6 什么时候可以移除 Caddy

同时满足才动手：

1. 6.4 的 1–5 全部通过；
2. 真实流式/非流式 AI 请求验收通过（6.4 末尾）；
3. Caddy 已停止发出 `h3=":1443"` 正向广告，并在保留的 HTTP/1.1、HTTP/2 响应上返回 `Alt-Svc: clear`（不保证缓存 H3 客户端均已收到）；自最后一次可能的正向广告起至少 30 天；
4. 可信流量/连接遥测显示排空期内无真实客户端请求且无存续的 1443 连接。手工健康检查不计作客户端缺席证据；没有可信遥测时不得移除；
5. 所有对外的 `:1443` 健康 URL（`docs/deployment.md` 排障一节、主工作树改动的 `scripts/deploy.sh`）已改为无端口 443 形式，回环探测不受影响，并验证过一次发布。

移除动作本身是停用并删除 `gptl-proxy` 容器与 `/opt/gptl-proxy`（保留备份），随后确认 1443 不再监听、443 仍正常。**本次交付不包含该动作。**

## 7. 隔离验证结论与不成立的结论

U003 在 Hostinger 上用**同一个生产镜像**（`sha256:3edf8b84…`，revision `78be6afb`）在隔离命名空间做了完整验证（U003 result（本地 `../.tmp/godoxy-ingress_20260928/units/U003/result.md`））：

- 首轮运行通过功能断言，但无法排除 Docker resolver 的上游转发，因此**保留该顾虑**；
- 第二轮在启动前架设命名空间级 IPv4/IPv6 防火墙并做 53 端口抓包：pcap **0 包**，DNS 与非回环出站计数全 0，run-owned resolver 只指向命名空间回环；该轮为其捕获窗口提供了可接受的独立证据，**不能消除首轮是否发生上游 DNS 转发的历史不确定性**；
- 默认 60s 超时对照失败 / `2h` 成功、TLS 测试 CA 与主机名校验、SSE 事件先于 EOF、取消后后端观察到断连、非 2xx 状态/头/正文透传、单次 UDP 往返均通过；真实 `127.0.0.1:3001/health` 200（补充项，未经过隔离入口）；
- 全部使用 run-owned 合成后端，**没有任何 AI 供应商请求**；测试 CA 只证明 TLS 校验，不证明生产 ACME 签发/续期；生产容器/路由/配置前后未变。

**本文档没有证明的事**：真实生产切换、生产 ACME 签发与续期、真实 AI 流式/非流式请求、Caddy 移除后的长期运行、Alt-Svc 缓存客户端的实际表现。

## 8. 证据索引

下列 `.tmp/` 文件为本次工作树保留的本地运行记录，不随 Git 提交发布；克隆仓库后不会包含这些文件。可共享的验证结论见第 7 节，探针与测试脚本随仓库提供。

- 本单元证据：`../.tmp/godoxy-ingress_20260928/units/U004/evidence/`（本地 `../.tmp/godoxy-ingress_20260928/units/U004/evidence/`）
  - `public-probe.txt`：443/1443/80 实测头与正文
  - `tls-cert-probe.txt`：443/1443 证书 subject/issuer/有效期
  - `server-config-facts.txt`：生产 `config.yml` 的 include、entrypoint middleware/access_log、autocert 概要与 `vhosts.yml`（未输出任何凭据）
  - `workspace-state.txt`：本工作树与主工作树待合并改动
- 上游单元：U001 超时与片段（本地 `../.tmp/godoxy-ingress_20260928/units/U001/result.md`）、U002 探针（本地 `../.tmp/godoxy-ingress_20260928/units/U002/result.md`）、U003 隔离运行（本地 `../.tmp/godoxy-ingress_20260928/units/U003/result.md`）
- GoDoxy 源码（revision `78be6afb`，本地 `/tmp/godoxy-inspect-78be6afb`）：`internal/route/port.go`、`internal/route/route.go`、`internal/route/provider/file.go`、`internal/config/types/config.go`、`internal/config/events.go`、`internal/config/state.go`、`internal/types/http_config.go`、`internal/routeimpl/reverse_proxy.go`、`internal/autocert/config.go`、`internal/autocert/provider.go`、`internal/entrypoint/sni_passthrough.go`、`internal/net/gphttp/middleware/redirect_http.go`、`internal/net/gphttp/middleware/middleware.go`、`goutils/http/reverseproxy/reverse_proxy.go`、`goutils/server/server.go`
