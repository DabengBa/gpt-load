# 自建实例部署（gptl.tanyaleoallen.cloud）

本文只描述本仓库维护者自建实例的发布流程；上游通用部署（`ghcr.io/tbphp/gpt-load:2`、原生二进制）见 `README_CN.md` 的「部署与数据」。

## 拓扑

- 反代：Caddy 容器 `gptl-proxy`（配置 `/opt/gptl-proxy/Caddyfile`），`https://gptl.tanyaleoallen.cloud:1443` → `127.0.0.1:3001`。
- 应用：Compose 项目 `/opt/gpt-load`，容器 `gpt-load`，镜像为自建 `gpt-load:dev-<短sha>`，数据在具名卷 `gpt-load_gpt-load-data`（含 `gpt-load.db`、`auth.key`、`encryption.key`）。
- 源码与构建：`/opt/gpt-load-src`，跟踪 `dev` 分支，只作为构建源；每次发布由脚本 `git reset --hard origin/dev`。
- 该容器已用 `com.centurylinklabs.watchtower.enable: "false"` 关闭自动更新，镜像只通过下面的脚本切换。

## 发布

```bash
ssh vps-kl /opt/gpt-load/deploy.sh
```

脚本按顺序执行：拉取 `dev` 最新提交 → `docker build --build-arg VERSION=dev-<短sha> -t gpt-load:dev-<短sha>` → 备份当前 `docker-compose.yml` → 改 `image` 标签与来源注释 → `docker compose up -d` → 轮询容器健康并打印 `/health`。健康未通过时脚本以非零退出，按下面的回滚步骤处理。

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
cp docker-compose.yml.bak-<时间戳>-dev-<旧短sha> docker-compose.yml
docker compose up -d
```

旧镜像不会自动清理，`docker images | grep gpt-load` 可直接看到待回滚的版本。回滚只切镜像标签，`gpt-load_gpt-load-data` 卷保持不动。

## 注意

- 持久化数据与 `encryption.key` 必须一起备份，缺失密钥会导致已有渠道凭据无法解密。
- `docker-compose.yml.bak-*` 是回滚点，确认新版本稳定前不要删除。
- 发布后先看 `docker logs` 的错误计数和首屏 `/health`，再观察真实请求日志中的失败重试情况。
