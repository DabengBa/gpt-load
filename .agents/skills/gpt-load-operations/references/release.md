# Hostinger Release

`scripts/deploy.sh [branch]`（默认 `dev`）；完整流程另见脚本注释与部署前检查。

## 不可反推约束

- GitHub merge API 不能原子 pin base，合并前避免并发 merge；以 pinned commit 的 `git archive` 构建，只有 pushed commit 能进镜像。
- 本地 unix docker endpoint 构建；上传前本地 verify previous image。
- 切换前守卫必须全部满足：远端 image ID 与本地一致、旧容器仍在运行、旧 image 未被覆盖、mount 与 volume 从运行中的容器读取、备份不在 tmpfs、free space > 2×data。
- 切换序列：stop 旧容器（30 秒宽限）→ 确认无 writer → `tar` 整个 data 目录 → compose 旁置/重指 image → `up --no-build --pull never` → 72×5s 本地健康 + 公网健康。
- 镜像版本必须是 `branch-shortsha`；切换成功后无自动回滚（DB 可能已迁移），单实例窗口非零停机。
- optional rehearsal 不替代 pre-switch 备份。

## 边界

GoDoxy ingress 切换与回滚观测、Caddy 移除历史见迁移归档（`.tmp/l5-doc-governance-migration/docs/root/godoxy-ingress.md`），仅作历史证据。
