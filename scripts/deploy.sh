#!/usr/bin/env bash
# 构建并切换 gptl.tanyaleoallen.cloud 上的 gpt-load 镜像。
#
# 用法（在本机调用，脚本在服务器上执行）：
#   ssh vps-kl /opt/gpt-load-src/scripts/deploy.sh [分支]    # 分支默认 dev
#
# 脚本自身位于构建源 /opt/gpt-load-src 内，而它会把该目录 reset 到目标提交。
# 为避免 bash 边读边执行一个刚被上层 reset 覆盖的脚本，开跑前先把自己复制到
# /tmp 再重新执行；/tmp 里的副本整轮运行不再变化。
#
# 拓扑、验证与回滚见 docs/deployment.md。
set -euo pipefail

if [ -z "${GPTLOAD_DEPLOY_REEXEC:-}" ]; then
  self=/tmp/gpt-load-deploy.$$.sh
  cp -- "$0" "$self"
  GPTLOAD_DEPLOY_REEXEC=1 exec bash "$self" "$@"
fi

branch=${1:-dev}
src=/opt/gpt-load-src
dst=/opt/gpt-load
compose=$dst/docker-compose.yml
prefix=${branch//\//-}

git -C "$src" fetch -q origin "$branch"
git -C "$src" reset -q --hard "origin/$branch"
sha=$(git -C "$src" rev-parse --short HEAD)
tag="gpt-load:$prefix-$sha"
old=$(sed -n 's/^    image: gpt-load:\(.*\)$/\1/p' "$compose" | head -1)

echo "本次构建: $tag（当前运行: ${old:-未知}）"
docker build --build-arg "VERSION=$prefix-$sha" -t "$tag" "$src"

cp -- "$compose" "$dst/docker-compose.yml.bak-$(date +%Y%m%d%H%M%S)-${old:-unknown}"
sed -i "s|^    image: gpt-load:.*|    image: $tag|" "$compose"
sed -i "s|^    # 自建镜像:.*|    # 自建镜像:$prefix@$sha，由 /opt/gpt-load-src/scripts/deploy.sh $branch 构建。|" "$compose"
docker compose -f "$compose" up -d

status=unknown
for _ in $(seq 1 24); do
  sleep 5
  status=$(docker inspect gpt-load --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)
  [ "$status" = healthy ] && break
done
echo "容器健康: $status"
curl -s --max-time 10 http://127.0.0.1:3001/health
echo
[ "$status" = healthy ] || { echo "健康检查未通过，回滚方式见 docs/deployment.md" >&2; exit 1; }
