#!/usr/bin/env bash
# 在本地构建已推送的提交，通过 SSH 上传镜像并切换 Hostinger 服务。
# vps-kl 是 Hostinger 服务器。用法与两阶段发布见 docs/deployment.md。
set -euo pipefail

branch=${1:-dev}
mode=${2:-all}
expected_commit=${3:-}
case "$mode" in
  all|--prepare-only) [ "$#" -le 2 ] || { echo '参数过多' >&2; exit 2; } ;;
  --activate-only)
    [ "$#" -eq 3 ] && [[ "$expected_commit" =~ ^[0-9a-f]{40}$ ]] || {
      echo '用法: scripts/deploy.sh <分支> --activate-only <已演练的完整提交SHA>' >&2; exit 2;
    } ;;
  *) echo "未知发布模式: $mode" >&2; exit 2 ;;
esac
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
git check-ref-format --branch "$branch" >/dev/null
git -C "$src" fetch -q origin "refs/heads/$branch"
commit=$(git -C "$src" rev-parse FETCH_HEAD)
if [ "$mode" = --activate-only ] && [ "$commit" != "$expected_commit" ]; then
  echo '远端分支已变化，拒绝切换；请重新准备并演练目标提交' >&2
  exit 1
fi
sha=$(git -C "$src" rev-parse --short "$commit")
prefix=${branch//\//-}
tag="gpt-load:$prefix-$sha"
[[ "$prefix-$sha" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]] || {
  echo '分支名无法用作 Docker 镜像标签' >&2; exit 1;
}

if [ "$mode" != --activate-only ]; then
  # 固定使用当前 Docker context 的默认 builder，禁止把构建转发到 SSH/TCP daemon。
  if [ -n "${DOCKER_CONTEXT:-}" ]; then
    endpoint=$(docker context inspect "$DOCKER_CONTEXT" --format '{{.Endpoints.docker.Host}}')
  else
    endpoint=${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}
  fi
  [[ "$endpoint" == unix://* ]] || { echo '发布需要本机 Unix socket Docker daemon' >&2; exit 1; }
  export DOCKER_HOST="$endpoint"
  unset DOCKER_CONTEXT
  arch=$(ssh vps-kl 'uname -m')
  case "$arch" in
    x86_64) platform=linux/amd64 ;;
    aarch64|arm64) platform=linux/arm64 ;;
    *) echo "不支持的服务器架构: $arch" >&2; exit 1 ;;
  esac

  echo "本地构建: $tag ($platform，提交 $commit)"
  # 只构建已推送提交；不将本地未提交文件或构建产物混入镜像。
  git -C "$src" archive "$commit" |
    docker buildx build --builder default --load --platform "$platform" \
      --build-arg "VERSION=$prefix-$sha" -t "$tag" -

  echo "上传镜像: $tag"
  docker save "$tag" | gzip -1 | ssh vps-kl 'docker load'
fi

if [ "$mode" = --prepare-only ]; then
  echo "准备完成，生产容器未切换: $tag (提交 $commit)"
  echo "快照演练和最终备份完成后执行: scripts/deploy.sh $branch --activate-only $commit"
  exit 0
fi

# 镜像加载成功后才切换；脚本通过 stdin 发送，服务器不再需要源码副本。
ssh vps-kl bash -s -- "$tag" <<'REMOTE'
set -euo pipefail
tag=$1
trap '' HUP
umask 077
cd /opt/gpt-load
compose=docker-compose.yml
docker image inspect "$tag" >/dev/null
[ "$(docker inspect gpt-load --format '{{.State.Running}}')" = true ] || {
  echo '发布必须从运行中的旧容器开始，禁止预先手工停机' >&2; exit 1;
}
old=$(sed -n 's/^    image: gpt-load:\(.*\)$/\1/p' "$compose" | head -1)
[ -n "$old" ] || { echo '未找到现有 gpt-load 镜像配置' >&2; exit 1; }
data=$(docker volume inspect gpt-load_gpt-load-data --format '{{.Mountpoint}}')
[ "$(docker inspect gpt-load --format '{{range .Mounts}}{{if eq .Destination "/app/data"}}{{.Source}}{{end}}{{end}}')" = "$data" ] || {
  echo '生产数据挂载与预期卷不一致' >&2; exit 1;
}
[ "$(findmnt -n -o FSTYPE -T "$PWD")" != tmpfs ]
size=$(du -sb "$data" | cut -f1)
avail=$(df -B1 --output=avail "$PWD" | tail -n 1)
[ "$avail" -gt "$((size * 2))" ] || { echo '最终备份空间不足' >&2; exit 1; }
backup=$(mktemp -d "$PWD/pre-upgrade.XXXXXXXX")
cp -- "$compose" "$backup/docker-compose.yml"
stopped=false
switched=false
recover_before_switch() {
  result=$?
  if [ "$stopped" = true ] && [ "$switched" = false ]; then
    echo '切换前失败，源库未修改，立即恢复旧容器' >&2
    cp -- "$backup/docker-compose.yml" "$compose" || true
    docker start gpt-load || true
  fi
  exit "$result"
}
trap recover_before_switch EXIT
echo "停写并备份: $backup"
stopped=true
docker compose -p gpt-load -f "$compose" stop -t 30 gpt-load
[ "$(docker inspect gpt-load --format '{{.State.Running}}')" = false ]
[ -z "$(docker ps -q --filter volume=gpt-load_gpt-load-data)" ] || {
  echo '数据卷仍有运行中的容器写入者' >&2; exit 1;
}
tar -C "$data" -cf "$backup/data.tar" .
tar -tf "$backup/data.tar" >/dev/null
echo '最终备份完成，立即切换镜像'
cp -- "$compose" "$compose.bak-$(date +%Y%m%d%H%M%S)-$old"
sed -i "s|^    image: gpt-load:.*|    image: $tag|" "$compose"
sed -i "s|^    # 自建镜像:.*|    # 自建镜像:$tag，由本地 scripts/deploy.sh 构建并上传。|" "$compose"
switched=true
docker compose -p gpt-load -f "$compose" up -d --no-build --pull never

status=unknown
for _ in $(seq 1 24); do
  sleep 5
  status=$(docker inspect gpt-load --format '{{.State.Health.Status}}' 2>/dev/null || echo unknown)
  [ "$status" = healthy ] && break
done
echo "容器健康: $status"
curl -fsS --max-time 10 http://127.0.0.1:3001/health
echo
[ "$status" = healthy ] || { echo "健康检查未通过，最终备份: $backup；数据库可能已变化，按 docs/deployment.md 排障与回滚" >&2; exit 1; }
REMOTE

echo '公网健康检查:'
curl -fsS --max-time 10 https://gptl.tanyaleoallen.cloud/health
echo
