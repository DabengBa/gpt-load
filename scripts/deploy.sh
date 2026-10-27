#!/usr/bin/env bash
# 本地构建已推送提交，本地验证镜像，上传后停写备份并切换 Hostinger 服务。
# vps-kl 是 Hostinger 服务器。用法与回滚见 docs/deployment.md。
set -euo pipefail

branch=${1:-dev}
mode=${2:-all}
expected=${3:-}
case "$mode" in
  all|--prepare-only) [ "$#" -le 2 ] || { echo '参数过多' >&2; exit 2; } ;;
  --activate-only)
    [ "$#" -eq 3 ] && [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || {
      echo '用法: scripts/deploy.sh <分支> --activate-only <已演练的完整提交SHA>' >&2; exit 2
    } ;;
  *) echo "未知发布模式: $mode" >&2; exit 2 ;;
esac
src=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
git check-ref-format --branch "$branch" >/dev/null

# 构建上下文、验证脚本和切换脚本都取自目标提交：先把提交导出到临时目录，再由导出的
# 副本执行。本地未提交的改动因此进不了镜像，也改不了验证与切换逻辑。
if [ -z "${DEPLOY_COMMIT:-}" ]; then
  git -C "$src" fetch -q origin "refs/heads/$branch"
  commit=$(git -C "$src" rev-parse FETCH_HEAD)
  sha=$(git -C "$src" rev-parse --short "$commit")
  if [ "$mode" = --activate-only ] && [ "$commit" != "$expected" ]; then
    echo '远端分支已变化，拒绝切换；请重新准备并演练目标提交' >&2
    exit 1
  fi
  bundle=$(mktemp -d)
  trap 'rm -rf "$bundle"' EXIT
  (cd "$src" && git archive "$commit") | tar -x -C "$bundle"
  echo "构建与验证使用提交 $commit 自带的脚本"
  DEPLOY_COMMIT=$commit DEPLOY_SHA=$sha bash "$bundle/scripts/deploy.sh" "$@"
  exit 0
fi
commit=$DEPLOY_COMMIT
sha=$DEPLOY_SHA

prefix=${branch//\//-}
tag="gpt-load:$prefix-$sha"
version="$prefix-$sha"
[[ "$version" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}$ ]] || {
  echo '分支名无法用作 Docker 镜像标签' >&2; exit 1
}

# 固定使用当前 Docker context 的默认 builder，禁止把构建转发到 SSH/TCP daemon。
if [ -n "${DOCKER_CONTEXT:-}" ]; then
  endpoint=$(docker context inspect "$DOCKER_CONTEXT" --format '{{.Endpoints.docker.Host}}')
else
  endpoint=${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}
fi
[[ "$endpoint" == unix://* ]] || { echo '发布需要本机 Unix socket Docker daemon' >&2; exit 1; }
export DOCKER_HOST="$endpoint"
unset DOCKER_CONTEXT

if [ "$mode" = --activate-only ]; then
  # 切换此前上传的镜像；验证照旧执行，切换不是绕开验证的捷径。
  docker image inspect "$tag" >/dev/null 2>&1 || {
    echo "本地没有待切换镜像: $tag；请先执行 scripts/deploy.sh $branch --prepare-only" >&2
    exit 1
  }
else
  arch=$(ssh vps-kl 'uname -m')
  case "$arch" in
    x86_64) platform=linux/amd64 ;;
    aarch64|arm64) platform=linux/arm64 ;;
    *) echo "不支持的服务器架构: $arch" >&2; exit 1 ;;
  esac
  echo "本地构建: $tag ($platform，提交 $commit)"
  docker buildx build --builder default --load --platform "$platform" \
    --build-arg "VERSION=$version" -t "$tag" "$src"
fi

echo "本地验证: $tag"
previous=$(ssh vps-kl "docker inspect gpt-load --format '{{.Image}}'")
[[ "$previous" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo '无法确定旧容器镜像 ID' >&2; exit 1; }
# 用真实旧镜像建库再演练升级；本地没有就先从服务器取回。
if ! docker image inspect "$previous" >/dev/null 2>&1; then
  ssh vps-kl "docker save '$previous'" | docker load
fi
image_id=$(docker image inspect "$tag" --format '{{.Id}}')
[[ "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]] || { echo '无法确定本地镜像 ID' >&2; exit 1; }
python3 "$src/scripts/verify-built-image.py" \
  --image "$tag" --previous-image "$previous" --expected-version "$version"

if [ "$mode" = --activate-only ]; then
  echo "使用已在服务器上的已验证镜像: $tag"
else
  echo "上传镜像: $tag"
  docker save "$tag" | gzip -1 | ssh vps-kl 'docker load'
fi

if [ "$mode" = --prepare-only ]; then
  echo "准备完成，生产容器未切换: $tag (提交 $commit)"
  echo "需要切换时执行: scripts/deploy.sh $branch --activate-only $commit"
  exit 0
fi

# 镜像加载成功后才切换；脚本通过 stdin 发送，服务器不需要源码副本。
ssh vps-kl bash -s -- "$tag" "$image_id" "$previous" "$version" <<'REMOTE'
set -euo pipefail
tag=$1
image_id=$2
previous=$3
version=$4
trap '' HUP
umask 077
cd /opt/gpt-load
compose=docker-compose.yml
docker image inspect "$tag" >/dev/null
# 服务器上的镜像必须就是本地验证过的那一个，避免上传被替换或截断。
[ "$(docker image inspect "$tag" --format '{{.Id}}')" = "$image_id" ] || {
  echo '服务器镜像 ID 与本地验证的镜像不一致，拒绝切换' >&2; exit 1
}
[ "$(docker inspect gpt-load --format '{{.State.Running}}')" = true ] || {
  echo '发布必须从运行中的旧容器开始，禁止预先手工停机' >&2; exit 1
}
# 升级验证基于旧镜像的数据库；停机前确认生产仍是该镜像，否则结论不再适用。
[ "$(docker inspect gpt-load --format '{{.Image}}')" = "$previous" ] || {
  echo '生产旧镜像在验证后被替换，拒绝切换；请重新执行发布' >&2; exit 1
}
old=$(sed -n 's/^    image: gpt-load:\(.*\)$/\1/p' "$compose" | head -1)
[ -n "$old" ] || { echo '未找到现有 gpt-load 镜像配置' >&2; exit 1; }
# 数据卷名由运行中的容器决定，不硬编码。
data=$(docker inspect gpt-load --format '{{range .Mounts}}{{if eq .Destination "/app/data"}}{{.Source}}{{end}}{{end}}')
volume=$(docker inspect gpt-load --format '{{range .Mounts}}{{if eq .Destination "/app/data"}}{{.Name}}{{end}}{{end}}')
[ -n "$data" ] && [ -n "$volume" ] || { echo '未找到运行容器的数据挂载' >&2; exit 1; }
size=$(du -sb "$data" | cut -f1)
avail=$(df -B1 --output=avail "$PWD" | tail -n 1)
[ "$avail" -gt "$((size * 2))" ] || { echo '最终备份空间不足' >&2; exit 1; }
# 同盘备份不能落在内存盘上，否则既不是恢复点也可能耗尽内存。
filesystem=$(findmnt -n -o FSTYPE -T "$PWD")
case "$filesystem" in
  tmpfs|ramfs) echo "备份目录位于 $filesystem，不能作为恢复点" >&2; exit 1 ;;
esac
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
[ -z "$(docker ps -q --filter volume="$volume")" ] || {
  echo '数据卷仍有运行中的容器写入者' >&2; exit 1
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
[ "$status" = healthy ] || {
  echo "健康检查未通过，最终备份: $backup；数据库可能已迁移，按 docs/deployment.md 排障与回滚" >&2
  exit 1
}
health=$(curl -fsS --max-time 10 http://127.0.0.1:3001/health)
echo "$health"
# 健康但跑着旧镜像同样是发布失败。
case "$health" in
  *"\"version\":\"$version\""*) ;;
  *) echo "运行版本不是目标版本 $version，最终备份: $backup" >&2; exit 1 ;;
esac
REMOTE

echo '公网健康检查:'
public=$(curl -fsS --max-time 10 https://gptl.tanyaleoallen.cloud/health)
echo "$public"
case "$public" in
  *"\"version\":\"$version\""*) ;;
  *) echo "公网版本不是 $version" >&2; exit 1 ;;
esac
