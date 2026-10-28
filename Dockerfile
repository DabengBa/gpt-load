FROM --platform=$BUILDPLATFORM node:26.12.1-alpine3.24@sha256:40cd011b55415bc894e45b02fdd5a5dfde056921b7d23a64c74f80497af48797 AS web-builder

WORKDIR /build
RUN npm install --global pnpm@latest

COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./web/
RUN pnpm --dir web install --frozen-lockfile

COPY internal/webui/page_routes.json ./internal/webui/page_routes.json
COPY web ./web
RUN pnpm --dir web run build


FROM --platform=$BUILDPLATFORM golang:1.27.3-alpine3.24@sha256:d200d2dd76b03a99b52651fd1325aff7dcd35422c35c01bac56571b0507f3deb AS go-builder

ARG VERSION=2.0.0-dev
ARG TARGETOS
ARG TARGETARCH
# GOPROXY 使用竖线分隔：任何错误（含网络错误）都回退到下一个源。
# 默认的逗号只在 404/410 时回退，模块代理瞬时故障会直接中断构建。
ENV GO111MODULE=on \
    CGO_ENABLED=0 \
    GOPROXY="https://proxy.golang.org|direct"

WORKDIR /build

COPY go.mod go.sum ./
COPY third_party/bifrost-core/go.mod third_party/bifrost-core/go.sum ./third_party/bifrost-core/
RUN go mod download

COPY *.go ./
COPY internal ./internal
COPY third_party/bifrost-core ./third_party/bifrost-core
COPY --from=web-builder /build/internal/webui/dist ./internal/webui/dist
RUN GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build \
    -ldflags "-s -w -X gpt-load/internal/platform/version.Version=${VERSION}" \
    -o gpt-load


# runtime 是两条打包路径共用的唯一 runtime 定义：默认的源码自包含构建，
# 以及发布流程复用 build-binaries 预编译产物的 prebuilt。两者只有二进制来源
# 不同，其余 runtime 配置全部在这里定义一次，避免出现两套配置各自漂移。
FROM alpine:3.24.1@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b AS runtime

WORKDIR /app
RUN apk add --no-cache libcrypto3=3.5.9-r0 libssl3=3.5.9-r0 \
    && apk add --no-cache ca-certificates tzdata \
    && update-ca-certificates \
    && addgroup -S -g 10001 gpt-load \
    && adduser -S -D -H -u 10001 -G gpt-load gpt-load \
    && mkdir -p /app/data \
    && chown 10001:10001 /app/data \
    && chmod 0700 /app/data

ENV HOST=0.0.0.0
ENV DATA_DIR=/app/data
COPY LICENSE THIRD_PARTY_NOTICES.md /app/licenses/
COPY LICENSES/Apache-2.0.txt /app/licenses/Apache-2.0.txt
COPY LICENSES/MIT.txt /app/licenses/MIT.txt
EXPOSE 3001
USER 10001:10001
ENTRYPOINT ["/app/gpt-load"]


# 发布路径：直接打包 build-binaries 已交叉编译好的二进制，不在镜像内重复编译。
# 需要 build context 中存在 release/gpt-load-linux-<arch>。
FROM runtime AS prebuilt

ARG TARGETARCH
# GitHub Actions 的 artifact 不保留可执行位，必须在复制时显式恢复为 0755。
COPY --chmod=0755 release/gpt-load-linux-${TARGETARCH} /app/gpt-load


# 默认 target：自包含的源码构建，供本地 `docker build .` 与用户自建使用。
# 必须保持在文件末尾，否则默认构建会落到 prebuilt 而要求预编译产物。
FROM runtime AS source-build

COPY --from=go-builder /build/gpt-load .
