# Contributing

Thanks for your interest in GPT-Load.

## Before you start

- **Security issues**: do not open a public issue. Follow the reporting guidance in [SECURITY.md](SECURITY.md).
- **Usage questions**: check the [official docs](https://www.gpt-load.com/docs) and the Telegram group linked from [README.md](README.md#project-support).
- **Large changes**: open an issue first to discuss the approach before investing in a pull request.

## Product boundaries

- This repository keeps the standard API and cloud-credential model plus external CLIProxyAPI gateways. Do not reintroduce built-in subscription authorization, OAuth account observation, or embedded CPA execution. Codex / Claude Code are downstream clients, not upstream subscription channels.
- Breaking storage changes have no in-place migration: the converter path (`gpt-load convert-legacy-db`, offline only) or a fresh database are the only supported upgrade routes. Converter changes must cover the retained/dropped table lists, refusal paths, and a post-conversion startup test. Tests must use temporary databases and never touch a live deployment database. A startup refusal does not promise zero SQLite/WAL side effects.

## Development

Build, test, lint, and governance commands are owned by [AGENTS.md](AGENTS.md). Toolchain versions are owned by [`go.mod`](go.mod) and the `engines` field of [`web/package.json`](web/package.json). In short: `make dev` runs the app, `make build` builds, `make test` runs Go tests, `make check` runs the deployment regression plus frontend dependency checks.

## 参与贡献

感谢你对 GPT-Load 的兴趣。

## 开始之前

- **安全问题**：不要提交公开 issue，按 [SECURITY.md](SECURITY.md) 的报告指引操作。
- **使用问题**：查阅[官方文档](https://www.gpt-load.com/docs)及 [README.md](README.md#project-support) 中的 Telegram 群。
- **较大改动**：先开 issue 讨论方案，再投入 pull request。

## 产品边界

- 本仓库保留标准 API 与云凭据模型及外部 CLIProxyAPI 网关。不要重新引入内置订阅授权、OAuth 账号观察或嵌入式 CPA 执行。Codex / Claude Code 是下游客户端，不是上游订阅渠道。
- 破坏性存储变更没有原地迁移：仅支持离线转换（`gpt-load convert-legacy-db`）或全新数据库。转换器改动必须覆盖保留/丢弃表清单、拒绝路径与转换后启动测试。测试必须使用临时数据库，不得连接业务部署库。启动拒绝不承诺 SQLite/WAL 零副作用。

## 本地开发

构建、测试、lint 与治理命令由 [AGENTS.md](AGENTS.md) 拥有。工具链版本由 [`go.mod`](go.mod) 与 [`web/package.json`](web/package.json) 的 `engines` 字段拥有。简言之：`make dev` 运行应用，`make build` 构建，`make test` 跑 Go 测试，`make check` 跑部署回归加前端依赖检查。
