# 事实依据与未核实项

- 现有分组和模型入口、保存后的模型行测活：`.docs/db/pages/groups.md`。
- 现有 OpenAI Compatible 的默认探测契约为 `protocol.OpenAICompletions`，针对指定模型发起非流式测活；其他专用渠道按各自契约和预设探测，不一定是 Chat：`internal/channel/modules/openai_compatible.go`、`internal/channel/modules/google_vertex.go`、`.docs/tech/realistic-probe.md`。它不是统一的可编辑请求工作台：`.docs/db/features/realistic-probe.md`。
- 旧库不能原地打开，需离线转换或使用新库：`PRODUCT.md`。这与建议在未来精简方案中不静默迁移已有组是两个不同的边界。
- Open WebUI、LiteLLM、Dify 的比较是用户提供的研究线索；因离线约束未联网验证，未在界面作为事实使用，也未做外部截图参照。
