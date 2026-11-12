# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## 定位与用户

GPT-Load 是自托管的 LLM 网关；管理界面是其控制面，不是营销网站。管理员配置上游、凭据和访问密钥，管理调度，并通过用量和请求日志排查问题；访问密钥（`access_key`）主体使用只读、权限限定的视图。成功意味着配置、查找、检查、修改和验证都清晰可达，生产状态不被隐藏，也不会让操作者误判。


## 产品与上游边界

- **分组、渠道/Provider、协议不是同义词。** 分组是管理员维护的上游配置和调度单位，包含渠道配置、凭据、模型和运行控制；渠道/Provider 指上游服务及其执行能力；客户端协议描述下游调用形状和授权，不代表一个分组或一个 Provider。访问密钥可限制可用分组和客户端协议。参见 [分组语义](.docs/db/pages/groups.md) 和 [支持范围](README.md#client-protocols)。
- 标准 API key 与结构化云凭据经 Bifrost 执行；外部 CLIProxyAPI 是独立运营、以 base URL 和 API key 连接的网关。内置 Codex/Claude/Antigravity/Grok 订阅授权、账号观察和嵌入式 CPA 执行已移除。Codex、Claude Code 仍是下游客户端，不是上游订阅渠道。
- 此破坏性版本不能原地打开旧数据库。停止旧实例后，用 `gpt-load convert-legacy-db` 离线转换（保留分组、API 凭据、访问密钥和额度状态；丢弃请求日志及用量历史），或使用全新数据库；不提供原地迁移、自动删除或兼容层。详见 [启动警告](README.md#quick-start) 和 [旧库转换](.docs/tech/legacy-db-conversion.md)。

## 当前界面与设计方向

**当前已实现：** Astryx 是唯一管理界面，采用顶部导航；首页提供实例摘要、网关连接及近 30 天花费/模型排行，空实例有欢迎引导，访问密钥主体可见其权限范围内的信息。首页花费固定近 30 天，没有时间切换器；其客户端和所选访问密钥使用 URL 状态。见 [HomeView](web/src/frontends/astryx/features/home/HomeView.tsx)、[HomeSpend](web/src/frontends/astryx/features/home/HomeSpend.tsx) 与 [首页路由状态](web/src/shared/routing/home-route.ts)。

**参考图启发的通用管理界面：** 借鉴布局、层级和细边框，以明亮的白色/浅灰表面与蓝色导航和控件统一现有路由；截图只提供视觉风格，不要求建设运营总览或新增路由。侧栏是可探索的导航呈现，须保留现有入口、角色边界和功能。[分组管理演示](docs/design/management-demo.html) 使用合成数据，离线展示列表表格、搜索和详情；它不是生产界面或实时数据。视觉依据和边界见 [DESIGN.md](DESIGN.md)。

保留首页已有信息及其可达性；费用是估算而非账单，缺失用量、部分数据和未定价不能当成零。统计和费用细节分别由 [用量核算](.docs/tech/usage-accounting.md) 与 [冻结模型定价](.docs/tech/frozen-model-pricing.md) 负责。

## 体验原则

- 专业、紧凑、可操作：信息密度服务任务；以适当留白建立层级，表格和配置页保持高效扫描，不以旧版 Vue 的像素密度作为唯一视觉标准。
- 参考图用于构图与层级，而非复制其产品语义、装饰或窗口外壳；生产共享主题已采用亮色方向，侧栏演示仍是提案。
- 集合的过滤、分页等可分享状态以 URL 为准；首页当前 URL 只编码客户端及访问密钥选择，勿假定所有选择都已进入 URL。渐进披露保留列表上下文，详情层和筛选抽屉按需展开。
- 状态、权限、单位和数据时间范围清楚可见；禁用、空、失败、过期和未知不能伪装成成功或零。
- WCAG AA 为目标：可见键盘焦点、可访问名称、正确地标，弹层管理并恢复焦点，尊重减少动态效果偏好。色彩之外仍需文字或形状表达状态。

反例：营销式大标题、装饰渐变、玩偶、过度阴影、为了好看牺牲行密度的巨型卡片，以及无业务意义的说明文案。更多界面组织边界见 [前端结构](.docs/tech/frontend-structure.md)。
