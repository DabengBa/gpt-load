---
target: web/src/frontends/astryx/features/monitor/ScheduleView.tsx
mode: Operate
total_score: 25
max_score: 40
na_heuristics: ""
p0_count: 0
p1_count: 1
method: dual-agent
assessment_a: 01a16b9d-7749-7759-a5f3-49c8ef8d8ab7
assessment_b: 01a16b9d-774e-7759-a5f3-49caadfe813a
---

# 调度中心 UI 检查

本报告先在对话中交付，再归档。仅检查，不修改产品代码。
手工归档，不生成内容指纹、hash 或 baseline；没有修改既有用户文件。

## 设计判断

紧凑表格、克制配色、行内编辑和高级选项折叠符合 GPT-Load 运维控制台定位。
不需要营销式大卡片、装饰或增加留白。优先改善草稿保护、窄屏排障效率和关键信息完整呈现。

## Nielsen 评分

| 维度 | 分数 /4 | 依据 |
|---|---:|---|
| 系统状态可见性 | 3 | 有状态、观测时间和草稿预览 |
| 业务语言匹配 | 3 | 术语准确，占比含义仍可加强说明 |
| 用户控制与自由 | 2 | 推理策略草稿保护不完整 |
| 一致性 | 3 | 视觉统一，配置提交方式不完全一致 |
| 错误预防 | 2 | 有校验，但部分操作会丢失草稿 |
| 识别优于记忆 | 2 | 截断与横滚增加对象辨认成本 |
| 操作效率 | 3 | 批量测活、URL 上下文 |
| 美观与极简 | 3 | 内容相关、视觉克制 |
| 错误诊断与恢复 | 2 | 非法数值缺少可见纠错说明 |
| 就地帮助 | 2 | 并发有说明，其他关键规则较弱 |
| 总计 | 25/40 | 可用，但存在明显操作摩擦 |

技术暂评 14/20：可访问性 3、性能 3、主题 3、响应式 3、实现完整性 2。
未完成完整性能、读屏、对比度测试，不作为合规认证。

## 优先问题

### [P1] 推理策略草稿无提示丢失

已用浏览器和请求记录复现两条路径：

1. 推理策略改为 high 后，切换模型再切回，或刷新页面，恢复原值，无确认、无 PATCH。
2. 推理策略改为 high 后行启停成功，revision 更新时清空尚未保存的推理草稿；PATCH 仅有 enabled。

位置：`ScheduleView.tsx:181`、`SchedulePanelDetail.tsx:931,1154`，
均在 `web/src/frontends/astryx/features/monitor/`。

原因：本地 reasoningDrafts 没有完整参与 URL 草稿、切换确认和重新同步保留逻辑。
建议统一所有待保存配置的草稿及离开保护；保留现有保存栏外观。命令：harden。

### [P2] 中等屏幕分离候选身份与诊断操作

900px 宽度仍保留约 240px 索引，详情表约有 518px 内部横向溢出。
状态和恢复在屏外，横滚后缺少固定身份列，增加跨分组误对应风险。
位置：`SchedulePanel.tsx:43`、`SchedulePanelDetail.tsx:444`。
建议中等宽度收起索引，或固定候选身份列。不是页面级横向溢出。命令：adapt。

### [P2] 名称和统计截断，缺少完整披露

长模型、分组名称仅保留共同前缀，成功率易被截断；手机双列也存在。
位置：`SchedulePanelDetail.tsx:526,1530,1562`。
建议保持紧凑，为完整名称提供键盘及触屏可用披露，独立保留成功率。
不能只增加 hover title。命令：adapt / clarify。

### [P2] 手机先经过完整模型列表才能进入详情

390px、五模型情景下首个候选约从 y=537 开始，已指定模型仍需经过所有索引。
位置：`SchedulePanel.tsx:278`。
建议默认显示当前模型和切换入口，按需展开列表，不缩小触控操作。命令：adapt / distill。

### [P2] 错误反馈断点

1. 非法优先级 0 红框且禁用保存，范围说明通过辅助技术关联，但截图不可见。
   位置：`SchedulePanelDetail.tsx:626,780`。
2. unknown schedule_model 显示空详情标题，没有解释和重新选择引导。
   位置：`SchedulePanel.tsx:134`。

建议错误时就地显示范围；失效模型显示明确空态和重新选择入口。命令：clarify / harden。

## 其他观察

[P2] 手机推理 Selector 内部触发按钮测得约 114×20 CSS px。
位置：`SchedulePanelDetail.tsx:1573`。建议检查并扩大手机命中区域。
未验证外围命中区、间距例外和物理设备，不直接判 WCAG 违规。

## 优点与使用者视角

- 候选配置、运行态同处一行，适合跨分组比较。
- 占比草稿有预览标记，不把未保存值伪装成服务器事实。
- 高级并发渐进展开，批量测活对停用组确认范围。
- 明暗主题、手机卡片化已有基础，不需要替换视觉语言。

Alex：中等屏幕来回横滚；Sam：重复控件需更明确分组上下文；
Casey：完整模型列表和名称截断拖慢现场排障。
认知负荷中等，在中等屏幕排障时偏高；主要来自信息分离，不是密度本身。

## 证据

现有 E2E：35/35 通过，输出 `/tmp/gpt-load-schedule-ui-audit-tests/`。
静态 detector 单轮检查四个 TSX，退出 0，零命中；不覆盖草稿行为缺陷。
浏览器使用真实当前 Vite 前端和模拟 API，未访问生产。
覆盖 1440px、900px、390px，以及明暗主题、长名称、异常状态和保存错误。
完整读屏、物理触控和生产性能未测。

- 中文桌面：`/tmp/critA/desktop-1440.png`
- 中文中等屏幕：`/tmp/critA/tablet-900.png`
- 中文手机：`/tmp/critA/mobile-390.png`
- 非法输入：`/tmp/critA/desktop-dirty.png`
- 推理草稿交互记录：`/tmp/assessment-b-extra.json`
- 明暗截图：`/tmp/assessment-b-1440-light.png`、`/tmp/assessment-b-1440-dark.png`、
  `/tmp/assessment-b-390-light.png`、`/tmp/assessment-b-390-dark.png`

## 待用户选择

1. 下一步优先修草稿丢失、优化窄屏布局，还是一起处理？
2. 范围选择：仅处理高优先级、处理全部问题，或先给布局方案不改代码？
