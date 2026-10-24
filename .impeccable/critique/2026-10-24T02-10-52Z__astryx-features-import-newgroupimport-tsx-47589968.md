---
target: "/import?mode=new"
resolved_target: "web/src/frontends/astryx/features/import/NewGroupImport.tsx"
slug: "astryx-features-import-newgroupimport-tsx-47589968"
date: "2026-10-24T02:10:52Z"
total_score: 27
max_score: 40
na_heuristics: ""
p0_count: 0
p1_count: 0
p2_count: 5
---

# 新建分组导入页检查

Method: dual-agent (A: 01a16b9e-156e-7759-a5f3-49cde90316c0 · B: 01a16b9e-158f-7759-a5f3-49cf423a3c35)

## 结论与范围

保留现有紧凑控制台风格，优先修复默认状态、错误反馈和移动端弹层，不需要整体重做。

已检查桌面 1440×900、移动端 390×844；设计检查覆盖浅色和深色主题。
使用模拟渠道数据和假密钥，未创建真实分组、未修改业务代码。
渠道字段参考仓库真实 OpenAI / OpenAI Compatible schema，非生产目录抓取。

## 设计健康评分：27/40

| 启发式 | 得分 /4 | 主要依据 |
|---|---:|---|
| 系统状态可见 | 2 | 有持续反馈，但凭据诊断相互矛盾 |
| 符合用户心智 | 3 | 渠道 → 凭据 → 模型顺序合理 |
| 用户控制 | 3 | 可编辑、可返回，不强制分步向导 |
| 一致性 | 3 | 共享控件与分区风格统一 |
| 错误预防 | 2 | 默认地址和搜索替代项容易误导 |
| 识别优于记忆 | 3 | 当前渠道、必填及可选状态清楚 |
| 操作效率 | 3 | 预设、搜索、发现和手工添加并存 |
| 简洁性 | 3 | 适合运维工具，无装饰性干扰 |
| 错误恢复 | 2 | 提示具体但存在错误归因 |
| 上下文帮助 | 3 | 地址、密钥用途和存储说明位置合理 |

交互结构符合 GPT-Load 配置任务，而非通用仪表盘模板。机械检测器返回零发现，
不代表可访问性或响应式检查全部通过。以下问题由人工、源码和浏览器交叉检查发现。

## 优先问题

### P2：自定义上游地址默认开启，给官方渠道制造额外必填项

OpenAI 已显示默认地址，却同时开启一个空的自定义地址输入框。填写一个有效格式
的假密钥后仍不能创建；关闭开关后即可继续。

- 位置：`web/src/frontends/astryx/features/import/NewGroupImport.tsx:168–170,274`。
- 影响：用户需重复填写默认地址，或自行发现关闭开关才能直接使用默认地址。
- 建议：可选地址默认使用渠道默认值，仅在草稿已有覆盖值时开启自定义。
  需要地址的兼容渠道仍保持必填。适合 `harden`。
- 截图：`/tmp/gpt-load-import-audit/A-desktop-light-initial-v2.png`、
  `B2-openai-valid-default.png`、`B2-override-empty.png`（后两项在同一目录）。

### P2：移动端其他渠道弹层裁切内容

390px 宽度下，弹层内搜索框前部、搜索图标及渠道图标被裁切，等待动画结束后仍存在。
主页面没有横向溢出，不代表弹层正常。

- 位置：`web/src/frontends/astryx/components/ChannelPresetPicker.tsx:124–128,382–391`。
- 影响：不常用渠道的识别线索缺失，搜索控件显得损坏。
- 建议：弹层宽度受可用视口约束，位置允许避让，内部最小宽度不得撑破容器。
  保留紧凑列表。适合 `adapt`。
- 截图：`/tmp/gpt-load-import-audit/A-mobile-selector-settled-v2.png`。

### P2：凭据说明、字段错误与底栏提示矛盾

1. 文案承诺相同密钥重复行按一个计算，实际重复两行也会拒绝。
2. 不完整 JSON 被提示删除其他凭据，没有指出 JSON 格式错误。
3. 两条不同密钥时，字段要求删除其他凭据，底栏却要求至少填写一条凭据。

- 位置：`NewGroupImport.tsx:235–237,319–329`、`CredentialTextarea.tsx:156,177–195`、
  `single-credential-input.ts:3–12`（均在 `web/src/frontends/astryx/features/import/`），
  `web/src/shared/i18n/locales/zh-CN/import.ts:110,130`。
- 影响：用户按提示操作也可能无法解决问题。
- 建议：区分空输入、多条凭据和 JSON 格式错误，统一字段、步骤摘要与底栏的诊断；
  删除与当前单凭据规则不符的旧文案。适合 `clarify`、`harden`。
- 截图：`/tmp/gpt-load-import-audit/A-desktop-light-invalid-v2.png`、
  同目录 `B2-duplicate-error.png`、`B2-json-error.png`。

### P2：凭据输入框未关联帮助、错误及必填状态

浏览器确认 textarea 有可访问名称和 aria-invalid，但没有 required / aria-required，
也没有 aria-describedby 关联存储说明及错误文本。

- 位置：`web/src/frontends/astryx/features/import/CredentialTextarea.tsx:177–197`。
- 影响：辅助技术用户不能获得与视觉用户同等明确的字段上下文。
- 建议：使用现有 Field 机制关联帮助、错误和必填状态，并验证实际读屏输出。
  适合 `harden`；DOM 检查不等同于完整 WCAG 认证。
- 原始证据：`/tmp/gpt-load-import-audit/B2-evidence.json`。

### P2：渠道搜索无结果时，兼容渠道看起来像普通匹配结果

输入不存在的渠道名称仍出现 OpenAI Compatible，没有说明它是替代方案。

- 位置：`web/src/frontends/astryx/components/ChannelPresetPicker.tsx:255–267`。
- 影响：用户可能误以为系统识别了供应商，或已经确认其兼容性。
- 建议：先显示未找到匹配渠道，再明确提供使用 OpenAI 兼容渠道的替代入口。
  适合 `clarify`。
- 截图：`/tmp/gpt-load-import-audit/A-desktop-light-nomatch-v2.png`。

## 应保留的部分

- 三个编号区块和必填 / 可选标记清晰，不必改成多页向导。
- 紧凑布局符合管理控制台定位；浅深主题的整体层级一致。
- 窄屏主页面没有横向溢出；抽查的主要移动端按钮高度为 44px。
- 创建前显示目标分组名称；提交及登录恢复已有测试覆盖。

## 认知负担与用户视角

单一主任务、三段分组、明确层级、同屏上下文和可选模型均有效。
四个常用渠道加溢出入口共五个选择超过技能的字面四项阈值，但有搜索辅助，
不应据此降低管理控制台效率。可选地址默认展开并变为必填增加了不必要负担。

- 新手：困惑于默认地址和自定义开关，或误解兼容搜索结果。
- 辅助技术用户：缺少字段说明关联，且底栏与字段的纠错指令矛盾。
- 移动用户：渠道弹层裁切识别线索；软键盘影响未验证。

操作旅程由清楚的三个区块建立信心，在额外地址要求和错误提示矛盾处受挫，
到可创建摘要时恢复明确。主要问题不是视觉风格或信息密度。

## 次要观察

- 移动端只有一个创建按钮时，底栏动作区仍占部分宽度，可后续微调布局。
- 底栏会在中间滚动位置覆盖内容，但滚动到底后仍可访问最后的操作，未证实持续遮挡。
- 未测数值对比度，不能仅凭暗色辅助文本截图宣称 WCAG 违规。
- 未将明文凭据编辑本身列为缺陷：共享屏幕场景需要进一步需求证据。

## 验证

- 导入页 E2E：16/16 通过。
- 单凭据解析测试：2/2 通过；重复行拒绝是当前明确行为。
- Impeccable 检测器：exit 0，零发现，无误报；原始 JSON 在
  `/tmp/gpt-load-import-audit/B-detector.json`。
- 未验证真实读屏、手机软键盘、实体触摸、数值对比度或生产提交。
- 初次预览启动错误已纠正；临时预览已停止，无用户可见的检测覆盖层。
- 本报告记录本次检查，不新增源码指纹、冻结合同或回归基线。

## 待确认

1. 下一步范围：仅修复默认值、校验文案和字段语义，还是连同移动弹层与搜索提示一起修复？
2. 官方渠道地址：默认关闭自定义覆盖，还是保留开启并预填默认地址？
