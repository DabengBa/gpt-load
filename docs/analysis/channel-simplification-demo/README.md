# 渠道精简：需求探索初稿

从 [六格方向总览](thumbs.html) 打开，或直接选择：

- [A 连接工作台（推荐）](dir-1.html)：配置和测试按顺序放在一起，右侧同步展示边界。[首屏截图](dir-1-first.png) / [整页截图](dir-1-1440.png)
- [B 模型验证台](dir-2.html)：结果优先，配置在下，现状与边界在左侧。[首屏截图](dir-2-first.png) / [整页截图](dir-2-1440.png)
- [C 变更边界清单](dir-3.html)：先比较现状与风险，再并排配置与测试。[首屏截图](dir-3-first.png) / [整页截图](dir-3-1440.png)
- [六格截图](thumbs-first.png)；D / E / F 是缩略备选，尚无完整交互。各页面也有 `*-375.png` 窄屏截图。

**需要用户选择方向；推荐不等于已选择。** 可选择 A/B/C，也可选一张备选或说明希望混合的部分。

浏览器直接打开 HTML 即可，不需要服务、依赖、网络或外部资源。各完整方向均可切换连接类型，输入虚构 Base URL 和模型，保存至内存后模拟三种结果。Key 固定为 `demo-key`，不能输入真实 Key；请勿在其他输入框粘贴真实凭据。没有网络请求、持久化、模型发现或真实测试。

## 复验

环境需已有 Playwright 和其 Chromium；无须安装或联网。示例（将 `NODE_PATH` 换成本机 Playwright 所在 `node_modules`）：

```sh
NODE_PATH=/mnt/projects/repos/pi-web-develop/node_modules node /mnt/projects/repos/pi-web-develop/node_modules/playwright/cli.js test docs/analysis/channel-simplification-demo/demo.test.cjs --workers=1 --reporter=line
NODE_PATH=/mnt/projects/repos/pi-web-develop/node_modules node docs/analysis/channel-simplification-demo/verify.cjs
```

浏览器测试覆盖保存前校验、四个专用入口、三种模拟结果、配置修改立即标为未保存并使旧结果过期、长模型名换行、B 版键盘前往测试、刷新清空、控制台错误和网络请求；截图脚本检查 1440 与 375 宽的页面打开、横向溢出、错误与网络请求，并生成首屏和整页截图。`build.py` 用标准库生成三个独立 HTML；浏览不需要执行它。

设计依据、六张方向卡、局限和尚未通过的选择门见 [directions.md](directions.md)。本 demo 是可丢弃的设计探索，不是生产功能实现。现有 OpenAI Compatible 默认对指定模型进行 Chat 非流式测活；其他专用渠道有各自的探测预设。本演示所有类型只模拟 Chat 非流式结果，不代表专用渠道的实际探测。
