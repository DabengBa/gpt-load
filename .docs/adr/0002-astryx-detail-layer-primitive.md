---
description: Astryx 详情层原语决策——Dialog 右侧面板 vs BottomSheet vs swizzled 侧栏
kind: adr
topic: astryx-migration
status: accepted
date: 2026-09-25
---

# ADR-0002：详情层以 Astryx `Dialog` 做右侧面板，不引入 BottomSheet 或 swizzled 侧栏

## 背景

Phase 1 spike(b) 要求为日志详情层在 `Dialog` / `BottomSheet` / swizzled
侧栏三者间选定原语。该选择影响后续所有详情面（日志、组详情、编辑表单等），
一旦上层组件定型再换原语成本高，属于难逆转决策。

参照物：classic `AppDrawer`（reka-ui `Dialog` 全自定义为右侧面板——
焦点陷阱、Esc、scrim 关闭、焦点归还、`dismissible` 守卫、≤520px 全幅）。

## 方案对比

| 方案 | 焦点陷阱/Esc/归还 | 面板外壳 | swizzle | 结论 |
|---|---|---|---|---|
| `Dialog` `position={{end:0}}` | 原生 `<dialog>` 全给 | xstyle：全高/无圆角/左边线/≤520px 全幅 | 0 | **选定** |
| `BottomSheet` | 原生 dialog 生命周期同给 | 底部锚定+snap+手势——移动形态，抑制手势属重覆盖 | 1+ | 否决 |
| swizzled 侧栏（自造 reka 等价物） | 需自实现 | 全自定义 | 1（且复制已成历史包袱的模式） | 否决 |

## 原生 `<dialog>` 的三个实测要点

1. **常驻挂载**：Dialog 在 `isOpen` 上升沿捕获 `document.activeElement`、
   下降沿归还焦点；条件渲染卸载会跳过该生命周期——宿主必须以
   `isOpen={selected !== undefined}` 驱动，而非 `{selected && <Panel/>}`。
2. **`data-autofocus` 标记**：组件级 autofocus 在 `showModal` 前 commit
   被静默丢弃，Dialog 仅在打开后对焦第一个 `[data-autofocus]` 后代——
   详情层把该标记放在 `LayoutContent`（滚动区，等价 reka 聚焦 content 根）。
3. **边界焦点静默逃逸**：Chromium 原生 modal 在 tab 序边界把焦点落到
   `<body>` 且不派发 focus 事件，`focusin` 重定向无法捕捉；与 reka 哨兵
   同机制，在 `Dialog onKeyDown` 拦截 Tab/Shift+Tab 做硬收容
   （`DetailPanel` 内实现，另留 `focusin` 守卫兜事件化逃逸）。

## 结论

- 详情层原语 = Astryx `Dialog` 右侧定位 + `DetailPanel` 封装
  （`title`/`subtitle`/`dismissible`→`purpose` info|required/`footer`）。
- 本 spike swizzle 计数 **0**（gate #5 输入）。
- 焦点行为契约由 `web/e2e/astryx-log-detail.spec.ts` 钉死：
  activeElement 全程不出 dialog、Esc/scrim 关闭、焦点归还触发行、
  `?selected_request_id=` 深链直达、480px 全幅。
