# Spec: Astryx 迁移 Phase 2(低耦合域:settings / models / home + model-prices 嵌入)

## 意图与核心流程

一句话:在 Phase 1 已批准的脚手架与共存契约上,按方案文档「Phase 2: Low-coupling domains」把 `settings`(约 2.9k)、`models`(约 4.7k)、`home`(约 4.7k)三个路由域迁移到 Astryx 前端,`model-prices`(约 1.1k)作为嵌入面随宿主域一并迁移。

- 触发条件:Phase 1 gate 已通过(7/7,记录在 `.docs/tech/astryx-migration-plan.md` Phase 1 outcome),用户答复 "go";Windows 编译限制已由 WSL2 构建真实 linux 二进制解决(CSP 15/15 实测复验)。
- 主路径:逐域迁移,每域独立完成「迁移 → manifest flag → e2e 双项目 → 评审」,顺序 `settings` → `models` → `home`(耦合从低到高;model-prices 嵌入 settings/models 的相应段落随宿主落地)。
- 成功状态:三域在 astryx flag 下达到 classic parity(行为/URL 契约/密度/键盘可达);classic 保持默认可达、功能完整;两前端 Playwright 项目各自全绿。

## 范围 / 不做范围

本次要做:

1. **settings 域**:`/settings` 整页——`section` query 驱动的六个区段(routing / connection / reliability / browser-access / data-maintenance / system),`use-settings-controller` 的读写流(patch 语义、read_only、dirty tracking、错误/冲突呈现),以及嵌入其中的 model-prices 相关段落。路由 query 契约按 `settings-route.ts` 复刻(canonical 化、默认 `section=routing` 不落 URL)。
2. **models 域**:`/models` 整页——模型树/集合、discovery/upstream drawer、alias editor、probe dialogs、price status 嵌入段落;`models-route.ts` 的 query 契约复刻。
3. **home 域**:`/` 整页——欢迎/汇总/attention/spend/连接信息/订阅账户卡片等区段,access_key 只读视图;`home-route.ts` 的 query 契约复刻。
4. **每域同步**:manifest `astryx: true`、前端 flag 下走新文档、对应 e2e spec 覆盖与 classic parity 断言、shared 层缺口补齐(资源/presenter 若仍未抽取)。

不做(明确推迟):

- Phase 3-6:`access-keys`/`monitor`/`schedule`/`logs` 完整域、`groups`/`import`、默认前端切换、classic 删除。
- 产品行为、后端 API 契约、页面语义变化;`.docs/db` 语义文档不改内容(仅 traceability/`code.paths` 指向新文件)。
- 超出 parity 的视觉重设计;不引入新依赖(除方案文档已列清单内的缺漏项)。
- unsaved-guard:import 域不在本 slice;settings 的脏表单若 classic 有离开确认行为,则 astryx 侧用 `useBlocker`+确认层实现等价语义(接线合同已在 Phase 1 评审中更正:resolve 语义 + `withResolver`)。

## 边界规则 / 验收

### 域级 definition of done(逐域)

- manifest 条目 `astryx: true`;该域 e2e spec 在 `classic` 与 `astryx` 双项目全绿(或 astryx 侧有等价 spec)。
- URL query 契约一致:`parse`/`serialize`/canonical 化行为与 classic 相同(含默认值不落 URL、非法值剔除、重复键处理);`resetScroll:false` 用于 query-only 导航。
- 导航移交:`RouteLink`/`isAstryxNavigable` 策略不变——从已迁移域跳到未迁移域是 document 导航。
- 密度/键盘/a11y:与 classic 相同量级的指标不漂移;焦点管理(overlay/返回)符合 DetailPanel/Layer 约定;每域至少一次浏览器手工复核(zh-CN + en-US 至少一个、light+dark 至少一个)。
- 共享原则:双前端都需要的逻辑落 `shared/`;astryx 侧不做 call-site xstyle 密度补偿(theme 层解决)。
- CSP/性能:迁移后 `go-csp.spec.ts` 对该路由的清扫仍零违规;首屏 gzip 变化记入 plan。

### 共存期边界(沿用)

- `gpt-load.frontend` cookie 仅选择静态文档;feature-freeze 规则不变(域已迁移后新功能只进 astryx)。
- 浏览器状态键冻结;不允许新增未声明共享键。
- 回滚路径:单域出问题=该域 manifest flag 回退,不影响其它域。

## 架构 / 约束(增量于方案文档)

- 组件复用优先序:既有 astryx `components/`(DetailPanel、ChannelIcon、CredentialHealthBar)→ Astryx 公共 API 组合 → 真的需要才评估 swizzle(swizzle 预算全局 ≤3,当前 0,新增必须在 plan 记录原因与版本)。
- Detail overlay 一律走 `DetailPanel`(ADR-0002),不自建 Drawer。
- 表单区:沿用 Astryx `Field`/`Input*`/`Selector`/`Switch` + `InputStatus` 错误呈现;不可见 label 仅用于表格行内等密度敏感处(可访问名必须存在)。
- `unsavedChanges` 控制器接线语义:`shouldBlockFn` 返回 resolve 的 promise(`useBlocker` withResolver/对称合同),离开-确认 UI 用 Astryx Dialog。
- i18n:新域 namespace 走 `ensureNamespaces`;`verify:astryx-i18n` 保持 key 对齐;`react-intl` ICU 兼容规则不变(`{placeholder}` 变量式)。
- 仓库契约同 Phase 1:TS/ESLint/build/CI 命令集不破;依赖精确版本 + 7 天规则;`internal/webui/dist` 不入库;`gptl-providers-models.html` 不动。

## 数据 / 集成

- `page_routes.json`:本 slice 结束时 `settings`、`models`、`home` 置 `astryx: true`(随各域完成逐项开启)。
- 共享资源:`shared/control/resources/{settings,models,model-prices,home,providers,model-probe,route-inspection,system-info}.ts` 已存在;缺口在 presenter 层按域补抽。
- e2e:每域新 spec 文件命名 `astryx-<domain>.spec.ts`,进 `astryx` 项目;fixture 走 `e2e/fixtures/` 现有模式(请求拦截 + deterministic DTO)。
- WSL2:Go 产物编译/真机 CSP 复验经 WSL2(`go build` → `GPT_LOAD_ORIGIN` 指向 wsl 运行的二进制);harness `internal/webui/cmd/webui` 仍用于快速迭代。

## 验证

| 门槛 | 命令 / 检查 | 时机 |
| --- | --- | --- |
| 类型/构建/lint | `pnpm --dir web exec vue-tsc -p tsconfig.app.json --noEmit` / `tsc -p tsconfig.astryx.json --noEmit` / `tsc -p tsconfig.node.json --noEmit`;`eslint` | 每改动 |
| 单域 e2e | `pnpm --dir web exec playwright test --project=astryx e2e/astryx-<domain>.spec.ts` | 每域 |
| 全量 e2e | `--project=astryx` + `--project=classic` 全套 | 每域完成、切片收尾 |
| CSP(真机) | WSL2 `go build` → `GPT_LOAD_ORIGIN` 矩阵(3 项目 × manifest 驱动路由表) | 每域收尾 |
| i18n | `verify:i18n-icu`、`verify:astryx-i18n` | 每域 |
| 契约 | `test:astryx-routes`、`test:search-codec`、`test:channel-contract`、`test:connection-json` | 每域 |
| 密度/a11y | `astryx-density.spec.ts`、`astryx-visual-audit.spec.ts`(按需扩面)+ 手工浏览器复核 | 每域 |
| 语义文档 | `doc-compiler.js check/build`(若动 `.docs/db`) | 改动时 |
