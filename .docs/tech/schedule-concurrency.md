---
description: "调度行并发上限的运行时语义、亲和优先边界和请求生命周期。"
kind: technical
topic: schedule-concurrency
code:
  paths:
    - internal/state/capacity.go
    - internal/scheduler/scheduler.go
    - internal/gateway/capacity.go
    - internal/gateway/attempts.go
    - internal/gateway/websocket_turn.go
    - internal/control/model_route_schedule.go
---
# 调度行并发上限

## Owner 与范围

本文档拥有调度行并发限制的实现事实：配置贯通、进程内容量记录、原子准入、候选跳过、释放和容量错误。产品可见的配置与调度行为由 [调度行并发上限](../db/features/dispatch-concurrency.md) 拥有；亲和绑定、身份代际和健康绕过仍由 [Affinity Binding](affinity-binding.md) 拥有。

## 容量键与配置

容量按 `RouteEntryKey`（`GroupID + EntryID`）记录。`concurrency_limit` 存在于模型条目 JSON、运行时 `ModelConfig` 和 `RouteTarget`，缺失归一为 0；0 表示不限。数值范围是 0 到 2147483647。调度中心 PATCH 的字段缺失保持原值，显式 0 解除限制，`null` 被拒绝。

容量记录属于单个服务进程。配置快照发布和容量状态更新在同一进程锁边界内协调；已有 reservation 不因降低上限、切换 0 或删除条目而中断。删除条目禁止新的准入，旧 reservation 释放时仍使用自己的记录；重新创建的条目不会被旧释放误扣。

## 准入顺序

调度候选先经过既有协议、权限、模型、启用、权重、优先级、健康和亲和资格。普通候选在正式尝试前申请容量；容量不足只跳过当前 `RouteEntryKey`，不会写入 scheduler 的 credential/upstream tried pair，也不会增加执行重试预算或产生 provider attempt。所有候选都被容量阻塞时，网关返回 `503 schedule_capacity_exhausted`。

本地 payload、凭据解密和代理准备完成后仍须再次进行原子准入，防止预检查与实际发送之间超发。准入成功后才标记正式 attempt。容量拒绝不执行健康反馈；已有真实上游失败时，原失败证据保留，容量原因作为独立的请求级结果补充。

## 亲和与固定身份

有效亲和请求沿用 [Affinity Binding](affinity-binding.md) 的绑定身份和资格检查；匹配 group、credential、identity generation 且当前条目可用时，即使达到普通上限也可申请名额，并继续计入实际占用。容量超限本身不会产生健康绕过或绑定接管。

WebSocket 的 `binding` 或 `requiredRef` 固定身份是连续性约束，不自动等同于有效亲和。固定身份没有亲和豁免且条目满载时，连接返回本地容量错误，不改投其他身份；只有没有固定身份的普通 turn 才能在容量阻塞后选择其他候选。有效亲和固定身份可以超限。

## 生命周期

HTTP 和流式请求在实际 Forward/ForwardStream 期间各持有一个 reservation；返回、失败、取消、超时和预派发终止均幂等释放。认证刷新、首供应商重放和内部重试释放旧 reservation 后重新申请。WebSocket 在握手和活动 ExecuteTurn 期间占用，等待下一条 turn 的空闲连接不占用。释放使用 reservation 自带的记录，不重新读取当前条目配置。

## 验证

容量单元测试覆盖原子竞争、亲和超限、条目精确跳过、热更新/删除、HTTP/流、WebSocket 固定身份及重放生命周期。控制台测试覆盖配置 round-trip、非法输入、草稿 URL、冲突保留和字段投影。系统 Chrome 可启动新增并发 UI 用例；完整 schedule E2E 仍有一个未改动的 models mock 投影契约失败。
