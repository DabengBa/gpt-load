---
description: AccessKey collection 读路径规模实测与「内存过滤 vs SQL 下推」决策
kind: adr
topic: collection-read-model
status: accepted
date: 2026-09-25
---

# ADR-0001：集合读模型保留「快照内全量捕获 + 内存过滤」，暂不做 SQL 下推

## 背景

管理面集合读路径按资源重复三件套：`*_http`（query 解析）→
`*_collection`（`withReadSnapshot` 全量捕获）→ `*_query`（内存过滤/排序/分页）。
架构评审候选 #4 提出收敛为共享读模型；前置门槛是先用实测数据回答
「全量捕获 + 内存过滤在当前规模是否可接受」。

## 实测

基准：`internal/storage/models/access_key_collection_scale_test.go`
（`BenchmarkAccessKeyCollectionScale`），复刻
`captureAccessKeyCollectionRecords` 的完整查询形状——
`access_keys` 全量 `Select` + 每行对 `request_logs` 的相关子查询
`MAX(completed_at_ms)` + 成本规则表 `Find` + 内存分组/排序。
环境：Windows amd64，Ryzen 5 2600X，文件型 SQLite + WAL，4 连接池。

| 规模 | 延迟/op | 说明 |
|---|---|---|
| 1k keys，0 request_logs | ~6.0 ms | 典型规模 |
| 1k keys，500k request_logs | ~19.9 ms | 相关子查询经 `idx_request_logs_access_completed_id` 索引探测 |
| 10k keys，500k request_logs | ~153.6 ms | 10× 压力，线性放大 |

## 结论

1. **典型规模（≤1k AccessKey）下全量捕获 ~6–20ms**，对管理页列表完全可接受；
   内存过滤/排序/分页在该量级为 µs 级，不是成本项。
2. **唯一随规模增长的项是逐行相关 `MAX` 探测**（~14µs/行 @500k logs）。
   `request_logs.access_key_id` 已有覆盖索引
   `idx_request_logs_access_completed_id(access_key_id, completed_at_ms DESC, id DESC)`，
   探测为 O(log) 索引查找而非全表扫描，10× 规模仍线性、无悬崖。
3. **决定**：保留声明式读模型 + 内存过滤；P3 可以推进「收敛三件套」试点，
   但不得把过滤/分页下推到 SQL 作为前提。
4. `withReadSnapshot` 一致性语义必须保留——无论内存过滤还是未来下推，
   捕获都在同一读事务内完成。

## 复测触发条件（出现任一即重开本决策）

- AccessKey/Group/Credential 任一集合在实测中突破 **10k 行**；
- 集合接口 P95 延迟实测 > **200ms**；
- `request_logs` 规模使逐行探测项主导延迟（当前 ~14µs/行，万级 keys 时 ~150ms）。

届时的首选修复不是给每行做子查询，而是单次 `LEFT JOIN + GROUP BY`
聚合 `MAX(completed_at_ms)`，或将 `last_request_at_ms` 降级为
写入侧维护的非规范化列——两者都不改变读快照语义。

## 关联

- 评审报告：`docs/analysis/architecture-review-briefing.md`（候选 #4）
- 实施计划：`docs/design/architecture-deepening-plan.md`（P3 节）
- 读路径代码：`internal/control/access_key_collection.go`、`internal/control/service.go` `withReadSnapshot`
