# 事件报告：schema_migrations 台账与注册表位次失配（2026-10-02）

## 摘要

`dev-700d11a1`（含 PR #155）部署到生产后，`gpt-load` 容器启动即在迁移阶段崩溃并进入 restart 循环，公网 502 约 17 分钟。根因是 **PR #155 删除了迁移 `0009_price_multipliers`，但迁移执行器仍按数组位次严格比对 DB 台账**，凡应用过该迁移的库都无法启动新版本。处置方式为在线备份后删除台账中的 `0009` 记录行，服务恢复。

同日晚间第二次停机（约 18 分钟）为 `sqlite-maintenance.sh` 离线维护的设计行为（不带 `--start-command` 时保持停机），非故障。

## 时间线（UTC）

| 时间 | 事件 |
|------|------|
| 22:59 | `deploy.sh` 将镜像切至 `dev-700d11a1` |
| ~23:00 | 启动失败：`schema_migrations contains unknown or non-contiguous migration "0009_price_multipliers"`，容器循环重启，公网 502 |
| 23:17 | `docker stop` → `DELETE FROM schema_migrations WHERE id='0009_price_multipliers'` → 启动，health 通过 |
| 23:55 | 执行离线维护（tar 归档 3.3G + WAL checkpoint + VACUUM + integrity_check），按设计停机 |
| 次日 00:13 | `docker start`，服务恢复；DB 体积 6.95G → 260M |

## 技术根因

1. `0009_price_multipliers` 由 #584 引入，为 `groups`/`access_keys` 增加 `price_multiplier_micros` 列。
2. PR #155（`3464036f`）删除了该迁移文件与注册表项，并把 `validateMigrationRegistry` 放宽为"编号递增即可"（允许注册表留空位）。
3. 但 `applyMigrationsLocked` 中的台账比对未同步放宽，仍是位次一一对应：

   ```go
   for index, id := range applied {
       if index >= len(entries) || entries[index].ID != id {
           return fmt.Errorf("schema_migrations contains unknown or non-contiguous migration %q", id)
       }
   }
   ```

   台账第 9 位是 `0009_price_multipliers`，注册表第 9 位已是 `0010_...` → 判定 non-contiguous → abort。

4. **影响面**：先运行过含 #584 的任意版本（台账写入 `0009`）、再升级到 ≥`3464036f` 的实例必崩；全新部署不受影响。以后每次"删除迁移"都会踩同一个坑。

## 修复建议

- **首选**：`applyMigrationsLocked` 增加 known-removed 集合，如 `removedMigrationIDs = {"0009_price_multipliers"}`。遍历台账时：ID 在移除集内则跳过且不消耗注册表位次；否则仍严格比对。配套单测覆盖"台账含已移除迁移"与"台账含未知迁移仍报错"两个场景。
- **备选**：删除迁移时保留 tombstone 占位项 `{ID, Up: no-op}`，维持台账与注册表位次一致（fresh install 会写入该行，行为统一但台账略冗余）。
- 孤儿列 `price_multiplier_micros` 仍残留在 `groups`/`access_keys` 表，无害；如需清理可另起迁移 `DROP COLUMN`。

## 运维备忘

- 本生产库台账已删除 `0009` 行（现 23 行，与 `dev≥3464036f` 对齐）。**回滚到 `dev-41c70d7c` 及更早版本前须先插回该行**，否则旧代码的严格连续校验同样失败。
- 恢复点保留：`/opt/gpt-load/backups/gpt-load-offline-KqJ6JHm7.tar.gz`（3.3G，维护前完整数据目录，含密钥）。
- `sqlite-maintenance.sh` 不带 `--start-command` 时按设计保持停机，等运维手动拉起并验证 `/health` 与加密配置读取；期望自动拉起需补 `--start-command/--health-url/--verify-command`。
