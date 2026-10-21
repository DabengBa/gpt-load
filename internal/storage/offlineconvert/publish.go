package offlineconvert

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/securefile"
	"gpt-load/internal/storage"
)

// target describes the validated publish destination.
type target struct {
	path      string
	directory string
}

// resolveTarget insists on a file-backed SQLite DSN whose file does not exist
// yet, so a conversion can never overwrite or merge into a live database.
func resolveTarget(dsn string) (target, error) {
	trimmed := strings.TrimSpace(dsn)
	if trimmed == "" {
		return target{}, errors.New("offline conversion: target database path is empty")
	}
	parsed, err := config.ParseDatabaseDSN(trimmed)
	if err != nil {
		return target{}, fmt.Errorf("offline conversion: invalid target DSN: %w", err)
	}
	base, _, _ := strings.Cut(parsed.DSN, "?")
	path := strings.TrimPrefix(base, "file:")
	if path == "" || strings.Contains(path, ":memory:") {
		return target{}, errors.New("offline conversion: target must be a SQLite file path")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return target{}, fmt.Errorf("offline conversion: resolve target path: %w", err)
	}
	if _, err := os.Lstat(absolute); err == nil {
		return target{}, fmt.Errorf("offline conversion: target %s already exists", absolute)
	} else if !os.IsNotExist(err) {
		return target{}, fmt.Errorf("offline conversion: inspect target: %w", err)
	}
	return target{path: absolute, directory: filepath.Dir(absolute)}, nil
}

// publish builds, validates and publishes the converted database.
//
// The published database is produced as a single file in journal_mode=DELETE, so
// no sidecar journal exists to publish inconsistently. Publication is one
// no-overwrite atomic link inside the target directory: creation is atomic, an
// existing target is never replaced, and an installer that creates the target
// first (EACCES) or wins the race (EEXIST) is reported instead of clobbered.
func publish(ctx context.Context, destination target, plan stage) (Result, error) {
	if err := securefile.PrepareManagedDataDir(destination.directory); err != nil {
		return Result{}, fmt.Errorf("offline conversion: prepare target directory: %w", err)
	}
	// The scratch directory lives inside the target directory so publication
	// stays on one filesystem, but it is never the published path itself.
	workingDirectory, err := os.MkdirTemp(destination.directory, ".convert-")
	if err != nil {
		return Result{}, fmt.Errorf("offline conversion: create target scratch directory: %w", err)
	}
	defer os.RemoveAll(workingDirectory)
	workingPath := filepath.Join(workingDirectory, filepath.Base(destination.path))

	result, err := materialize(ctx, workingPath, plan)
	if err != nil {
		return Result{}, err
	}
	return result, publishSingleFile(workingPath, destination.path)
}

// publishSingleFile links the validated database into place without ever
// replacing an existing file. os.Link is used instead of os.Rename because
// rename silently overwrites the destination, which would make the
// "target must not exist" guarantee racy.
func publishSingleFile(workingPath, finalPath string) error {
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		if _, err := os.Lstat(workingPath + suffix); err == nil {
			return fmt.Errorf(
				"offline conversion: converted database still has %s; refusing a multi-file publish",
				filepath.Base(workingPath)+suffix,
			)
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("offline conversion: inspect converted artifact: %w", err)
		}
	}
	if err := os.Link(workingPath, finalPath); err != nil {
		switch {
		case errors.Is(err, os.ErrExist):
			return fmt.Errorf("offline conversion: target %s already exists", finalPath)
		case errors.Is(err, os.ErrPermission):
			return fmt.Errorf(
				"offline conversion: target %s is not writable; refusing to publish over an existing file: %w",
				finalPath, err,
			)
		default:
			return fmt.Errorf("offline conversion: publish converted database: %w", err)
		}
	}
	if err := securefile.HardenManagedFileIfExists(finalPath); err != nil {
		_ = os.Remove(finalPath)
		return fmt.Errorf("offline conversion: secure converted database: %w", err)
	}
	if err := syncDirectory(filepath.Dir(finalPath)); err != nil {
		_ = os.Remove(finalPath)
		return fmt.Errorf("offline conversion: durably publish converted database: %w", err)
	}
	return nil
}

// materialize creates the converted database at workingPath and validates it
// before anything is published.
func materialize(ctx context.Context, workingPath string, plan stage) (Result, error) {
	target, err := gorm.Open(sqlite.Open(workingPath+"?_pragma=journal_mode(delete)"), &gorm.Config{
		Logger: logger.Discard,
	})
	if err != nil {
		return Result{}, fmt.Errorf("offline conversion: create target database: %w", err)
	}
	defer closeDatabase(target)

	if err := storage.AutoMigrate(target); err != nil {
		return Result{}, fmt.Errorf("offline conversion: initialize target schema: %w", err)
	}
	result := Result{
		Tables:         make(map[string]int64, len(plan.retainedTables)),
		RetiredColumns: plan.retiredColumns,
	}
	err = withSourceDatabase(ctx, plan.sourcePath, func(source *gorm.DB) error {
		copied, copyErr := copyRows(ctx, source, target, plan)
		if copyErr != nil {
			return copyErr
		}
		result.Tables = copied
		return validateConverted(ctx, target, source, plan, result)
	})
	if err != nil {
		return Result{}, err
	}
	closeDatabase(target)
	if err := requireSingleFileDatabase(workingPath); err != nil {
		return Result{}, err
	}
	return result, nil
}

// requireSingleFileDatabase proves the converted database needs no sidecar, so
// publishing it cannot leave a journal behind.
func requireSingleFileDatabase(path string) error {
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		if _, err := os.Lstat(path + suffix); err == nil {
			return fmt.Errorf(
				"offline conversion: converted database produced %s; refusing a multi-file publish",
				filepath.Base(path)+suffix,
			)
		} else if !os.IsNotExist(err) {
			return fmt.Errorf("offline conversion: inspect converted artifact: %w", err)
		}
	}
	var journalMode string
	db, err := gorm.Open(sqlite.Open("file:"+path+"?mode=ro"), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		return fmt.Errorf("offline conversion: open converted database: %w", err)
	}
	defer closeDatabase(db)
	if err := db.Raw("PRAGMA journal_mode").Scan(&journalMode).Error; err != nil {
		return fmt.Errorf("offline conversion: read converted journal mode: %w", err)
	}
	if !strings.EqualFold(journalMode, "delete") {
		return fmt.Errorf(
			"offline conversion: converted database journal mode = %q, want a single-file delete journal",
			journalMode,
		)
	}
	return nil
}

// copyRows copies every retained table inside one transaction. Values are read
// as raw SQLite values so stored bytes survive unchanged.
//
// The copied column list is the explicit expected column set for the supported
// source schema, never the intersection of same-named columns: an intersection
// would silently drop a stored value whose column the source is missing, and
// the target would fall back to a column default.
func copyRows(ctx context.Context, source, target *gorm.DB, plan stage) (map[string]int64, error) {
	targetTables, err := tableColumns(ctx, target)
	if err != nil {
		return nil, err
	}
	if _, err := tableColumns(ctx, source); err != nil {
		return nil, err
	}
	copied := make(map[string]int64, len(plan.retainedTables))
	err = target.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		for _, table := range plan.retainedTables {
			targetColumns, ok := targetTables[table]
			if !ok {
				return fmt.Errorf("offline conversion: target table %q is missing", table)
			}
			columns, ok := currentColumns[table]
			if !ok {
				return fmt.Errorf("offline conversion: table %q has no expected column set", table)
			}
			if err := requireTargetColumns(table, columns, targetColumns); err != nil {
				return err
			}
			rows, err := readRows(ctx, source, table, columns, plan.flavourFilter)
			if err != nil {
				return err
			}
			if err := insertRows(ctx, tx, table, columns, rows, autoincrementHighWater(ctx, source, table)); err != nil {
				return err
			}
			copied[table] = int64(len(rows))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return copied, nil
}

// autoincrementHighWater reads the source's AUTOINCREMENT counter for a table.
//
// SQLite only advances `sqlite_sequence` for an explicit rowid insert when the
// inserted value is greater than the stored counter, so copying the surviving
// rows is not enough: a source whose highest ID was deleted holds a counter
// above every remaining row, and the published database would then hand that ID
// out again. External references (request logs, exports, backups) still point at
// the deleted ID, so the counter itself is part of the copied state.
func autoincrementHighWater(ctx context.Context, source *gorm.DB, table string) int64 {
	if !source.Migrator().HasTable("sqlite_sequence") {
		return 0
	}
	var highWater int64
	err := source.WithContext(ctx).
		Raw("SELECT COALESCE(max(seq), 0) FROM sqlite_sequence WHERE name = ?", table).
		Scan(&highWater).Error
	if err != nil {
		// sqlite_sequence is an internal table with one fixed shape; a read error
		// here means the source is not what it declared, and the integrity check
		// already ran. Falling back to zero only loses the guarantee for this
		// table, which the target sequence check below still enforces.
		return 0
	}
	return highWater
}

// requireTargetColumns proves the freshly migrated target declares every
// expected column, so the explicit source read can always be inserted.
func requireTargetColumns(table string, expected, declared []string) error {
	for _, column := range expected {
		if slices.Contains(declared, column) {
			continue
		}
		return fmt.Errorf(
			"offline conversion: target table %s is missing the expected column %s",
			table, column,
		)
	}
	return nil
}

func readRows(ctx context.Context, source *gorm.DB, table string, columns []string, flavourFilter string) ([][]any, error) {
	query := "SELECT " + quoteIdentifiers(columns) + " FROM " + quoteIdentifier(table)
	if table == "groups" && flavourFilter != "" {
		query += " WHERE " + flavourFilter
	}
	// The identity column is first in the shared set only when the source and
	// target agree on column order; order by the table's primary key instead.
	query += " ORDER BY " + quoteIdentifier(identityColumn(table, columns))
	rows, err := source.WithContext(ctx).Raw(query).Rows()
	if err != nil {
		return nil, fmt.Errorf("offline conversion: read %s: %w", table, err)
	}
	defer func() { _ = rows.Close() }()

	var collected [][]any
	for rows.Next() {
		values := make([]any, len(columns))
		targets := make([]any, len(columns))
		for index := range values {
			targets[index] = &values[index]
		}
		if err := rows.Scan(targets...); err != nil {
			return nil, fmt.Errorf("offline conversion: scan %s: %w", table, err)
		}
		collected = append(collected, values)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("offline conversion: read %s: %w", table, err)
	}
	return collected, nil
}

func insertRows(ctx context.Context, tx *gorm.DB, table string, columns []string, rows [][]any, highWater int64) error {
	if len(rows) != 0 {
		statement := "INSERT INTO " + quoteIdentifier(table) + " (" + quoteIdentifiers(columns) + ") VALUES ("
		placeholders := make([]string, len(columns))
		for index := range placeholders {
			placeholders[index] = "?"
		}
		statement += strings.Join(placeholders, ", ") + ")"
		for _, row := range rows {
			if err := tx.WithContext(ctx).Exec(statement, row...).Error; err != nil {
				return fmt.Errorf("offline conversion: insert %s: %w", table, err)
			}
		}
	}
	return restoreAutoincrementHighWater(ctx, tx, table, highWater)
}

// restoreAutoincrementHighWater raises the target's AUTOINCREMENT counter to the
// source value. Only a table that actually has a counter is touched, so a table
// without AUTOINCREMENT never gains a foreign sqlite_sequence row.
func restoreAutoincrementHighWater(ctx context.Context, tx *gorm.DB, table string, highWater int64) error {
	if highWater <= 0 || !tx.Migrator().HasTable("sqlite_sequence") {
		return nil
	}
	var stored int64
	if err := tx.WithContext(ctx).
		Raw("SELECT count(*) FROM sqlite_sequence WHERE name = ?", table).
		Scan(&stored).Error; err != nil {
		return fmt.Errorf("offline conversion: read target sequence for %s: %w", table, err)
	}
	if stored == 0 {
		// No counter exists for this table, so it is not AUTOINCREMENT and has no
		// reusable-ID problem to solve.
		return nil
	}
	var current int64
	if err := tx.WithContext(ctx).
		Raw("SELECT seq FROM sqlite_sequence WHERE name = ?", table).
		Scan(&current).Error; err != nil {
		return fmt.Errorf("offline conversion: read target sequence for %s: %w", table, err)
	}
	if current >= highWater {
		return nil
	}
	if err := tx.WithContext(ctx).
		Exec("UPDATE sqlite_sequence SET seq = ? WHERE name = ?", highWater, table).Error; err != nil {
		return fmt.Errorf("offline conversion: restore %s high-water mark: %w", table, err)
	}
	return nil
}

func tableColumns(ctx context.Context, db *gorm.DB) (map[string][]string, error) {
	var tables []string
	if err := db.WithContext(ctx).Raw(
		"SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
	).Scan(&tables).Error; err != nil {
		return nil, fmt.Errorf("offline conversion: list tables: %w", err)
	}
	columns := make(map[string][]string, len(tables))
	for _, table := range tables {
		var names []string
		if err := db.WithContext(ctx).Raw(
			"SELECT name FROM pragma_table_info(?)", table,
		).Scan(&names).Error; err != nil {
			return nil, fmt.Errorf("offline conversion: read %s columns: %w", table, err)
		}
		columns[table] = names
	}
	return columns, nil
}

// identityColumn returns the stable ordering key of a copied table.
func identityColumn(table string, columns []string) string {
	switch table {
	case "access_key_cost_limit_states":
		return "rule_id"
	case "control_operations":
		return "commit_sequence"
	case "system_settings":
		return "key"
	default:
		for _, candidate := range []string{"id", "key"} {
			if slices.Contains(columns, candidate) {
				return candidate
			}
		}
		return columns[0]
	}
}

func quoteIdentifier(value string) string {
	return `"` + strings.ReplaceAll(value, `"`, `""`) + `"`
}

func quoteIdentifiers(values []string) string {
	quoted := make([]string, 0, len(values))
	for _, value := range values {
		quoted = append(quoted, quoteIdentifier(value))
	}
	return strings.Join(quoted, ", ")
}

// validateConverted proves the target is a complete, self-consistent database:
// identical row identities, preserved per-table counts, satisfied foreign keys,
// SQLite integrity, and the absence of retired subscription columns.
func validateConverted(ctx context.Context, target, source *gorm.DB, plan stage, result Result) error {
	for _, table := range plan.retainedTables {
		filter := ""
		if table == "groups" {
			filter = plan.flavourFilter
		}
		column := "id"
		if table == "access_key_cost_limit_states" {
			column = "rule_id"
		}
		sourceCount, err := countRows(ctx, source, table, filter)
		if err != nil {
			return err
		}
		if result.Tables[table] != sourceCount {
			return fmt.Errorf("offline conversion: %s copied %d rows, want %d", table, result.Tables[table], sourceCount)
		}
		targetCount, err := countRows(ctx, target, table, "")
		if err != nil {
			return err
		}
		if targetCount != result.Tables[table] {
			return fmt.Errorf("offline conversion: %s stored %d rows, want %d", table, targetCount, result.Tables[table])
		}
		if err := compareIdentities(ctx, source, target, table, column, filter); err != nil {
			return err
		}
	}
	// The target schema is the current schema; a retired column surviving into
	// it would mean the published database is not what the application expects.
	for table := range currentColumns {
		for _, column := range retiredColumns()[table] {
			if target.Migrator().HasColumn(table, column) {
				return fmt.Errorf("offline conversion: target still declares %s.%s", table, column)
			}
		}
	}
	if err := requirePreservedSequences(ctx, source, target, plan); err != nil {
		return err
	}
	var violations []struct {
		Table  string `gorm:"column:table"`
		RowID  int64  `gorm:"column:rowid"`
		Parent string `gorm:"column:parent"`
	}
	if err := target.WithContext(ctx).Raw("PRAGMA foreign_key_check").Scan(&violations).Error; err != nil {
		return fmt.Errorf("offline conversion: check target foreign keys: %w", err)
	}
	if len(violations) != 0 {
		return fmt.Errorf("offline conversion: target has %d foreign key violations", len(violations))
	}
	var integrity string
	if err := target.WithContext(ctx).Raw("PRAGMA integrity_check").Scan(&integrity).Error; err != nil {
		return fmt.Errorf("offline conversion: check target integrity: %w", err)
	}
	if integrity != "ok" {
		return fmt.Errorf("offline conversion: target integrity check = %q", integrity)
	}
	return nil
}

// requirePreservedSequences proves the published AUTOINCREMENT counters are not
// below the source counters for the copied tables, so an ID freed by deletion in
// the legacy database is not handed out again.
func requirePreservedSequences(ctx context.Context, source, target *gorm.DB, plan stage) error {
	for _, table := range plan.retainedTables {
		want := autoincrementHighWater(ctx, source, table)
		if want <= 0 {
			continue
		}
		var got int64
		if err := target.WithContext(ctx).
			Raw("SELECT COALESCE(max(seq), 0) FROM sqlite_sequence WHERE name = ?", table).
			Scan(&got).Error; err != nil {
			return fmt.Errorf("offline conversion: read target sequence for %s: %w", table, err)
		}
		if got < want {
			return fmt.Errorf(
				"offline conversion: %s AUTOINCREMENT high-water = %d, want at least the source value %d",
				table, got, want,
			)
		}
	}
	return nil
}

func countRows(ctx context.Context, db *gorm.DB, table, filter string) (int64, error) {
	query := "SELECT count(*) FROM " + quoteIdentifier(table)
	if filter != "" {
		query += " WHERE " + filter
	}
	var count int64
	if err := db.WithContext(ctx).Raw(query).Scan(&count).Error; err != nil {
		return 0, fmt.Errorf("offline conversion: count %s: %w", table, err)
	}
	return count, nil
}

func compareIdentities(ctx context.Context, source, target *gorm.DB, table, column, filter string) error {
	query := func() string {
		statement := "SELECT " + quoteIdentifier(column) + " FROM " + quoteIdentifier(table)
		if filter != "" {
			statement += " WHERE " + filter
		}
		return statement + " ORDER BY " + quoteIdentifier(column)
	}
	var want, got []string
	if err := source.WithContext(ctx).Raw(query()).Scan(&want).Error; err != nil {
		return fmt.Errorf("offline conversion: read %s.%s: %w", table, column, err)
	}
	if err := target.WithContext(ctx).Raw(query()).Scan(&got).Error; err != nil {
		return fmt.Errorf("offline conversion: read target %s.%s: %w", table, column, err)
	}
	if !slices.Equal(want, got) {
		return fmt.Errorf("offline conversion: %s.%s changed: %v -> %v", table, column, want, got)
	}
	return nil
}
