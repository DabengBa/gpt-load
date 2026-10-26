// Package offlineconvert converts a legacy GPT-Load SQLite database into a
// fresh database for the current schema.
//
// Preconditions:
//   - The legacy gateway process is stopped. The source is only read (through one
//     SQLite read transaction that takes VACUUM INTO), but a source that a live
//     writer is mutating cannot be proven consistent, so the tool surfaces it: any
//     side file created by the read itself or any database byte change is a hard
//     failure, never a silent success.
//   - The target path does not exist. Publication is a single-file no-replace
//     atomic publish, so an existing target is never overwritten or merged.
//
// The source is snapshot with SQLite's own "VACUUM INTO" (read transaction
// semantics, includes committed WAL pages) into a private scratch directory, and
// the snapshot — not the original — is what the copy reads. The published
// database is a single file in journal_mode=DELETE, so there is no sidecar to
// publish inconsistently. Subscriptions and request logs are intentionally not
// migrated; stored subscription material is reported as an explicit refusal
// instead of being discarded.
package offlineconvert

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/storage"
	"gpt-load/internal/storage/migrations"
)

// retiredMigrationID is the migration the subscription removal deleted. Every
// database written by the last legacy release carries it.
const retiredMigrationID = "0003_remove_observation_fresh_until"

// requiredMigrationIDs is the pinned ledger a supported source must contain:
// the last legacy release's chain, which is the current active chain plus the
// retired 0003 subscription migration. It is an explicit list and not
// storage.RetiredMigrationIDs() plus the running chain, because deriving it from the running code would
// silently reinterpret every future chain change as "the old chain", and a
// future migration could then be applied to a source that never had it.
var requiredMigrationIDs = []string{
	"0001_initial",
	"0002_access_key_cost_limits",
	retiredMigrationID,
	"0004_usage_stats_group_activity_index",
	"0005_proxy_config",
	"0006_error_decision",
	"0007_access_key_lifecycle",
	"0008_remove_inject_usage_options",
	"0010_single_credential_per_group",
	"0011_usage_latency",
	"0012_debug_captures",
	"0013_usage_attempt_stats",
	"0014_affinity_observability",
	"0015_provider_url",
	"0016_affinity_key",
	"0017_remove_validation_interval",
	"0018_usage_journal_bucket_index",
	"0019_request_log_operation_index",
	"0020_agent_credentials",
	"0021_agent_change_proposals",
	"0022_usage_first_response",
	"0023_billing_failure_category",
	"0024_provider_feedback",
	"0025_affinity_diagnostics",
}

// deletedSubscriptionChannelIDs are the built-in subscription channels that the
// current release no longer defines. A group using one of them is refused even
// when its connection_type was never marked as a subscription.
var deletedSubscriptionChannelIDs = []string{"codex", "antigravity", "grok"}

// deletedSubscriptionConnectionTypes are connection flavours the current
// release can no longer express. Any other unknown flavour is reported with its
// own group and value instead of being guessed.
var deletedSubscriptionConnectionTypes = []string{"subscription", "oauth"}

// Options selects the conversion endpoints. TargetDSN must not exist; it is the
// SQLite file path (or DSN) the application will open inside its DATA_DIR.
type Options struct {
	SourceDSN string
	TargetDSN string
}

// Result reports what the conversion published: copied rows per retained table
// and the retired columns the source declared, which were not migrated.
type Result struct {
	Tables         map[string]int64
	RetiredColumns []string
}

// Convert builds a new database from a legacy source and publishes it under
// the requested target path.
func Convert(ctx context.Context, options Options) (Result, error) {
	if ctx == nil {
		return Result{}, errors.New("offline conversion: context is required")
	}
	plan, err := inspectSource(ctx, options.SourceDSN)
	if err != nil {
		return Result{}, err
	}
	target, err := resolveTarget(options.TargetDSN)
	if err != nil {
		return Result{}, err
	}
	return publish(ctx, target, plan)
}

// stage is the validated copy plan derived from one legacy source.
type stage struct {
	sourcePath     string
	retainedTables []string
	retiredColumns []string
}

// convertibleFlavours are the stored connection types a legacy source may carry
// without describing subscription material. The retired "subscription" flavour
// is deliberately absent.
var convertibleFlavours = []string{"api_key"}

func inspectSource(ctx context.Context, dsn string) (stage, error) {
	sourcePath, err := sqliteFilePath(dsn)
	if err != nil {
		return stage{}, err
	}
	if err := requireRegularSource(sourcePath); err != nil {
		return stage{}, err
	}
	// Refuse before opening anything: an open writer makes the snapshot
	// unprovable, and opening the database at all would itself create the side
	// files that would then hide the difference.
	if err := refuseActiveWriter(sourcePath); err != nil {
		return stage{}, err
	}
	result := stage{sourcePath: sourcePath}
	err = withSourceDatabase(ctx, sourcePath, func(source *gorm.DB) error {
		if err := requireKnownLedger(source); err != nil {
			return err
		}
		// The declared schema is validated before integrity, so a missing table
		// or column is reported as itself instead of as the foreign key damage
		// it causes.
		if result.retiredColumns, err = droppedSourceColumns(source); err != nil {
			return err
		}
		if result.retainedTables, err = classifySourceTables(source); err != nil {
			return err
		}
		if err := requireSourceIntegrity(source); err != nil {
			return err
		}
		if err := requireOrdinaryGroups(source); err != nil {
			return err
		}
		if err := requireNoRetiredSubscriptionData(source); err != nil {
			return err
		}
		if err := requireOrdinaryCredentials(source); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		return stage{}, err
	}
	// The snapshot is taken: the reader's own empty side files are allowed, but a
	// writer that appeared mid-snapshot is not.
	if err := refuseActiveWriter(sourcePath); err != nil {
		return stage{}, err
	}
	return result, nil
}

// requireKnownLedger accepts one legacy chain: the pinned required ledger of the
// last legacy release, in ascending order, with no gaps or duplicates, plus
// nothing but retired migration IDs this repository itself removed.
//
// The retired IDs are optional because the running chain skips them instead of
// deleting their ledger rows, so a database that applied one of them before its
// removal keeps it forever. Accepting them is not accepting unknown history: an
// ID that lives in neither the required chain nor the retired set is refused,
// and so is a ledger with a gap in the required chain.
func requireKnownLedger(source *gorm.DB) error {
	if !source.Migrator().HasTable("schema_migrations") {
		return errors.New("offline conversion: source has no schema_migrations ledger")
	}
	var applied []string
	if err := source.Table("schema_migrations").Order("id ASC").Pluck("id", &applied).Error; err != nil {
		return fmt.Errorf("offline conversion: read source ledger: %w", err)
	}
	retired := storage.RetiredMigrationIDs()
	required := make(map[string]struct{}, len(requiredMigrationIDs))
	for _, id := range requiredMigrationIDs {
		required[id] = struct{}{}
	}
	seen := make(map[string]struct{}, len(applied))
	for index, id := range applied {
		if index > 0 && applied[index-1] >= id {
			return fmt.Errorf(
				"offline conversion: source ledger entry %q is duplicated or out of order",
				id,
			)
		}
		if _, ok := required[id]; ok {
			seen[id] = struct{}{}
			continue
		}
		if !slices.Contains(retired, id) {
			return fmt.Errorf(
				"offline conversion: source ledger entry %q is not part of the known legacy migration chain",
				id,
			)
		}
	}
	missing := make([]string, 0, len(requiredMigrationIDs))
	for _, id := range requiredMigrationIDs {
		if _, ok := seen[id]; !ok {
			missing = append(missing, id)
		}
	}
	if len(missing) != 0 {
		return fmt.Errorf(
			"offline conversion: source ledger is not the known legacy migration chain (missing %s)",
			missing[0],
		)
	}
	return nil
}

// requireOrdinaryGroups rejects any stored group whose flavour this release can
// no longer express.
//
// The stored connection type is a retired column: the current schema
// authenticates every group exactly one code-owned way per channel, so the
// value is never copied. A legacy source that still declares it is read here
// only to refuse subscription material, because publishing a database with
// those groups silently missing is exactly what this tool must not do. A source
// that already dropped the column is inspected on channel identity alone.
func requireOrdinaryGroups(source *gorm.DB) error {
	hasFlavour := source.Migrator().HasColumn("groups", "connection_type")
	query := source.Table("groups").Select("id", "channel_id")
	if hasFlavour {
		query = query.Select("id", "channel_id", "connection_type")
	}
	var rows []struct {
		ID             uint   `gorm:"column:id"`
		ChannelID      string `gorm:"column:channel_id"`
		ConnectionType string `gorm:"column:connection_type"`
	}
	if err := query.Order("id ASC").Find(&rows).Error; err != nil {
		return fmt.Errorf("offline conversion: read source groups: %w", err)
	}
	registry := channel.NewRegistry()
	for _, row := range rows {
		if hasFlavour {
			if slices.Contains(deletedSubscriptionConnectionTypes, row.ConnectionType) {
				return fmt.Errorf(
					"offline conversion: group %d uses retired subscription connection type %q; subscriptions are unsupported, refusing to drop it",
					row.ID, row.ConnectionType,
				)
			}
			if !slices.Contains(convertibleFlavours, row.ConnectionType) {
				return fmt.Errorf(
					"offline conversion: group %d stores the unknown connection type %q; refusing to drop it",
					row.ID, row.ConnectionType,
				)
			}
		}
		if strings.Contains(strings.ToLower(row.ChannelID), "subscription") {
			return fmt.Errorf(
				"offline conversion: group %d stores a subscription channel %q; subscriptions are unsupported, refusing to drop it",
				row.ID, row.ChannelID,
			)
		}
		if slices.Contains(deletedSubscriptionChannelIDs, row.ChannelID) {
			return fmt.Errorf(
				"offline conversion: group %d stores the deleted subscription channel %q; subscriptions are unsupported, refusing to drop it",
				row.ID, row.ChannelID,
			)
		}
		if _, known := registry.Get(channel.ID(row.ChannelID)); !known {
			return fmt.Errorf(
				"offline conversion: group %d stores the unknown channel %q; refusing to guess what it authenticates",
				row.ID, row.ChannelID,
			)
		}
	}
	return nil
}

// requireOrdinaryCredentials refuses stored credentials that are evidently
// subscription material. Ciphertext cannot be inspected without the operator
// key, so the detection uses durable non-secret markers: credential identity
// fingerprints are keyed hashes of the plaintext credential, while a
// subscription credential carried a retired subscription auth marker.
//
// The retired subscription tables are not queried per credential: their columns
// differ per table (the real `credential_stages` is keyed by a string stage id
// and links to a group, not to a credential), and any row at all in them is
// subscription state this release cannot express. requireNoRetiredSubscriptionData
// refuses those tables wholesale, which is both stricter and schema-independent.
func requireOrdinaryCredentials(source *gorm.DB) error {
	if !source.Migrator().HasColumn("credentials", "auth_state") {
		return nil
	}
	var flagged []uint
	if err := source.Raw(`SELECT id FROM credentials
		WHERE auth_state <> 'ready' OR auth_error_code <> ''
		ORDER BY id ASC`).Scan(&flagged).Error; err != nil {
		return fmt.Errorf("offline conversion: read retired subscription credential state: %w", err)
	}
	if len(flagged) != 0 {
		return fmt.Errorf(
			"offline conversion: credentials %v store subscription auth state; subscriptions are unsupported, refusing to drop them",
			flagged,
		)
	}
	return nil
}

// requireNoRetiredSubscriptionData refuses a source that still stores rows in a
// retired subscription table.
//
// The retired tables are dropped by the conversion, so a stored row there is
// stored data the operator cannot get back. Discarding subscription account
// material is not authorized by an API-key-only conversion, and no marker in the
// retained tables proves the row is unimportant, so presence alone is refused.
// An empty retired table is the normal state of an API-key-only deployment and
// is accepted.
func requireNoRetiredSubscriptionData(source *gorm.DB) error {
	for _, table := range retiredSubscriptionTables() {
		if !source.Migrator().HasTable(table) {
			continue
		}
		var stored int64
		if err := source.Raw("SELECT count(*) FROM " + quoteIdentifier(table)).Scan(&stored).Error; err != nil {
			return fmt.Errorf("offline conversion: count retired subscription rows in %s: %w", table, err)
		}
		if stored != 0 {
			return fmt.Errorf(
				"offline conversion: source stores %d %s rows; subscriptions are unsupported, refusing to drop stored subscription data",
				stored, table,
			)
		}
	}
	return nil
}

// retiredSubscriptionTables are the tables the retired subscription feature
// owned. The conversion drops them, so they must be empty.
func retiredSubscriptionTables() []string {
	return migrations.RetiredBaselineTables()
}

// droppedSourceColumns lists the retired columns the source actually declares,
// so the operator sees exactly which stored values the conversion did not
// migrate.
func droppedSourceColumns(source *gorm.DB) ([]string, error) {
	var columns []string
	for table := range currentColumns {
		for _, column := range retiredColumns()[table] {
			if source.Migrator().HasColumn(table, column) {
				columns = append(columns, table+"."+column)
			}
		}
	}
	slices.Sort(columns)
	return columns, nil
}

// classifySourceTables partitions every source table into retained and dropped
// sets and rejects anything the converter does not know about, so a legacy
// table can never be dropped by accident.
func classifySourceTables(source *gorm.DB) ([]string, error) {
	var tables []string
	if err := source.Raw(
		"SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
	).Scan(&tables).Error; err != nil {
		return nil, fmt.Errorf("offline conversion: list source tables: %w", err)
	}
	if err := requireExactLegacyColumns(source, tables); err != nil {
		return nil, err
	}
	if err := requireEveryRetainedTable(source, tables); err != nil {
		return nil, err
	}
	var kept []string
	for _, table := range tables {
		switch {
		case slices.Contains(migratedTables, table):
			kept = append(kept, table)
		case slices.Contains(droppedTables, table):
		default:
			return nil, fmt.Errorf(
				"offline conversion: source table %q is not classified as migrated or dropped; refusing to guess",
				table,
			)
		}
	}
	return kept, nil
}

// requireEveryRetainedTable insists that a supported source version declares
// every migrated table.
//
// Without this check the copy simply skips a table the source does not have and
// publishes the target with an empty table, silently losing the operator's rows.
// Which tables exist is part of the supported source schema, exactly like which
// columns exist, so a missing table is a refusal and not an empty copy.
func requireEveryRetainedTable(source *gorm.DB, tables []string) error {
	missing := make([]string, 0, len(migratedTables))
	for _, table := range migratedTables {
		if slices.Contains(tables, table) {
			continue
		}
		missing = append(missing, table)
	}
	if len(missing) == 0 {
		return nil
	}
	slices.Sort(missing)
	return fmt.Errorf(
		"offline conversion: source table %q is missing; the supported legacy schema declares it, so its rows would be dropped",
		missing[0],
	)
}

// requireExactLegacyColumns pins the source schema to the known legacy schema.
//
// Both directions are checked, because the copy intersects source and target
// columns: an unexpected extra column would be silently dropped, and a missing
// required column would let the target fall back to a column default instead of
// the stored value. Only the retired columns the current schema deliberately
// dropped may be absent or, when present, are reported and skipped.
func requireExactLegacyColumns(source *gorm.DB, tables []string) error {
	for _, table := range tables {
		if !slices.Contains(migratedTables, table) {
			continue
		}
		current, known := currentColumns[table]
		if !known {
			return fmt.Errorf("offline conversion: table %q has no recorded current columns", table)
		}
		var declared []string
		if err := source.Raw("SELECT name FROM pragma_table_info(?)", table).Scan(&declared).Error; err != nil {
			return fmt.Errorf("offline conversion: read %s columns: %w", table, err)
		}
		for _, column := range declared {
			if slices.Contains(current, column) || isRetiredColumn(table, column) {
				continue
			}
			return fmt.Errorf(
				"offline conversion: source column %s.%s is not part of the current schema and is not a retired column; refusing to drop it",
				table, column,
			)
		}
		for _, column := range current {
			if slices.Contains(declared, column) {
				continue
			}
			return fmt.Errorf(
				"offline conversion: source column %s.%s is missing; the stored value would be replaced by a column default",
				table, column,
			)
		}
	}
	return nil
}

// retiredColumns are the columns the current schema deliberately dropped, keyed
// by the table that used to declare them.
//
// The retirement is per table and column, never a bare column-name allowlist: a
// source that never declared `groups.price_multiplier_micros` but does declare
// `credentials.price_multiplier_micros` stores a column this release cannot
// interpret, and it must stay an unclassified column instead of being silently
// dropped just because another table retired the same name.
func retiredColumns() map[string][]string {
	return map[string][]string{
		// Subscription credential auth state.
		"credentials": {"auth_state", "auth_error_code"},
		// The product no longer models a per-group connection type; groups
		// authenticate exactly one code-owned way per channel. Sources still
		// declare the column, so it is retired here and never copied.
		"groups": {"connection_type", "price_multiplier_micros"},
		// Configurable price multipliers, retired by the frozen-pricing change
		// (`0009_price_multipliers`). The stored value is never re-read: new
		// receipts use frozen model prices only, so the value is retired with the
		// column and no historical amount is recomputed.
		"access_keys": {"price_multiplier_micros"},
	}
}

// isRetiredColumn reports whether table used to declare column before this
// release dropped it.
func isRetiredColumn(table, column string) bool {
	return slices.Contains(retiredColumns()[table], column)
}

// retiredColumnPairs flattens the retired declaration to `table.column` names.
func retiredColumnPairs() []string {
	var pairs []string
	for table, columns := range retiredColumns() {
		for _, column := range columns {
			pairs = append(pairs, table+"."+column)
		}
	}
	slices.Sort(pairs)
	return pairs
}

// currentColumns records the columns the current schema declares for every
// migrated table. It is the allowlist that makes column loss impossible: a
// stored column outside this list (and outside retiredColumns) stops the
// conversion.
var currentColumns = map[string][]string{
	"groups": {
		"id", "name", "channel_id", "params", "models", "overrides",
		"proxy_config", "provider_url", "enabled", "weight_manual", "validation_model",
		"created_at_ms", "updated_at_ms",
	},
	"credentials": {
		"id", "group_id", "data", "fingerprint", "identity_fingerprint", "secret_version",
		"status", "weight_manual", "proxy_config", "created_at_ms", "updated_at_ms",
	},
	"access_keys": {
		"id", "name", "key_value", "key_hash", "key_suffix", "status", "filters", "rpm_limit",
		"daily_cost_limit_nano_usd", "monthly_cost_limit_nano_usd", "expires_at_ms",
		"created_at_ms", "updated_at_ms",
	},
	"access_key_cost_limit_rules": {
		"id", "access_key_id", "kind", "limit_nano_usd", "period_seconds", "rule_revision",
		"created_at_ms", "updated_at_ms",
	},
	"access_key_cost_limit_states": {
		"rule_id", "rule_revision", "used_nano_usd", "window_started_at_ms", "window_ends_at_ms",
		"window_generation", "snapshot_version", "updated_at_ms",
	},
	"model_prices": {
		"id", "channel_id", "model_id", "input_price_nano_usd_per_million_tokens",
		"output_price_nano_usd_per_million_tokens", "cache_read_price_nano_usd_per_million_tokens",
		"cache_write_price_nano_usd_per_million_tokens", "context_price_tiers",
		"mode_price_schedules", "is_manual", "created_at_ms", "updated_at_ms",
	},
	"system_settings": {"key", "value", "updated_at_ms"},
	"control_operations": {
		"commit_sequence", "operation_id", "idempotency_key", "digest_version", "request_digest",
		"operation_kind", "resource_identity", "proposal_id", "canonical_result", "required_stages",
		"last_completed_stage", "failed_stage", "completed_at_ms", "compacted_at_ms",
		"created_at_ms", "updated_at_ms",
	},
	"jobs": {
		"id", "type", "status", "payload", "result", "error", "created_at_ms", "started_at_ms",
		"finished_at_ms",
	},
	"agent_credentials": {
		"id", "name", "secret_hash", "scopes", "status", "expires_at_ms", "disabled_at_ms",
		"created_at_ms", "updated_at_ms",
	},
	"agent_change_proposals": {
		"id", "created_at_ms", "updated_at_ms", "base_snapshot_revision", "creator_credential_id",
		"state", "updates", "approved_runtime_epoch", "approved_by", "approved_at_ms",
	},
}

// migratedTables lists every source table whose rows are copied. Tables that do
// not exist yet in the legacy source are created empty by the current
// migration chain, and tables that did not exist then simply are not copied.
var migratedTables = []string{
	"access_key_cost_limit_rules",
	"access_key_cost_limit_states",
	"access_keys",
	"agent_change_proposals",
	"agent_credentials",
	"control_operations",
	"credentials",
	"groups",
	"jobs",
	"model_prices",
	"system_settings",
}

// droppedTables lists source tables whose data is deliberately not migrated:
// retired subscription state and request/usage logs.
var droppedTables = []string{
	"credential_attempt_stats",
	"credential_observations",
	"credential_reset_operations",
	"credential_stages",
	"debug_capture_attempts",
	"debug_capture_chunks",
	"debug_capture_sessions",
	"debug_captures",
	"request_log_attempts",
	"request_logs",
	"schema_migrations",
	"usage_aggregation_journal",
	"usage_attempt_aggregation_journals",
	"usage_attempt_stats",
	"usage_stats",
}

// refuseActiveWriter enforces the documented "stop the legacy gateway"
// precondition instead of hoping for it.
//
// A non-empty "-wal" holds committed pages that a live connection has not
// checkpointed yet, so the writer is definitely still running and the converter
// cannot prove its snapshot is a state that writer ever settled on. A
// "-journal" exists only while a transaction is being written, or after a crash
// during one, and is refused for the same reason. Both are reported with the
// remedy instead of being converted around.
//
// An empty "-wal" or a "-shm" advertises no unflushed committed data — SQLite
// recreates them and removes them by itself and they carry no source content —
// so they are not treated as a writer. A "-shm" is deliberately not a signal:
// any read-only open of a WAL-mode database creates one (32 KiB, no pages), so
// its presence says nothing about whether a writer is attached. This check runs
// before and after the snapshot, so a writer that starts mid-conversion is still
// caught.
func refuseActiveWriter(path string) error {
	// A non-empty write-ahead log holds committed pages that a live connection has
	// not checkpointed yet: the writer is still running.
	if info, err := os.Stat(path + "-wal"); err == nil {
		if info.Size() != 0 {
			return fmt.Errorf(
				"offline conversion: source has a non-empty %s, so a writer still has the database open; stop the legacy gateway (or any other SQLite client) and retry",
				filepath.Base(path)+"-wal",
			)
		}
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("offline conversion: inspect source write-ahead log: %w", err)
	}
	// A rollback journal only exists while a transaction is being written, or
	// after a crash during one.
	if _, err := os.Stat(path + "-journal"); err == nil {
		return fmt.Errorf(
			"offline conversion: source has %s, so a transaction is being written or was interrupted; stop the legacy gateway and retry",
			filepath.Base(path)+"-journal",
		)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("offline conversion: inspect source rollback journal: %w", err)
	}
	// An empty -wal or a -shm advertises no unflushed committed data; SQLite
	// recreates and removes them on its own and they carry no source content.
	return nil
}

func requireRegularSource(path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("offline conversion: inspect source database: %w", err)
	}
	if !info.Mode().IsRegular() {
		return errors.New("offline conversion: source must be a regular file")
	}
	return nil
}

func closeDatabase(db *gorm.DB) {
	sqlDB, err := db.DB()
	if err != nil {
		return
	}
	_ = sqlDB.Close()
}

func sqliteFilePath(dsn string) (string, error) {
	trimmed := strings.TrimSpace(dsn)
	if trimmed == "" {
		return "", errors.New("offline conversion: source database path is empty")
	}
	if strings.Contains(trimmed, ":memory:") {
		return "", errors.New("offline conversion: source must be a SQLite file")
	}
	base, _, _ := strings.Cut(trimmed, "?")
	if strings.HasPrefix(base, "file:") {
		base = strings.TrimPrefix(base, "file:")
	}
	if base == "" {
		return "", errors.New("offline conversion: source must be a SQLite file")
	}
	if _, err := config.ParseDatabaseDSN(base); err != nil {
		return "", fmt.Errorf("offline conversion: invalid SQLite source path: %w", err)
	}
	return base, nil
}
