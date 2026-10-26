package offlineconvert_test

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/encryption"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage"
	"gpt-load/internal/storage/migrations"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/storage/offlineconvert"
)

// The legacy chain below is frozen at the last commit that still shipped the
// retired subscription migrations. Only the retired 0003 identifier is listed
// because it is the only migration this repository deleted; every other ID is
// resolved from the current registry, so a future chain change cannot silently
// change what "known legacy chain" means.
const retiredSubscriptionMigrationID = "0003_remove_observation_fresh_until"

// retiredPriceMultiplierMigrationID is the migration removed from the active
// chain by the pricing change. A database that applied it before that removal
// keeps the row forever, because the current chain skips retired IDs instead of
// deleting them.
const retiredPriceMultiplierMigrationID = "0009_price_multipliers"

// buildLegacyDatabase creates the exact last legacy schema: the current chain
// with the retired subscription migration 0003 present in the ledger and
// subscription tables/columns still declared.
func buildLegacyDatabase(t *testing.T, path string) *gorm.DB {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open legacy database: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("legacy database pool: %v", err)
	}
	sqlDB.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = sqlDB.Close() })

	if err := storage.AutoMigrate(db); err != nil {
		t.Fatalf("apply current migrations to legacy fixture: %v", err)
	}
	// Reconstruct the pre-refactor ledger: every active migration plus the
	// retired subscription migration, in ascending order.
	var ledger []string
	if err := db.Table("schema_migrations").Order("id").Pluck("id", &ledger).Error; err != nil {
		t.Fatalf("read fixture ledger: %v", err)
	}
	if err := db.Exec("DELETE FROM schema_migrations").Error; err != nil {
		t.Fatalf("reset fixture ledger: %v", err)
	}
	merged := append(slices.Clone(ledger), retiredSubscriptionMigrationID)
	slices.Sort(merged)
	for _, id := range merged {
		if err := db.Exec("INSERT INTO schema_migrations (id) VALUES (?)", id).Error; err != nil {
			t.Fatalf("insert fixture ledger %s: %v", id, err)
		}
	}

	for _, statement := range []string{
		`ALTER TABLE groups ADD COLUMN connection_type varchar(32) NOT NULL DEFAULT 'api_key'`,
		`ALTER TABLE credentials ADD COLUMN auth_state varchar(32) NOT NULL DEFAULT 'ready'`,
		`ALTER TABLE credentials ADD COLUMN auth_error_code varchar(64) NOT NULL DEFAULT ''`,
	} {
		if err := db.Exec(statement).Error; err != nil {
			t.Fatalf("declare legacy subscription schema: %v", err)
		}
	}
	// The legacy source still stores the retired per-group connection type; the
	// converter reports it as a retired column instead of copying it.
	// The retired subscription tables are declared from the schema the supported
	// legacy release actually created, not from a hand-written stand-in: the real
	// `credential_stages` has a string `id` primary key and no `credential_id`
	// column at all, and the real `credential_reset_operations` is keyed by
	// `idempotency_key`.
	for _, model := range migrations.RetiredBaselineModels() {
		if err := db.AutoMigrate(model); err != nil {
			t.Fatalf("declare legacy subscription schema %T: %v", model, err)
		}
	}
	return db
}

// assertFixtureMatchesRetiredBaseline fails if the fixture's retired tables no
// longer reproduce the historical schema, so a fixture cannot drift back into a
// shape that only the converter agrees with.
func assertFixtureMatchesRetiredBaseline(t *testing.T, db *gorm.DB) {
	t.Helper()
	want, err := migrations.RetiredBaselineColumns(db)
	if err != nil {
		t.Fatalf("read retired baseline columns: %v", err)
	}
	for table, columns := range want {
		var declared []string
		if err := db.Raw("SELECT name FROM pragma_table_info(?)", table).Scan(&declared).Error; err != nil {
			t.Fatalf("read fixture %s columns: %v", table, err)
		}
		slices.Sort(columns)
		slices.Sort(declared)
		if !slices.Equal(columns, declared) {
			t.Fatalf("fixture %s columns = %v, want the historical schema %v", table, declared, columns)
		}
	}
}

type fixture struct {
	ordinaryGroupID  uint
	groups           []uint
	credentials      []uint
	accessKeys       []uint
	rules            []uint
	modelPrices      []uint
	agentCredentials []uint
	proposals        []string
	operations       []uint64
	jobs             []string
	settings         []string
	credentialPlain  map[uint]string
}

func buildLegacyFixture(t *testing.T, path string, cipher encryption.Service) fixture {
	t.Helper()
	db := buildLegacyDatabase(t, path)
	want := fixture{credentialPlain: make(map[uint]string)}

	for index, channelID := range []string{"openai", "cliproxyapi", "gemini"} {
		group := models.Group{
			Name:        fmt.Sprintf("group-%d", index+1),
			ChannelID:   channelID,
			Params:      models.JSON(`{"base_url":"https://example.invalid"}`),
			Models:      models.JSON(`[{"id":"m1","alias":""}]`),
			Enabled:     index != 2,
			CreatedAtMS: int64(1_700_000_000_000 + index),
			UpdatedAtMS: int64(1_700_000_001_000 + index),
		}
		if err := db.Create(&group).Error; err != nil {
			t.Fatalf("create legacy group: %v", err)
		}
		want.groups = append(want.groups, group.ID)
		if index == 0 {
			want.ordinaryGroupID = group.ID
		}

		plaintext := fmt.Sprintf(`{"api_key":"sk-legacy-%d"}`, index)
		encrypted, err := cipher.Encrypt(plaintext)
		if err != nil {
			t.Fatalf("encrypt legacy credential: %v", err)
		}
		credential := models.Credential{
			GroupID:     group.ID,
			Data:        encrypted,
			Fingerprint: cipher.Hash(plaintext),
		}
		if err := db.Create(&credential).Error; err != nil {
			t.Fatalf("create legacy credential: %v", err)
		}
		want.credentials = append(want.credentials, credential.ID)
		want.credentialPlain[credential.ID] = plaintext
	}

	accessKey := models.AccessKey{
		Name: "key-one", KeyValue: "encrypted-key-value", KeyHash: "hash-one",
		KeySuffix: "0f0f", Status: "active", RPMLimit: 7,
		Filters:     models.JSON(`{"groups":[1]}`),
		CreatedAtMS: 1_700_000_002_000, UpdatedAtMS: 1_700_000_003_000,
	}
	if err := db.Create(&accessKey).Error; err != nil {
		t.Fatalf("create legacy access key: %v", err)
	}
	want.accessKeys = append(want.accessKeys, accessKey.ID)

	rule := models.AccessKeyCostLimitRule{
		AccessKeyID: accessKey.ID, Kind: models.AccessKeyCostLimitKindPeriodic,
		LimitNanoUSD: 5_000_000, PeriodSeconds: 3600, RuleRevision: 3,
		CreatedAtMS: 1_700_000_004_000, UpdatedAtMS: 1_700_000_005_000,
	}
	if err := db.Create(&rule).Error; err != nil {
		t.Fatalf("create legacy cost rule: %v", err)
	}
	want.rules = append(want.rules, rule.ID)
	windowStart, windowEnd := int64(1_700_000_006_000), int64(1_700_000_007_000)
	state := models.AccessKeyCostLimitState{
		RuleID: rule.ID, RuleRevision: rule.RuleRevision, UsedNanoUSD: 1_234_567,
		WindowStartedAtMS: &windowStart, WindowEndsAtMS: &windowEnd,
		WindowGeneration: 4, SnapshotVersion: 9, UpdatedAtMS: 1_700_000_008_000,
	}
	if err := db.Create(&state).Error; err != nil {
		t.Fatalf("create legacy cost state: %v", err)
	}

	inputPrice, outputPrice := int64(150), int64(600)
	price := models.ModelPrice{
		ChannelID: "openai", ModelID: "gpt-legacy",
		InputPriceNanoUSDPerMillionTokens:  &inputPrice,
		OutputPriceNanoUSDPerMillionTokens: &outputPrice,
		ContextPriceTiers:                  models.JSON(`[{"threshold_tokens":1000,"input_price_nano_usd_per_million_tokens":150,"output_price_nano_usd_per_million_tokens":600,"cache_read_price_nano_usd_per_million_tokens":10,"cache_write_price_nano_usd_per_million_tokens":20}]`),
		IsManual:                           true,
		CreatedAtMS:                        1_700_000_009_000, UpdatedAtMS: 1_700_000_010_000,
	}
	if err := db.Create(&price).Error; err != nil {
		t.Fatalf("create legacy model price: %v", err)
	}
	want.modelPrices = append(want.modelPrices, price.ID)

	for _, setting := range []models.SystemSetting{
		{Key: "request_log_retention_days", Value: "14", UpdatedAtMS: 1_700_000_011_000},
		{Key: models.InternalSystemSettingPrefix + "bootstrap.default_access_key.v1", Value: "true", UpdatedAtMS: 1_700_000_012_000},
		{Key: models.InternalSystemSettingPrefix + "affinity.binding." + strings.Repeat("a", 64), Value: `{"group_id":1,"credential_id":1,"identity_generation":1}`, UpdatedAtMS: 1_700_000_013_000},
	} {
		if err := db.Create(&setting).Error; err != nil {
			t.Fatalf("create legacy system setting: %v", err)
		}
		want.settings = append(want.settings, setting.Key)
	}

	agentCredential := models.AgentCredential{
		Name: "agent", SecretHash: "agent-hash", Scopes: models.JSON(`["read"]`),
		Status: "active", CreatedAtMS: 1_700_000_014_000, UpdatedAtMS: 1_700_000_014_500,
	}
	if err := db.Create(&agentCredential).Error; err != nil {
		t.Fatalf("create legacy agent credential: %v", err)
	}
	want.agentCredentials = append(want.agentCredentials, agentCredential.ID)

	proposal := models.AgentChangeProposal{
		ID:                   "5c2c9f43-5d84-4c56-9c8b-1f1d0e6f7a11",
		CreatedAtMS:          1_700_000_015_000,
		UpdatedAtMS:          1_700_000_016_000,
		BaseSnapshotRevision: 12,
		CreatorCredentialID:  agentCredential.ID,
		State:                "pending",
		Updates:              models.JSON(`[]`),
	}
	if err := db.Create(&proposal).Error; err != nil {
		t.Fatalf("create legacy proposal: %v", err)
	}
	want.proposals = append(want.proposals, proposal.ID)

	operation := models.ControlOperation{
		OperationID:        "6f6d5b6d-5f5c-4b4a-8a3d-2f1f0e9d8c22",
		IdempotencyKey:     "7a7b8c8d-9d9e-4f4f-8b8b-3c3c2d2d1e1e",
		DigestVersion:      1,
		RequestDigest:      make([]byte, 32),
		OperationKind:      "group_create",
		ResourceIdentity:   "group",
		RequiredStages:     models.JSON(`["config"]`),
		LastCompletedStage: "config",
		CreatedAtMS:        1_700_000_017_000,
		UpdatedAtMS:        1_700_000_018_000,
	}
	if err := db.Create(&operation).Error; err != nil {
		t.Fatalf("create legacy control operation: %v", err)
	}
	want.operations = append(want.operations, operation.CommitSequence)

	job := models.Job{
		ID: "9c9d0e0f-1a1b-4c4d-8e8f-5a5a6b6b7c7c", Type: "group_discovery",
		Status: "completed", Payload: models.JSON(`{"group_id":1}`),
		CreatedAtMS: 1_700_000_019_000,
	}
	if err := db.Create(&job).Error; err != nil {
		t.Fatalf("create legacy job: %v", err)
	}
	want.jobs = append(want.jobs, job.ID)

	// Request logs and the subscription tables are deliberately populated so a
	// conversion that copies them, or a validator that ignores them, is visible.
	if err := db.Exec(`INSERT INTO request_logs (id, completed_at_ms, access_key_id, group_id, channel_id,
		credential_id, protocol, operation, client_model, upstream_model, status, status_code, stream, duration_ms, error_summary)
		VALUES ('log-1', 1700000020000, ?, ?, 'openai', ?, 'openai', 'chat', 'm1', 'm1', 'success', 200, 0, 12, '')`,
		accessKey.ID, want.ordinaryGroupID, want.credentials[0]).Error; err != nil {
		t.Fatalf("create legacy request log: %v", err)
	}
	return want
}

type sourceSnapshot struct {
	hash     string
	ledger   []string
	walBytes int64
}

func snapshotSource(t *testing.T, path string) sourceSnapshot {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read source database: %v", err)
	}
	var walBytes int64
	if info, statErr := os.Stat(path + "-wal"); statErr == nil {
		walBytes = info.Size()
	}
	sum := sha256.Sum256(raw)
	db := openReadOnly(t, path)
	var ledger []string
	if err := db.Table("schema_migrations").Order("id").Pluck("id", &ledger).Error; err != nil {
		t.Fatalf("read source ledger: %v", err)
	}
	return sourceSnapshot{hash: hex.EncodeToString(sum[:]), ledger: ledger, walBytes: walBytes}
}

func assertSourceUnchanged(t *testing.T, path string, before sourceSnapshot) {
	t.Helper()
	after := snapshotSource(t, path)
	if after.hash != before.hash {
		t.Fatalf("source database bytes changed: %s -> %s", before.hash, after.hash)
	}
	if !slices.Equal(after.ledger, before.ledger) {
		t.Fatalf("source ledger changed: %v -> %v", before.ledger, after.ledger)
	}
	if after.walBytes != 0 {
		t.Fatalf("source WAL gained %d bytes", after.walBytes)
	}
	if info, err := os.Stat(path + "-shm"); err == nil && info.Size() != 0 {
		t.Fatalf("source SHM gained %d bytes", info.Size())
	}
	if _, err := os.Lstat(path + "-journal"); err == nil {
		t.Fatalf("source gained a rollback journal")
	}
}

// assertSingleFileTarget proves the published database needs no sidecar.
func assertSingleFileTarget(t *testing.T, path string) {
	t.Helper()
	for _, suffix := range []string{"-wal", "-shm", "-journal"} {
		if _, err := os.Lstat(path + suffix); err == nil {
			t.Fatalf("published target has %s side file", suffix)
		}
	}
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("read target directory: %v", err)
	}
	for _, entry := range entries {
		if entry.Name() != filepath.Base(path) {
			t.Fatalf("unexpected leftover artifact %s", entry.Name())
		}
	}
}

func openReadOnly(t *testing.T, path string) *gorm.DB {
	t.Helper()
	db, err := gorm.Open(sqlite.Open("file:"+path+"?mode=ro"), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open %s read-only: %v", path, err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("pool %s: %v", path, err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	return db
}

func writeSource(t *testing.T, path, statement string, args ...any) {
	t.Helper()
	db, err := gorm.Open(sqlite.Open(path), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open source for mutation: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("source pool: %v", err)
	}
	defer func() { _ = sqlDB.Close() }()
	if err := db.Exec(statement, args...).Error; err != nil {
		t.Fatalf("mutate source fixture: %v", err)
	}
}

func setLedger(t *testing.T, path string, ledger []string) {
	t.Helper()
	writeSource(t, path, "DELETE FROM schema_migrations")
	for _, id := range ledger {
		writeSource(t, path, "INSERT INTO schema_migrations (id) VALUES (?)", id)
	}
}

func assertTargetSchema(t *testing.T, db *gorm.DB) {
	t.Helper()
	for _, table := range migrations.RetiredBaselineTables() {
		if db.Migrator().HasTable(table) {
			t.Errorf("target still declares %s", table)
		}
	}
	for _, column := range []string{"auth_state", "auth_error_code"} {
		if db.Migrator().HasColumn("credentials", column) {
			t.Errorf("target still declares credentials.%s", column)
		}
	}
	var ledger []string
	if err := db.Table("schema_migrations").Order("id").Pluck("id", &ledger).Error; err != nil {
		t.Fatalf("read target ledger: %v", err)
	}
	if len(ledger) == 0 {
		t.Fatal("target ledger is empty")
	}
	if slices.Contains(ledger, retiredSubscriptionMigrationID) || slices.Contains(ledger, "0009_price_multipliers") {
		t.Fatalf("target ledger contains a retired migration: %v", ledger)
	}
}

func assertTargetContent(t *testing.T, db *gorm.DB, cipher encryption.Service, want fixture) {
	t.Helper()
	var groups []models.Group
	if err := db.Order("id").Find(&groups).Error; err != nil {
		t.Fatalf("read target groups: %v", err)
	}
	if got := groupIDs(groups); !slices.Equal(got, want.groups) {
		t.Fatalf("target group ids = %v, want %v", got, want.groups)
	}
	for index, group := range groups {
		if group.CreatedAtMS != int64(1_700_000_000_000+index) {
			t.Errorf("group %d created_at_ms = %d", group.ID, group.CreatedAtMS)
		}
		if string(group.Params) != `{"base_url":"https://example.invalid"}` {
			t.Errorf("group %d params = %s", group.ID, group.Params)
		}
	}

	var credentials []models.Credential
	if err := db.Order("id").Find(&credentials).Error; err != nil {
		t.Fatalf("read target credentials: %v", err)
	}
	if len(credentials) != len(want.credentials) {
		t.Fatalf("target credentials = %d, want %d", len(credentials), len(want.credentials))
	}
	for _, credential := range credentials {
		plaintext, err := cipher.Decrypt(credential.Data)
		if err != nil {
			t.Fatalf("decrypt target credential %d: %v", credential.ID, err)
		}
		if plaintext != want.credentialPlain[credential.ID] {
			t.Errorf("credential %d plaintext changed", credential.ID)
		}
		if credential.IdentityFingerprint != credential.Fingerprint {
			t.Errorf("credential %d identity fingerprint = %q", credential.ID, credential.IdentityFingerprint)
		}
	}

	var rules []models.AccessKeyCostLimitRule
	if err := db.Order("id").Find(&rules).Error; err != nil {
		t.Fatalf("read target cost rules: %v", err)
	}
	if len(rules) != 1 || rules[0].RuleRevision != 3 || rules[0].LimitNanoUSD != 5_000_000 || rules[0].PeriodSeconds != 3600 {
		t.Fatalf("target cost rules = %+v", rules)
	}
	var states []models.AccessKeyCostLimitState
	if err := db.Order("rule_id").Find(&states).Error; err != nil {
		t.Fatalf("read target cost states: %v", err)
	}
	if len(states) != 1 || states[0].UsedNanoUSD != 1_234_567 || states[0].WindowGeneration != 4 || states[0].SnapshotVersion != 9 {
		t.Fatalf("target cost states = %+v", states)
	}
	if states[0].WindowStartedAtMS == nil || states[0].WindowEndsAtMS == nil {
		t.Fatal("target cost state window checkpoint lost")
	}

	var prices []models.ModelPrice
	if err := db.Order("id").Find(&prices).Error; err != nil {
		t.Fatalf("read target model prices: %v", err)
	}
	if len(prices) != 1 || prices[0].ModelID != "gpt-legacy" ||
		prices[0].InputPriceNanoUSDPerMillionTokens == nil ||
		*prices[0].InputPriceNanoUSDPerMillionTokens != 150 {
		t.Fatalf("target model prices = %+v", prices)
	}

	var settings []models.SystemSetting
	if err := db.Order("key").Find(&settings).Error; err != nil {
		t.Fatalf("read target settings: %v", err)
	}
	if len(settings) != len(want.settings) {
		t.Fatalf("target settings = %d, want %d", len(settings), len(want.settings))
	}

	var agentCredentials []models.AgentCredential
	if err := db.Find(&agentCredentials).Error; err != nil || len(agentCredentials) != 1 {
		t.Fatalf("target agent credentials = %d, %v", len(agentCredentials), err)
	}
	var proposals []models.AgentChangeProposal
	if err := db.Find(&proposals).Error; err != nil || len(proposals) != 1 || proposals[0].State != "pending" {
		t.Fatalf("target proposals = %+v, %v", proposals, err)
	}
	var operations []models.ControlOperation
	if err := db.Find(&operations).Error; err != nil || len(operations) != 1 ||
		operations[0].CommitSequence != want.operations[0] {
		t.Fatalf("target operations = %+v, %v", operations, err)
	}
	var jobs []models.Job
	if err := db.Find(&jobs).Error; err != nil || len(jobs) != 1 || jobs[0].Status != "completed" {
		t.Fatalf("target jobs = %+v, %v", jobs, err)
	}
	var accessKeys []models.AccessKey
	if err := db.Find(&accessKeys).Error; err != nil || len(accessKeys) != 1 || accessKeys[0].RPMLimit != 7 {
		t.Fatalf("target access keys = %+v, %v", accessKeys, err)
	}
	var logs int64
	if err := db.Table("request_logs").Count(&logs).Error; err != nil {
		t.Fatalf("count target request logs: %v", err)
	}
	if logs != 0 {
		t.Errorf("target request logs = %d, want 0 (log tables are not migrated)", logs)
	}

	assertDatabaseIntegrity(t, db)
}

func groupIDs(groups []models.Group) []uint {
	ids := make([]uint, 0, len(groups))
	for _, group := range groups {
		ids = append(ids, group.ID)
	}
	return ids
}

func assertDatabaseIntegrity(t *testing.T, db *gorm.DB) {
	t.Helper()
	var violations []struct {
		Table  string `gorm:"column:table"`
		RowID  int64  `gorm:"column:rowid"`
		Parent string `gorm:"column:parent"`
	}
	if err := db.Raw("PRAGMA foreign_key_check").Scan(&violations).Error; err != nil {
		t.Fatalf("foreign key check: %v", err)
	}
	if len(violations) != 0 {
		t.Fatalf("target has foreign key violations: %+v", violations)
	}
	var integrity string
	if err := db.Raw("PRAGMA integrity_check").Scan(&integrity).Error; err != nil {
		t.Fatalf("integrity check: %v", err)
	}
	if integrity != "ok" {
		t.Fatalf("integrity check = %q", integrity)
	}
}

// assertTargetLoadable proves the published database is what the application
// actually starts on: the current migration chain accepts it unchanged and the
// production state loader publishes it with the operator's own key.
//
// Decrypting the stored ciphertext is not the same check. The startup loader
// validates every persisted credential against the channel's live credential
// schema and recomputes the stored fingerprint, so a conversion that dropped or
// mangled a value fails here even though the rows decrypted.
func assertTargetLoadable(t *testing.T, targetPath string, cipher encryption.Service, want fixture) {
	t.Helper()
	runtime, err := storage.Open(targetPath)
	if err != nil {
		t.Fatalf("open converted target with the application: %v", err)
	}
	sqlDB, err := runtime.DB()
	if err != nil {
		t.Fatalf("target pool: %v", err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	if err := storage.AutoMigrate(runtime); err != nil {
		t.Fatalf("apply migrations to converted target: %v", err)
	}

	var groups []models.Group
	if err := runtime.Order("id").Find(&groups).Error; err != nil || len(groups) != len(want.groups) {
		t.Fatalf("runtime groups = %d, %v", len(groups), err)
	}
	var credentials []models.Credential
	if err := runtime.Order("id").Find(&credentials).Error; err != nil {
		t.Fatalf("runtime credentials: %v", err)
	}
	for _, credential := range credentials {
		if _, err := cipher.Decrypt(credential.Data); err != nil {
			t.Fatalf("runtime cannot decrypt credential %d: %v", credential.ID, err)
		}
	}

	manager := state.NewManager()
	registry := state.NewCredentialRegistry()
	channels := channel.NewRegistry()
	if err := stateloader.NewWithCredentialValidation(
		runtime, manager, registry, channels, cipher,
	).Load(t.Context()); err != nil {
		t.Fatalf("startup state loader rejected the converted database: %v", err)
	}
	snapshot := manager.Current()
	if snapshot == nil {
		t.Fatal("startup state loader published no configuration snapshot")
	}
	if len(snapshot.Groups) != len(want.groups) {
		t.Fatalf("published groups = %d, want %d", len(snapshot.Groups), len(want.groups))
	}
	published := registry.Snapshot()
	if len(published) != len(want.credentials) {
		t.Fatalf("published credentials = %d, want %d", len(published), len(want.credentials))
	}
	for _, credential := range published {
		if want.credentialPlain[credential.ID] == "" {
			t.Errorf("published unknown credential %d", credential.ID)
		}
	}
	var affinity models.SystemSetting
	if err := runtime.Where("key LIKE ?", models.InternalSystemSettingPrefix+"affinity.binding.%").Take(&affinity).Error; err != nil {
		t.Fatalf("runtime affinity binding: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal([]byte(affinity.Value), &decoded); err != nil || decoded["credential_id"] != float64(1) {
		t.Fatalf("runtime affinity binding payload = %s, %v", affinity.Value, err)
	}
}

func newTestCipher(t *testing.T, keyMaterial string) encryption.Service {
	t.Helper()
	cipher, err := encryption.NewService(keyMaterial)
	if err != nil {
		t.Fatalf("build encryption service: %v", err)
	}
	return cipher
}

// A source that still holds an open WAL connection is the documented "gateway was
// not stopped" case. The converter cannot prove such a snapshot is a state the
// writer ever settled on, so it refuses, names the remedy, publishes nothing, and
// leaves the source bytes untouched.
func TestConvertRefusesLiveWalSourceAndLeavesItUntouched(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("7f", 32)))

	// Keep a real WAL-mode connection open, which is exactly the state an
	// unstopped legacy gateway leaves behind.
	live, err := gorm.Open(sqlite.Open(sourcePath+"?_pragma=journal_mode(wal)"), &gorm.Config{Logger: logger.Discard})
	if err != nil {
		t.Fatalf("open live WAL source: %v", err)
	}
	sqlDB, err := live.DB()
	if err != nil {
		t.Fatalf("live source pool: %v", err)
	}
	defer func() { _ = sqlDB.Close() }()
	if err := live.Exec(`INSERT INTO system_settings (key, value, updated_at_ms)
		VALUES ('conversion_probe', '1', 1700000200000)`).Error; err != nil {
		t.Fatalf("write into live WAL source: %v", err)
	}
	before := snapshotSource(t, sourcePath)
	sourceFiles := listSideFiles(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err = offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "stop") {
		t.Fatalf("source holding an open WAL connection was not refused with the stop-writers remedy: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("refused conversion published a target: %v", statErr)
	}
	if after := snapshotSource(t, sourcePath); after.hash != before.hash {
		t.Fatalf("source database bytes changed: %s -> %s", before.hash, after.hash)
	}
	for _, name := range sourceFiles {
		if _, statErr := os.Lstat(filepath.Join(dir, name)); statErr != nil {
			t.Fatalf("source side file %s disappeared: %v", name, statErr)
		}
	}
}

// A stopped source in WAL mode (journal mode persisted, no open connection and
// no side files left) is snapshotted consistently and the snapshot is what the
// copy reads. The read transaction may create SQLite's own side files, so the
// source is allowed to gain them only when they are empty and the database bytes
// are unchanged; the published target still has exactly one file.
func TestConvertSnapshotsStoppedWalModeSource(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	want := buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("6d", 32)))
	// Persist WAL as the database journal mode, then close every connection so
	// no -wal/-shm remains: this is a stopped legacy deployment.
	writeSource(t, sourcePath, "PRAGMA journal_mode=WAL")
	writeSource(t, sourcePath, `INSERT INTO system_settings (key, value, updated_at_ms)
		VALUES ('request_log_retention_days', '30', 1700000200000)
		ON CONFLICT(key) DO UPDATE SET value = excluded.value`)
	writeSource(t, sourcePath, "PRAGMA wal_checkpoint(TRUNCATE)")
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		t.Fatalf("convert stopped WAL-mode source: %v", err)
	}
	after := snapshotSource(t, sourcePath)
	if after.hash != before.hash {
		t.Fatalf("source database bytes changed: %s -> %s", before.hash, after.hash)
	}
	if after.walBytes != 0 {
		t.Fatalf("source WAL gained %d bytes", after.walBytes)
	}
	if result.Tables["system_settings"] != int64(len(want.settings)) {
		t.Fatalf("system_settings copied = %d, want %d", result.Tables["system_settings"], len(want.settings))
	}
	assertSingleFileTarget(t, targetDSN)
	target := openReadOnly(t, targetDSN)
	var value string
	if err := target.Raw("SELECT value FROM system_settings WHERE key = 'request_log_retention_days'").Scan(&value).Error; err != nil {
		t.Fatalf("read converted setting: %v", err)
	}
	if value != "30" {
		t.Fatalf("converted setting = %q, want the latest written value", value)
	}
}

func listSideFiles(t *testing.T, path string) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("read source directory: %v", err)
	}
	var files []string
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), filepath.Base(path)) {
			files = append(files, entry.Name())
		}
	}
	slices.Sort(files)
	return files
}

func TestConvertLegacyDatabase(t *testing.T) {
	sourceDir, targetDir := t.TempDir(), t.TempDir()
	sourcePath := filepath.Join(sourceDir, "gpt-load.db")
	targetDSN := filepath.Join(targetDir, "data", "gpt-load.db")

	keyMaterial := strings.Repeat("ab", 32)
	cipher := newTestCipher(t, keyMaterial)
	want := buildLegacyFixture(t, sourcePath, cipher)
	sourceSnapshot := snapshotSource(t, sourcePath)

	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		t.Fatalf("convert legacy database: %v", err)
	}

	assertSourceUnchanged(t, sourcePath, sourceSnapshot)
	// Publication is a single file: no journal sidecar may exist for the target.
	assertSingleFileTarget(t, targetDSN)

	target := openReadOnly(t, targetDSN)
	assertTargetSchema(t, target)
	assertTargetContent(t, target, cipher, want)
	assertTargetLoadable(t, targetDSN, cipher, want)

	if result.Tables["groups"] != int64(len(want.groups)) {
		t.Errorf("groups copied = %d, want %d", result.Tables["groups"], len(want.groups))
	}
	if result.Tables["request_logs"] != 0 {
		t.Errorf("request_logs copied = %d, want 0", result.Tables["request_logs"])
	}
	if _, ok := result.Tables["credential_observations"]; ok {
		t.Error("subscription table reported as copied")
	}
	if !slices.Contains(result.RetiredColumns, "credentials.auth_error_code") {
		t.Errorf("retired columns = %v, want credentials.auth_error_code", result.RetiredColumns)
	}
	var journalMode string
	if err := target.Raw("PRAGMA journal_mode").Scan(&journalMode).Error; err != nil {
		t.Fatalf("read target journal mode: %v", err)
	}
	if !strings.EqualFold(journalMode, "delete") {
		t.Errorf("target journal mode = %q, want a single-file delete journal", journalMode)
	}
	if _, statErr := os.Lstat(filepath.Join(targetDir, "data", "encryption.key")); !os.IsNotExist(statErr) {
		t.Errorf("converter must not copy key material: %v", statErr)
	}
	if _, statErr := os.Stat(filepath.Join(sourceDir, "encryption.key")); !os.IsNotExist(statErr) {
		t.Errorf("test fixture must not create key material in DATA_DIR: %v", statErr)
	}
}

func TestConvertRejectsSubscriptionGroups(t *testing.T) {
	for _, channelID := range []string{"codex", "anthropic-subscription", "antigravity", "grok"} {
		t.Run(channelID, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("cd", 32)))

			writeSource(t, sourcePath, `INSERT INTO groups (id, name, channel_id, connection_type, params, models, enabled, created_at_ms, updated_at_ms)
				VALUES (900, 'subscription', ?, 'api_key', '{}', '[]', 1, 1, 1)`, channelID)
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err == nil || !strings.Contains(err.Error(), "subscription") {
				t.Fatalf("subscription group accepted: %v", err)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
			assertSourceUnchanged(t, sourcePath, before)
		})
	}
}

func TestConvertRejectsSubscriptionStageOnOrdinaryCredential(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("aa", 32)))
	// The real legacy `credential_stages` row: string primary key, the OAuth
	// column that only the retired subscription flow wrote, and no
	// `credential_id` column at all.
	writeSource(t, sourcePath, `INSERT INTO credential_stages
		(id, channel_id, connection_type, authorization_method, status, encrypted_payload,
		 safe_summary_json, expires_at_ms, created_at_ms, updated_at_ms)
		VALUES ('2f1b0a54-0c4a-4b4f-9d3f-1a2b3c4d5e6f', 'codex', 'subscription', 'browser_oauth',
			'pending_authorization', 'encrypted-oauth-payload', '{}', 1700000900000, 1700000800000, 1700000800000)`)
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "subscription") {
		t.Fatalf("stored subscription stage accepted: %v", err)
	}
	if strings.Contains(err.Error(), "no column named") {
		t.Fatalf("converter queried a column the real legacy schema does not have: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("rejected conversion published a target: %v", statErr)
	}
	assertSourceUnchanged(t, sourcePath, before)
}

func TestConvertRejectsSubscriptionResetOperation(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	want := buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("bb", 32)))
	writeSource(t, sourcePath, `INSERT INTO credential_reset_operations
		(idempotency_key, request_digest, group_id, credential_id, redeem_request_id, state,
		 created_at_ms, updated_at_ms)
		VALUES ('3a2b1c0d-1e2f-4a5b-8c9d-0e1f2a3b4c5d', x'00', ?, ?, '4b3c2d1e-2f3a-4b5c-9d0e-1f2a3b4c5d6e',
			'succeeded', 1700000800000, 1700000900000)`, want.ordinaryGroupID, want.credentials[0])
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "subscription") {
		t.Fatalf("subscription reset operation accepted: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("rejected conversion published a target: %v", statErr)
	}
	assertSourceUnchanged(t, sourcePath, before)
}

// Every retired subscription table is refused conservatively: any stored row
// there belongs to a feature this release cannot express, and discarding it is
// not authorized. The fixture uses the real historical schema for each table.
func TestConvertRejectsEveryStoredSubscriptionTable(t *testing.T) {
	for _, test := range []struct {
		name      string
		statement string
	}{
		{
			name: "stage",
			statement: `INSERT INTO credential_stages
				(id, channel_id, connection_type, authorization_method, status, encrypted_payload,
				 safe_summary_json, expires_at_ms, created_at_ms, updated_at_ms)
				VALUES ('11111111-1111-4111-8111-111111111111', 'codex', 'subscription', 'device_oauth',
					'consumed', 'p', '{}', 1, 1, 1)`,
		},
		{
			name: "consumed stage without group",
			statement: `INSERT INTO credential_stages
				(id, channel_id, connection_type, authorization_method, status, encrypted_payload,
				 safe_summary_json, expires_at_ms, created_at_ms, updated_at_ms)
				VALUES ('22222222-2222-4222-8222-222222222222', 'codex', 'subscription', 'browser_oauth',
					'outcome_unknown', 'p', '{}', 1, 1, 1)`,
		},
		{
			name: "observation",
			statement: `INSERT INTO credential_observations
				(credential_id, identity_fingerprint, snapshot_json, state, updated_at_ms)
				VALUES (1, 'fp', '{}', 'fresh', 1)`,
		},
		{
			name: "reset operation",
			statement: `INSERT INTO credential_reset_operations
				(idempotency_key, request_digest, group_id, credential_id, redeem_request_id, state,
				 created_at_ms, updated_at_ms)
				VALUES ('33333333-3333-4333-8333-333333333333', x'00', 1, 1,
					'44444444-4444-4444-8444-444444444444', 'prepared', 1, 1)`,
		},
		{
			name: "stage without a group",
			statement: `INSERT INTO credential_stages
				(id, channel_id, connection_type, authorization_method, status, encrypted_payload,
				 safe_summary_json, expires_at_ms, created_at_ms, updated_at_ms)
				VALUES ('55555555-5555-4555-8555-555555555555', 'codex', 'subscription', 'oauth_file',
					'failed', 'p', '{}', 1, 1, 1)`,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("5e", 32)))
			writeSource(t, sourcePath, test.statement)
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err == nil || !strings.Contains(err.Error(), "subscription") {
				t.Fatalf("stored %s accepted: %v", test.name, err)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
			assertSourceUnchanged(t, sourcePath, before)
		})
	}
}

// An empty retired subscription table is the normal state of an API-key-only
// deployment: it must convert, and the published target must not declare the
// retired table at all.
func TestConvertAcceptsEmptyRetiredSubscriptionTables(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	cipher := newTestCipher(t, strings.Repeat("7d", 32))
	want := buildLegacyFixture(t, sourcePath, cipher)
	assertFixtureMatchesRetiredBaseline(t, openReadOnly(t, sourcePath))

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	if _, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	}); err != nil {
		t.Fatalf("source with empty retired subscription tables rejected: %v", err)
	}
	target := openReadOnly(t, targetDSN)
	for _, table := range migrations.RetiredBaselineTables() {
		if target.Migrator().HasTable(table) {
			t.Errorf("published target still declares retired table %s", table)
		}
	}
	var credentials int64
	if err := target.Table("credentials").Count(&credentials).Error; err != nil {
		t.Fatalf("count converted credentials: %v", err)
	}
	if credentials != int64(len(want.credentials)) {
		t.Fatalf("converted credentials = %d, want %d", credentials, len(want.credentials))
	}
}

func TestConvertRejectsSubscriptionAuthState(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("cc", 32)))
	writeSource(t, sourcePath, `UPDATE credentials SET auth_state = 'reauthorization_required', auth_error_code = 'legacy_error'
		WHERE id = (SELECT max(id) FROM credentials)`)
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "subscription") {
		t.Fatalf("subscription auth state accepted: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("rejected conversion published a target: %v", statErr)
	}
	assertSourceUnchanged(t, sourcePath, before)
}

func TestConvertRejectsUntraceableSubscriptionCredential(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	cipher := newTestCipher(t, strings.Repeat("ef", 32))
	want := buildLegacyFixture(t, sourcePath, cipher)

	encrypted, err := cipher.Encrypt(`{"access_token":"subscription-oauth-token"}`)
	if err != nil {
		t.Fatalf("encrypt subscription credential: %v", err)
	}
	// A subscription credential is stored inside an ordinary API group with no
	// retired subscription bookkeeping and no auth marker: the converter can
	// still identify it because its identity fingerprint is derived from
	// subscription material instead of the group's connection type.
	writeSource(t, sourcePath, `UPDATE credentials SET data = ?, fingerprint = 'fp-subscription', identity_fingerprint = 'fp-subscription'
		WHERE id = ?`, encrypted, want.credentials[0])
	writeSource(t, sourcePath, `UPDATE groups SET channel_id = 'credential-parent-of-subscription' WHERE id = ?`, want.ordinaryGroupID)
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err = offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "subscription") {
		t.Fatalf("untraceable subscription credential accepted: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("rejected conversion published a target: %v", statErr)
	}
	assertSourceUnchanged(t, sourcePath, before)
}

func TestConvertRefusesToOverwriteExistingTarget(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("12", 32)))

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	if err := os.MkdirAll(filepath.Dir(targetDSN), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(targetDSN, []byte("attacker-controlled"), 0o600); err != nil {
		t.Fatal(err)
	}

	_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "already exists") {
		t.Fatalf("existing target accepted: %v", err)
	}
	content, readErr := os.ReadFile(targetDSN)
	if readErr != nil || string(content) != "attacker-controlled" {
		t.Fatalf("existing target modified: %q, %v", content, readErr)
	}
}

func TestConvertRejectsUnknownLegacyChain(t *testing.T) {
	for _, test := range []struct {
		name   string
		ledger []string
	}{
		{"missing active migrations", []string{"0001_initial"}},
		{"unknown retired migration", []string{"0001_initial", "0002_access_key_cost_limits", "0009_unknown_retired"}},
		{"retired migration for the wrong schema", []string{"0001_initial", "0002_access_key_cost_limits", retiredSubscriptionMigrationID, "0004_usage_stats_group_activity_index"}},
		{"unknown prefix", []string{"0001_initial", "0002_access_key_cost_limits", "0003_other_migration"}},
		{"retired price multiplier without the active chain", []string{"0001_initial", retiredPriceMultiplierMigrationID}},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("34", 32)))
			setLedger(t, sourcePath, test.ledger)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err == nil {
				t.Fatalf("unknown legacy chain %v accepted", test.ledger)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
		})
	}
}

func TestConvertRejectsIncompleteSourceLedger(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("56", 32)))
	writeSource(t, sourcePath, "DELETE FROM schema_migrations WHERE id = ?", retiredSubscriptionMigrationID)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "0003") {
		t.Fatalf("missing retired migration accepted: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("rejected conversion published a target: %v", statErr)
	}
}

// A database that ran while the price-multiplier migration still existed keeps
// `0009_price_multipliers` in its ledger: the current chain skips retired IDs
// instead of deleting their rows. Both shapes must be accepted, everything else
// about the chain must still be exact.
func TestConvertAcceptsRetiredPriceMultiplierLedgerEntry(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	cipher := newTestCipher(t, strings.Repeat("57", 32))
	want := buildLegacyFixture(t, sourcePath, cipher)

	ledger := snapshotSource(t, sourcePath).ledger
	withRetired := append(slices.Clone(ledger), retiredPriceMultiplierMigrationID)
	slices.Sort(withRetired)
	setLedger(t, sourcePath, withRetired)
	if !slices.Equal(snapshotSource(t, sourcePath).ledger, withRetired) {
		t.Fatalf("fixture ledger = %v, want %v", snapshotSource(t, sourcePath).ledger, withRetired)
	}
	before := snapshotSource(t, sourcePath)

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		t.Fatalf("source with the retired price-multiplier ledger entry rejected: %v", err)
	}
	if result.Tables["credentials"] != int64(len(want.credentials)) {
		t.Fatalf("credentials copied = %d, want %d", result.Tables["credentials"], len(want.credentials))
	}
	assertSourceUnchanged(t, sourcePath, before)
	assertTargetLoadable(t, targetDSN, cipher, want)
}

// Only the known retired IDs are optional. A gap in the active chain, an extra
// retired entry that this repository never created, or a missing mandatory entry
// is still refused.
func TestConvertRejectsLedgerWithRetiredEntryAndGap(t *testing.T) {
	for _, test := range []struct {
		name string
		drop string
	}{
		{"gap before the retired entry", "0008_remove_inject_usage_options"},
		{"gap after the retired entry", "0010_single_credential_per_group"},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("58", 32)))
			ledger := append(snapshotSource(t, sourcePath).ledger, retiredPriceMultiplierMigrationID)
			slices.Sort(ledger)
			ledger = slices.DeleteFunc(ledger, func(id string) bool { return id == test.drop })
			setLedger(t, sourcePath, ledger)
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err == nil {
				t.Fatalf("ledger with a gap accepted: %v", ledger)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
			assertSourceUnchanged(t, sourcePath, before)
		})
	}
}

func TestConvertRejectsUnclassifiedSourceTable(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("78", 32)))
	writeSource(t, sourcePath, "CREATE TABLE mystery (id integer PRIMARY KEY)")
	writeSource(t, sourcePath, "INSERT INTO mystery VALUES (1)")

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "mystery") {
		t.Fatalf("unclassified source table accepted: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("rejected conversion published a target: %v", statErr)
	}
}

// A legacy source missing a required retained column would let the target fall
// back to a column default instead of the stored value, so the conversion must
// refuse rather than silently publish defaults.
func TestConvertRejectsMissingRequiredSourceColumn(t *testing.T) {
	for _, test := range []struct {
		name      string
		statement string
		want      string
	}{
		{
			name:      "group provider url",
			statement: "ALTER TABLE groups DROP COLUMN provider_url",
			want:      "groups.provider_url",
		},
		{
			name:      "group proxied column",
			statement: "ALTER TABLE groups DROP COLUMN proxy_config",
			want:      "groups.proxy_config",
		},
		{
			name:      "access key expiry",
			statement: "ALTER TABLE access_keys DROP COLUMN expires_at_ms",
			want:      "access_keys.expires_at_ms",
		},
		{
			name:      "cost limit rule period",
			statement: "ALTER TABLE access_key_cost_limit_states DROP COLUMN window_generation",
			want:      "access_key_cost_limit_states.window_generation",
		},
		{
			name:      "model price manual flag",
			statement: "ALTER TABLE model_prices DROP COLUMN is_manual",
			want:      "model_prices.is_manual",
		},
		{
			name:      "agent credential scopes",
			statement: "ALTER TABLE agent_credentials DROP COLUMN scopes",
			want:      "agent_credentials.scopes",
		},
		{
			name:      "system setting value",
			statement: "ALTER TABLE system_settings DROP COLUMN value",
			want:      "system_settings.value",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()
			sourcePath := filepath.Join(dir, "gpt-load.db")
			buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("9b", 32)))
			writeSource(t, sourcePath, test.statement)
			before := snapshotSource(t, sourcePath)

			targetDSN := filepath.Join(dir, "out", "gpt-load.db")
			_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("source missing %s accepted: %v", test.want, err)
			}
			if !strings.Contains(err.Error(), "missing") {
				t.Fatalf("error does not name the missing column: %v", err)
			}
			if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
				t.Fatalf("rejected conversion published a target: %v", statErr)
			}
			assertSourceUnchanged(t, sourcePath, before)
		})
	}
}

// The retired subscription columns may be absent from an already-converted
// source; that is the one allowed missing-column case.
func TestConvertAcceptsSourceWithoutRetiredColumns(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	want := buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("9c", 32)))
	writeSource(t, sourcePath, "ALTER TABLE credentials DROP COLUMN auth_state")
	writeSource(t, sourcePath, "ALTER TABLE credentials DROP COLUMN auth_error_code")
	writeSource(t, sourcePath, "ALTER TABLE groups DROP COLUMN connection_type")

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err != nil {
		t.Fatalf("source without retired columns rejected: %v", err)
	}
	if len(result.RetiredColumns) != 0 {
		t.Fatalf("retired columns = %v, want none reported", result.RetiredColumns)
	}
	target := openReadOnly(t, targetDSN)
	var credentials int64
	if err := target.Table("credentials").Count(&credentials).Error; err != nil {
		t.Fatalf("count converted credentials: %v", err)
	}
	if credentials != int64(len(want.credentials)) {
		t.Fatalf("converted credentials = %d, want %d", credentials, len(want.credentials))
	}
}

// Concurrent publishers must not produce a partial or replaced target: exactly
// one wins, the loser fails, and the winner's file is a complete database.
func TestConvertConcurrentPublishHasSingleCompleteWinner(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("9d", 32)))
	targetDSN := filepath.Join(dir, "out", "gpt-load.db")

	type outcome struct {
		result offlineconvert.Result
		err    error
	}

	const publishers = 4
	results := make(chan outcome, publishers)
	for range publishers {
		go func() {
			result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
				SourceDSN: sourcePath,
				TargetDSN: targetDSN,
			})
			results <- outcome{result: result, err: err}
		}()
	}
	var winners int
	for range publishers {
		select {
		case outcome := <-results:
			if outcome.err == nil {
				winners++
			} else if !strings.Contains(outcome.err.Error(), "already exists") {
				t.Errorf("loser failed for the wrong reason: %v", outcome.err)
			}
		case <-time.After(2 * time.Minute):
			t.Fatal("concurrent conversion did not finish")
		}
	}
	if winners != 1 {
		t.Fatalf("winners = %d, want exactly 1", winners)
	}
	assertSingleFileTarget(t, targetDSN)
	target := openReadOnly(t, targetDSN)
	var integrity string
	if err := target.Raw("PRAGMA integrity_check").Scan(&integrity).Error; err != nil || integrity != "ok" {
		t.Fatalf("published target integrity = %q, %v", integrity, err)
	}
	var groups int64
	if err := target.Table("groups").Count(&groups).Error; err != nil || groups == 0 {
		t.Fatalf("published target groups = %d, %v", groups, err)
	}
}

func TestConvertRejectsUnclassifiedSourceColumn(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("9a", 32)))
	writeSource(t, sourcePath, "ALTER TABLE groups ADD COLUMN legacy_only TEXT NOT NULL DEFAULT ''")

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	_, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	})
	if err == nil || !strings.Contains(err.Error(), "groups.legacy_only") {
		t.Fatalf("unclassified source column accepted: %v", err)
	}
	if _, statErr := os.Lstat(targetDSN); !os.IsNotExist(statErr) {
		t.Fatalf("rejected conversion published a target: %v", statErr)
	}
}

// SQLite only advances `sqlite_sequence` for an explicit rowid insert when the
// inserted value exceeds the stored high-water mark. A source whose highest ID
// was deleted therefore carries a high-water mark above every remaining row, and
// the converter must copy it. Otherwise the very next created row reuses an ID
// that external references (logs, backups, audit exports) still point at.
func TestConvertPreservesAutoincrementHighWaterMark(t *testing.T) {
	dir := t.TempDir()
	sourcePath := filepath.Join(dir, "gpt-load.db")
	want := buildLegacyFixture(t, sourcePath, newTestCipher(t, strings.Repeat("5f", 32)))

	// Delete the highest IDs of an AUTOINCREMENT table so the source's remaining
	// max id and its high-water mark deliberately disagree.
	deletedGroup := want.groups[len(want.groups)-1]
	deletedCredential := want.credentials[len(want.credentials)-1]
	writeSource(t, sourcePath, "DELETE FROM credentials WHERE id = ?", deletedCredential)
	writeSource(t, sourcePath, "DELETE FROM groups WHERE id = ?", deletedGroup)
	groupHighWater, credentialHighWater := sourceSequence(t, sourcePath)
	if groupHighWater <= int64(slices.Max(want.groups[:len(want.groups)-1])) {
		t.Fatalf("fixture high-water %d does not exceed the remaining max group id", groupHighWater)
	}
	if credentialHighWater <= int64(slices.Max(want.credentials[:len(want.credentials)-1])) {
		t.Fatalf("fixture high-water %d does not exceed the remaining max credential id", credentialHighWater)
	}

	targetDSN := filepath.Join(dir, "out", "gpt-load.db")
	if _, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: sourcePath,
		TargetDSN: targetDSN,
	}); err != nil {
		t.Fatalf("convert source with deleted highest ids: %v", err)
	}

	target := openReadOnly(t, targetDSN)
	gotGroupHighWater := sequenceFor(t, target, "groups")
	gotCredentialHighWater := sequenceFor(t, target, "credentials")
	if gotGroupHighWater != groupHighWater {
		t.Errorf("target groups high-water = %d, want the source value %d", gotGroupHighWater, groupHighWater)
	}
	if gotCredentialHighWater != credentialHighWater {
		t.Errorf("target credentials high-water = %d, want the source value %d", gotCredentialHighWater, credentialHighWater)
	}
	// A dropped table is not copied, so its sequence must not be copied either.
	for _, table := range migrations.RetiredBaselineTables() {
		if sequenceExists(t, target, table) {
			t.Errorf("target carries a sqlite_sequence row for the dropped table %s", table)
		}
	}
	assertTargetSequencesConsistent(t, target)
}

// sourceSequence returns the source high-water marks of the two AUTOINCREMENT
// tables the conversion copies.
func sourceSequence(t *testing.T, path string) (int64, int64) {
	t.Helper()
	db := openReadOnly(t, path)
	return sequenceFor(t, db, "groups"), sequenceFor(t, db, "credentials")
}

func sequenceFor(t *testing.T, db *gorm.DB, table string) int64 {
	t.Helper()
	var value int64
	if err := db.Raw("SELECT seq FROM sqlite_sequence WHERE name = ?", table).Scan(&value).Error; err != nil {
		t.Fatalf("read sqlite_sequence for %s: %v", table, err)
	}
	return value
}

func quoteColumn(name string) string {
	return "`" + strings.ReplaceAll(name, "`", "``") + "`"
}

func sequenceExists(t *testing.T, db *gorm.DB, table string) bool {
	t.Helper()
	var count int64
	if err := db.Raw("SELECT count(*) FROM sqlite_sequence WHERE name = ?", table).Scan(&count).Error; err != nil {
		t.Fatalf("count sqlite_sequence rows for %s: %v", table, err)
	}
	return count != 0
}

// assertTargetSequencesConsistent proves no copied table starts below a stored
// row, which would make the next insert collide with existing data.
func assertTargetSequencesConsistent(t *testing.T, db *gorm.DB) {
	t.Helper()
	var rows []struct {
		Name string `gorm:"column:name"`
		Seq  int64  `gorm:"column:seq"`
	}
	if err := db.Raw("SELECT name, seq FROM sqlite_sequence").Scan(&rows).Error; err != nil {
		t.Fatalf("read target sqlite_sequence: %v", err)
	}
	for _, row := range rows {
		// AUTOINCREMENT only appears on integer primary keys, which are exactly
		// the tables that carry a sequence row, but a composite key table can
		// still be listed; the pragma reports its real key column.
		var keyColumns []string
		if err := db.Raw("SELECT name FROM pragma_table_info(?) WHERE pk > 0 ORDER BY pk", row.Name).Scan(&keyColumns).Error; err != nil {
			t.Fatalf("read %s primary key: %v", row.Name, err)
		}
		if len(keyColumns) != 1 {
			continue
		}
		var maxID int64
		if err := db.Raw("SELECT COALESCE(max(" + quoteColumn(keyColumns[0]) + "), 0) FROM " + quoteColumn(row.Name)).Scan(&maxID).Error; err != nil {
			t.Fatalf("read max key of %s: %v", row.Name, err)
		}
		if row.Seq < maxID {
			t.Errorf("target %s high-water %d is below its stored max key %d", row.Name, row.Seq, maxID)
		}
	}
}
