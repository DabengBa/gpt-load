package control

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/encryption"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

// ––––– 精确快照 helper –––––
//
// The helpers below capture the complete persisted and runtime state that a
// channel switch may touch, so "unchanged" is proven by comparing real rows and
// published objects instead of inferring stability from a revision counter.

func captureGroupRow(t *testing.T, fixture serviceFixture, groupID uint) models.Group {
	t.Helper()
	var group models.Group
	if err := fixture.db.First(&group, groupID).Error; err != nil {
		t.Fatalf("capture group %d: %v", groupID, err)
	}
	return group
}

func captureGroupCredentialRows(t *testing.T, fixture serviceFixture, groupID uint) []models.Credential {
	t.Helper()
	var rows []models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Order("id ASC").Find(&rows).Error; err != nil {
		t.Fatalf("capture credentials for group %d: %v", groupID, err)
	}
	return rows
}

func captureModelPriceRows(t *testing.T, fixture serviceFixture) []models.ModelPrice {
	t.Helper()
	var rows []models.ModelPrice
	if err := fixture.db.Order("id ASC").Find(&rows).Error; err != nil {
		t.Fatalf("capture model price rows: %v", err)
	}
	return rows
}

// ensureManualPrice upserts one manual price row, keeping the existing row ID
// and created_at_ms when a catalog/automatic row already owns the identity.
func ensureManualPrice(t *testing.T, db *gorm.DB, channelID, modelID string, value int64) {
	t.Helper()
	price := models.ModelPrice{
		ChannelID: channelID, ModelID: modelID,
		InputPriceNanoUSDPerMillionTokens: &value, IsManual: true,
	}
	err := db.Clauses(clause.OnConflict{
		Columns: []clause.Column{{Name: "channel_id"}, {Name: "model_id"}},
		DoUpdates: clause.Assignments(map[string]any{
			"is_manual": true,
			"input_price_nano_usd_per_million_tokens": value,
		}),
	}).Create(&price).Error
	if err != nil {
		t.Fatalf("ensure manual price %s/%s: %v", channelID, modelID, err)
	}
}

func captureCredentialRegistryEntries(
	t *testing.T,
	fixture serviceFixture,
	groupID uint,
) []state.CredentialEntry {
	t.Helper()
	var ids []uint
	if err := fixture.db.Model(&models.Credential{}).
		Where("group_id = ?", groupID).Order("id ASC").Pluck("id", &ids).Error; err != nil {
		t.Fatalf("capture credential ids for group %d: %v", groupID, err)
	}
	if len(ids) == 0 {
		return nil
	}
	entries, err := fixture.registry.SnapshotGroupCredentialEntriesExact(groupID, ids)
	if err != nil {
		t.Fatalf("capture runtime credential entries for group %d: %v", groupID, err)
	}
	return entries
}

// groupSwitchState is the full state bundle asserted by failure-path tests.
type groupSwitchState struct {
	Group              models.Group
	Credentials        []models.Credential
	ModelPrices        []models.ModelPrice
	CredentialEntries  []state.CredentialEntry
	Snapshot           *state.ConfigSnapshot
	View               state.GroupView
	PriceTable         *pricing.Table
	CredentialRegistry int
}

func captureGroupSwitchState(
	t *testing.T,
	fixture serviceFixture,
	groupID uint,
) groupSwitchState {
	t.Helper()
	snapshot := fixture.manager.Current()
	view, ok := snapshot.LookupGroup(groupID)
	if !ok {
		t.Fatalf("group %d missing from runtime snapshot", groupID)
	}
	entries := captureCredentialRegistryEntries(t, fixture, groupID)
	return groupSwitchState{
		Group:              captureGroupRow(t, fixture, groupID),
		Credentials:        captureGroupCredentialRows(t, fixture, groupID),
		ModelPrices:        captureModelPriceRows(t, fixture),
		CredentialEntries:  entries,
		Snapshot:           snapshot,
		View:               view,
		PriceTable:         fixture.priceRuntime.Load(),
		CredentialRegistry: fixture.registry.CredentialCountsByGroup([]uint{groupID})[groupID],
	}
}

// assertGroupSwitchStateEqual proves a failed switch changed nothing: complete
// persisted Group row, complete credential rows, complete model price rows,
// published config snapshot (identity and group view), runtime credential
// registry entries, and the published price table.
func assertGroupSwitchStateEqual(t *testing.T, before, after groupSwitchState) {
	t.Helper()
	if !reflect.DeepEqual(before.Group, after.Group) {
		t.Fatalf("group row changed:\nbefore=%+v\nafter= %+v", before.Group, after.Group)
	}
	if !reflect.DeepEqual(before.Credentials, after.Credentials) {
		t.Fatalf("credential rows changed:\nbefore=%+v\nafter= %+v", before.Credentials, after.Credentials)
	}
	if !reflect.DeepEqual(before.ModelPrices, after.ModelPrices) {
		t.Fatalf("model price rows changed:\nbefore=%+v\nafter= %+v", before.ModelPrices, after.ModelPrices)
	}
	if before.Snapshot != after.Snapshot {
		t.Fatalf("runtime snapshot was republished: before rev=%d after rev=%d",
			before.Snapshot.Revision, after.Snapshot.Revision)
	}
	if !reflect.DeepEqual(before.View, after.View) {
		t.Fatalf("runtime group view changed:\nbefore=%+v\nafter= %+v", before.View, after.View)
	}
	if before.CredentialRegistry != after.CredentialRegistry {
		t.Fatalf("runtime credential registry size changed: %d -> %d",
			before.CredentialRegistry, after.CredentialRegistry)
	}
	if !reflect.DeepEqual(before.CredentialEntries, after.CredentialEntries) {
		t.Fatalf("runtime credential entries changed:\nbefore=%+v\nafter= %+v",
			before.CredentialEntries, after.CredentialEntries)
	}
	if before.PriceTable != after.PriceTable {
		t.Fatal("published price table was replaced by a failed switch")
	}
	// The two runtime objects above are compared by identity; equality of the
	// persisted rows that feed them is asserted above as well.
	if before.Group.ChannelID != after.Group.ChannelID ||
		before.Group.ConnectionType != after.Group.ConnectionType {
		t.Fatal("channel_id/connection_type changed by a failed switch")
	}
}

func assertCredentialRowsIdentical(t *testing.T, before, after []models.Credential) {
	t.Helper()
	if len(before) != len(after) {
		t.Fatalf("credential row count changed: before=%d after=%d", len(before), len(after))
	}
	for index := range before {
		if !reflect.DeepEqual(before[index], after[index]) {
			t.Fatalf("credential[%d] changed:\nbefore=%+v\nafter= %+v", index, before[index], after[index])
		}
		if before[index].Data != after[index].Data {
			t.Fatalf("credential[%d] ciphertext changed", index)
		}
	}
}

// assertSuccessfulGroupSwitchState compares the successful before/after
// contract. Only the target fields, target-derived runtime identity, snapshot
// revision, and publication of an equivalent price table are allowed to differ.
func assertSuccessfulGroupSwitchState(
	t *testing.T,
	fixture serviceFixture,
	groupID uint,
	before, after groupSwitchState,
	target channel.ID,
	targetConnectionType models.ConnectionType,
	targetParams models.JSON,
) {
	t.Helper()

	expectedGroup := before.Group
	expectedGroup.ChannelID = string(target)
	expectedGroup.ConnectionType = targetConnectionType
	expectedGroup.Params = append(models.JSON(nil), targetParams...)
	// GORM may advance this timestamp for the successful group write.
	expectedGroup.UpdatedAtMS = after.Group.UpdatedAtMS
	if !reflect.DeepEqual(expectedGroup, after.Group) {
		t.Fatalf("group changed outside the switch contract:\nexpected=%+v\nafter=%+v", expectedGroup, after.Group)
	}
	if string(after.Group.Models) != string(before.Group.Models) {
		t.Fatalf("Models JSON changed:\nbefore=%s\nafter=%s", before.Group.Models, after.Group.Models)
	}
	if after.Group.ChannelID != string(target) || after.Group.ConnectionType != targetConnectionType {
		t.Fatalf("group target = %q/%q, want %q/%q", after.Group.ChannelID, after.Group.ConnectionType, target, targetConnectionType)
	}
	if string(after.Group.Params) != string(targetParams) || string(after.Group.Params) == `{}` {
		t.Fatalf("target params = %s, want preserved non-empty %s", after.Group.Params, targetParams)
	}

	if !reflect.DeepEqual(before.Credentials, after.Credentials) {
		t.Fatalf("full credential rows changed:\nbefore=%+v\nafter=%+v", before.Credentials, after.Credentials)
	}
	assertCredentialRowsIdentical(t, before.Credentials, after.Credentials)
	if !reflect.DeepEqual(before.ModelPrices, after.ModelPrices) {
		t.Fatalf("persisted ModelPrice rows changed:\nbefore=%+v\nafter=%+v", before.ModelPrices, after.ModelPrices)
	}
	if !reflect.DeepEqual(before.PriceTable, after.PriceTable) {
		t.Fatalf("PriceRuntime rules changed:\nbefore=%+v\nafter=%+v", before.PriceTable, after.PriceTable)
	}

	if len(before.CredentialEntries) != len(after.CredentialEntries) {
		t.Fatalf("runtime credential entry count changed: before=%d after=%d", len(before.CredentialEntries), len(after.CredentialEntries))
	}
	for index := range before.CredentialEntries {
		expectedEntry := before.CredentialEntries[index]
		expectedEntry.IdentityGeneration = stateloader.CredentialIdentityGeneration(
			before.Credentials[index].IdentityFingerprint,
			string(target),
			string(targetConnectionType),
			json.RawMessage(targetParams),
		)
		if !reflect.DeepEqual(expectedEntry, after.CredentialEntries[index]) {
			t.Fatalf("runtime CredentialEntry[%d] changed outside target identity:\nexpected=%+v\nafter=%+v", index, expectedEntry, after.CredentialEntries[index])
		}
	}

	if before.Snapshot == nil || after.Snapshot == nil {
		t.Fatal("successful switch must have before and after ConfigSnapshot")
	}
	if after.Snapshot == before.Snapshot || after.Snapshot.Revision <= before.Snapshot.Revision {
		t.Fatalf("ConfigSnapshot was not republished with a higher revision: before=%p/%d after=%p/%d",
			before.Snapshot, before.Snapshot.Revision, after.Snapshot, after.Snapshot.Revision)
	}
	beforeSnapshotView, ok := before.Snapshot.Groups[groupID]
	if !ok {
		t.Fatalf("group %d missing from before ConfigSnapshot", groupID)
	}
	afterSnapshotView, ok := after.Snapshot.Groups[groupID]
	if !ok {
		t.Fatalf("group %d missing from after ConfigSnapshot", groupID)
	}
	assertRuntimeGroupViewSwitch(t, fixture, beforeSnapshotView, afterSnapshotView, target, targetConnectionType, targetParams)
	if !reflect.DeepEqual(before.Snapshot.GroupCatalog[groupID], after.Snapshot.GroupCatalog[groupID]) {
		t.Fatalf("target GroupCatalog entry changed:\nbefore=%+v\nafter=%+v",
			before.Snapshot.GroupCatalog[groupID], after.Snapshot.GroupCatalog[groupID])
	}
	assertRuntimeGroupViewSwitch(t, fixture, before.View, after.View, target, targetConnectionType, targetParams)
}

// assertRuntimeGroupViewSwitch compares every GroupView field other than the
// target-derived channel, params, and ResolvedTarget, then checks target
// resolution against the channel registry.
func assertRuntimeGroupViewSwitch(
	t *testing.T,
	fixture serviceFixture,
	before, after state.GroupView,
	target channel.ID,
	targetConnectionType models.ConnectionType,
	targetParams models.JSON,
) {
	t.Helper()
	beforeComparable, afterComparable := before, after
	beforeComparable.ChannelID, afterComparable.ChannelID = "", ""
	beforeComparable.ConnectionType, afterComparable.ConnectionType = "", ""
	beforeComparable.Params, afterComparable.Params = nil, nil
	beforeComparable.ClientProtocols, afterComparable.ClientProtocols = nil, nil
	beforeComparable.ResolvedTarget, afterComparable.ResolvedTarget = channel.ResolvedTarget{}, channel.ResolvedTarget{}
	if !reflect.DeepEqual(beforeComparable, afterComparable) {
		t.Fatalf("runtime GroupView changed outside target fields:\nbefore=%+v\nafter=%+v", before, after)
	}
	if after.ChannelID != target || after.ConnectionType != string(targetConnectionType) ||
		string(after.Params) != string(targetParams) {
		t.Fatalf("runtime GroupView target = %q/%q/%s, want %q/%q/%s",
			after.ChannelID, after.ConnectionType, after.Params, target, targetConnectionType, targetParams)
	}
	expectedTarget, err := fixture.channelRegistry.Resolve(target, json.RawMessage(targetParams))
	if err != nil {
		t.Fatalf("resolve expected runtime target: %v", err)
	}
	if after.ResolvedTarget.ChannelID != expectedTarget.ChannelID ||
		after.ResolvedTarget.ProviderKind != expectedTarget.ProviderKind ||
		after.ResolvedTarget.CatalogProviderID != expectedTarget.CatalogProviderID ||
		!reflect.DeepEqual(after.ResolvedTarget.ResponsesWebsocket, expectedTarget.ResponsesWebsocket) ||
		!reflect.DeepEqual(after.ResolvedTarget.TargetConfig, expectedTarget.TargetConfig) {
		t.Fatalf("runtime ResolvedTarget = %+v, want target channel resolution %+v", after.ResolvedTarget, expectedTarget)
	}
}

// decryptCallTracker makes the reverse credential-backed test observable: a
// connection-type mismatch must return before the credential is decrypted.
type decryptCallTracker struct {
	encryption.Service
	decryptCalls int
}

func (tracker *decryptCallTracker) Decrypt(ciphertext string) (string, error) {
	tracker.decryptCalls++
	return tracker.Service.Decrypt(ciphertext)
}

// ––––– R1: API-key 通道间兼容切换并精确保留 –––––

func TestUpdateGroupSettingsSwitchChannelAPIKeyCompatible(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	const legacyBaseURL = "https://legacy-compatible.example/v1"
	name := "channel-switch-compatible"
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name: stringPointer(name), ChannelID: channel.OpenAI,
		Params:      json.RawMessage(`{"base_url":"` + legacyBaseURL + `"}`),
		Models:      optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gpt-4o", Alias: "legacy-model", AliasEnabled: true}}},
		Credentials: "sk-switch-compatible", ConnectionType: models.ConnectionTypeAPIKey,
	})
	if err != nil {
		t.Fatalf("CreateGroup() error = %v", err)
	}
	groupID := created.GroupID
	// Bind deterministic manual rows to the success path. This promotes any
	// existing OpenAI catalog row in place and avoids claiming automatic rows
	// are immutable when reconciliation may legitimately reprice them.
	ensureManualPrice(t, fixture.db, string(channel.OpenAI), "gpt-4o", 2_000_000_000)
	ensureManualPrice(t, fixture.db, string(channel.Anthropic), "gpt-4o", 3_000_000_000)
	fixture.priceRuntime.Publish(mustLoadPriceTable(t, fixture.db))

	before := captureGroupSwitchState(t, fixture, groupID)
	if len(before.Credentials) != 1 || len(before.CredentialEntries) != 1 {
		t.Fatalf("fixture must create one credential row/entry: rows=%d entries=%d", len(before.Credentials), len(before.CredentialEntries))
	}
	if string(before.Group.Params) != `{"base_url":"`+legacyBaseURL+`"}` {
		t.Fatalf("fixture params = %s, want non-empty legacy base_url", before.Group.Params)
	}
	if len(before.Group.Models) == 0 || string(before.Group.Models) == "[]" {
		t.Fatalf("fixture must create a non-empty models list, got %s", before.Group.Models)
	}
	for _, row := range before.ModelPrices {
		if !row.IsManual {
			t.Fatalf("success fixture unexpectedly contains an automatic price row: %+v", row)
		}
	}
	beforeRevision := before.Snapshot.Revision
	targetParams := models.JSON(`{"base_url":"` + legacyBaseURL + `"}`)

	result, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.Anthropic},
	})
	if err != nil {
		t.Fatalf("UpdateGroupSettings(anthropic) error = %v", err)
	}
	if result.ChannelID != channel.Anthropic || string(result.Params) != string(targetParams) {
		t.Fatalf("result target = %q/%s, want %q/%s", result.ChannelID, result.Params, channel.Anthropic, targetParams)
	}

	after := captureGroupSwitchState(t, fixture, groupID)
	assertSuccessfulGroupSwitchState(t, fixture, groupID, before, after,
		channel.Anthropic, models.ConnectionTypeAPIKey, targetParams)
	if after.Snapshot.Revision <= beforeRevision {
		t.Fatalf("runtime revision did not advance: before=%d after=%d", beforeRevision, after.Snapshot.Revision)
	}
}

// ––––– R1: 模型与价格状态在兼容切换后保持可用 –––––

func TestUpdateGroupSettingsSwitchChannelKeepsPriceState(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-price")

	// Manual rows are the deterministic, non-catalog-owned price state: the
	// existing reconciliation never deletes them. The fresh group already owns an
	// automatic row for openai/gpt-4o, so it is promoted to manual in place.
	ensureManualPrice(t, fixture.db, string(channel.OpenAI), "gpt-4o", 2_000_000_000)
	ensureManualPrice(t, fixture.db, string(channel.Anthropic), "gpt-4o", 3_000_000_000)
	fixture.priceRuntime.Publish(mustLoadPriceTable(t, fixture.db))

	beforeGroup := captureGroupRow(t, fixture, groupID)
	beforePrices := captureModelPriceRows(t, fixture)
	beforeTable := fixture.priceRuntime.Load()
	openaiBefore, ok := beforeTable.Lookup(pricing.Identity{ChannelID: string(channel.OpenAI), ModelID: "gpt-4o"})
	if !ok {
		t.Fatal("manual OpenAI price missing before switch")
	}
	anthropicBefore, ok := beforeTable.Lookup(pricing.Identity{ChannelID: string(channel.Anthropic), ModelID: "gpt-4o"})
	if !ok {
		t.Fatal("manual Anthropic price missing before switch")
	}

	if _, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.Anthropic},
	}); err != nil {
		t.Fatalf("UpdateGroupSettings(anthropic) error = %v", err)
	}

	afterGroup := captureGroupRow(t, fixture, groupID)
	if string(afterGroup.Models) != string(beforeGroup.Models) {
		t.Fatalf("group models changed:\nbefore=%s\nafter= %s", beforeGroup.Models, afterGroup.Models)
	}
	afterPrices := captureModelPriceRows(t, fixture)
	if !reflect.DeepEqual(beforePrices, afterPrices) {
		t.Fatalf("manual price rows changed:\nbefore=%+v\nafter= %+v", beforePrices, afterPrices)
	}
	afterTable := fixture.priceRuntime.Load()
	if afterTable == nil {
		t.Fatal("price table missing after switch")
	}
	openaiAfter, ok := afterTable.Lookup(pricing.Identity{ChannelID: string(channel.OpenAI), ModelID: "gpt-4o"})
	if !ok || !reflect.DeepEqual(openaiBefore, openaiAfter) {
		t.Fatalf("OpenAI manual price changed: before=%+v after=%+v", openaiBefore, openaiAfter)
	}
	anthropicAfter, ok := afterTable.Lookup(pricing.Identity{ChannelID: string(channel.Anthropic), ModelID: "gpt-4o"})
	if !ok || !reflect.DeepEqual(anthropicBefore, anthropicAfter) {
		t.Fatalf("Anthropic manual price changed: before=%+v after=%+v", anthropicBefore, anthropicAfter)
	}
}

// ––––– R1: 同通道 settings 更新无回归 –––––

func TestUpdateGroupSettingsSameChannelNoRegression(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-same")

	beforeGroup := captureGroupRow(t, fixture, groupID)
	beforeCreds := captureGroupCredentialRows(t, fixture, groupID)
	beforeEntries := captureCredentialRegistryEntries(t, fixture, groupID)
	beforeRevision := fixture.manager.Current().Revision

	result, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		Params: optionalField[json.RawMessage]{Set: true, Value: json.RawMessage(`{}`)},
	})
	if err != nil {
		t.Fatalf("UpdateGroupSettings(same channel) error = %v", err)
	}
	if result.ChannelID != channel.OpenAI {
		t.Fatalf("channel_id = %q, want %q", result.ChannelID, channel.OpenAI)
	}

	afterGroup := captureGroupRow(t, fixture, groupID)
	if string(afterGroup.Models) != string(beforeGroup.Models) {
		t.Fatalf("group models changed:\nbefore=%s\nafter= %s", beforeGroup.Models, afterGroup.Models)
	}
	if afterGroup.ChannelID != beforeGroup.ChannelID ||
		afterGroup.ConnectionType != beforeGroup.ConnectionType {
		t.Fatalf("channel unchanged expectation violated: %q/%q -> %q/%q",
			beforeGroup.ChannelID, beforeGroup.ConnectionType,
			afterGroup.ChannelID, afterGroup.ConnectionType)
	}
	assertCredentialRowsIdentical(t, beforeCreds, captureGroupCredentialRows(t, fixture, groupID))
	afterEntries := captureCredentialRegistryEntries(t, fixture, groupID)
	if !reflect.DeepEqual(beforeEntries, afterEntries) {
		t.Fatalf("runtime credential entries changed:\nbefore=%+v\nafter= %+v",
			beforeEntries, afterEntries)
	}
	if fixture.manager.Current().Revision <= beforeRevision {
		t.Fatalf("runtime revision did not advance: before=%d after=%d",
			beforeRevision, fixture.manager.Current().Revision)
	}
}

// ––––– R2: 无效/空白/null channel_id –––––

func TestUpdateGroupSettingsSwitchChannelInvalidID(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-invalid")

	before := captureGroupSwitchState(t, fixture, groupID)

	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.ID("nonexistent")},
	})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want validation", err)
	}
	assertGroupSwitchStateEqual(t, before, captureGroupSwitchState(t, fixture, groupID))
}

func TestUpdateGroupSettingsNullChannelIDRejected(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-null")

	before := captureGroupSwitchState(t, fixture, groupID)

	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Null: true},
	})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want validation", err)
	}
	assertGroupSwitchStateEqual(t, before, captureGroupSwitchState(t, fixture, groupID))
}

func TestUpdateGroupSettingsEmptyChannelIDRejected(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-empty")

	before := captureGroupSwitchState(t, fixture, groupID)

	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.ID("  ")},
	})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want validation", err)
	}
	assertGroupSwitchStateEqual(t, before, captureGroupSwitchState(t, fixture, groupID))
}

// ––––– R2: 目标 params 不兼容回滚 –––––

func TestUpdateGroupSettingsSwitchChannelParamsRejected(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-params-reject")

	before := captureGroupSwitchState(t, fixture, groupID)

	// OpenAI (optional base_url) -> OpenAICompatible (required base_url) with {}
	// params must fail target-schema validation.
	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.OpenAICompatible},
	})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want validation", err)
	}
	after := captureGroupSwitchState(t, fixture, groupID)
	assertGroupSwitchStateEqual(t, before, after)
	if after.Group.ChannelID != string(channel.OpenAI) || string(after.Group.Params) != `{}` {
		t.Fatalf("target leak after params rejection: channel_id=%q params=%s",
			after.Group.ChannelID, after.Group.Params)
	}
}

// ––––– R2: 目标 credential schema 不兼容回滚（进入 decodeCredential） –––––

func TestUpdateGroupSettingsSwitchChannelCredentialSchemaRejected(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	const secret = "sk-switch-cred-reject"
	groupID := createGroupWithCredentials(t, fixture, secret)

	before := captureGroupSwitchState(t, fixture, groupID)
	if len(before.Credentials) == 0 {
		t.Fatal("fixture must create one credential row")
	}

	// OpenAI (api_key, api_key field) -> GoogleVertex (api_key,
	// service_account_json field): same connection type, so the source/target
	// type guard does not reject; the failure is produced by the target
	// API-key credential schema inside decodeCredential.
	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.GoogleVertex},
	})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want ErrValidation", err)
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("error text leaked credential plaintext: %q", err.Error())
	}
	after := captureGroupSwitchState(t, fixture, groupID)
	assertGroupSwitchStateEqual(t, before, after)
	assertCredentialRowsIdentical(t, before.Credentials, after.Credentials)
}

// ––––– R2: candidate/update 形成后由既有 DB 约束失败并整体回滚 –––––

// TestUpdateGroupSettingsSwitchChannelRollsBackOnDuplicateNameConstraint covers
// the rollback boundary the reviewer required: target channel, params and
// credential validation all pass, the candidate update set (channel_id,
// connection_type, params, name) is already formed, and the transaction then
// fails on the pre-existing unique index on groups.name. Nothing may be
// partially written and no runtime object may be republished.
func TestUpdateGroupSettingsSwitchChannelRollsBackOnDuplicateNameConstraint(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-rollback")

	// Pre-existing DB constraint holder: a second group already owns the name.
	createPriceTestGroup(t, fixture.db, models.Group{
		Name: "taken-name", ChannelID: string(channel.Anthropic),
		Params: models.JSON(`{}`), Models: models.JSON(`[{"id":"taken-model"}]`),
		Overrides: models.JSON(`{}`), Enabled: true,
	})
	priceValue := int64(1_000_000_000)
	if err := fixture.db.Create(&models.ModelPrice{
		ChannelID: string(channel.Anthropic), ModelID: "taken-model", IsManual: true,
		InputPriceNanoUSDPerMillionTokens: &priceValue,
	}).Error; err != nil {
		t.Fatal(err)
	}

	before := captureGroupSwitchState(t, fixture, groupID)

	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.Anthropic},
		Name:      optionalField[string]{Set: true, Value: "taken-name"},
	})
	if err == nil {
		t.Fatal("UpdateGroupSettings with duplicate name: expected DB constraint failure, got nil")
	}
	// ErrDuplicateResource can only come from the persisted unique index, which
	// proves the failure happened after candidate formation, not in validation.
	if !errors.Is(err, app_errors.ErrDuplicateResource) {
		t.Fatalf("error type = %T (%v), want ErrDuplicateResource from the DB constraint", err, err)
	}

	after := captureGroupSwitchState(t, fixture, groupID)
	assertGroupSwitchStateEqual(t, before, after)
	if after.Group.ChannelID != string(channel.OpenAI) ||
		after.Group.ConnectionType != models.ConnectionTypeAPIKey {
		t.Fatalf("channel change leaked after rollback: channel_id=%q connection_type=%q",
			after.Group.ChannelID, after.Group.ConnectionType)
	}
	if after.Group.Name != before.Group.Name {
		t.Fatalf("name change leaked after rollback: %q -> %q", before.Group.Name, after.Group.Name)
	}
}

// ––––– R2: API-key ↔ subscription 双向零凭据回滚 –––––

func TestUpdateGroupSettingsSwitchChannelConnectionTypeRejected_ZeroCreds_APIKeyToSubscription(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)

	// API-key group with no credential rows, created directly in the DB so the
	// runtime credential registry is genuinely empty.
	group := createPriceTestGroup(t, fixture.db, models.Group{
		Name:           "zero-creds-apikey-to-sub",
		ChannelID:      string(channel.OpenAI),
		ConnectionType: models.ConnectionTypeAPIKey,
		Params:         models.JSON(`{}`),
		Models:         models.JSON(`[]`),
		Overrides:      models.JSON(`{}`),
		Enabled:        true,
	})
	initialInput, err := stateloader.BuildCompileInput(t.Context(), fixture.db)
	if err != nil {
		t.Fatalf("BuildCompileInput: %v", err)
	}
	if _, err := fixture.manager.Publish(initialInput); err != nil {
		t.Fatalf("initial publish: %v", err)
	}

	before := captureGroupSwitchState(t, fixture, group.ID)
	if len(before.Credentials) != 0 || before.CredentialRegistry != 0 {
		t.Fatalf("fixture must be zero-credential, got rows=%d registry=%d",
			len(before.Credentials), before.CredentialRegistry)
	}

	// OpenAI (api_key) -> Codex (subscription) must fail on the connection type
	// invariant alone, with no credential rows to inspect.
	_, err = fixture.service.UpdateGroupSettings(t.Context(), group.ID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.Codex},
	})
	if err == nil {
		t.Fatal("UpdateGroupSettings(subscription) with zero creds: expected error, got nil")
	}
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want ErrValidation", err)
	}
	after := captureGroupSwitchState(t, fixture, group.ID)
	assertGroupSwitchStateEqual(t, before, after)
	if after.Group.ConnectionType != models.ConnectionTypeAPIKey {
		t.Fatalf("connection_type changed to %q after zero-credential rejection", after.Group.ConnectionType)
	}
}

func TestUpdateGroupSettingsSwitchChannelConnectionTypeRejected_ZeroCreds_SubscriptionToAPIKey(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)

	// Create a subscription group directly in DB (no OAuth flow needed).
	group := createPriceTestGroup(t, fixture.db, models.Group{
		Name:           "zero-creds-sub-to-apikey",
		ChannelID:      string(channel.Codex),
		ConnectionType: models.ConnectionTypeSubscription,
		Params:         models.JSON(`{}`),
		Models:         models.JSON(`[]`),
		Overrides:      models.JSON(`{}`),
		Enabled:        true,
	})

	initialInput, err := stateloader.BuildCompileInput(t.Context(), fixture.db)
	if err != nil {
		t.Fatalf("BuildCompileInput: %v", err)
	}
	if _, err := fixture.manager.Publish(initialInput); err != nil {
		t.Fatalf("initial publish: %v", err)
	}

	before := captureGroupSwitchState(t, fixture, group.ID)
	if len(before.Credentials) != 0 {
		t.Fatalf("fixture must be zero-credential, got %d rows", len(before.Credentials))
	}

	result, err := fixture.service.UpdateGroupSettings(t.Context(), group.ID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.OpenAI},
	})
	if err == nil {
		t.Fatalf("UpdateGroupSettings(api_key) with zero creds: expected error, got nil (result=%+v)", result)
	}
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want ErrValidation", err)
	}
	after := captureGroupSwitchState(t, fixture, group.ID)
	assertGroupSwitchStateEqual(t, before, after)
	if after.Group.ConnectionType != models.ConnectionTypeSubscription {
		t.Fatalf("connection_type changed to %q after zero-credential rejection", after.Group.ConnectionType)
	}
}

// ––––– R2: API-key ↔ subscription 有凭据时双向回滚 –––––

func TestUpdateGroupSettingsSwitchChannelConnectionTypeRejected_APIKeyToSubscription(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-conn-reject")

	before := captureGroupSwitchState(t, fixture, groupID)

	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.Codex},
	})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want validation", err)
	}
	after := captureGroupSwitchState(t, fixture, groupID)
	assertGroupSwitchStateEqual(t, before, after)
	assertCredentialRowsIdentical(t, before.Credentials, after.Credentials)
}

func TestUpdateGroupSettingsSwitchChannelConnectionTypeRejected_SubscriptionToAPIKey(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	stage := mustImportSubscriptionStage(t, fixture, "switch-reverse", "switch-reverse@example.com")
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name:                stringPointer("sub-to-apikey-reject"),
		ChannelID:           channel.Codex,
		ConnectionType:      models.ConnectionTypeSubscription,
		Models:              optionalGroupModels{Set: true, Values: []GroupModel{{ID: "gpt-5.2"}}},
		StagedCredentialIDs: []string{stage.StageID},
	})
	if err != nil {
		t.Fatalf("CreateGroup(subscription) error = %v", err)
	}
	groupID := created.GroupID
	before := captureGroupSwitchState(t, fixture, groupID)
	if len(before.Credentials) != 1 || before.CredentialRegistry != 1 {
		t.Fatalf("fixture must contain one real subscription credential: rows=%d registry=%d",
			len(before.Credentials), before.CredentialRegistry)
	}

	// Use a real stage-backed row, then make decrypt and SQL access observable.
	// The registry connection-type guard must reject before it reads this row.
	credentialQueries := 0
	credentialIDPlucks := 0
	credentialRowReads := 0
	const callbackName = "test:channel_switch_reverse_no_credential_read"
	if err := fixture.db.Callback().Query().Before("gorm:query").Register(callbackName, func(tx *gorm.DB) {
		if tx.Statement.Table != "credentials" {
			return
		}
		credentialQueries++
		if _, ok := tx.Statement.Dest.(*[]uint); ok {
			credentialIDPlucks++
			return
		}
		credentialRowReads++
	}); err != nil {
		t.Fatalf("register credential query observer: %v", err)
	}
	t.Cleanup(func() { _ = fixture.db.Callback().Query().Remove(callbackName) })
	tracker := &decryptCallTracker{Service: fixture.encryption}
	fixture.service.encryption = tracker
	result, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.OpenAI},
	})
	if err == nil {
		t.Fatalf("UpdateGroupSettings(api_key) with subscription credential: expected error, got nil (result=%+v)", result)
	}
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want validation", err)
	}
	if credentialRowReads != 0 {
		t.Fatalf("connection-type rejection read full credentials %d time(s)", credentialRowReads)
	}
	if credentialQueries != credentialIDPlucks {
		t.Fatalf("credential query classification = %d total/%d id-only", credentialQueries, credentialIDPlucks)
	}
	if tracker.decryptCalls != 0 {
		t.Fatalf("connection-type rejection decrypted %d credential rows", tracker.decryptCalls)
	}
	after := captureGroupSwitchState(t, fixture, groupID)
	assertGroupSwitchStateEqual(t, before, after)
	if after.Group.ChannelID != string(channel.Codex) ||
		after.Group.ConnectionType != models.ConnectionTypeSubscription {
		t.Fatalf("target leaked after rejection: %q/%q", after.Group.ChannelID, after.Group.ConnectionType)
	}
}

// ––––– R2: 按目标 schema 校验每条凭据（fingerprint 不匹配） –––––

func TestUpdateGroupSettingsSwitchChannelFingerprintMismatchRejected(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-switch-fp")

	// Corrupt the stored fingerprint so canonical validation fails. Capture the
	// full state AFTER corruption: a failed switch must not mutate anything.
	corruptedFingerprint := "deadbeef"
	if err := fixture.db.Model(&models.Credential{}).
		Where("group_id = ?", groupID).
		Update("fingerprint", corruptedFingerprint).Error; err != nil {
		t.Fatal(err)
	}
	before := captureGroupSwitchState(t, fixture, groupID)
	if len(before.Credentials) == 0 || before.Credentials[0].Fingerprint != corruptedFingerprint {
		t.Fatalf("fingerprint corruption failed: %+v", before.Credentials)
	}

	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		ChannelID: optionalField[channel.ID]{Set: true, Value: channel.Anthropic},
	})
	if err == nil {
		t.Fatal("UpdateGroupSettings with corrupted fingerprint: expected error, got nil")
	}
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("error type = %T, want ErrValidation", err)
	}
	after := captureGroupSwitchState(t, fixture, groupID)
	assertGroupSwitchStateEqual(t, before, after)
	assertCredentialRowsIdentical(t, before.Credentials, after.Credentials)
}

// ––––– 安全：目标凭据解析失败时不泄漏 secret –––––

// TestUpdateGroupSettingsSwitchChannelCredentialErrorDoesNotLeakSecret covers
// the two real decodeCredential failure paths while switching between two
// API-key channels (so the connection-type guard cannot short-circuit first):
// undecryptable stored ciphertext, and a stored API-key payload that the target
// API-key credential schema cannot parse. In both cases the caller must receive
// the generic validation error and never the plaintext secret or ciphertext.
//
// Each subtest runs on its own migrated in-memory fixture (newServiceFixture),
// so deliberately corrupting one credential row cannot leak into other tests.
func TestUpdateGroupSettingsSwitchChannelCredentialErrorDoesNotLeakSecret(t *testing.T) {
	t.Parallel()

	subtests := []struct {
		name    string
		corrupt func(t *testing.T, fixture serviceFixture, groupID uint)
		target  channel.ID
	}{
		{
			name: "undecryptable ciphertext",
			corrupt: func(t *testing.T, fixture serviceFixture, groupID uint) {
				t.Helper()
				if err := fixture.db.Model(&models.Credential{}).
					Where("group_id = ?", groupID).
					Update("data", "not-a-valid-ciphertext").Error; err != nil {
					t.Fatal(err)
				}
			},
			target: channel.Anthropic,
		},
		{
			name: "target api_key schema cannot parse stored payload",
			corrupt: func(t *testing.T, fixture serviceFixture, groupID uint) {
				t.Helper()
				// No corruption needed: GoogleVertex is an API-key channel whose
				// required credential field is service_account_json, so the stored
				// api_key payload reaches decodeCredential and fails the target schema.
			},
			target: channel.GoogleVertex,
		},
	}

	for _, subtest := range subtests {
		t.Run(subtest.name, func(t *testing.T) {
			t.Parallel()
			fixture := newServiceFixture(t)
			const secret = "sk-switch-leak-plaintext"
			groupID := createGroupWithCredentials(t, fixture, secret)
			subtest.corrupt(t, fixture, groupID)
			before := captureGroupSwitchState(t, fixture, groupID)
			if len(before.Credentials) == 0 {
				t.Fatal("fixture must create one credential row")
			}
			storedCiphertext := before.Credentials[0].Data

			_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
				ChannelID: optionalField[channel.ID]{Set: true, Value: subtest.target},
			})
			if err == nil {
				t.Fatal("UpdateGroupSettings: expected credential validation error, got nil")
			}
			if !errors.Is(err, app_errors.ErrValidation) {
				t.Fatalf("error type = %T, want ErrValidation", err)
			}
			errText := err.Error()
			for label, fragment := range map[string]string{
				"plaintext secret":  secret,
				"secret prefix":     "sk-switch",
				"stored ciphertext": storedCiphertext,
				"schema field name": "service_account_json",
			} {
				if strings.Contains(errText, fragment) {
					t.Fatalf("error text leaked %s: %q", label, errText)
				}
			}
			// The API surface is the generic validation message only; the
			// underlying decrypt/schema error is folded and not returned.
			if errText != app_errors.ErrValidation.Error() {
				t.Fatalf("error text = %q, want the generic validation message %q",
					errText, app_errors.ErrValidation.Error())
			}

			after := captureGroupSwitchState(t, fixture, groupID)
			assertGroupSwitchStateEqual(t, before, after)
			assertCredentialRowsIdentical(t, before.Credentials, after.Credentials)
		})
	}
}
