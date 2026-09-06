package control

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/catalog"
	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

func TestGetGroupModelsReturnsClientNamesAndPricingStatus(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	fixture.catalogRuntime.Publish(&catalog.Snapshot{Providers: map[string]catalog.Provider{
		"openai": {
			ID: "openai",
			Models: map[string]catalog.Model{
				"gpt-4o": {
					ID: "gpt-4o",
					Cost: &catalog.ModelCost{Prices: pricing.Prices{
						Input: pricing.Price{Set: true, NanoUSDPerMillion: 1},
					}},
				},
			},
		},
	}})
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAI,
		Params:    json.RawMessage(`{}`),
		Models: optionalGroupModels{Set: true, Values: []GroupModel{
			{ID: "gpt-4o", Alias: "default", AliasEnabled: true},
			{ID: "missing-price", Alias: ""},
		}},
		Credentials: "sk-model-read", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatal(err)
	}

	got, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
	if err != nil {
		t.Fatalf("GetGroupModels() error = %v", err)
	}
	want := GroupModelsResponse{
		Items: []GroupModelResponse{
			{ID: "gpt-4o", Alias: "default", AliasEnabled: true, ClientModel: "default", PricingStatus: PricingStatusConfigured},
			{ID: "missing-price", Alias: "", AliasEnabled: false, ClientModel: "missing-price", PricingStatus: PricingStatusPending},
		},
		Total:   2,
		Pending: 1,
	}
	// entry_id 由 GET 懒回填（设计 §2.2），其余字段与配置一一对应。
	entryIDPattern := regexp.MustCompile(`^e[0-9a-f]{12}$`)
	if len(got.Items) != len(want.Items) {
		t.Fatalf("GetGroupModels() items = %d, want %d", len(got.Items), len(want.Items))
	}
	for index, item := range got.Items {
		if !entryIDPattern.MatchString(item.EntryID) {
			t.Fatalf("item %d entry_id = %q, want lazy-backfilled e+12hex", index, item.EntryID)
		}
		want.Items[index].EntryID = item.EntryID
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("GetGroupModels() = %#v, want %#v", got, want)
	}
}

func TestMapGroupModelsResponseTreatsContextTierOnlyPriceAsConfigured(t *testing.T) {
	t.Parallel()
	result, err := mapGroupModelsResponse(
		string(channel.OpenAI),
		[]groupModelEntry{{ID: "tiered-model"}},
		modelPriceRows{
			{ChannelID: string(channel.OpenAI), ModelID: "tiered-model"}: {
				ChannelID:         string(channel.OpenAI),
				ModelID:           "tiered-model",
				ContextPriceTiers: models.JSON(`[{"threshold_tokens":1000,"input_price_nano_usd_per_million_tokens":1,"output_price_nano_usd_per_million_tokens":null,"cache_read_price_nano_usd_per_million_tokens":null,"cache_write_price_nano_usd_per_million_tokens":null}]`),
			},
		},
	)
	if err != nil {
		t.Fatalf("mapGroupModelsResponse() error = %v", err)
	}
	want := GroupModelsResponse{
		Items: []GroupModelResponse{{
			ID: "tiered-model", ClientModel: "tiered-model", PricingStatus: PricingStatusConfigured,
		}},
		Total: 1,
	}
	if !reflect.DeepEqual(result, want) {
		t.Fatalf("mapGroupModelsResponse() = %#v, want %#v", result, want)
	}
}

func TestNormalizeGroupModelsAppliesAliasSwitchAndReportsStableConflicts(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name          string
		values        []GroupModel
		want          []GroupModel
		wantConflicts []ModelNameConflict
		wantError     error
	}{
		{
			// 同一对外名映射到不同上游模型是合法的多条目路由（设计 §3）。
			name: "multi-mapping under one external name no longer conflicts",
			values: []GroupModel{
				{ID: "a", Alias: ""},
				{ID: "b", Alias: "a", AliasEnabled: true},
			},
			want: []GroupModel{
				{ID: "a", Alias: ""},
				{ID: "b", Alias: "a"},
			},
		},
		{
			name: "duplicate upstream pair under one external name conflicts",
			values: []GroupModel{
				{ID: "a", Alias: "x", AliasEnabled: true},
				{ID: "a", Alias: "x", AliasEnabled: true},
			},
			wantConflicts: []ModelNameConflict{{ClientModel: "x", Indexes: []int{0, 1}}},
			wantError:     app_errors.ErrModelNameConflict,
		},
		{
			name: "model names remain case sensitive",
			values: []GroupModel{
				{ID: "a", Alias: "X", AliasEnabled: true},
				{ID: "b", Alias: "x", AliasEnabled: true},
			},
			want: []GroupModel{
				{ID: "a", Alias: "X"},
				{ID: "b", Alias: "x"},
			},
		},
		{
			// 旧 1:1 冲突在新语义下是合法多条目：不同上游共用对外名。
			name: "trimmed aliases multi-mapping does not conflict",
			values: []GroupModel{
				{ID: " a ", Alias: ""},
				{ID: "b", Alias: " a ", AliasEnabled: true},
			},
			want: []GroupModel{
				{ID: "a", Alias: ""},
				{ID: "b", Alias: "a"},
			},
		},
		{
			name:      "enabled alias cannot be blank after trimming",
			values:    []GroupModel{{ID: "a", Alias: " ", AliasEnabled: true}},
			wantError: app_errors.ErrValidation,
		},
		{
			name: "multiple conflicts use first occurrence order",
			values: []GroupModel{
				{ID: "a"},
				{ID: "a", Alias: "a", AliasEnabled: true},
				{ID: "c"},
				{ID: "c", Alias: "c", AliasEnabled: true},
			},
			wantConflicts: []ModelNameConflict{
				{ClientModel: "a", Indexes: []int{0, 1}},
				{ClientModel: "c", Indexes: []int{2, 3}},
			},
			wantError: app_errors.ErrModelNameConflict,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := normalizeGroupModels(test.values)
			if test.wantError == nil {
				if err != nil {
					t.Fatalf("normalizeGroupModels() error = %v", err)
				}
				if !reflect.DeepEqual(got, test.want) {
					t.Fatalf("normalizeGroupModels() = %#v, want %#v", got, test.want)
				}
				return
			}

			var apiErr *app_errors.APIError
			if !errors.As(err, &apiErr) || apiErr.Code != test.wantError.(*app_errors.APIError).Code {
				t.Fatalf("normalizeGroupModels() error = %#v, want %q", err, test.wantError)
			}
			if test.wantConflicts == nil {
				return
			}
			data, ok := apiErr.Data.(ModelNameConflictData)
			if !ok || !reflect.DeepEqual(data.Conflicts, test.wantConflicts) {
				t.Fatalf("conflict data = %#v, want %#v", apiErr.Data, test.wantConflicts)
			}
		})
	}
}

func TestNormalizeGroupModelsAllowsMultiMappingAndRejectsDuplicatePairs(t *testing.T) {
	t.Parallel()
	// 同一对外名映射到不同上游模型是合法的多条目路由（设计 §3）。
	got, err := normalizeGroupModels([]GroupModel{
		{ID: "provider-a", Alias: "public", AliasEnabled: true},
		{ID: "provider-b", Alias: "public", AliasEnabled: true},
		{ID: "public"},
	})
	if err != nil {
		t.Fatalf("normalizeGroupModels() error = %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("normalizeGroupModels() = %#v, want 3 entries", got)
	}
	// 同一对外名下重复同一上游模型才冲突。
	for _, values := range [][]GroupModel{
		{
			{ID: "provider-a", Alias: "public", AliasEnabled: true},
			{ID: "provider-a", Alias: "public", AliasEnabled: true},
		},
		{{ID: "public"}, {ID: "public"}},
	} {
		var apiErr *app_errors.APIError
		if _, err := normalizeGroupModels(values); !errors.As(err, &apiErr) ||
			apiErr.Code != app_errors.ErrModelNameConflict.Code {
			t.Fatalf("normalizeGroupModels(%#v) error = %#v", values, err)
		}
	}
}

func TestUpdateGroupModelsRequiresNonNullModelsField(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupForCredentialImport(t, fixture, "sk-required-models")
	for _, request := range []GroupModelsUpdateRequest{
		{},
		{Models: optionalGroupModels{Set: false}},
	} {
		if _, err := fixture.service.UpdateGroupModels(t.Context(), groupID, request); !errors.Is(err, app_errors.ErrValidation) {
			t.Fatalf("request %#v error = %v", request, err)
		}
	}

	var request GroupModelsUpdateRequest
	err := json.Unmarshal([]byte(`{"models":null}`), &request)
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("null models error = %v, want ErrValidation", err)
	}
}

func TestUpdateGroupModelsReplacesAuthoritativeListAndPublishesOnce(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAICompatible,
		Params:    json.RawMessage(`{"base_url":"https://model-save.example.com/v1"}`),
		Models: optionalGroupModels{
			Set:    true,
			Values: []GroupModel{{ID: "provider-old", Alias: "old-public", AliasEnabled: true}},
		},
		Credentials: "sk-model-save-a\nsk-model-save-b", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatal(err)
	}
	validation := "validation-model-must-stay"
	if _, err := fixture.service.UpdateGroupSettings(t.Context(), created.GroupID, GroupSettingsUpdateRequest{
		ValidationModel: optionalField[string]{Set: true, Value: validation},
	}); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.Model(&models.Group{}).
		Where("id = ?", created.GroupID).
		Update("overrides", models.JSON(`{
			"stream_idle_timeout":45,
			"inject_usage_options":false,
			"header_rules":{"remove":["X-Trace"]}
		}`)).Error; err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.Create(&models.SystemSetting{
		Key: state.SettingRequestTimeout, Value: "701",
	}).Error; err != nil {
		t.Fatal(err)
	}
	beforeRevision := fixture.manager.Current().Revision
	beforeRegistry := fixture.registry.Snapshot()
	var beforeCredentials []models.Credential
	if err := fixture.db.Where("group_id = ?", created.GroupID).Order("id ASC").Find(&beforeCredentials).Error; err != nil {
		t.Fatal(err)
	}

	got, err := fixture.service.UpdateGroupModels(t.Context(), created.GroupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{
			Set: true,
			Values: []GroupModel{
				{ID: "provider-b", Alias: "public-b", AliasEnabled: true},
				{ID: "provider-a", Alias: "public-a", AliasEnabled: true},
			},
		},
	})
	if err != nil {
		t.Fatalf("UpdateGroupModels() error = %v", err)
	}
	wantModels := []GroupModel{
		{ID: "provider-b", Alias: "public-b"},
		{ID: "provider-a", Alias: "public-a"},
	}
	want := GroupModelsResponse{
		Items: []GroupModelResponse{
			{ID: "provider-b", Alias: "public-b", AliasEnabled: true, ClientModel: "public-b", PricingStatus: PricingStatusPending},
			{ID: "provider-a", Alias: "public-a", AliasEnabled: true, ClientModel: "public-a", PricingStatus: PricingStatusPending},
		},
		Total:   2,
		Pending: 2,
	}
	// entry_id 懒回填后响应携带服务端生成的标识（设计 §2.2）。
	entryIDPattern := regexp.MustCompile(`^e[0-9a-f]{12}$`)
	if len(got.Items) != len(want.Items) {
		t.Fatalf("models response items = %d, want %d", len(got.Items), len(want.Items))
	}
	for index, item := range got.Items {
		if !entryIDPattern.MatchString(item.EntryID) {
			t.Fatalf("item %d entry_id = %q, want lazy-backfilled e+12hex", index, item.EntryID)
		}
		want.Items[index].EntryID = item.EntryID
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("models response = %#v, want %#v", got, want)
	}
	settings, err := fixture.service.GetGroupSettings(t.Context(), created.GroupID)
	if err != nil {
		t.Fatalf("GetGroupSettings() error = %v", err)
	}
	summary, err := fixture.service.GetGroupSummary(t.Context(), created.GroupID)
	if err != nil {
		t.Fatalf("GetGroupSummary() error = %v", err)
	}
	if settings.ValidationModel == nil || *settings.ValidationModel != validation || summary.CredentialCount != 2 {
		t.Fatalf("settings/summary = %#v/%#v", settings, summary)
	}
	streamIdle, ok := settings.Overrides[state.SettingStreamIdleTimeout].(json.Number)
	if len(settings.Overrides) != 3 || !ok || streamIdle.String() != "45" ||
		settings.Overrides[state.SettingHeaderRules] == nil ||
		settings.Overrides[state.SettingInjectUsageOptions] != false {
		t.Fatalf("preserved sparse config = %#v", settings.Overrides)
	}
	if settings.Effective.FirstByteTimeout != 120 ||
		settings.Effective.RequestTimeout != 701 ||
		settings.Effective.StreamIdleTimeout != 45 ||
		settings.Effective.InjectUsageOptions ||
		len(settings.Effective.HeaderRules.Set) != 0 ||
		!reflect.DeepEqual(settings.Effective.HeaderRules.Remove, []string{"X-Trace"}) {
		t.Fatalf("post-write effective config = %#v", settings.Effective)
	}
	if settings.Effective.HeaderRules.Set == nil || settings.Effective.HeaderRules.Remove == nil {
		t.Fatalf("effective header collections = %#v", settings.Effective.HeaderRules)
	}
	// entry_id 懒回填后存储行携带服务端生成的标识（设计 §2.2）。
	stored := loadCreatedGroupModels(t, fixture, created.GroupID)
	if len(stored) != len(wantModels) {
		t.Fatalf("stored models = %d entries, want %d", len(stored), len(wantModels))
	}
	for index, model := range stored {
		if !entryIDPattern.MatchString(model.EntryID) {
			t.Fatalf("stored model %d entry_id = %q, want lazy-backfilled e+12hex", index, model.EntryID)
		}
		wantModels[index].EntryID = model.EntryID
	}
	if !reflect.DeepEqual(stored, wantModels) {
		t.Fatalf("stored models = %#v, want %#v", stored, wantModels)
	}
	if fixture.manager.Current().Revision != beforeRevision+1 {
		t.Fatalf("revision = %d, want %d", fixture.manager.Current().Revision, beforeRevision+1)
	}
	if !reflect.DeepEqual(fixture.registry.Snapshot(), beforeRegistry) {
		t.Fatal("models save changed Registry")
	}
	var afterCredentials []models.Credential
	if err := fixture.db.Where("group_id = ?", created.GroupID).Order("id ASC").Find(&afterCredentials).Error; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(afterCredentials, beforeCredentials) {
		t.Fatalf("credentials changed: got=%#v want=%#v", afterCredentials, beforeCredentials)
	}
	snapshot := fixture.manager.Current()
	view := snapshot.Groups[created.GroupID]
	if settings.Effective.RequestTimeout != int64(view.Timeouts.Request/time.Second) ||
		settings.Effective.StreamIdleTimeout != int64(view.Timeouts.StreamIdle/time.Second) ||
		settings.Effective.InjectUsageOptions != view.InjectUsageOptions ||
		settings.Effective.AffinityEnabled != view.AffinityEnabled ||
		!reflect.DeepEqual(settings.Effective.HeaderRules.Remove, view.HeaderRules.Remove) {
		t.Fatalf("effective/snapshot = %#v/%#v", settings.Effective, view)
	}
	targets := snapshot.ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]
	if len(targets) != 2 ||
		targets["public-a"][0].UpstreamModelID != "provider-a" ||
		targets["public-b"][0].UpstreamModelID != "provider-b" {
		t.Fatalf("candidate mapping = %#v", targets)
	}
	if _, exists := targets["old-public"]; exists {
		t.Fatalf("authoritative replacement retained old model: %#v", targets)
	}
	routes := snapshot.ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion]
	if len(routes) != 2 ||
		routes["public-a"][0].UpstreamModelID != "provider-a" ||
		routes["public-b"][0].UpstreamModelID != "provider-b" {
		t.Fatalf("route catalog = %#v", routes)
	}
}

func TestUpdateGroupModelsAllowsEmptyList(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAICompatible,
		Params:    json.RawMessage(`{"base_url":"https://empty-models.example.com/v1"}`),
		Models: optionalGroupModels{
			Set:    true,
			Values: []GroupModel{{ID: "provider-old", Alias: "old-public", AliasEnabled: true}},
		},
		Credentials: "sk-empty-models", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatal(err)
	}
	before := fixture.manager.Current().Revision
	got, err := fixture.service.UpdateGroupModels(t.Context(), created.GroupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{Set: true, Values: []GroupModel{}},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := GroupModelsResponse{Items: []GroupModelResponse{}, Total: 0, Pending: 0}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("models response = %#v, want %#v", got, want)
	}
	if fixture.manager.Current().Revision != before+1 {
		t.Fatalf("revision = %d, want %d", fixture.manager.Current().Revision, before+1)
	}
	if len(fixture.manager.Current().ExecutionCandidates[protocol.OpenAICompletions][execution.OperationChatCompletion]) != 0 ||
		len(fixture.manager.Current().ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion]) != 0 {
		t.Fatalf("model indexes = candidates:%#v routes:%#v",
			fixture.manager.Current().ExecutionCandidates, fixture.manager.Current().ExecutionRouteCatalog)
	}
}

func TestUpdateGroupModelsNeverCallsDiscoveryOrChangesAccessKeyFilters(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	groupID := createGroupForCredentialImport(t, fixture, "sk-no-discovery")
	access, err := fixture.service.CreateAccessKey(t.Context(), AccessKeyCreateRequest{
		Name: "filtered",
		Filters: &AccessKeyFilters{
			Groups: []uint{groupID},
			Models: []string{"old-public"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var beforeAccess models.AccessKey
	if err := fixture.db.First(&beforeAccess, access.ID).Error; err != nil {
		t.Fatal(err)
	}
	fixture.service.executor = newRecordingDiscoveryExecutor(&recordingDiscoveryExecutorTarget{
		value: protocol.OpenAICompletions,
		listFn: func(context.Context, string, string, state.HeaderRules) ([]string, error) {
			t.Fatal("UpdateGroupModels must not call model discovery")
			return nil, nil
		},
	})

	_, err = fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{
			Set:    true,
			Values: []GroupModel{{ID: "provider-new", Alias: "new-public", AliasEnabled: true}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var afterAccess models.AccessKey
	if err := fixture.db.First(&afterAccess, access.ID).Error; err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(afterAccess, beforeAccess) {
		t.Fatalf("persisted AccessKey changed: got=%#v want=%#v", afterAccess, beforeAccess)
	}
	filters, err := decodeStoredAccessKeyFilters(afterAccess.Filters)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(filters.Models, []string{"old-public"}) {
		t.Fatalf("filters = %#v", filters)
	}
}

func TestUpdateGroupModelsFailuresDoNotPublish(t *testing.T) {
	t.Parallel()
	t.Run("external collision", func(t *testing.T) {
		fixture := newServiceFixture(t)
		groupID := createGroupForCredentialImport(t, fixture, "sk-invalid-models")
		beforeRevision := fixture.manager.Current().Revision
		beforeRegistry := fixture.registry.Snapshot()
		beforeModels := loadCreatedGroupModels(t, fixture, groupID)

		_, err := fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
			Models: optionalGroupModels{
				Set: true,
				Values: []GroupModel{
					{ID: "provider-a", Alias: "public", AliasEnabled: true},
					{ID: "provider-a", Alias: "public", AliasEnabled: true},
				},
			},
		})
		var apiErr *app_errors.APIError
		if !errors.As(err, &apiErr) || apiErr.Code != app_errors.ErrModelNameConflict.Code {
			t.Fatalf("UpdateGroupModels() error = %#v, want MODEL_NAME_CONFLICT", err)
		}
		assertModelsUpdateStateUnchanged(t, fixture, groupID, beforeRevision, beforeRegistry, beforeModels)
	})

	t.Run("full compile failure", func(t *testing.T) {
		fixture := newServiceFixture(t)
		groupID := createGroupForCredentialImport(t, fixture, "sk-compile-models")
		corrupt := validControlGroup("model-save-corrupt-other")
		if err := fixture.db.Create(corrupt).Error; err != nil {
			t.Fatal(err)
		}
		if err := fixture.db.Exec("UPDATE groups SET channel_id = ? WHERE id = ?", "unknown", corrupt.ID).Error; err != nil {
			t.Fatal(err)
		}
		beforeRevision := fixture.manager.Current().Revision
		beforeRegistry := fixture.registry.Snapshot()
		beforeModels := loadCreatedGroupModels(t, fixture, groupID)

		_, err := fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
			Models: optionalGroupModels{
				Set:    true,
				Values: []GroupModel{{ID: "provider-new", Alias: "new-public", AliasEnabled: true}},
			},
		})
		if err == nil {
			t.Fatal("UpdateGroupModels() error = nil, want full Compile failure")
		}
		assertModelsUpdateStateUnchanged(t, fixture, groupID, beforeRevision, beforeRegistry, beforeModels)
	})

	t.Run("commit failure", func(t *testing.T) {
		fixture, dsn := newFileServiceFixture(t)
		created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
			ChannelID: channel.OpenAICompatible,
			Params:    json.RawMessage(`{"base_url":"https://commit-failure-models.example.com/v1"}`),
			Models: optionalGroupModels{
				Set:    true,
				Values: []GroupModel{{ID: "provider-old", Alias: "old-public", AliasEnabled: true}},
			},
			Credentials: "sk-commit-models", ConnectionType: "api_key",
		})
		if err != nil {
			t.Fatal(err)
		}
		beforeRevision := fixture.manager.Current().Revision
		beforeRegistry := fixture.registry.Snapshot()
		beforeModels := loadCreatedGroupModels(t, fixture, created.GroupID)
		releaseReader := holdRollbackJournalReadLock(t, fixture.db, dsn)

		_, err = fixture.service.UpdateGroupModels(t.Context(), created.GroupID, GroupModelsUpdateRequest{
			Models: optionalGroupModels{
				Set:    true,
				Values: []GroupModel{{ID: "provider-new", Alias: "new-public", AliasEnabled: true}},
			},
		})
		var apiErr *app_errors.APIError
		if !errors.As(err, &apiErr) || apiErr.Code != app_errors.ErrDatabase.Code {
			t.Fatalf("UpdateGroupModels() error = %#v, want DATABASE_ERROR", err)
		}
		releaseReader()
		assertModelsUpdateStateUnchanged(
			t, fixture, created.GroupID, beforeRevision, beforeRegistry, beforeModels,
		)
	})
}

func assertModelsUpdateStateUnchanged(
	t *testing.T,
	fixture serviceFixture,
	groupID uint,
	wantRevision uint64,
	wantRegistry []state.CredentialRuntimeView,
	wantModels []GroupModel,
) {
	t.Helper()
	if fixture.manager.Current().Revision != wantRevision {
		t.Fatalf("revision = %d, want unchanged %d", fixture.manager.Current().Revision, wantRevision)
	}
	if !reflect.DeepEqual(fixture.registry.Snapshot(), wantRegistry) {
		t.Fatal("Registry changed")
	}
	if got := loadCreatedGroupModels(t, fixture, groupID); !reflect.DeepEqual(got, wantModels) {
		t.Fatalf("persisted models changed: got=%#v want=%#v", got, wantModels)
	}
}

func TestMapGroupModelsResponseCarriesRouteEntryWeightAndPriority(t *testing.T) {
	t.Parallel()
	result, err := mapGroupModelsResponse(
		string(channel.OpenAI),
		[]groupModelEntry{
			{ID: "entry-a", Alias: "public", Weight: intPointer(30), Priority: intPointer(2)},
			{ID: "entry-b"},
		},
		modelPriceRows{},
	)
	if err != nil {
		t.Fatalf("mapGroupModelsResponse() error = %v", err)
	}
	want := GroupModelsResponse{
		Items: []GroupModelResponse{
			{
				ID: "entry-a", Alias: "public", AliasEnabled: true, ClientModel: "public",
				Weight: intPointer(30), Priority: intPointer(2), PricingStatus: PricingStatusPending,
			},
			{
				ID: "entry-b", AliasEnabled: false, ClientModel: "entry-b", PricingStatus: PricingStatusPending,
			},
		},
		Total:   2,
		Pending: 2,
	}
	if !reflect.DeepEqual(result, want) {
		t.Fatalf("mapGroupModelsResponse() = %#v, want %#v", result, want)
	}
}

// loadStoredGroupModelsJSON reads the persisted models column verbatim. Unlike
// loadCreatedGroupModels it also works for rows carrying route entry fields.
func loadStoredGroupModelsJSON(t *testing.T, fixture serviceFixture, groupID uint) string {
	t.Helper()
	var group models.Group
	if err := fixture.db.First(&group, groupID).Error; err != nil {
		t.Fatalf("query group %d: %v", groupID, err)
	}
	return string(group.Models)
}

func loadLoaderGroupModels(
	t *testing.T,
	fixture serviceFixture,
	groupID uint,
) []state.ModelConfig {
	t.Helper()
	input, err := stateloader.BuildCompileInput(t.Context(), fixture.db)
	if err != nil {
		t.Fatalf("BuildCompileInput() error = %v", err)
	}
	for _, group := range input.Groups {
		if group.ID == groupID {
			return group.Models
		}
	}
	t.Fatalf("BuildCompileInput() missing group %d", groupID)
	return nil
}

func TestUpdateGroupModelsRoundTripsLegacyPayloadWithoutRouteFields(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAICompatible,
		Params:    json.RawMessage(`{"base_url":"https://legacy-models.example.com/v1"}`),
		Models: optionalGroupModels{
			Set:    true,
			Values: []GroupModel{{ID: "provider-old", Alias: "old-public", AliasEnabled: true}},
		},
		Credentials: "sk-legacy-models", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatal(err)
	}

	assertStoredGroupModelsLegacyShape := func(stage string) {
		t.Helper()
		if stored := loadStoredGroupModelsJSON(t, fixture, created.GroupID); strings.Contains(stored, "weight") || strings.Contains(stored, "priority") {
			t.Fatalf("stored models after %s = %s, want legacy shape without route fields", stage, stored)
		}
	}
	assertStoredGroupModelsLegacyShape("create")
	got, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
	if err != nil {
		t.Fatalf("GetGroupModels() error = %v", err)
	}
	if len(got.Items) != 1 || got.Items[0].Weight != nil || got.Items[0].Priority != nil {
		t.Fatalf("models response = %#v, want nil route fields", got.Items)
	}
	runtimeModels := loadLoaderGroupModels(t, fixture, created.GroupID)
	if len(runtimeModels) != 1 || runtimeModels[0].Weight != nil || runtimeModels[0].Priority != nil {
		t.Fatalf("loader models = %#v, want nil route fields", runtimeModels)
	}

	if _, err := fixture.service.UpdateGroupModels(t.Context(), created.GroupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{Set: true, Values: []GroupModel{{ID: "provider-new"}}},
	}); err != nil {
		t.Fatalf("UpdateGroupModels() error = %v", err)
	}
	assertStoredGroupModelsLegacyShape("update")
}

func TestGroupModelRouteFieldsRoundTripThroughStorageAndRuntime(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAICompatible,
		Params:    json.RawMessage(`{"base_url":"https://route-fields.example.com/v1"}`),
		Models: optionalGroupModels{
			Set:    true,
			Values: []GroupModel{{ID: "provider-old", Alias: "old-public", AliasEnabled: true}},
		},
		Credentials: "sk-route-fields", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.Model(&models.Group{}).
		Where("id = ?", created.GroupID).
		Update("models", models.JSON(`[{"id":"entry-a","alias":"public","weight":30,"priority":2},{"id":"entry-b","alias":"public","weight":0}]`)).Error; err != nil {
		t.Fatal(err)
	}

	got, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
	if err != nil {
		t.Fatalf("GetGroupModels() error = %v", err)
	}
	// 懒回填：直接改库写入的存量条目在首次 GET 时获得服务端生成的 entry_id。
	entryIDPattern := regexp.MustCompile(`^e[0-9a-f]{12}$`)
	wantItems := []GroupModelResponse{
		{ID: "entry-a", Alias: "public", AliasEnabled: true, ClientModel: "public", Weight: intPointer(30), Priority: intPointer(2), PricingStatus: PricingStatusPending},
		{ID: "entry-b", Alias: "public", AliasEnabled: true, ClientModel: "public", Weight: intPointer(0), PricingStatus: PricingStatusPending},
	}
	if len(got.Items) != len(wantItems) {
		t.Fatalf("models response items = %#v, want %d items", got.Items, len(wantItems))
	}
	for index, item := range got.Items {
		want := wantItems[index]
		if item.EntryID == "" || !entryIDPattern.MatchString(item.EntryID) {
			t.Fatalf("item %d entry_id = %q, want lazy-backfilled e+12hex", index, item.EntryID)
		}
		if item.ID != want.ID || item.Alias != want.Alias || item.AliasEnabled != want.AliasEnabled ||
			item.ClientModel != want.ClientModel || item.PricingStatus != want.PricingStatus {
			t.Fatalf("item %d = %#v, want core fields %#v", index, item, want)
		}
		if !reflect.DeepEqual(item.Weight, want.Weight) || !reflect.DeepEqual(item.Priority, want.Priority) {
			t.Fatalf("item %d route fields = %v/%v, want %v/%v", index, item.Weight, item.Priority, want.Weight, want.Priority)
		}
	}
	reread, err := fixture.service.GetGroupModels(t.Context(), created.GroupID)
	if err != nil {
		t.Fatalf("second GetGroupModels() error = %v", err)
	}
	for index, item := range reread.Items {
		if item.EntryID != got.Items[index].EntryID {
			t.Fatalf("reread item %d entry_id = %q, want stable %q", index, item.EntryID, got.Items[index].EntryID)
		}
	}

	var row models.Group
	if err := fixture.db.First(&row, created.GroupID).Error; err != nil {
		t.Fatal(err)
	}
	candidate, err := mapGroupRowToState(row)
	if err != nil {
		t.Fatalf("mapGroupRowToState() error = %v", err)
	}
	wantModels := []state.ModelConfig{
		{ID: "entry-a", Alias: "public", Weight: intPointer(30), Priority: intPointer(2)},
		{ID: "entry-b", Alias: "public", Weight: intPointer(0)},
	}
	if len(candidate.Models) != len(wantModels) {
		t.Fatalf("state models count = %d, want %d", len(candidate.Models), len(wantModels))
	}
	for index, model := range candidate.Models {
		if !entryIDPattern.MatchString(model.EntryID) {
			t.Fatalf("state model %d entry_id = %q, want lazy-backfilled", index, model.EntryID)
		}
		want := wantModels[index]
		if model.ID != want.ID || model.Alias != want.Alias ||
			!reflect.DeepEqual(model.Weight, want.Weight) || !reflect.DeepEqual(model.Priority, want.Priority) {
			t.Fatalf("state model %d = %#v, want core fields %#v", index, model, want)
		}
	}
	runtimeModels := loadLoaderGroupModels(t, fixture, created.GroupID)
	if len(runtimeModels) != len(wantModels) {
		t.Fatalf("loader models count = %d, want %d", len(runtimeModels), len(wantModels))
	}
	for index, model := range runtimeModels {
		if model.EntryID != candidate.Models[index].EntryID {
			t.Fatalf("loader model %d entry_id = %q, want %q", index, model.EntryID, candidate.Models[index].EntryID)
		}
	}
}

func TestValidateGroupRowCandidateEnforcesRouteEntryRules(t *testing.T) {
	fixture := newServiceFixture(t)
	tests := []struct {
		name    string
		models  string
		wantErr string
	}{
		{
			name:    "duplicate external and upstream pair",
			models:  `[{"id":"a","alias":"x"},{"id":"a","alias":"x"}]`,
			wantErr: `group 7 (route-rules) has duplicate route entry for external model "x" and upstream model "a"`,
		},
		{
			name:    "external model with only zero weights",
			models:  `[{"id":"a","alias":"x","weight":0},{"id":"b","alias":"x","weight":0}]`,
			wantErr: `group 7 (route-rules) external model "x" entry weights must sum to a positive value`,
		},
		{
			name:    "negative weight",
			models:  `[{"id":"a","weight":-1}]`,
			wantErr: `group 7 (route-rules) model "a": weight must be between 0 and 100`,
		},
		{
			name:    "priority below one",
			models:  `[{"id":"a","priority":0}]`,
			wantErr: `group 7 (route-rules) model "a": priority must be at least 1`,
		},
		{
			name:    "empty model id",
			models:  `[{"id":" ","alias":"x"}]`,
			wantErr: `group 7 (route-rules) model entry 0: model id is required`,
		},
		{
			name:   "weighted entries within limits pass the gate",
			models: `[{"id":"a","alias":"x","weight":100,"priority":1}]`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			group := models.Group{
				ID: 7, Name: "route-rules", ChannelID: string(channel.OpenAI),
				ConnectionType: models.ConnectionTypeAPIKey,
				Params:         models.JSON(`{}`), Models: models.JSON(test.models),
			}
			err := validateGroupRowCandidate(t.Context(), fixture.db, group, fixture.channelRegistry)
			if test.wantErr == "" {
				if err != nil {
					t.Fatalf("validateGroupRowCandidate() error = %v", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), test.wantErr) {
				t.Fatalf("validateGroupRowCandidate() error = %v, want substring %q", err, test.wantErr)
			}
		})
	}
}

func TestUpdateGroupModelsRejectsUnroutableStoredEntries(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAICompatible,
		Params:    json.RawMessage(`{"base_url":"https://unroutable-entries.example.com/v1"}`),
		Models: optionalGroupModels{
			Set:    true,
			Values: []GroupModel{{ID: "provider-a", Alias: "public", AliasEnabled: true}},
		},
		Credentials: "sk-unroutable-entries", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.Model(&models.Group{}).
		Where("id = ?", created.GroupID).
		Update("models", models.JSON(`[{"id":"provider-a","alias":"public","weight":0}]`)).Error; err != nil {
		t.Fatal(err)
	}
	beforeRevision := fixture.manager.Current().Revision
	beforeRegistry := fixture.registry.Snapshot()
	beforeStored := loadStoredGroupModelsJSON(t, fixture, created.GroupID)

	_, err = fixture.service.UpdateGroupModels(t.Context(), created.GroupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{Set: true, Values: []GroupModel{{ID: "provider-b"}}},
	})
	var apiErr *app_errors.APIError
	if !errors.As(err, &apiErr) || apiErr.Code != app_errors.ErrInternalServer.Code {
		t.Fatalf("UpdateGroupModels() error = %#v, want existing-row gate failure", err)
	}
	// assertModelsUpdateStateUnchanged 不能用于带路由字段的存储行（GroupModel 解码
	// 拒绝未知键），这里直接断言 revision/registry/存储原文不变。
	if fixture.manager.Current().Revision != beforeRevision {
		t.Fatalf("revision = %d, want unchanged %d", fixture.manager.Current().Revision, beforeRevision)
	}
	if !reflect.DeepEqual(fixture.registry.Snapshot(), beforeRegistry) {
		t.Fatal("Registry changed")
	}
	if got := loadStoredGroupModelsJSON(t, fixture, created.GroupID); got != beforeStored {
		t.Fatalf("persisted models changed: got=%s want=%s", got, beforeStored)
	}

	// 存储里的路由字段不阻塞保存路径：修复权重后可继续保存旧报文。
	if err := fixture.db.Model(&models.Group{}).
		Where("id = ?", created.GroupID).
		Update("models", models.JSON(`[{"id":"provider-a","alias":"public","weight":5}]`)).Error; err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.service.UpdateGroupModels(t.Context(), created.GroupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{Set: true, Values: []GroupModel{{ID: "provider-b"}}},
	}); err != nil {
		t.Fatalf("UpdateGroupModels() after repair error = %v", err)
	}
}

func intPointer(value int) *int { return &value }
