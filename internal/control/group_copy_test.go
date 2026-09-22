package control

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func createGroupCopySource(
	t *testing.T,
	fixture serviceFixture,
	name string,
) GroupCreateResult {
	t.Helper()
	result, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name:      stringPointer(name),
		ChannelID: channel.OpenAICompatible,
		Params:    json.RawMessage(`{"base_url":"https://` + name + `.example/v1"}`),
		Models: optionalGroupModels{Set: true, Values: []GroupModel{
			{ID: "provider-model", Alias: "public-model", AliasEnabled: true},
		}},
		Credentials:    "sk-" + name,
		ConnectionType: models.ConnectionTypeAPIKey,
		ProviderURL:    optionalField[string]{Set: true, Value: "https://provider.example"},
	})
	if err != nil {
		t.Fatalf("create copy source %q: %v", name, err)
	}
	return result
}

func loadGroupByID(t *testing.T, fixture serviceFixture, id uint) models.Group {
	t.Helper()
	var group models.Group
	if err := fixture.db.Take(&group, id).Error; err != nil {
		t.Fatalf("load group %d: %v", id, err)
	}
	return group
}

func loadGroupCredentials(t *testing.T, fixture serviceFixture, groupID uint) []models.Credential {
	t.Helper()
	var rows []models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Order("id ASC").Find(&rows).Error; err != nil {
		t.Fatalf("load credentials for group %d: %v", groupID, err)
	}
	return rows
}

func TestCopyGroupIdempotentClonesConfigAndCredential(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	source := createGroupCopySource(t, fixture, "copy-source")
	if err := fixture.db.Model(&models.Group{}).
		Where("id = ?", source.GroupID).
		Update("enabled", false).Error; err != nil {
		t.Fatalf("disable source group: %v", err)
	}
	const key = "318f47a2-9c35-4d6e-8b1a-1234567890ab"

	first, err := fixture.service.CopyGroupIdempotent(t.Context(), key, source.GroupID)
	if err != nil {
		t.Fatalf("CopyGroupIdempotent() error = %v", err)
	}
	if first.GroupID == source.GroupID || first.GroupName != "copy-source-copy" ||
		first.CredentialsAdded != 1 || first.CredentialsDuplicated != 0 {
		t.Fatalf("copy result = %#v, want new group copy-source-copy with one credential", first)
	}

	original := loadGroupByID(t, fixture, source.GroupID)
	clone := loadGroupByID(t, fixture, first.GroupID)
	if clone.Name != "copy-source-copy" ||
		clone.ChannelID != original.ChannelID ||
		clone.ConnectionType != original.ConnectionType ||
		string(clone.Params) != string(original.Params) ||
		string(clone.Overrides) != string(original.Overrides) ||
		clone.Enabled != false ||
		clone.ProviderURL == nil || *clone.ProviderURL != *original.ProviderURL {
		t.Fatalf("clone row = %#v, want verbatim config copy of %#v", clone, original)
	}
	if (clone.ProxyConfig == nil) != (original.ProxyConfig == nil) ||
		clone.ProxyConfig != nil && *clone.ProxyConfig != *original.ProxyConfig {
		t.Fatalf("clone proxy = %v, want %v", clone.ProxyConfig, original.ProxyConfig)
	}
	var originalModels, cloneModels []groupModelEntry
	if err := json.Unmarshal(original.Models, &originalModels); err != nil {
		t.Fatalf("decode source models: %v", err)
	}
	if err := json.Unmarshal(clone.Models, &cloneModels); err != nil {
		t.Fatalf("decode clone models: %v", err)
	}
	if len(originalModels) != len(cloneModels) {
		t.Fatalf("model counts = source:%d clone:%d", len(originalModels), len(cloneModels))
	}
	for index := range originalModels {
		if originalModels[index].TestAlias == "" || cloneModels[index].TestAlias == "" || originalModels[index].TestAlias == cloneModels[index].TestAlias {
			t.Fatalf("model %d test aliases = %q/%q, want distinct generated values", index, originalModels[index].TestAlias, cloneModels[index].TestAlias)
		}
		originalModels[index].TestAlias = ""
		cloneModels[index].TestAlias = ""
	}
	if !reflect.DeepEqual(originalModels, cloneModels) {
		t.Fatalf("clone models differ beyond test aliases: source=%#v clone=%#v", originalModels, cloneModels)
	}
	if *clone.PriceMultiplierMicros != *original.PriceMultiplierMicros {
		t.Fatalf("clone price multiplier = %d, want %d",
			*clone.PriceMultiplierMicros, *original.PriceMultiplierMicros)
	}

	sourceCredentials := loadGroupCredentials(t, fixture, source.GroupID)
	cloneCredentials := loadGroupCredentials(t, fixture, clone.ID)
	if len(sourceCredentials) != 1 || len(cloneCredentials) != 1 {
		t.Fatalf("credential counts = source:%d clone:%d, want 1/1",
			len(sourceCredentials), len(cloneCredentials))
	}
	if cloneCredentials[0].Data != sourceCredentials[0].Data ||
		cloneCredentials[0].Fingerprint != sourceCredentials[0].Fingerprint ||
		cloneCredentials[0].IdentityFingerprint != sourceCredentials[0].IdentityFingerprint ||
		cloneCredentials[0].AuthState != sourceCredentials[0].AuthState {
		t.Fatalf("cloned credential = %#v, want verbatim credential of %#v",
			cloneCredentials[0], sourceCredentials[0])
	}
	if original.Name != "copy-source" {
		t.Fatalf("source group mutated: %#v", original)
	}

	replayed, err := fixture.service.CopyGroupIdempotent(t.Context(), key, source.GroupID)
	if err != nil {
		t.Fatalf("replay CopyGroupIdempotent() error = %v", err)
	}
	if !reflect.DeepEqual(replayed, first) {
		t.Fatalf("replay = %#v, first = %#v", replayed, first)
	}
	var groupCount int64
	if err := fixture.db.Model(&models.Group{}).Count(&groupCount).Error; err != nil {
		t.Fatalf("count groups: %v", err)
	}
	if groupCount != 2 {
		t.Fatalf("group count = %d, want 2 (source + one clone)", groupCount)
	}

	second, err := fixture.service.CopyGroupIdempotent(
		t.Context(),
		"418f47a2-9c35-4d6e-8b1a-1234567890ab",
		source.GroupID,
	)
	if err != nil {
		t.Fatalf("second CopyGroupIdempotent() error = %v", err)
	}
	if second.GroupName != "copy-source-copy-2" {
		t.Fatalf("second copy name = %q, want copy-source-copy-2", second.GroupName)
	}
}

func TestCopyGroupIdempotentGeneratesDistinctTestAliases(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	source := createGroupCopySource(t, fixture, "copy-test-alias")
	sourceModels, err := fixture.service.GetGroupModels(t.Context(), source.GroupID)
	if err != nil {
		t.Fatalf("GetGroupModels(source) error = %v", err)
	}
	if len(sourceModels.Items) != 1 || sourceModels.Items[0].TestAlias == "" {
		t.Fatalf("source models = %#v, want test alias", sourceModels.Items)
	}
	copyResult, err := fixture.service.CopyGroupIdempotent(t.Context(), "818f47a2-9c35-4d6e-8b1a-1234567890ab", source.GroupID)
	if err != nil {
		t.Fatalf("CopyGroupIdempotent() error = %v", err)
	}
	copyModels, err := fixture.service.GetGroupModels(t.Context(), copyResult.GroupID)
	if err != nil {
		t.Fatalf("GetGroupModels(copy) error = %v", err)
	}
	if len(copyModels.Items) != 1 || copyModels.Items[0].TestAlias == "" || copyModels.Items[0].TestAlias == sourceModels.Items[0].TestAlias {
		t.Fatalf("copy models = %#v, want distinct generated test alias from %q", copyModels.Items, sourceModels.Items[0].TestAlias)
	}
}

func TestCopyGroupIdempotentLoadsEncryptedProxyPolicies(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	source := createGroupCopySource(t, fixture, "copy-proxy")
	encryptProxy := func(proxy outboundproxy.Config) string {
		t.Helper()
		encoded, err := outboundproxy.Encode(proxy)
		if err != nil {
			t.Fatalf("encode proxy: %v", err)
		}
		ciphertext, err := fixture.encryption.Encrypt(encoded)
		if err != nil {
			t.Fatalf("encrypt proxy: %v", err)
		}
		return ciphertext
	}
	groupProxy := encryptProxy(outboundproxy.Config{
		Mode: outboundproxy.ModeCustom,
		URL:  "http://group-proxy.example.com:8080",
	})
	if err := fixture.db.Model(&models.Group{}).
		Where("id = ?", source.GroupID).
		Update("proxy_config", groupProxy).Error; err != nil {
		t.Fatalf("persist group proxy: %v", err)
	}
	if err := fixture.db.Create(&models.SystemSetting{
		Key: outboundproxy.SystemSettingKey,
		Value: encryptProxy(outboundproxy.Config{
			Mode: outboundproxy.ModeCustom,
			URL:  "http://global-proxy.example.com:8080",
		}),
	}).Error; err != nil {
		t.Fatalf("persist global proxy: %v", err)
	}

	if _, err := fixture.service.CopyGroupIdempotent(
		t.Context(),
		"618f47a2-9c35-4d6e-8b1a-1234567890ab",
		source.GroupID,
	); err != nil {
		t.Fatalf("CopyGroupIdempotent() with encrypted proxies error = %v", err)
	}
}

func TestCopyGroupIdempotentNormalizesRefreshingCredential(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	source := createGroupCopySource(t, fixture, "copy-refreshing")
	credentials := loadGroupCredentials(t, fixture, source.GroupID)
	if len(credentials) != 1 {
		t.Fatalf("source credentials = %d, want 1", len(credentials))
	}
	if err := fixture.db.Model(&models.Credential{}).
		Where("id = ?", credentials[0].ID).
		Updates(map[string]any{
			"auth_state":      models.CredentialAuthStateRefreshing,
			"auth_error_code": "",
		}).Error; err != nil {
		t.Fatalf("mark source credential refreshing: %v", err)
	}

	result, err := fixture.service.CopyGroupIdempotent(
		t.Context(),
		"718f47a2-9c35-4d6e-8b1a-1234567890ab",
		source.GroupID,
	)
	if err != nil {
		t.Fatalf("CopyGroupIdempotent() error = %v", err)
	}
	cloned := loadGroupCredentials(t, fixture, result.GroupID)
	if len(cloned) != 1 {
		t.Fatalf("cloned credentials = %d, want 1", len(cloned))
	}
	if cloned[0].AuthState != models.CredentialAuthStateOutcomeUnknown ||
		cloned[0].AuthErrorCode != "refresh_interrupted" {
		t.Fatalf("cloned credential auth state = %q/%q, want outcome_unknown/refresh_interrupted", cloned[0].AuthState, cloned[0].AuthErrorCode)
	}
}

func TestCopyGroupIdempotentRejectsMissingGroup(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	fixture.service.operationRandom = bytes.NewReader(bytes.Repeat([]byte{0x72}, 16))
	_, err := fixture.service.CopyGroupIdempotent(
		t.Context(),
		"518f47a2-9c35-4d6e-8b1a-1234567890ab",
		9999,
	)
	assertAPIErrorCode(t, err, app_errors.ErrResourceNotFound.Code)
}

func TestGroupCopyHTTPClonesThroughAPI(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	source := createGroupCopySource(t, fixture, "http-copy")
	engine := gin.New()
	NewServer(
		&config.Config{AuthKey: authTestKey},
		fixture.service,
	).RegisterRoutes(engine)

	request := httptest.NewRequest(
		http.MethodPost,
		"/api/groups/"+strconv.FormatUint(uint64(source.GroupID), 10)+"/copy",
		bytes.NewReader([]byte(`{}`)),
	)
	request.Header.Set("Authorization", "Bearer "+authTestKey)
	request.Header.Set("Content-Type", "application/json")
	setRequiredTestIdempotencyHeader(request)
	recorder := httptest.NewRecorder()
	engine.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("copy response = %d %s, want 200", recorder.Code, recorder.Body.String())
	}
	var envelope struct {
		Data struct {
			GroupID   uint   `json:"group_id"`
			GroupName string `json:"group_name"`
		} `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode copy response: %v", err)
	}
	if envelope.Data.GroupID == source.GroupID || envelope.Data.GroupName != "http-copy-copy" {
		t.Fatalf("copy data = %#v, want cloned group http-copy-copy", envelope.Data)
	}
	if got := loadGroupByID(t, fixture, envelope.Data.GroupID); got.Name != "http-copy-copy" {
		t.Fatalf("cloned group = %#v", got)
	}

	missing := httptest.NewRequest(http.MethodPost, "/api/groups/9999/copy", bytes.NewReader([]byte(`{}`)))
	missing.Header.Set("Authorization", "Bearer "+authTestKey)
	setRequiredTestIdempotencyHeader(missing)
	missingRecorder := httptest.NewRecorder()
	engine.ServeHTTP(missingRecorder, missing)
	if missingRecorder.Code != http.StatusNotFound {
		t.Fatalf("missing group copy = %d %s, want 404", missingRecorder.Code, missingRecorder.Body.String())
	}

	noKey := httptest.NewRequest(
		http.MethodPost,
		"/api/groups/"+strconv.FormatUint(uint64(source.GroupID), 10)+"/copy",
		nil,
	)
	noKey.Header.Set("Authorization", "Bearer "+authTestKey)
	noKeyRecorder := httptest.NewRecorder()
	engine.ServeHTTP(noKeyRecorder, noKey)
	if noKeyRecorder.Code == http.StatusOK {
		t.Fatalf("copy without idempotency key succeeded: %s", noKeyRecorder.Body.String())
	}
}
