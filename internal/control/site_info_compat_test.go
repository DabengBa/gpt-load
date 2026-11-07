package control

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/storage/models"
)

func TestLegacyInvalidProviderDoesNotBlockUnrelatedSettings(t *testing.T) {
	initControlI18n(t)
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-legacy-provider")
	if err := f.db.Model(&models.Group{}).Where("id = ?", id).Update("provider_url", "https:").Error; err != nil {
		t.Fatal(err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
	w := serveGroupSettingsRequest(t, engine, http.MethodPut, "/api/groups/"+stringGroupID(id)+"/settings", "test-auth-key", `{"enabled":false}`)
	if w.Code != http.StatusOK {
		t.Fatalf("legacy update: status=%d body=%s", w.Code, w.Body)
	}
	var group models.Group
	if err := f.db.First(&group, id).Error; err != nil {
		t.Fatal(err)
	}
	if group.Enabled || group.ProviderURL == nil || *group.ProviderURL != "https:" {
		t.Fatalf("unexpected group: %+v", group)
	}
	w = serveGroupSettingsRequest(t, engine, http.MethodPut, "/api/groups/"+stringGroupID(id)+"/settings", "test-auth-key", `{"site_access":{"user_id":"42","access_token":"secret"}}`)
	if w.Code != http.StatusBadRequest {
		t.Fatalf("invalid root accepted for site access: %d %s", w.Code, w.Body)
	}
}

func TestProviderPortWriteAndLegacyCompatibility(t *testing.T) {
	initControlI18n(t)
	f := newServiceFixture(t)
	id := createGroupWithCredentials(t, f, "sk-legacy-port")
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, f.service).RegisterRoutes(engine)
	settings := "/api/groups/" + stringGroupID(id) + "/settings"
	validCreate := `{"name":"port-test-valid","channel_id":"openai","params":{"base_url":"https://port-test.example/v1"},"models":[],"credential":"sk-port-65535","provider_url":"https://example.com:65535"}`
	if w := performSiteInfoMutation(t, engine, http.MethodPost, "/api/groups", "00000000-0000-4000-8000-00000000c100", validCreate); w.Code != http.StatusOK {
		t.Fatalf("valid port create: %d %s", w.Code, w.Body)
	}
	if _, err := f.service.UpdateGroupSettings(t.Context(), id, GroupSettingsUpdateRequest{
		ProviderURL: optionalField[string]{Set: true, Value: "https://example.com:65535"},
	}); err != nil {
		t.Fatalf("valid port settings update: %v", err)
	}
	for _, port := range []string{"65536", "99999"} {
		create := `{"name":"port-test","channel_id":"openai","params":{},"models":[],"credential":"sk-new-port-` + port + `","provider_url":"https://example.com:` + port + `"}`
		w := performSiteInfoMutation(t, engine, http.MethodPost, "/api/groups", "00000000-0000-4000-8000-00000000c"+map[string]string{"65536": "101", "99999": "102"}[port], create)
		if w.Code != http.StatusBadRequest {
			t.Errorf("create port %s: %d %s", port, w.Code, w.Body)
		}
		w = serveGroupSettingsRequest(t, engine, http.MethodPut, settings, "test-auth-key", `{"provider_url":"https://example.com:`+port+`"}`)
		if w.Code != http.StatusBadRequest {
			t.Errorf("update port %s: %d %s", port, w.Code, w.Body)
		}
	}
	legacy := "https://example.com:99999/path"
	if err := f.db.Model(&models.Group{}).Where("id = ?", id).Update("provider_url", legacy).Error; err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct{ method, body string }{{http.MethodGet, ""}, {http.MethodPut, `{"name":"legacy renamed"}`}} {
		w := serveGroupSettingsRequest(t, engine, tc.method, settings, "test-auth-key", tc.body)
		if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), legacy) {
			t.Errorf("legacy %s: %d %s", tc.method, w.Code, w.Body)
		}
	}
}

// Exercise the actual wire boundary with the exact old client field whitelist.
func TestSiteInfoResponseOptInCompatibility(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-compat-site")
	if err := fixture.db.Model(&models.Group{}).Where("id = ?", id).Update("provider_url", "https://example.org/v1").Error; err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.Create(&models.GroupSiteConfig{GroupID: id, ConfigVersion: 1, UserID: "42", AccessToken: "cipher", SiteRootURL: "https://example.org", Status: "pending", UpdatedAtMS: 1}).Error; err != nil {
		t.Fatal(err)
	}
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	settings := "/api/groups/" + stringGroupID(id) + "/settings"
	for _, tc := range []struct {
		method, path, body, field string
		allowed                   []string
	}{
		{http.MethodGet, "/api/groups", "", "site_info", []string{"id", "name", "channel_id", "params", "provider_url", "status", "model_count", "client_model_count", "credential_configured", "credential_status"}},
		{http.MethodGet, settings, "", "site_access", []string{"name", "channel_id", "params", "provider_url", "enabled", "overrides", "effective", "proxy"}},
		{http.MethodPut, settings, `{"enabled":true}`, "site_access", []string{"name", "channel_id", "params", "provider_url", "enabled", "overrides", "effective", "proxy"}},
	} {
		t.Run(tc.method+tc.path, func(t *testing.T) {
			for _, include := range []bool{false, true} {
				path := tc.path
				if include {
					path += "?include=site_info"
				}
				w := serveGroupSettingsRequest(t, engine, tc.method, path, "test-auth-key", tc.body)
				if w.Code != http.StatusOK {
					t.Fatalf("status=%d body=%s", w.Code, w.Body)
				}
				var envelope struct {
					Data json.RawMessage `json:"data"`
				}
				if err := json.Unmarshal(w.Body.Bytes(), &envelope); err != nil {
					t.Fatal(err)
				}
				data := envelope.Data
				if tc.field == "site_info" {
					var collection struct {
						Items []json.RawMessage `json:"items"`
					}
					if err := json.Unmarshal(data, &collection); err != nil || len(collection.Items) != 1 {
						t.Fatalf("collection=%s err=%v", data, err)
					}
					data = collection.Items[0]
				}
				var fields map[string]json.RawMessage
				if err := json.Unmarshal(data, &fields); err != nil {
					t.Fatal(err)
				}
				_, present := fields[tc.field]
				if present != include {
					t.Fatalf("include=%v fields=%s", include, data)
				}
				if !include {
					allowed := map[string]bool{}
					for _, field := range tc.allowed {
						allowed[field] = true
					}
					for field := range fields {
						if !allowed[field] {
							t.Fatalf("old client rejects %q: %s", field, data)
						}
					}
					if len(fields) != len(allowed) {
						t.Fatalf("old baseline missing fields: %s", data)
					}
				}
			}
		})
	}
}

// The create and copy envelopes never carried site fields. Opting in must not
// widen them, otherwise a request from the new client could still hand an old
// client a payload it rejects.
func TestSiteInfoCreateAndCopyResponsesStayBaseline(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)

	body := `{"name":"site-create","channel_id":"openai_compatible",` +
		`"params":{"base_url":"https://site-create.example/v1"},` +
		`"models":[{"id":"gpt-4o","alias":"site-create-public","alias_enabled":true}],` +
		`"credential":"sk-site-create","provider_url":"https://site-create.example/v1",` +
		`"site_access":{"user_id":"42","access_token":"site-access-secret"}}`
	key := "00000000-0000-4000-8000-00000000c001"
	var created map[string]json.RawMessage
	for attempt := 0; attempt < 2; attempt++ {
		recorder := performSiteInfoMutation(t, engine, http.MethodPost, "/api/groups?include=site_info", key, body)
		data := assertSiteInfoBaselineMutationEnvelope(t, recorder, "group_id", "group_name", "credential_id")
		if attempt == 0 {
			created = data
			continue
		}
		if string(data["group_id"]) != string(created["group_id"]) {
			t.Fatalf("replay group_id = %s, want %s", data["group_id"], created["group_id"])
		}
	}

	var groupID uint
	if err := json.Unmarshal(created["group_id"], &groupID); err != nil || groupID == 0 {
		t.Fatalf("decode created group_id: %v, value=%s", err, created["group_id"])
	}

	copied := performSiteInfoMutation(
		t,
		engine,
		http.MethodPost,
		"/api/groups/"+stringGroupID(groupID)+"/copy?include=site_info",
		"00000000-0000-4000-8000-00000000c002",
		`{}`,
	)
	copiedData := assertSiteInfoBaselineMutationEnvelope(t, copied, "group_id", "group_name", "credential_id")
	if string(copiedData["group_id"]) == string(created["group_id"]) {
		t.Fatalf("copy reused the source group id: %#v", copiedData)
	}

	// The create request still applied site_access; only the response is narrow.
	settings := serveGroupSettingsRequest(
		t,
		engine,
		http.MethodGet,
		"/api/groups/"+stringGroupID(groupID)+"/settings?include=site_info",
		"test-auth-key",
		"",
	)
	if settings.Code != http.StatusOK {
		t.Fatalf("settings status=%d body=%s", settings.Code, settings.Body)
	}
	var settingsEnvelope struct {
		Data struct {
			SiteAccess GroupSiteAccessResponse `json:"site_access"`
		} `json:"data"`
	}
	if err := json.Unmarshal(settings.Body.Bytes(), &settingsEnvelope); err != nil {
		t.Fatal(err)
	}
	siteAccess := settingsEnvelope.Data.SiteAccess
	if siteAccess.UserID != "42" || !siteAccess.TokenConfigured ||
		siteAccess.SiteRootURL != "https://site-create.example" || siteAccess.Snapshot == nil ||
		siteAccess.Snapshot.Status != "pending" {
		t.Fatalf("site_access = %#v", siteAccess)
	}
}

// Only a single exact include=site_info value opts in. Every other form keeps
// the baseline settings payload, and the strict collection parser rejects it so
// a typo cannot silently downgrade a new client.
func TestSiteInfoIncludeParameterOptsInOnlyForExactSingleValue(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-compat-include")
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)
	settings := "/api/groups/" + stringGroupID(id) + "/settings"

	for _, query := range []string{
		"include=",
		"include=site_access",
		"include=site_info&include=site_info",
		"include=site_info&include=site_access",
	} {
		w := serveGroupSettingsRequest(t, engine, http.MethodGet, settings+"?"+query, "test-auth-key", "")
		if w.Code != http.StatusOK {
			t.Fatalf("%s: status=%d body=%s", query, w.Code, w.Body)
		}
		var envelope struct {
			Data map[string]json.RawMessage `json:"data"`
		}
		if err := json.Unmarshal(w.Body.Bytes(), &envelope); err != nil {
			t.Fatal(err)
		}
		if _, present := envelope.Data["site_access"]; present {
			t.Fatalf("%s opts in: %s", query, w.Body)
		}
		if len(envelope.Data) != 8 {
			t.Fatalf("%s fields=%s", query, w.Body)
		}
	}

	exact := serveGroupSettingsRequest(t, engine, http.MethodGet, settings+"?include=site_info", "test-auth-key", "")
	if exact.Code != http.StatusOK || !strings.Contains(exact.Body.String(), `"site_access"`) {
		t.Fatalf("exact include: status=%d body=%s", exact.Code, exact.Body)
	}

	got, apiErr := parseGroupCollectionQuery("include=site_info", false)
	if apiErr != nil {
		t.Fatalf("parseGroupCollectionQuery(include=site_info) error = %v", apiErr)
	}
	if got.Query != "" || got.Sort != GroupCollectionSortRecent || got.Page != 1 || got.PageSize != 100 {
		t.Fatalf("include=site_info must not alter the collection query: %#v", got)
	}
	for _, query := range []string{
		"include=",
		"include=site_access",
		"include=site_info&include=site_info",
		"include=site_info&include=site_access",
	} {
		w := performGroupCollectionRequest(engine, "/api/groups?"+query, "Bearer test-auth-key")
		assertGroupCollectionHTTPError(t, w, http.StatusBadRequest, "BAD_REQUEST")
	}
}

// An opted-in group without a stored token must still surface an explicit null
// so the new client can tell "not configured" from "extension unsupported".
func TestSiteInfoOptInKeepsExplicitNullForUnconfiguredGroup(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	createGroupWithCredentials(t, fixture, "sk-compat-unconfigured")
	engine := gin.New()
	NewServer(&config.Config{AuthKey: "test-auth-key"}, fixture.service).RegisterRoutes(engine)

	optedIn := performGroupCollectionRequest(engine, "/api/groups?include=site_info", "Bearer test-auth-key")
	if optedIn.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", optedIn.Code, optedIn.Body)
	}
	var optedInEnvelope struct {
		Data struct {
			Items []map[string]json.RawMessage `json:"items"`
		} `json:"data"`
	}
	if err := json.Unmarshal(optedIn.Body.Bytes(), &optedInEnvelope); err != nil {
		t.Fatal(err)
	}
	if len(optedInEnvelope.Data.Items) != 1 {
		t.Fatalf("items=%s", optedIn.Body)
	}
	raw, present := optedInEnvelope.Data.Items[0]["site_info"]
	if !present || string(raw) != "null" {
		t.Fatalf("opted-in unconfigured site_info = %s present=%v", raw, present)
	}

	baseline := performGroupCollectionRequest(engine, "/api/groups", "Bearer test-auth-key")
	if baseline.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", baseline.Code, baseline.Body)
	}
	var baselineEnvelope struct {
		Data struct {
			Items []map[string]json.RawMessage `json:"items"`
		} `json:"data"`
	}
	if err := json.Unmarshal(baseline.Body.Bytes(), &baselineEnvelope); err != nil {
		t.Fatal(err)
	}
	if len(baselineEnvelope.Data.Items) != 1 {
		t.Fatalf("items=%s", baseline.Body)
	}
	if _, present := baselineEnvelope.Data.Items[0]["site_info"]; present {
		t.Fatalf("baseline leaked site_info: %s", baseline.Body)
	}
}

func performSiteInfoMutation(
	t *testing.T,
	engine *gin.Engine,
	method, path, idempotencyKey, body string,
) *httptest.ResponseRecorder {
	t.Helper()
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(method, path, strings.NewReader(body))
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer test-auth-key")
	request.Header.Set("Idempotency-Key", idempotencyKey)
	engine.ServeHTTP(recorder, request)
	return recorder
}

func assertSiteInfoBaselineMutationEnvelope(
	t *testing.T,
	recorder *httptest.ResponseRecorder,
	allowed ...string,
) map[string]json.RawMessage {
	t.Helper()
	if recorder.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body)
	}
	var envelope struct {
		Code int                        `json:"code"`
		Data map[string]json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &envelope); err != nil {
		t.Fatalf("decode mutation envelope: %v body=%s", err, recorder.Body)
	}
	if envelope.Code != 0 || len(envelope.Data) != len(allowed) {
		t.Fatalf("mutation envelope = %#v, want %d baseline keys", envelope, len(allowed))
	}
	for _, field := range allowed {
		if envelope.Data[field] == nil {
			t.Fatalf("mutation data lacks %q: %#v", field, envelope.Data)
		}
	}
	for _, forbidden := range []string{`"site_access"`, `"site_info"`, `"token_configured"`, "site-access-secret"} {
		if strings.Contains(recorder.Body.String(), forbidden) {
			t.Fatalf("mutation response exposes %q: %s", forbidden, recorder.Body)
		}
	}
	return envelope.Data
}
