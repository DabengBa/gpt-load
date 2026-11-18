package control

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/agent"
	"gpt-load/internal/platform/httproute"
	"gpt-load/internal/storage/models"
)

func TestAgentModelRoutesHTTP(t *testing.T) {
	initControlI18n(t)
	f := newServiceFixture(t)
	const upstreamKey = "secret-upstream-key-for-model-routes"
	const upstreamURL = "https://private-provider.example/v1"
	groupID := createGroupWithCredentials(t, f, upstreamKey)
	original := loadCreatedGroupModels(t, f, groupID)[0]
	// A pre-existing entry may have only its persisted test alias for identity.
	legacy, err := json.Marshal([]map[string]any{{
		"id": original.ID, "test_alias": original.TestAlias, "enabled": false,
		"concurrency_limit": 7, "future_field": 42,
	}})
	if err != nil {
		t.Fatal(err)
	}
	if err := f.db.Model(&models.Group{}).Where("id = ?", groupID).Updates(map[string]any{
		"provider_url": upstreamURL, "name": "private-provider.example", "enabled": false, "models": models.JSON(legacy),
	}).Error; err != nil {
		t.Fatal(err)
	}
	store := agent.NewCredentialStore(f.db, f.encryption)
	f.service.SetAgentCredentialStore(store)
	issue := func(scopes ...string) string {
		t.Helper()
		key := fmt.Sprintf("00000000-0000-4000-8000-%012x", testIdempotencySequence.Add(1))
		credential, err := f.service.CreateAgentCredentialIdempotent(t.Context(), key,
			AgentCredentialCreateRequest{Name: strings.Join(scopes, " "), Scopes: scopes})
		if err != nil {
			t.Fatal(err)
		}
		return credential.Secret
	}
	reader := issue("diagnostics:read")
	server := agent.NewServer(agent.Dependencies{Credentials: store, Snapshots: f.manager})
	server.SetModelRouteService(f.service)
	engine := gin.New()
	registry, err := httproute.NewRegistry(server.Module())
	if err != nil {
		t.Fatal(err)
	}
	if err := registry.Bind(engine); err != nil {
		t.Fatal(err)
	}
	request := func(method, path, secret, body string, status int) []byte {
		t.Helper()
		req := httptest.NewRequest(method, agent.APIPrefix+path, strings.NewReader(body))
		if secret != "" {
			req.Header.Set("Authorization", "Bearer "+secret)
		}
		rec := httptest.NewRecorder()
		engine.ServeHTTP(rec, req)
		if rec.Code != status {
			t.Fatalf("%s %s: got %d, want %d: %s", method, path, rec.Code, status, rec.Body)
		}
		if strings.Contains(rec.Body.String(), "private-provider.example") || strings.Contains(rec.Body.String(), upstreamKey) {
			t.Fatal("response disclosed upstream connection information")
		}
		return rec.Body.Bytes()
	}
	const inventory = "/model-routes"
	collection := fmt.Sprintf("/groups/%d/model-routes", groupID)
	for _, token := range []string{"", "test-admin-key", upstreamKey} {
		request(http.MethodGet, inventory, token, "", http.StatusUnauthorized)
		request(http.MethodPost, collection, token, `{}`, http.StatusUnauthorized)
	}
	list := request(http.MethodGet, inventory, reader, "", http.StatusOK)
	if !strings.Contains(string(list), `"enabled":false`) || !strings.Contains(string(list), original.TestAlias) {
		t.Fatalf("inventory missing disabled group or legacy identity: %s", list)
	}
	request(http.MethodPost, collection, reader, `{"id":"gpt-new"}`, http.StatusForbidden)
	writer := issue("routes:write")
	request(http.MethodGet, inventory, writer, "", http.StatusForbidden)
	request(http.MethodPost, collection, writer, `{"id":"gpt-new","provider_url":"`+upstreamURL+`"}`, http.StatusBadRequest)
	request(http.MethodPost, collection, writer, `{"id":"invalid\u0001model"}`, http.StatusBadRequest)
	request(http.MethodPatch, collection+"/"+original.TestAlias, writer, `{"credential":"`+upstreamKey+`"}`, http.StatusBadRequest)
	request(http.MethodPost, collection, writer, `{"id":"gpt-new","alias":"explore","reasoning_effort":"medium","priority":2,"weight":3}`, http.StatusOK)
	request(http.MethodGet, "/routes?external_model=explore&protocol=openai-completions&operation=chat_completion", reader, "", http.StatusOK)
	legacyPath := collection + "/" + original.TestAlias
	request(http.MethodPatch, legacyPath, writer, `{"alias":"`+strings.Repeat("x", 256)+`"}`, http.StatusBadRequest)
	request(http.MethodPatch, legacyPath, writer, `{"id":"gpt-renamed","alias":"leader","reasoning_effort":"high","priority":2,"weight":3}`, http.StatusOK)
	request(http.MethodPatch, legacyPath, writer, `{"reasoning_effort":null,"weight":null}`, http.StatusOK)
	request(http.MethodPatch, legacyPath, writer, `{"priority":0}`, http.StatusBadRequest)
	var group models.Group
	if err := f.db.First(&group, groupID).Error; err != nil {
		t.Fatal(err)
	}
	if group.ProviderURL == nil || *group.ProviderURL != upstreamURL || group.Enabled || !strings.Contains(string(group.Models), `"future_field":42`) {
		t.Fatalf("unrelated group state changed: %s", group.Models)
	}
	configured := f.manager.Current().GroupCatalog[groupID].Models
	if len(configured) != 2 || configured[0].ID != "gpt-renamed" || configured[0].Alias != "leader" || configured[0].ReasoningEffort != "" || configured[0].Weight != nil || configured[0].Priority == nil || *configured[0].Priority != 2 || configured[0].ConcurrencyLimit != 7 || configured[0].Enabled == nil || *configured[0].Enabled {
		t.Fatalf("unexpected published state: %+v", configured)
	}
	if configured[1].EntryID == "" || configured[1].Alias != "explore" || configured[1].ReasoningEffort != "medium" {
		t.Fatalf("created route not published: %+v", configured[1])
	}
	if _, err := f.service.GetGroupModels(t.Context(), groupID); err != nil {
		t.Fatal(err)
	}
	if err := f.db.First(&group, groupID).Error; err != nil {
		t.Fatal(err)
	}
	var backfilled []groupModelEntry
	if err := json.Unmarshal(group.Models, &backfilled); err != nil || len(backfilled) != 2 || backfilled[0].EntryID == "" {
		t.Fatalf("admin read did not backfill legacy identity: %s (%v)", group.Models, err)
	}
	request(http.MethodPatch, legacyPath, writer, `{"priority":3}`, http.StatusOK)
	request(http.MethodDelete, legacyPath, writer, "", http.StatusOK)
	request(http.MethodDelete, legacyPath, writer, "", http.StatusNotFound)
	if got := f.manager.Current().GroupCatalog[groupID].Models; len(got) != 1 || got[0].EntryID != configured[1].EntryID {
		t.Fatalf("delete affected unrelated route: %+v", got)
	}
}
