package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/storage/models"
)

// credentialActivityStub is the narrow reader the credential detail path uses
// to project the ordinary API credential's last upstream failure summary.
type credentialActivityStub struct {
	activities map[uint]requestlog.CredentialActivity
	err        error
	queries    []requestlog.CredentialActivityQuery
}

func (stub *credentialActivityStub) QueryCredentialActivity(
	_ context.Context,
	query requestlog.CredentialActivityQuery,
) (map[uint]requestlog.CredentialActivity, error) {
	stub.queries = append(stub.queries, query)
	if stub.err != nil {
		return nil, stub.err
	}
	return stub.activities, nil
}

func newCredentialDetailSummaryFixture(t *testing.T, stub *credentialActivityStub) (serviceFixture, uint, uint) {
	t.Helper()
	initControlI18n(t)
	fixture := newServiceFixture(t)
	fixture.service.credentialActivity = stub
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		Name:           stringPointer("detail summary"),
		ChannelID:      channel.OpenAI,
		Params:         json.RawMessage(`{}`),
		Models:         optionalGroupModels{Set: true},
		Credentials:    "sk-detail-summary",
		ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatal(err)
	}
	var credential models.Credential
	if err := fixture.db.Where("group_id = ?", created.GroupID).Take(&credential).Error; err != nil {
		t.Fatal(err)
	}
	return fixture, created.GroupID, credential.ID
}

func TestCredentialDetailKeepsRedactedFailureSummary(t *testing.T) {
	t.Parallel()
	var credentialID uint
	stub := &credentialActivityStub{}
	fixture, groupID, id := newCredentialDetailSummaryFixture(t, stub)
	credentialID = id
	stub.activities = map[uint]requestlog.CredentialActivity{
		credentialID: {CredentialID: credentialID, LastFailureSummary: "upstream rejected sk-****mary with 401"},
	}
	server := NewServer(&config.Config{AuthKey: "summary-auth"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)

	response := serveCredentialRequest(
		t, engine, http.MethodGet,
		fmt.Sprintf("/api/groups/%d/credential", groupID),
		"", "summary-auth", "", 0,
	)
	if response.Code != http.StatusOK {
		t.Fatalf("credential response = %d %s", response.Code, response.Body.String())
	}
	var envelope struct {
		Data GroupCredentialResponse `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Data.Credential == nil {
		t.Fatalf("credential detail missing: %s", response.Body.String())
	}
	if got := envelope.Data.Credential.LastFailureSummary; got != "upstream rejected sk-****mary with 401" {
		t.Fatalf("last_failure_summary = %q, want the redacted upstream summary", got)
	}
	if len(stub.queries) != 1 || len(stub.queries[0].CredentialIDs) != 1 ||
		stub.queries[0].CredentialIDs[0] != credentialID {
		t.Fatalf("activity queries = %#v, want one query for credential %d", stub.queries, credentialID)
	}
}

func TestCredentialDetailSurvivesActivityLookupFailure(t *testing.T) {
	t.Parallel()
	stub := &credentialActivityStub{err: errors.New("activity unavailable")}
	fixture, groupID, _ := newCredentialDetailSummaryFixture(t, stub)
	server := NewServer(&config.Config{AuthKey: "lookup-auth"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)

	response := serveCredentialRequest(
		t, engine, http.MethodGet,
		fmt.Sprintf("/api/groups/%d/credential", groupID),
		"", "lookup-auth", "", 0,
	)
	if response.Code != http.StatusOK {
		t.Fatalf("lookup failure must not fail the read: %d %s", response.Code, response.Body.String())
	}
	var envelope struct {
		Data GroupCredentialResponse `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Data.Credential == nil || envelope.Data.Credential.LastFailureSummary != "" {
		t.Fatalf("credential = %#v, want a credential detail without a summary", envelope.Data.Credential)
	}
}
