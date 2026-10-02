package control

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func TestCredentialExpectedIDRequired(t *testing.T) {
	initControlI18n(t)
	t.Parallel()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-current")
	current, err := fixture.service.currentGroupCredentialID(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	for _, header := range []string{"", "invalid", "0", strconv.FormatUint(uint64(current+1), 10), strconv.FormatUint(uint64(current), 10)} {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodDelete, "/", nil)
		c.Request.Header.Set("X-Credential-ID", header)
		c.Params = gin.Params{{Key: "group_id", Value: strconv.FormatUint(uint64(id), 10)}}
		(&Server{service: fixture.service}).resolveGroupCredential(c)
		wantAccepted := header == strconv.FormatUint(uint64(current), 10)
		if c.IsAborted() == wantAccepted {
			t.Errorf("header %q aborted = %v, want accepted %v", header, c.IsAborted(), wantAccepted)
		}
		if wantAccepted && c.Param("credential_id") != header {
			t.Errorf("pinned ID = %q, want %q", c.Param("credential_id"), header)
		}
	}
}

func TestCredentialUpdateRejectsDuplicateLinesWithoutMutation(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-before")
	var before, after models.Credential
	if err := fixture.db.Where("group_id = ?", id).Take(&before).Error; err != nil {
		t.Fatal(err)
	}
	_, err := fixture.service.UpdateGroupCredential(t.Context(), id, before.ID, CredentialUpdateRequest{Credentials: optionalField[string]{Set: true, Value: "sk-after\nsk-after"}})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Errorf("duplicate update error = %v, want validation", err)
	}
	if err := fixture.db.Take(&after, before.ID).Error; err != nil {
		t.Fatal(err)
	}
	if after.Data != before.Data || after.SecretVersion != before.SecretVersion || after.UpdatedAtMS != before.UpdatedAtMS {
		t.Errorf("duplicate update changed data/version")
	}
}

func TestCredentialPinnedIDRejectsReplacement(t *testing.T) {
	initControlI18n(t)
	t.Parallel()
	fixture := newServiceFixture(t)
	group := createGroupWithCredentials(t, fixture, "sk-old")
	oldID, err := fixture.service.currentGroupCredentialID(t.Context(), group)
	if err != nil {
		t.Fatal(err)
	}
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodDelete, "/", nil)
	c.Request.Header.Set("X-Credential-ID", strconv.FormatUint(uint64(oldID), 10))
	c.Params = gin.Params{{Key: "group_id", Value: strconv.FormatUint(uint64(group), 10)}}
	(&Server{service: fixture.service}).resolveGroupCredential(c)
	if c.IsAborted() {
		t.Fatal("current ID rejected")
	}
	if err := fixture.service.DeleteGroupCredential(t.Context(), group, oldID); err != nil {
		t.Fatal(err)
	}
	replacement, err := fixture.service.ImportGroupCredentials(t.Context(), group, CredentialImportRequest{Credentials: "sk-new"})
	if err != nil {
		t.Fatal(err)
	}
	pinned, err := strconv.ParseUint(c.Param("credential_id"), 10, strconv.IntSize)
	if err != nil {
		t.Fatal(err)
	}
	if err := fixture.service.DeleteGroupCredential(t.Context(), group, uint(pinned)); err == nil {
		t.Fatal("stale deletion succeeded")
	}
	current, err := fixture.service.currentGroupCredentialID(t.Context(), group)
	if err != nil || current != replacement.CredentialID {
		t.Fatalf("replacement ID = %d, error = %v", current, err)
	}
}

func TestCredentialConnectRejectsStaleExpectedIDAtomically(t *testing.T) {
	t.Parallel()
	fixture, group, current := newSubscriptionCredentialFixture(t)
	stage := mustImportSubscriptionStage(t, fixture, "account-observation", "observation@example.com")
	_, err := fixture.service.ConnectGroupCredentialsIdempotent(t.Context(), "818f47a2-9c35-4d6e-8b1a-1234567890ab", group, current+1, stage.StageID)
	if !errors.Is(err, app_errors.ErrCredentialVersionConflict) {
		t.Fatalf("stale connect error = %v", err)
	}
	var after models.CredentialStage
	if err := fixture.db.Take(&after, "id = ?", stage.StageID).Error; err != nil {
		t.Fatal(err)
	}
	if after.Status == models.CredentialStageConsumed || after.EncryptedPayload == "" {
		t.Fatal("stale connect consumed stage")
	}
}
func TestGroupCopyReturnsCopiedCredentialAndEmptyNull(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	id := createGroupWithCredentials(t, fixture, "sk-source")
	result, err := fixture.service.CopyGroupIdempotent(t.Context(), "818f47a2-9c35-4d6e-8b1a-1234567890ab", id)
	if err != nil {
		t.Fatal(err)
	}
	var copied models.Credential
	if err := fixture.db.Where("group_id = ?", result.GroupID).Take(&copied).Error; err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		t.Fatal(err)
	}
	var body struct {
		CredentialID *uint `json:"credential_id"`
	}
	if err := json.Unmarshal(encoded, &body); err != nil {
		t.Fatal(err)
	}
	if body.CredentialID == nil || *body.CredentialID != copied.ID {
		t.Errorf("copy result = %s, want credential %d", encoded, copied.ID)
	}
	current, err := fixture.service.currentGroupCredentialID(t.Context(), id)
	if err != nil {
		t.Fatal(err)
	}
	if err := fixture.service.DeleteGroupCredential(t.Context(), id, current); err != nil {
		t.Fatal(err)
	}
	empty, err := fixture.service.CopyGroupIdempotent(t.Context(), "818f47a2-9c35-4d6e-8b1a-1234567890ac", id)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err = json.Marshal(empty)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatal(err)
	}
	if string(fields["credential_id"]) != "null" {
		t.Errorf("empty copy = %s, want credential_id:null", encoded)
	}
}

func TestCredentialConnectRequiresExpectedIDHeader(t *testing.T) {
	initControlI18n(t)
	t.Parallel()
	fixture, id, current := newSubscriptionCredentialFixture(t)
	if err := fixture.service.DeleteGroupCredential(t.Context(), id, current); err != nil {
		t.Fatal(err)
	}
	stage := mustImportSubscriptionStage(t, fixture, "connect-header", "connect-header@example.com")
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Params = gin.Params{{Key: "group_id", Value: strconv.FormatUint(uint64(id), 10)}}
	c.Request = httptest.NewRequest(http.MethodPost, "/", strings.NewReader(`{"staged_credential_id":"`+stage.StageID+`"}`))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("Idempotency-Key", "818f47a2-9c35-4d6e-8b1a-1234567890ab")
	(&Server{service: fixture.service}).handleConnectGroupCredentials(c)
	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("connect without expected ID status = %d, want 400", recorder.Code)
	}
}

func TestSingularCredentialRouteContract(t *testing.T) {
	t.Parallel()
	module := (&Server{}).HTTPModule()
	want := map[string]bool{
		http.MethodGet + " /groups/:group_id/credential":    false,
		http.MethodPost + " /groups/:group_id/credential":   false,
		http.MethodPut + " /groups/:group_id/credential":    false,
		http.MethodDelete + " /groups/:group_id/credential": false,
	}
	for _, route := range module.Routes {
		if strings.Contains(route.Path, "/groups/:group_id/credentials") {
			t.Errorf("retired plural credential route remains: %s", route.Path)
		}
		for _, method := range route.Methods {
			key := method + " " + route.Path
			if _, exists := want[key]; exists {
				want[key] = true
			}
		}
	}
	for key, exists := range want {
		if !exists {
			t.Errorf("singular credential route missing: %s", key)
		}
	}
}

func TestSingularCredentialEmptyAndOccupiedGroup(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-current")
	var row models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Take(&row).Error; err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.service.ImportGroupCredentials(t.Context(), groupID, CredentialImportRequest{Credentials: "sk-current"}); !errors.Is(err, app_errors.ErrSingleCredentialRequired) {
		t.Fatalf("configure occupied group error = %v, want single credential required", err)
	}
	if err := fixture.service.DeleteGroupCredential(t.Context(), groupID, row.ID); err != nil {
		t.Fatal(err)
	}
	result, err := fixture.service.GetGroupCredential(t.Context(), groupID)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := json.Marshal(result)
	if err != nil || string(encoded) != `{"credential":null,"observation":null}` {
		t.Fatalf("empty group = %s, error = %v", encoded, err)
	}
	if _, err := fixture.service.GetGroupCredential(t.Context(), groupID+1000); err == nil {
		t.Fatal("missing group unexpectedly succeeded")
	}
	if _, err := fixture.service.ImportGroupCredentials(t.Context(), groupID, CredentialImportRequest{Credentials: "sk-new"}); err != nil {
		t.Fatalf("configure empty group: %v", err)
	}
	result, err = fixture.service.GetGroupCredential(t.Context(), groupID)
	if err != nil || result.Credential == nil || result.Credential.CredentialID == row.ID {
		t.Fatalf("reconfigured group = %#v, error = %v", result, err)
	}
}

func TestSingularSubscriptionConnectJSON(t *testing.T) {
	t.Parallel()
	var request CredentialConnectRequest
	if err := json.Unmarshal([]byte(`{"staged_credential_id":"stage-one"}`), &request); err != nil {
		t.Fatal(err)
	}
	if request.StagedCredentialID != "stage-one" {
		t.Fatalf("one-stage request = %#v", request)
	}
	if err := json.Unmarshal([]byte(`{"staged_credential_id":["one","two"]}`), &request); err == nil {
		t.Fatal("multi-stage request unexpectedly accepted")
	}
}

func TestSingularCredentialDTORejectsPluralFields(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		body   string
		target any
	}{
		{body: `{"credentials":"one"}`, target: &GroupCreateRequest{}},
		{body: `{"staged_credential_ids":["one"]}`, target: &GroupCreateRequest{}},
		{body: `{"credentials":"one"}`, target: &CredentialImportRequest{}},
		{body: `{"staged_credential_ids":["one"]}`, target: &CredentialConnectRequest{}},
	} {
		decoder := json.NewDecoder(strings.NewReader(test.body))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(test.target); err == nil {
			t.Errorf("retired plural input accepted: %s", test.body)
		}
	}
}
