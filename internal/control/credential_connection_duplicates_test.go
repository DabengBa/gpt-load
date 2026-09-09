package control

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"sort"
	"testing"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func TestCreateSubscriptionGroupRejectsDuplicateStagesAtomically(t *testing.T) {
	t.Parallel()
	for _, idempotent := range []bool{false, true} {
		t.Run(fmt.Sprintf("idempotent=%t", idempotent), func(t *testing.T) {
			t.Parallel()
			fixture := newServiceFixture(t)
			first := mustImportSubscriptionStage(t, fixture, "create-duplicate", "first@example.com")
			second := mustImportSubscriptionStage(t, fixture, "create-duplicate", "second@example.com")
			stageIDs := []string{first.StageID, second.StageID}
			beforeSnapshot := fixture.manager.Current()
			beforeRegistry := fixture.registry.Snapshot()
			var beforeStages []models.CredentialStage
			if err := fixture.db.Where("id IN ?", stageIDs).Order("id ASC").Find(&beforeStages).Error; err != nil {
				t.Fatal(err)
			}

			request := GroupCreateRequest{
				Name: stringPointer("subscription duplicate create"), ChannelID: channel.Codex,
				ConnectionType:      models.ConnectionTypeSubscription,
				Models:              optionalGroupModels{Set: true},
				StagedCredentialIDs: stageIDs,
			}
			var err error
			if idempotent {
				_, err = fixture.service.CreateGroupIdempotent(
					t.Context(), "00000000-0000-4000-8000-00000000d101", request,
				)
			} else {
				_, err = fixture.service.CreateGroup(t.Context(), request)
			}
			if !errors.Is(err, app_errors.ErrDuplicateCredentialIdentity) {
				t.Fatalf("create error = %v, want duplicate credential identity", err)
			}

			var groupCount, credentialCount int64
			if err := fixture.db.Model(&models.Group{}).Count(&groupCount).Error; err != nil {
				t.Fatal(err)
			}
			if err := fixture.db.Model(&models.Credential{}).Count(&credentialCount).Error; err != nil {
				t.Fatal(err)
			}
			if groupCount != 0 || credentialCount != 0 {
				t.Fatalf("created rows = group:%d credential:%d, want 0/0", groupCount, credentialCount)
			}
			var afterStages []models.CredentialStage
			if err := fixture.db.Where("id IN ?", stageIDs).Order("id ASC").Find(&afterStages).Error; err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(afterStages, beforeStages) {
				t.Fatalf("stages changed after rejection: before=%#v after=%#v", beforeStages, afterStages)
			}
			if fixture.manager.Current() != beforeSnapshot ||
				!reflect.DeepEqual(fixture.registry.Snapshot(), beforeRegistry) {
				t.Fatal("rejected create mutated snapshot or registry")
			}
		})
	}
}

func TestConnectSubscriptionGroupRejectsExistingCredentialBatchAtomically(t *testing.T) {
	t.Parallel()
	for _, idempotent := range []bool{false, true} {
		t.Run(fmt.Sprintf("idempotent=%t", idempotent), func(t *testing.T) {
			t.Parallel()
			fixture, groupID, _ := newSubscriptionCredentialFixture(t)
			existing := mustImportSubscriptionStage(t, fixture, "account-observation", "observation@example.com")
			newFirst := mustImportSubscriptionStage(t, fixture, "connect-new", "new-first@example.com")
			newSecond := mustImportSubscriptionStage(t, fixture, "connect-new", "new-second@example.com")
			stageIDs := []string{existing.StageID, newFirst.StageID, newSecond.StageID}
			beforeSnapshot := fixture.manager.Current()
			beforeRegistry := fixture.registry.Snapshot()
			var beforeCredential models.Credential
			if err := fixture.db.Where("group_id = ?", groupID).Take(&beforeCredential).Error; err != nil {
				t.Fatal(err)
			}
			var beforeStages []models.CredentialStage
			if err := fixture.db.Where("id IN ?", stageIDs).Order("id ASC").Find(&beforeStages).Error; err != nil {
				t.Fatal(err)
			}

			var err error
			if idempotent {
				_, err = fixture.service.ConnectGroupCredentialsIdempotent(
					t.Context(),
					"00000000-0000-4000-8000-00000000d102",
					groupID,
					stageIDs,
				)
			} else {
				_, err = fixture.service.ConnectGroupCredentials(t.Context(), groupID, stageIDs)
			}
			if !errors.Is(err, app_errors.ErrDuplicateCredentialIdentity) {
				t.Fatalf("connect error = %v, want duplicate credential identity", err)
			}

			var credentialCount int64
			if err := fixture.db.Model(&models.Credential{}).
				Where("group_id = ?", groupID).Count(&credentialCount).Error; err != nil {
				t.Fatal(err)
			}
			if credentialCount != 1 {
				t.Fatalf("credential count = %d, want 1", credentialCount)
			}
			var afterCredential models.Credential
			if err := fixture.db.Where("id = ?", beforeCredential.ID).Take(&afterCredential).Error; err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(afterCredential, beforeCredential) {
				t.Fatalf("credential changed after rejection: before=%#v after=%#v", beforeCredential, afterCredential)
			}
			var afterStages []models.CredentialStage
			if err := fixture.db.Where("id IN ?", stageIDs).Order("id ASC").Find(&afterStages).Error; err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(afterStages, beforeStages) {
				t.Fatalf("stages changed after rejection: before=%#v after=%#v", beforeStages, afterStages)
			}
			var consumedCount int64
			if err := fixture.db.Model(&models.CredentialStage{}).
				Where("id IN ? AND status = ?", stageIDs, models.CredentialStageConsumed).
				Count(&consumedCount).Error; err != nil {
				t.Fatal(err)
			}
			if consumedCount != 0 {
				t.Fatalf("consumed stage count = %d, want 0", consumedCount)
			}
			if fixture.manager.Current() != beforeSnapshot ||
				!reflect.DeepEqual(fixture.registry.Snapshot(), beforeRegistry) {
				t.Fatal("rejected connect mutated snapshot or registry")
			}
		})
	}
}

func TestConnectSubscriptionGroupReplacesCredentialThatNeedsReauthorization(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name           string
		authState      models.CredentialAuthState
		authErrorCode  string
		idempotencyKey string
	}{
		{
			name: "reauthorization required", authState: models.CredentialAuthStateReauthorizationRequired,
			authErrorCode: "refresh_rejected", idempotencyKey: "00000000-0000-4000-8000-00000000d103",
		},
		{
			name: "outcome unknown", authState: models.CredentialAuthStateOutcomeUnknown,
			authErrorCode: "refresh_outcome_unknown", idempotencyKey: "00000000-0000-4000-8000-00000000d104",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			fixture, groupID, credentialID := newSubscriptionCredentialFixture(t)
			var before models.Credential
			if err := fixture.db.First(&before, credentialID).Error; err != nil {
				t.Fatal(err)
			}
			if err := fixture.db.Model(&models.Credential{}).Where("id = ?", credentialID).Updates(map[string]any{
				"auth_state": test.authState, "auth_error_code": test.authErrorCode,
			}).Error; err != nil {
				t.Fatal(err)
			}
			stage, err := fixture.service.ImportCredentialStage(t.Context(), channel.Codex, []byte(fmt.Sprintf(
				`{"type":"codex","access_token":"replacement-access-%s","refresh_token":"replacement-refresh-%s","account_id":"account-observation","email":%q}`,
				test.authState,
				test.authState,
				fmt.Sprintf("replacement-%s@example.com", test.authState),
			)))
			if err != nil {
				t.Fatal(err)
			}

			inspection, err := fixture.service.InspectGroupCredentialConnection(
				t.Context(), groupID, []string{stage.StageID},
			)
			if err != nil {
				t.Fatal(err)
			}
			if len(inspection.DuplicatedStageIDs) != 0 {
				t.Errorf("inspection duplicates = %v, want none", inspection.DuplicatedStageIDs)
			}

			result, err := fixture.service.ConnectGroupCredentialsIdempotent(
				t.Context(), test.idempotencyKey, groupID, []string{stage.StageID},
			)
			if err != nil {
				t.Fatal(err)
			}
			if result.CredentialsAdded != 1 || result.CredentialsDuplicated != 0 {
				t.Errorf("result = %#v", result)
			}
			var after models.Credential
			if err := fixture.db.First(&after, credentialID).Error; err != nil {
				t.Fatal(err)
			}
			if after.Fingerprint == before.Fingerprint || after.Data == before.Data ||
				after.IdentityFingerprint != before.IdentityFingerprint ||
				after.SecretVersion != before.SecretVersion+1 ||
				after.AuthState != models.CredentialAuthStateReady || after.AuthErrorCode != "" {
				t.Errorf(
					"replacement state = fingerprint_changed:%t data_changed:%t identity_preserved:%t version:%d auth:%q error:%q",
					after.Fingerprint != before.Fingerprint,
					after.Data != before.Data,
					after.IdentityFingerprint == before.IdentityFingerprint,
					after.SecretVersion,
					after.AuthState,
					after.AuthErrorCode,
				)
			}
			var credentialCount int64
			if err := fixture.db.Model(&models.Credential{}).
				Where("group_id = ?", groupID).Count(&credentialCount).Error; err != nil {
				t.Fatal(err)
			}
			if credentialCount != 1 {
				t.Errorf("credential count = %d, want 1", credentialCount)
			}
			var consumed models.CredentialStage
			if err := fixture.db.Take(&consumed, "id = ?", stage.StageID).Error; err != nil {
				t.Fatal(err)
			}
			if consumed.Status != models.CredentialStageConsumed || consumed.EncryptedPayload != "" {
				t.Errorf("stage was not consumed after replacement: %#v", consumed)
			}
		})
	}
}

func TestInspectSubscriptionConnectionIdentifiesExactDuplicateStages(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture, groupID, _ := newSubscriptionCredentialFixture(t)
	existing := mustImportSubscriptionStage(t, fixture, "account-observation", "existing@example.com")
	newFirst := mustImportSubscriptionStage(t, fixture, "inspect-new", "new-first@example.com")
	newSecond := mustImportSubscriptionStage(t, fixture, "inspect-new", "new-second@example.com")
	stageIDs := []string{existing.StageID, newFirst.StageID, newSecond.StageID}

	engine := gin.New()
	const auth = "credential-duplicate-inspection-auth"
	NewServer(&config.Config{AuthKey: auth}, fixture.service).RegisterRoutes(engine)
	encoded, err := json.Marshal(CredentialConnectRequest{StagedCredentialIDs: stageIDs})
	if err != nil {
		t.Fatal(err)
	}
	response := serveCredentialRequest(
		t,
		engine,
		http.MethodPost,
		fmt.Sprintf("/api/groups/%d/credentials/connect/inspect", groupID),
		string(encoded),
		auth,
		"",
	)
	if response.Code != http.StatusOK {
		t.Fatalf("inspection = %d %s", response.Code, response.Body.String())
	}
	var envelope struct {
		Code int `json:"code"`
		Data struct {
			DuplicatedStageIDs []string `json:"duplicated_stage_ids"`
		} `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	want := []string{existing.StageID}
	if newFirst.StageID < newSecond.StageID {
		want = append(want, newSecond.StageID)
	} else {
		want = append(want, newFirst.StageID)
	}
	sort.Strings(want)
	sort.Strings(envelope.Data.DuplicatedStageIDs)
	if envelope.Code != 0 || !reflect.DeepEqual(envelope.Data.DuplicatedStageIDs, want) {
		t.Fatalf("inspection = %#v, want duplicate stages %#v", envelope, want)
	}
}
