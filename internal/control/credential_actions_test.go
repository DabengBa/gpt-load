package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/subscription/providers/codex"
	subscriptionruntime "gpt-load/internal/subscription/runtime"

	"github.com/gin-gonic/gin"
)

func TestDownloadGroupCredentialReturnsCanonicalJSONAndSafeFilename(t *testing.T) {
	t.Parallel()

	fixture, groupID, credentialID := newSubscriptionCredentialFixture(t)
	result, err := fixture.service.DownloadGroupCredential(t.Context(), groupID, credentialID)
	if err != nil {
		t.Fatal(err)
	}
	if result.Filename != "codex-observation-example.com.json" {
		t.Fatalf("filename = %q", result.Filename)
	}
	var credential map[string]any
	if err := json.Unmarshal(result.Credential, &credential); err != nil {
		t.Fatalf("decode credential = %v", err)
	}
	if credential["access_token"] == nil || credential["refresh_token"] == nil ||
		credential["account_id"] != "account-observation" {
		t.Fatalf("credential = %#v", credential)
	}
}

func TestSubscriptionCredentialFilenameFallsBackToCredentialID(t *testing.T) {
	t.Parallel()
	if got := subscriptionCredentialFilename("Claude", "", 42); got != "claude-credential-42.json" {
		t.Fatalf("filename = %q", got)
	}
	if got := subscriptionCredentialFilename("Claude", "Admin+tag@example.com", 42); got != "claude-admin-tag-example.com.json" {
		t.Fatalf("filename = %q", got)
	}
}

func TestDownloadGroupCredentialHTTPReturnsJSONObjectAndNoStoreHeaders(t *testing.T) {
	t.Parallel()
	initControlI18n(t)
	fixture, groupID, credentialID := newSubscriptionCredentialFixture(t)
	server := NewServer(&config.Config{AuthKey: "credential-download-auth"}, fixture.service)
	engine := gin.New()
	server.RegisterRoutes(engine)

	response := serveCredentialRequest(
		t,
		engine,
		http.MethodPost,
		fmt.Sprintf("/api/groups/%d/credentials/%d/download", groupID, credentialID),
		"{}",
		"credential-download-auth",
		"",
	)
	if response.Code != http.StatusOK {
		t.Fatalf("download response = %d %s", response.Code, response.Body.String())
	}
	if response.Header().Get("Cache-Control") != "no-store" || response.Header().Get("Pragma") != "no-cache" {
		t.Fatalf("secret response headers = %#v", response.Header())
	}
	var envelope struct {
		Code int `json:"code"`
		Data struct {
			Filename   string         `json:"filename"`
			Credential map[string]any `json:"credential"`
		} `json:"data"`
	}
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Code != 0 || envelope.Data.Filename == "" || envelope.Data.Credential["access_token"] == nil {
		t.Fatalf("download envelope = %#v", envelope)
	}
}

func TestDownloadAllGroupCredentialsRejectsAPIKeyGroup(t *testing.T) {
	t.Parallel()

	fixture := newServiceFixture(t)
	groupID := createGroupForCredentialImport(t, fixture, "sk-download-all-forbidden")
	if _, err := fixture.service.DownloadAllGroupCredentials(t.Context(), groupID); !errors.Is(err, app_errors.ErrForbidden) {
		t.Fatalf("DownloadAllGroupCredentials() error = %v, want forbidden", err)
	}
}

func TestRefreshGroupCredentialOnlyRefreshesToken(t *testing.T) {
	t.Parallel()

	fixture, groupID, credentialID := newSubscriptionCredentialFixture(t)
	recoveryCalls := 0
	fixture.service.recoverSubscriptionCredential = func(
		_ context.Context,
		channelID channel.ID,
		snapshot execution.CredentialSnapshot,
	) (subscriptionruntime.Credential, *execution.ErrorEvidence) {
		recoveryCalls++
		if channelID != channel.Codex {
			return subscriptionruntime.Credential{}, &execution.ErrorEvidence{
				Kind: execution.ErrorKindInternal,
				Code: "unexpected_subscription_channel",
			}
		}
		credential, err := codex.ParseCredentialJSON(snapshot.Data())
		if err != nil {
			return subscriptionruntime.Credential{}, &execution.ErrorEvidence{
				Kind: execution.ErrorKindInternal,
				Code: "invalid_test_credential",
			}
		}
		runtimeCredential, err := testRuntimeCredential(fixture.service, credential)
		if err != nil {
			return subscriptionruntime.Credential{}, &execution.ErrorEvidence{
				Kind: execution.ErrorKindInternal,
				Code: "invalid_test_credential",
			}
		}
		return runtimeCredential, nil
	}
	observationCalls := 0
	setCodexAccountObservation(fixture.service, func(context.Context, codex.Credential) (codex.AccountObservation, error) {
		observationCalls++
		return codex.AccountObservation{Payload: []byte(`{"quota_windows":[]}`)}, nil
	})

	if _, err := fixture.service.RefreshGroupCredential(t.Context(), groupID, credentialID); err != nil {
		t.Fatal(err)
	}
	if recoveryCalls != 1 || observationCalls != 0 {
		t.Fatalf("recovery calls = %d, observation calls = %d", recoveryCalls, observationCalls)
	}
}

func TestRefreshGroupCredentialCanRetryAfterTemporaryFailure(t *testing.T) {
	t.Parallel()
	fixture, groupID, credentialID := newSubscriptionCredentialFixture(t)
	calls := 0
	fixture.service.recoverSubscriptionCredential = func(
		context.Context,
		channel.ID,
		execution.CredentialSnapshot,
	) (subscriptionruntime.Credential, *execution.ErrorEvidence) {
		calls++
		if calls == 1 {
			return subscriptionruntime.Credential{}, &execution.ErrorEvidence{
				Kind: execution.ErrorKindHTTP, Hint: execution.FailureHintRefreshUnavailable,
				StatusCode: http.StatusTooManyRequests, Code: "refresh_temporarily_unavailable",
			}
		}
		return subscriptionruntime.Credential{}, nil
	}

	if _, err := fixture.service.RefreshGroupCredential(t.Context(), groupID, credentialID); !errors.Is(err, app_errors.ErrCredentialRefreshTemporarilyUnavailable) {
		t.Fatalf("first RefreshGroupCredential() error = %v", err)
	}
	if _, err := fixture.service.RefreshGroupCredential(t.Context(), groupID, credentialID); err != nil || calls != 2 {
		t.Fatalf("second RefreshGroupCredential() error/calls = %v/%d", err, calls)
	}
}
