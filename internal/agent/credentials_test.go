package agent

import (
	"context"
	"strings"
	"testing"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/platform/encryption"
	"gpt-load/internal/storage/models"
	"gpt-load/internal/testutil/encryptiontest"
	"gpt-load/internal/testutil/sqlitetest"
)

const agentTestKeyMaterial = "agent-credential-test-key-material-2026"

func newAgentTestStore(t *testing.T) (*CredentialStore, *gorm.DB) {
	t.Helper()
	db := sqlitetest.OpenMigrated(t)
	store := NewCredentialStore(db, encryptiontest.Service(t, agentTestKeyMaterial))
	return store, db
}

func createTestCredential(
	t *testing.T,
	store *CredentialStore,
	input CredentialCreateInput,
) (CredentialMetadata, string) {
	t.Helper()
	var metadata CredentialMetadata
	var secret string
	err := store.db.Transaction(func(tx *gorm.DB) error {
		var createErr error
		metadata, secret, createErr = store.CreateInTx(tx, input)
		return createErr
	})
	if err != nil {
		t.Fatalf("CreateInTx() error = %v", err)
	}
	if secret == "" {
		t.Fatal("CreateInTx() returned an empty secret")
	}
	return metadata, secret
}

func TestNormalizeScopesCanonicalizesAndRejectsBadInput(t *testing.T) {
	t.Parallel()
	if _, err := NormalizeScopes(nil); err == nil {
		t.Fatal("NormalizeScopes(nil) error = nil, want rejection")
	}
	if _, err := NormalizeScopes([]string{"diagnostics:read", "diagnostics:read"}); err == nil {
		t.Fatal("NormalizeScopes(duplicate) error = nil, want rejection")
	}
	if _, err := NormalizeScopes([]string{"evidence:raw"}); err == nil {
		t.Fatal("NormalizeScopes(unknown) error = nil, want rejection")
	}
	scopes, err := NormalizeScopes([]string{"changes:apply", "diagnostics:read"})
	if err != nil {
		t.Fatalf("NormalizeScopes() error = %v", err)
	}
	if len(scopes) != 2 || scopes[0] != ScopeDiagnosticsRead || scopes[1] != ScopeChangesApply {
		t.Fatalf("NormalizeScopes() = %v, want canonical grantable order", scopes)
	}
}

func TestAgentCredentialStorePersistsOnlyTheSecretDigest(t *testing.T) {
	t.Parallel()
	store, db := newAgentTestStore(t)
	metadata, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "diagnostics agent",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	if !strings.HasPrefix(secret, agentSecretPrefix) {
		t.Fatalf("secret = %q, want prefix %q", secret, agentSecretPrefix)
	}
	if len(secret) != agentSecretLength {
		t.Fatalf("len(secret) = %d, want %d", len(secret), agentSecretLength)
	}
	var row models.AgentCredential
	if err := db.First(&row, "id = ?", metadata.ID).Error; err != nil {
		t.Fatalf("load persisted credential: %v", err)
	}
	if row.SecretHash == secret {
		t.Fatal("persisted SecretHash equals the plaintext secret")
	}
	service, err := encryption.NewService(agentTestKeyMaterial)
	if err != nil {
		t.Fatalf("encryption.NewService() error = %v", err)
	}
	if row.SecretHash != service.Hash(secret) {
		t.Fatal("persisted SecretHash is not the HMAC digest of the secret")
	}
	if strings.Contains(string(row.Scopes), secret) {
		t.Fatal("persisted scopes contain plaintext secret material")
	}
	if metadata.Status != CredentialStatusActive {
		t.Fatalf("metadata.Status = %q, want %q", metadata.Status, CredentialStatusActive)
	}
}

func TestAuthenticateFailsClosedForEveryRejectedCredential(t *testing.T) {
	t.Parallel()
	store, db := newAgentTestStore(t)
	_, activeSecret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "active",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	disabled, disabledSecret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "disabled",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	if _, err := store.Disable(context.Background(), disabled.ID); err != nil {
		t.Fatalf("Disable() error = %v", err)
	}
	pastMS := time.Now().Add(-time.Hour).UnixMilli()
	_, expiredSecret := createTestCredential(t, store, CredentialCreateInput{
		Name:        "expired",
		Scopes:      []Scope{ScopeDiagnosticsRead},
		ExpiresAtMS: &pastMS,
	})
	unknown := agentSecretPrefix + strings.Repeat("a", agentSecretLength-len(agentSecretPrefix))

	rejected := [][2]string{
		{"unknown", unknown},
		{"disabled", disabledSecret},
		{"expired", expiredSecret},
		{"empty", ""},
		{"admin", "sk-admin-control-key"},
		{"short", agentSecretPrefix + "abc"},
		// Same length and prefix as a real secret, but the digest lookup
		// cannot match, so the outcome must still be a rejection.
		{"wrong-suffix", agentSecretPrefix + strings.Repeat("Z", agentSecretLength-len(agentSecretPrefix))},
	}
	for _, testCase := range rejected {
		if _, err := store.Authenticate(context.Background(), testCase[1]); err == nil {
			t.Fatalf("Authenticate(%s token) error = nil, want rejection", testCase[0])
		}
	}
	principal, err := store.Authenticate(context.Background(), activeSecret)
	if err != nil {
		t.Fatalf("Authenticate(active) error = %v", err)
	}
	if !principal.HasScope(ScopeDiagnosticsRead) || principal.HasScope(ScopeChangesApply) {
		t.Fatalf("principal scopes = %v, want exactly diagnostics:read", principal.Scopes)
	}
	var persistedDisabled models.AgentCredential
	if err := db.First(&persistedDisabled, "id = ?", disabled.ID).Error; err != nil {
		t.Fatalf("load disabled credential: %v", err)
	}
	if persistedDisabled.DisabledAtMS == nil {
		t.Fatal("disabled credential is missing disabled_at_ms")
	}
}

func TestDisableIsIdempotentAndReportsUnknownCredentialsAsNotFound(t *testing.T) {
	t.Parallel()
	store, _ := newAgentTestStore(t)
	metadata, _ := createTestCredential(t, store, CredentialCreateInput{
		Name:   "revocable",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	first, err := store.Disable(context.Background(), metadata.ID)
	if err != nil {
		t.Fatalf("first Disable() error = %v", err)
	}
	if first.Status != CredentialStatusDisabled || first.DisabledAtMS == nil {
		t.Fatalf("first Disable() = %#v, want disabled with timestamp", first)
	}
	second, err := store.Disable(context.Background(), metadata.ID)
	if err != nil {
		t.Fatalf("second Disable() error = %v", err)
	}
	if second.DisabledAtMS == nil || *second.DisabledAtMS != *first.DisabledAtMS {
		t.Fatal("second Disable() changed the disabled timestamp")
	}
	if _, err := store.Disable(context.Background(), metadata.ID+1000); err == nil {
		t.Fatal("Disable(unknown) error = nil, want not found")
	}
}

func TestOperationResultNeverContainsThePlaintextSecret(t *testing.T) {
	t.Parallel()
	store, _ := newAgentTestStore(t)
	_, secret := createTestCredential(t, store, CredentialCreateInput{
		Name:   "ledger",
		Scopes: []Scope{ScopeDiagnosticsRead},
	})
	items, err := store.List(context.Background())
	if err != nil {
		t.Fatalf("List() error = %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("len(List()) = %d, want 1", len(items))
	}
	if len(items) != 1 {
		t.Fatalf("len(List()) = %d, want 1", len(items))
	}
	if items[0].Name != "ledger" {
		t.Fatalf("List()[0].Name = %q, want the safe credential name", items[0].Name)
	}
	if strings.Contains(items[0].Name, secret) {
		t.Fatal("credential metadata exposes secret material")
	}
}
