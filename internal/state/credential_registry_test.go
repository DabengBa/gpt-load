package state

import "testing"

func TestCredentialRegistryKeepsAuthAndIdentityMetadata(t *testing.T) {
	registry := NewCredentialRegistry()
	entry := testCredential(1, 1)
	entry.AuthState = CredentialAuthStateRefreshing
	if err := registry.ReplaceCredentials([]CredentialEntry{entry}); err != nil {
		t.Fatal(err)
	}
	if got, ok := registry.CredentialAuthStateOf(1); !ok || got != CredentialAuthStateRefreshing {
		t.Fatalf("auth state = %q, %v", got, ok)
	}
}
