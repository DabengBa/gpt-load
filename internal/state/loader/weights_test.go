package loader_test

import (
	"context"
	"testing"

	"gpt-load/internal/state"
	"gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

func TestBuildGroupCredentialEntriesRejectsMultipleCredentials(t *testing.T) {
	db := openMigratedDatabase(t)
	group := models.Group{Name: "group", ChannelID: "openai", Params: models.JSON(`{}`), Models: models.JSON(`[]`), Enabled: true}
	mustCreate(t, db, &group)
	mustCreate(t, db, &models.Credential{GroupID: group.ID, Data: "one", Fingerprint: "one", IdentityFingerprint: "one", AuthState: models.CredentialAuthStateReady})
	mustCreate(t, db, &models.Credential{GroupID: group.ID, Data: "two", Fingerprint: "two", IdentityFingerprint: "two", AuthState: models.CredentialAuthStateReady})
	_, err := loader.BuildGroupCredentialEntries(context.Background(), db, group.ID)
	if err == nil {
		t.Fatal("BuildGroupCredentialEntries() error = nil")
	}
	_ = state.CredentialAuthStateReady
}
