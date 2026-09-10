package control

import (
	"context"
	"testing"
	"time"

	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

// 首页的「X/Y 个凭据可用」必须和健康页 classifyHealthKey 用同一套分桶：
// 待重新授权的凭据不参与调度，两页并排时必须保持同一结论。
func TestReadHomeBaseAvailableCredentialsMatchHealthClassification(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	now := time.Date(2026, time.August, 16, 9, 0, 0, 0, time.UTC)

	groups := []*models.Group{
		validControlGroup("home-available-parity"),
		validControlGroup("home-reauthorization-one"),
		validControlGroup("home-reauthorization-two"),
	}
	for _, group := range groups {
		if err := fixture.db.Create(group).Error; err != nil {
			t.Fatalf("create group: %v", err)
		}
	}

	credentials := []models.Credential{
		{ID: 1, GroupID: groups[0].ID, Data: "cipher-1", Fingerprint: "hash-1", AuthState: models.CredentialAuthStateReady},
		{ID: 2, GroupID: groups[1].ID, Data: "cipher-2", Fingerprint: "hash-2", AuthState: models.CredentialAuthStateReauthorizationRequired},
		{ID: 3, GroupID: groups[2].ID, Data: "cipher-3", Fingerprint: "hash-3", AuthState: models.CredentialAuthStateReauthorizationRequired},
	}
	if err := fixture.db.Create(&credentials).Error; err != nil {
		t.Fatalf("create credentials: %v", err)
	}
	if err := fixture.db.Order("id ASC").Find(&credentials).Error; err != nil {
		t.Fatalf("reload credentials: %v", err)
	}

	entries := make([]state.CredentialEntry, 0, len(credentials))
	for index, credential := range credentials {
		group := groups[index]
		entries = append(entries, state.CredentialEntry{
			ID: credential.ID, GroupID: group.ID,
			Version:            groupCollectionCredentialVersion(credential.SecretVersion),
			IdentityGeneration: groupCollectionCredentialIdentity(credential.IdentityFingerprint, *group),
			Fingerprint:        credential.Fingerprint,
			EncryptedValue:     "cipher-" + string(rune('1'+index)),
			AuthState:          state.CredentialAuthStateReady,
		})
	}
	// 2、3 号待重新授权；统一分组凭据模型下，每个状态使用独立分组承载。
	entries[1].AuthState = state.CredentialAuthStateReauthorizationRequired
	entries[2].AuthState = state.CredentialAuthStateReauthorizationRequired
	if err := fixture.registry.ReplaceCredentials(entries); err != nil {
		t.Fatalf("registry.ReplaceCredentials() error = %v", err)
	}

	input, err := stateloader.BuildCompileInput(context.Background(), fixture.db, fixture.channelRegistry)
	if err != nil {
		t.Fatalf("BuildCompileInput() error = %v", err)
	}
	if _, err := fixture.manager.Publish(input); err != nil {
		t.Fatalf("manager.Publish() error = %v", err)
	}
	fixture.service.registrySnapshot = fixture.registry.Snapshot

	base, err := fixture.service.ReadHomeBase(context.Background(), now.UnixMilli())
	if err != nil {
		t.Fatalf("ReadHomeBase() error = %v", err)
	}

	if base.Inventory.CredentialCount != 3 {
		t.Fatalf("CredentialCount = %d, want 3", base.Inventory.CredentialCount)
	}
	if base.Inventory.AvailableCredentialCount != 1 {
		t.Fatalf(
			"AvailableCredentialCount = %d, want 1 (待重新授权的凭据不可用)",
			base.Inventory.AvailableCredentialCount,
		)
	}

	// 与健康页的分桶逐条比对，确保两处结论一致而不只是数字凑巧相等。
	snapshot := fixture.manager.Current()
	var healthAvailable int64
	for _, view := range fixture.registry.Snapshot() {
		catalog, ok := snapshot.GroupCatalog[view.GroupID]
		if !ok {
			t.Fatalf("group %d missing from catalog", view.GroupID)
		}
		if classifyHealthKey(catalog, view, now) == healthBucketAvailable {
			healthAvailable++
		}
	}
	if base.Inventory.AvailableCredentialCount != healthAvailable {
		t.Fatalf(
			"home available = %d, health available = %d; 两处口径必须一致",
			base.Inventory.AvailableCredentialCount,
			healthAvailable,
		)
	}
}
