package control

import (
	"context"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

const bootstrapMarkerForTest = models.InternalSystemSettingPrefix + "bootstrap.default_access_key.v1"

func TestEnsureInitialStateDoesNotAutoCreateAccessKey(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	beforeRevision := fixture.manager.Current().Revision

	if err := fixture.service.EnsureInitialState(context.Background()); err != nil {
		t.Fatalf("EnsureInitialState() error = %v", err)
	}

	assertAccessKeyCount(t, fixture, 0)
	assertBootstrapMarkerCount(t, fixture, 0)
	if got := fixture.manager.Current().Revision; got != beforeRevision {
		t.Fatalf("snapshot revision = %d, want unchanged %d", got, beforeRevision)
	}
	if len(fixture.manager.Current().AccessKeysByHash) != 0 {
		t.Fatal("EnsureInitialState() published a runtime snapshot")
	}
}

func TestEnsureInitialStatePreservesExistingAccessKey(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	const plaintext = "gl-existing-access-key"
	ciphertext, err := fixture.encryption.Encrypt(plaintext)
	if err != nil {
		t.Fatalf("Encrypt(existing key) error = %v", err)
	}
	existing := models.AccessKey{
		Name:      "Existing",
		KeyValue:  ciphertext,
		KeyHash:   fixture.encryption.Hash(plaintext),
		KeySuffix: "0000",
		Status:    string(state.AccessKeyStatusActive),
		Filters:   models.JSON(`{"groups":[],"protocols":[],"models":[]}`),
	}
	if err := fixture.db.Create(&existing).Error; err != nil {
		t.Fatalf("create existing AccessKey: %v", err)
	}

	if err := fixture.service.EnsureInitialState(context.Background()); err != nil {
		t.Fatalf("EnsureInitialState() error = %v", err)
	}

	var rows []models.AccessKey
	if err := fixture.db.Order("id ASC").Find(&rows).Error; err != nil {
		t.Fatalf("query AccessKeys: %v", err)
	}
	if len(rows) != 1 || rows[0].ID != existing.ID {
		t.Fatalf("AccessKeys = %#v, want only existing row", rows)
	}
	decrypted, err := fixture.encryption.Decrypt(rows[0].KeyValue)
	if err != nil || decrypted != plaintext {
		t.Fatalf("existing credential = %q, %v, want unchanged", decrypted, err)
	}
	assertBootstrapMarkerCount(t, fixture, 0)
}

func TestEnsureInitialStateDoesNotRecreateDeletedFinalKey(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	if err := fixture.service.EnsureInitialState(context.Background()); err != nil {
		t.Fatalf("first EnsureInitialState() error = %v", err)
	}
	if err := fixture.db.Where("1 = 1").Delete(&models.AccessKey{}).Error; err != nil {
		t.Fatalf("delete final AccessKey: %v", err)
	}

	if err := fixture.service.EnsureInitialState(context.Background()); err != nil {
		t.Fatalf("second EnsureInitialState() error = %v", err)
	}
	assertAccessKeyCount(t, fixture, 0)
	assertBootstrapMarkerCount(t, fixture, 0)
}

func TestEnsureInitialStateRecoversInterruptedSubscriptionAuth(t *testing.T) {
	t.Parallel()

	fixture := newServiceFixture(t)
	now := time.UnixMilli(1_800_000_000_000)
	fixture.service.now = func() time.Time { return now }
	group := validControlGroup("subscription-recovery")
	group.ChannelID = string(channel.Codex)
	group.ConnectionType = models.ConnectionTypeSubscription
	if err := fixture.db.Create(group).Error; err != nil {
		t.Fatal(err)
	}
	credential := models.Credential{
		GroupID: group.ID, Data: "cipher", Fingerprint: "secret", IdentityFingerprint: "identity",
		SecretVersion: 2, AuthState: models.CredentialAuthStateRefreshing, CreatedAtMS: now.Add(-time.Hour).UnixMilli(), UpdatedAtMS: now.Add(-time.Hour).UnixMilli(),
	}
	if err := fixture.db.Create(&credential).Error; err != nil {
		t.Fatal(err)
	}
	stage := models.CredentialStage{
		ID: "00000000-0000-4000-8000-000000000901", ChannelID: "codex",
		ConnectionType: models.ConnectionTypeSubscription, AuthorizationMethod: "browser_oauth",
		Status: models.CredentialStageExchanging, EncryptedPayload: "encrypted-stage", PayloadSchemaVersion: 1,
		SafeSummaryJSON: models.JSON(`{}`), ExpiresAtMS: now.Add(time.Minute).UnixMilli(),
		CreatedAtMS: now.Add(-time.Minute).UnixMilli(), UpdatedAtMS: now.Add(-time.Minute).UnixMilli(),
	}
	if err := fixture.db.Create(&stage).Error; err != nil {
		t.Fatal(err)
	}
	if err := fixture.service.EnsureInitialState(t.Context()); err != nil {
		t.Fatal(err)
	}
	if err := fixture.db.Take(&credential, credential.ID).Error; err != nil {
		t.Fatal(err)
	}
	if credential.AuthState != models.CredentialAuthStateOutcomeUnknown || credential.AuthErrorCode != "refresh_interrupted" {
		t.Fatalf("credential = %#v", credential)
	}
	if err := fixture.db.Take(&stage, "id = ?", stage.ID).Error; err != nil {
		t.Fatal(err)
	}
	if stage.Status != models.CredentialStageOutcomeUnknown || stage.EncryptedPayload != "" || stage.ErrorCode != "authorization_exchange_interrupted" {
		t.Fatalf("stage = %#v", stage)
	}
}

func assertAccessKeyCount(t *testing.T, fixture serviceFixture, want int64) {
	t.Helper()
	var count int64
	if err := fixture.db.Model(&models.AccessKey{}).Count(&count).Error; err != nil {
		t.Fatalf("count AccessKeys: %v", err)
	}
	if count != want {
		t.Fatalf("AccessKey count = %d, want %d", count, want)
	}
}

func assertBootstrapMarkerCount(t *testing.T, fixture serviceFixture, want int64) {
	t.Helper()
	var count int64
	if err := fixture.db.Model(&models.SystemSetting{}).
		Where("key = ?", bootstrapMarkerForTest).Count(&count).Error; err != nil {
		t.Fatalf("count bootstrap marker: %v", err)
	}
	if count != want {
		t.Fatalf("bootstrap marker count = %d, want %d", count, want)
	}
}

func assertModelPriceCount(t *testing.T, fixture serviceFixture, want int64) {
	t.Helper()
	var count int64
	if err := fixture.db.Model(&models.ModelPrice{}).Count(&count).Error; err != nil {
		t.Fatalf("count ModelPrice rows: %v", err)
	}
	if count != want {
		t.Fatalf("ModelPrice count = %d, want %d", count, want)
	}
}
