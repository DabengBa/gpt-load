package agent

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/platform/encryption"
	"gpt-load/internal/platform/epochms"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

const (
	agentSecretPrefix = "gla_"
	agentSecretBytes  = 32
	agentSecretLength = len(agentSecretPrefix) + 43
)

// errAgentCredentialInvalid collapses every authentication failure into one
// fail-closed outcome so callers cannot probe credential state.
var errAgentCredentialInvalid = errors.New("agent credential is invalid")

// Principal is the resolved Agent identity for one request.
type Principal struct {
	CredentialID uint
	Name         string
	Scopes       []Scope
	ExpiresAtMS  *int64

	scopeSet map[Scope]struct{}
}

// HasScope reports whether the caller holds a grantable scope.
func (principal Principal) HasScope(scope Scope) bool {
	if principal.scopeSet == nil {
		return false
	}
	_, ok := principal.scopeSet[scope]
	return ok
}

// CredentialCreateInput is the validated, normalized create request.
type CredentialCreateInput struct {
	Name        string
	Scopes      []Scope
	ExpiresAtMS *int64
}

// CredentialStore persists and authenticates independent Agent credentials.
// It never stores or logs the plaintext secret.
type CredentialStore struct {
	db         *gorm.DB
	encryption encryption.Service
	random     io.Reader
	now        func() time.Time
}

// NewCredentialStore builds the Agent credential store over the shared database.
func NewCredentialStore(db *gorm.DB, encryptionService encryption.Service) *CredentialStore {
	return &CredentialStore{
		db:         db,
		encryption: encryptionService,
		random:     rand.Reader,
		now:        time.Now,
	}
}

// NormalizeScopes validates requested scopes and returns the canonical order.
// Empty, unknown, and duplicate scopes are rejected.
func NormalizeScopes(raw []string) ([]Scope, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("at least one agent scope is required")
	}
	seen := make(map[Scope]struct{}, len(raw))
	for _, value := range raw {
		scope := Scope(value)
		if !scope.Valid() {
			return nil, fmt.Errorf("agent scope %q is not grantable", value)
		}
		if _, exists := seen[scope]; exists {
			return nil, fmt.Errorf("agent scope %q is duplicated", value)
		}
		seen[scope] = struct{}{}
	}
	result := make([]Scope, 0, len(seen))
	for _, scope := range GrantableScopes {
		if _, exists := seen[scope]; exists {
			result = append(result, scope)
		}
	}
	return result, nil
}

func newAgentSecret(random io.Reader) (string, error) {
	if random == nil {
		return "", fmt.Errorf("generate agent secret: random source is nil")
	}
	value := make([]byte, agentSecretBytes)
	if _, err := io.ReadFull(random, value); err != nil {
		return "", fmt.Errorf("generate agent secret: %w", err)
	}
	return agentSecretPrefix + base64.RawURLEncoding.EncodeToString(value), nil
}

func encodeScopes(scopes []Scope) (models.JSON, error) {
	values := make([]string, 0, len(scopes))
	for _, scope := range scopes {
		values = append(values, string(scope))
	}
	raw, err := json.Marshal(values)
	if err != nil {
		return nil, fmt.Errorf("encode agent scopes: %w", err)
	}
	return models.JSON(raw), nil
}

func metadataFromRow(row models.AgentCredential) (CredentialMetadata, error) {
	var raw []string
	if err := json.Unmarshal(row.Scopes, &raw); err != nil {
		return CredentialMetadata{}, fmt.Errorf("decode agent scopes: %w", err)
	}
	scopes, err := NormalizeScopes(raw)
	if err != nil {
		return CredentialMetadata{}, fmt.Errorf("decode agent scopes: %w", err)
	}
	return CredentialMetadata{
		ID:           row.ID,
		Name:         row.Name,
		Scopes:       scopes,
		Status:       row.Status,
		ExpiresAtMS:  row.ExpiresAtMS,
		DisabledAtMS: row.DisabledAtMS,
		CreatedAtMS:  row.CreatedAtMS,
		UpdatedAtMS:  row.UpdatedAtMS,
	}, nil
}

// CreateInTx persists a new credential inside the caller's transaction and
// returns the plaintext secret exactly once. Only the HMAC digest is stored.
func (store *CredentialStore) CreateInTx(
	tx *gorm.DB,
	input CredentialCreateInput,
) (CredentialMetadata, string, error) {
	if store == nil || store.db == nil || store.encryption == nil {
		return CredentialMetadata{}, "", app_errors.ErrInternalServer
	}
	if tx == nil {
		return CredentialMetadata{}, "", app_errors.ErrInternalServer
	}
	secret, err := newAgentSecret(store.random)
	if err != nil {
		return CredentialMetadata{}, "", app_errors.ErrInternalServer
	}
	secretHash := store.encryption.Hash(secret)
	if secretHash == "" {
		return CredentialMetadata{}, "", app_errors.ErrInternalServer
	}
	scopes, err := encodeScopes(input.Scopes)
	if err != nil {
		return CredentialMetadata{}, "", app_errors.ErrInternalServer
	}
	nowMS, err := epochms.FromTime(store.now())
	if err != nil {
		return CredentialMetadata{}, "", app_errors.ErrInternalServer
	}
	row := models.AgentCredential{
		Name:        input.Name,
		SecretHash:  secretHash,
		Scopes:      scopes,
		Status:      CredentialStatusActive,
		ExpiresAtMS: cloneInt64(input.ExpiresAtMS),
		CreatedAtMS: nowMS,
		UpdatedAtMS: nowMS,
	}
	if err := tx.Create(&row).Error; err != nil {
		return CredentialMetadata{}, "", app_errors.ParseDBError(err)
	}
	metadata, err := metadataFromRow(row)
	if err != nil {
		return CredentialMetadata{}, "", app_errors.ErrInternalServer
	}
	return metadata, secret, nil
}

// Disable marks a credential disabled. Disabling an already disabled or
// unknown credential is reported as not found for unknown IDs and is
// idempotent for already disabled IDs.
func (store *CredentialStore) Disable(ctx context.Context, id uint) (CredentialMetadata, error) {
	if store == nil || store.db == nil || id == 0 {
		return CredentialMetadata{}, app_errors.ErrResourceNotFound
	}
	nowMS, err := epochms.FromTime(store.now())
	if err != nil {
		return CredentialMetadata{}, app_errors.ErrInternalServer
	}
	err = store.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		var row models.AgentCredential
		if err := tx.First(&row, "id = ?", id).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return errAgentCredentialNotFound
			}
			return err
		}
		if row.Status == CredentialStatusActive {
			if err := tx.Model(&models.AgentCredential{}).
				Where("id = ? AND status = ?", id, CredentialStatusActive).
				Updates(map[string]any{
					"status":         CredentialStatusDisabled,
					"disabled_at_ms": nowMS,
					"updated_at_ms":  nowMS,
				}).Error; err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		if errors.Is(err, errAgentCredentialNotFound) {
			return CredentialMetadata{}, app_errors.ErrResourceNotFound
		}
		return CredentialMetadata{}, app_errors.ParseDBError(err)
	}
	return store.Get(ctx, id)
}

var errAgentCredentialNotFound = errors.New("agent credential not found")

// Get returns one credential's safe metadata.
func (store *CredentialStore) Get(ctx context.Context, id uint) (CredentialMetadata, error) {
	if store == nil || store.db == nil || id == 0 {
		return CredentialMetadata{}, app_errors.ErrResourceNotFound
	}
	var row models.AgentCredential
	if err := store.db.WithContext(ctx).First(&row, "id = ?", id).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return CredentialMetadata{}, app_errors.ErrResourceNotFound
		}
		return CredentialMetadata{}, app_errors.ParseDBError(err)
	}
	metadata, err := metadataFromRow(row)
	if err != nil {
		return CredentialMetadata{}, app_errors.ErrInternalServer
	}
	return metadata, nil
}

// List returns every credential's safe metadata ordered by ID.
func (store *CredentialStore) List(ctx context.Context) ([]CredentialMetadata, error) {
	if store == nil || store.db == nil {
		return nil, app_errors.ErrInternalServer
	}
	var rows []models.AgentCredential
	if err := store.db.WithContext(ctx).Order("id ASC").Find(&rows).Error; err != nil {
		return nil, app_errors.ParseDBError(err)
	}
	result := make([]CredentialMetadata, 0, len(rows))
	for _, row := range rows {
		metadata, err := metadataFromRow(row)
		if err != nil {
			return nil, app_errors.ErrInternalServer
		}
		result = append(result, metadata)
	}
	return result, nil
}

// Authenticate resolves a Bearer secret to a Principal. Every failure mode
// (unknown, malformed, disabled, expired, corrupt scopes) is fail-closed.
func (store *CredentialStore) Authenticate(ctx context.Context, token string) (Principal, error) {
	if store == nil || store.db == nil || store.encryption == nil {
		return Principal{}, app_errors.ErrInternalServer
	}
	if len(token) != agentSecretLength || token[:len(agentSecretPrefix)] != agentSecretPrefix {
		return Principal{}, errAgentCredentialInvalid
	}
	fingerprint := store.encryption.Hash(token)
	if fingerprint == "" {
		return Principal{}, errAgentCredentialInvalid
	}
	var row models.AgentCredential
	if err := store.db.WithContext(ctx).
		Where("secret_hash = ?", fingerprint).
		Take(&row).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return Principal{}, errAgentCredentialInvalid
		}
		return Principal{}, app_errors.ParseDBError(err)
	}
	if row.Status != CredentialStatusActive {
		return Principal{}, errAgentCredentialInvalid
	}
	if row.ExpiresAtMS != nil && store.now().UnixMilli() >= *row.ExpiresAtMS {
		return Principal{}, errAgentCredentialInvalid
	}
	metadata, err := metadataFromRow(row)
	if err != nil {
		return Principal{}, errAgentCredentialInvalid
	}
	return principalFromMetadata(metadata), nil
}

func principalFromMetadata(metadata CredentialMetadata) Principal {
	scopeSet := make(map[Scope]struct{}, len(metadata.Scopes))
	for _, scope := range metadata.Scopes {
		scopeSet[scope] = struct{}{}
	}
	return Principal{
		CredentialID: metadata.ID,
		Name:         metadata.Name,
		Scopes:       append([]Scope(nil), metadata.Scopes...),
		ExpiresAtMS:  cloneInt64(metadata.ExpiresAtMS),
		scopeSet:     scopeSet,
	}
}

func cloneInt64(value *int64) *int64 {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}
