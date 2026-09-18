package storage

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"gorm.io/gorm"
	"gorm.io/gorm/clause"

	"gpt-load/internal/affinity"
	"gpt-load/internal/storage/models"
)

const affinityBindingSettingPrefix = models.InternalSystemSettingPrefix + "affinity.binding."

type affinityBindingValue struct {
	GroupID            uint   `json:"group_id"`
	CredentialID       uint   `json:"credential_id"`
	IdentityGeneration uint64 `json:"identity_generation"`
}

// AffinityStore persists one binding per HMAC-derived affinity key in the
// existing system_settings table. The raw key is never exposed in errors.
type AffinityStore struct {
	db *gorm.DB
}

func NewAffinityStore(db *gorm.DB) *AffinityStore {
	return &AffinityStore{db: db}
}

func (store *AffinityStore) Lookup(ctx context.Context, key affinity.Key) (affinity.Target, bool, error) {
	if err := validateAffinityKey(key); err != nil {
		return affinity.Target{}, false, err
	}
	if store == nil || store.db == nil {
		return affinity.Target{}, false, errors.New("affinity store is unavailable")
	}
	var row models.SystemSetting
	err := store.db.WithContext(ctx).Select("key", "value").
		Where("key = ?", affinityBindingSettingPrefix+string(key)).Take(&row).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return affinity.Target{}, false, nil
	}
	if err != nil {
		return affinity.Target{}, false, fmt.Errorf("lookup affinity binding: %w", err)
	}
	value, err := decodeAffinityBinding(row.Value)
	if err != nil {
		return affinity.Target{}, false, fmt.Errorf("decode affinity binding: %w", err)
	}
	return affinity.Target{GroupID: value.GroupID, CredentialID: value.CredentialID, IdentityGeneration: value.IdentityGeneration}, true, nil
}

func (store *AffinityStore) Upsert(ctx context.Context, key affinity.Key, target affinity.Target) error {
	if err := validateAffinityKey(key); err != nil {
		return err
	}
	if !target.Valid() {
		return errors.New("invalid affinity target")
	}
	if store == nil || store.db == nil {
		return errors.New("affinity store is unavailable")
	}
	payload, err := json.Marshal(affinityBindingValue{GroupID: target.GroupID, CredentialID: target.CredentialID, IdentityGeneration: target.IdentityGeneration})
	if err != nil {
		return fmt.Errorf("encode affinity binding: %w", err)
	}
	row := models.SystemSetting{
		Key: affinityBindingSettingPrefix + string(key), Value: string(payload),
		UpdatedAtMS: time.Now().UnixMilli(),
	}
	if err := store.db.WithContext(ctx).Clauses(clause.OnConflict{
		Columns:   []clause.Column{{Name: "key"}},
		DoUpdates: clause.AssignmentColumns([]string{"value", "updated_at_ms"}),
	}).Create(&row).Error; err != nil {
		return fmt.Errorf("upsert affinity binding: %w", err)
	}
	return nil
}

func validateAffinityKey(key affinity.Key) error {
	value := string(key)
	if len(value) != 64 {
		return errors.New("invalid affinity key")
	}
	for index := range value {
		character := value[index]
		if !((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f')) {
			return errors.New("invalid affinity key")
		}
	}
	return nil
}

func decodeAffinityBinding(raw string) (affinityBindingValue, error) {
	var value affinityBindingValue
	if err := json.Unmarshal([]byte(raw), &value); err != nil {
		return affinityBindingValue{}, err
	}
	if value.GroupID == 0 || value.CredentialID == 0 || value.IdentityGeneration == 0 {
		return affinityBindingValue{}, errors.New("invalid affinity target")
	}
	return value, nil
}
