package keypool

import (
	"fmt"
	"strconv"

	"gpt-load/internal/models"

	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
)

// KeyProviderProcessor adapts KeyProvider to implement TaskProcessor interface
type KeyProviderProcessor struct {
	provider *KeyProvider
}

// NewKeyProviderProcessor creates a new processor that wraps KeyProvider
func NewKeyProviderProcessor(provider *KeyProvider) *KeyProviderProcessor {
	return &KeyProviderProcessor{provider: provider}
}

// ProcessSuccess handles successful key usage - resets failure count
// Uses cache-first strategy: update cache first, then DB, rollback cache on DB failure
func (p *KeyProviderProcessor) ProcessSuccess(keyID uint, keyHashKey, activeKeysListKey string) error {
	keyDetails, err := p.provider.store.HGetAll(keyHashKey)
	if err != nil {
		return fmt.Errorf("failed to get key details from store: %w", err)
	}

	failureCount, _ := strconv.ParseInt(keyDetails["failure_count"], 10, 64)
	oldStatus := keyDetails["status"]
	isActive := oldStatus == models.KeyStatusActive

	// Skip if already in good state
	if failureCount == 0 && isActive {
		return nil
	}

	// Prepare updates
	updates := map[string]any{"failure_count": int64(0)}
	needRestoreToActive := !isActive

	if needRestoreToActive {
		updates["status"] = models.KeyStatusActive
	}

	// Step 1: Update cache first (optimistic)
	if err := p.provider.store.HSet(keyHashKey, updates); err != nil {
		return fmt.Errorf("failed to update key details in store: %w", err)
	}

	// If key needs to be restored to active pool, do it now
	if needRestoreToActive {
		if err := p.provider.store.LRem(activeKeysListKey, 0, keyID); err != nil {
			// Rollback cache
			p.rollbackCacheSuccess(keyHashKey, failureCount, oldStatus)
			return fmt.Errorf("failed to LRem key before LPush on recovery: %w", err)
		}
		if err := p.provider.store.LPush(activeKeysListKey, keyID); err != nil {
			// Rollback cache
			p.rollbackCacheSuccess(keyHashKey, failureCount, oldStatus)
			return fmt.Errorf("failed to LPush key back to active list: %w", err)
		}
	}

	// Step 2: Update database
	dbErr := p.provider.executeTransactionWithRetry(func(tx *gorm.DB) error {
		var key models.APIKey
		if err := tx.Set("gorm:query_option", "FOR UPDATE").First(&key, keyID).Error; err != nil {
			return fmt.Errorf("failed to lock key %d for update: %w", keyID, err)
		}

		dbUpdates := map[string]any{"failure_count": 0}
		if needRestoreToActive {
			dbUpdates["status"] = models.KeyStatusActive
		}

		if err := tx.Model(&key).Updates(dbUpdates).Error; err != nil {
			return fmt.Errorf("failed to update key in DB: %w", err)
		}

		return nil
	})

	// Step 3: If DB fails, rollback cache
	if dbErr != nil {
		logrus.WithFields(logrus.Fields{
			"keyID": keyID,
			"error": dbErr,
		}).Warn("DB update failed, rolling back cache")

		p.rollbackCacheSuccess(keyHashKey, failureCount, oldStatus)

		// If we added to active list, remove it
		if needRestoreToActive {
			p.provider.store.LRem(activeKeysListKey, 0, keyID)
		}

		return dbErr
	}

	if needRestoreToActive {
		logrus.WithField("keyID", keyID).Debug("Key has recovered and is being restored to active pool.")
	}

	return nil
}

// rollbackCacheSuccess restores cache to previous state after a failed success update
func (p *KeyProviderProcessor) rollbackCacheSuccess(keyHashKey string, oldFailureCount int64, oldStatus string) {
	rollback := map[string]any{
		"failure_count": oldFailureCount,
		"status":        oldStatus,
	}
	if err := p.provider.store.HSet(keyHashKey, rollback); err != nil {
		logrus.WithFields(logrus.Fields{
			"keyHashKey": keyHashKey,
			"error":      err,
		}).Error("Failed to rollback cache after DB failure")
	}
}


// ProcessFailure handles failed key usage - increments failure count and potentially blacklists
// Uses cache-first strategy: update cache first, then DB, rollback cache on DB failure
func (p *KeyProviderProcessor) ProcessFailure(task *StatusUpdateTask, keyHashKey, activeKeysListKey string) error {
	keyDetails, err := p.provider.store.HGetAll(keyHashKey)
	if err != nil {
		return fmt.Errorf("failed to get key details from store: %w", err)
	}

	// Skip if already invalid
	oldStatus := keyDetails["status"]
	if oldStatus == models.KeyStatusInvalid {
		return nil
	}

	oldFailureCount, _ := strconv.ParseInt(keyDetails["failure_count"], 10, 64)
	blacklistThreshold := task.BlacklistThresh
	newFailureCount := oldFailureCount + 1
	shouldBlacklist := blacklistThreshold > 0 && newFailureCount >= int64(blacklistThreshold)

	// Step 1: Update cache first (optimistic)
	// Increment failure count
	if _, err := p.provider.store.HIncrBy(keyHashKey, "failure_count", 1); err != nil {
		return fmt.Errorf("failed to increment failure count in store: %w", err)
	}

	// If should blacklist, update status and remove from active list
	if shouldBlacklist {
		if err := p.provider.store.HSet(keyHashKey, map[string]any{"status": models.KeyStatusInvalid}); err != nil {
			// Rollback failure count
			p.provider.store.HIncrBy(keyHashKey, "failure_count", -1)
			return fmt.Errorf("failed to update key status to invalid in store: %w", err)
		}

		if err := p.provider.store.LRem(activeKeysListKey, 0, task.KeyID); err != nil {
			// Rollback status and failure count
			p.rollbackCacheFailure(keyHashKey, oldFailureCount, oldStatus)
			return fmt.Errorf("failed to LRem key from active list: %w", err)
		}
	}

	// Step 2: Update database
	dbErr := p.provider.executeTransactionWithRetry(func(tx *gorm.DB) error {
		var key models.APIKey
		if err := tx.Set("gorm:query_option", "FOR UPDATE").First(&key, task.KeyID).Error; err != nil {
			return fmt.Errorf("failed to lock key %d for update: %w", task.KeyID, err)
		}

		dbUpdates := map[string]any{"failure_count": newFailureCount}
		if shouldBlacklist {
			dbUpdates["status"] = models.KeyStatusInvalid
		}

		if err := tx.Model(&key).Updates(dbUpdates).Error; err != nil {
			return fmt.Errorf("failed to update key stats in DB: %w", err)
		}

		return nil
	})

	// Step 3: If DB fails, rollback cache
	if dbErr != nil {
		logrus.WithFields(logrus.Fields{
			"keyID": task.KeyID,
			"error": dbErr,
		}).Warn("DB update failed, rolling back cache")

		p.rollbackCacheFailure(keyHashKey, oldFailureCount, oldStatus)

		// If we removed from active list, add it back
		if shouldBlacklist {
			p.provider.store.LPush(activeKeysListKey, task.KeyID)
		}

		return dbErr
	}

	if shouldBlacklist {
		logrus.WithFields(logrus.Fields{
			"keyID":     task.KeyID,
			"threshold": blacklistThreshold,
		}).Warn("Key has reached blacklist threshold, disabling.")
	}

	return nil
}

// rollbackCacheFailure restores cache to previous state after a failed failure update
func (p *KeyProviderProcessor) rollbackCacheFailure(keyHashKey string, oldFailureCount int64, oldStatus string) {
	rollback := map[string]any{
		"failure_count": oldFailureCount,
		"status":        oldStatus,
	}
	if err := p.provider.store.HSet(keyHashKey, rollback); err != nil {
		logrus.WithFields(logrus.Fields{
			"keyHashKey": keyHashKey,
			"error":      err,
		}).Error("Failed to rollback cache after DB failure")
	}
}
