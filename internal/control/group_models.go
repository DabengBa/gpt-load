package control

import (
	"context"
	"encoding/json"
	"fmt"

	"gorm.io/gorm"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/pricing"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

type GroupModelsUpdateRequest struct {
	Models optionalGroupModels `json:"models"`
}

type GroupModelResponse struct {
	ID             string                     `json:"id"`
	Alias          string                     `json:"alias"`
	AliasEnabled   bool                       `json:"alias_enabled"`
	ClientModel    string                     `json:"client_model"`
	EntryID        string                     `json:"entry_id"`
	Weight         *int                       `json:"weight"`
	Priority       *int                       `json:"priority"`
	CircuitBreaker *state.EntryCircuitBreaker `json:"circuit_breaker"`
	PricingStatus  PricingStatus              `json:"pricing_status"`
	PriceID        *uint                      `json:"price_id,omitempty"`
}

type GroupModelsResponse struct {
	Items   []GroupModelResponse `json:"items"`
	Total   int                  `json:"total"`
	Pending int                  `json:"pending"`
}

type ModelNameConflict struct {
	ClientModel string `json:"client_model"`
	Indexes     []int  `json:"indexes"`
}

type ModelNameConflictData struct {
	Conflicts []ModelNameConflict `json:"conflicts"`
}

// groupModelEntry decodes a persisted group model entry. It extends the
// legacy {id, alias} rows with the optional route-entry weight/priority
// fields (design §3) and, unlike GroupModel, tolerates unknown storage keys
// so older readers never fail on rows written with new fields.
type groupModelEntry struct {
	ID             string                     `json:"id"`
	Alias          string                     `json:"alias"`
	EntryID        string                     `json:"entry_id,omitempty"`
	Weight         *int                       `json:"weight,omitempty"`
	Priority       *int                       `json:"priority,omitempty"`
	CircuitBreaker *state.EntryCircuitBreaker `json:"circuit_breaker,omitempty"`
}

func cloneEntryCircuitBreaker(value *state.EntryCircuitBreaker) *state.EntryCircuitBreaker {
	if value == nil {
		return nil
	}
	result := &state.EntryCircuitBreaker{}
	if value.BlacklistThreshold != nil {
		v := *value.BlacklistThreshold
		result.BlacklistThreshold = &v
	}
	if value.CooldownSeconds != nil {
		v := *value.CooldownSeconds
		result.CooldownSeconds = &v
	}
	return result
}
func (model groupModelEntry) toModelConfig() state.ModelConfig {
	return state.ModelConfig{
		ID: model.ID, Alias: model.Alias, EntryID: model.EntryID,
		Weight: cloneInt(model.Weight), Priority: cloneInt(model.Priority),
		CircuitBreaker: cloneEntryCircuitBreaker(model.CircuitBreaker),
	}
}

func (s *Service) GetGroupModels(ctx context.Context, groupID uint) (GroupModelsResponse, error) {
	if groupID == 0 {
		return GroupModelsResponse{}, app_errors.ErrBadRequest
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	group, err := loadGroupRow(s.db.WithContext(ctx), groupID)
	if err != nil {
		return GroupModelsResponse{}, err
	}
	groupModels := make([]groupModelEntry, 0)
	if err := decodeGroupDiscoveryJSON(group.Models, &groupModels); err != nil {
		return GroupModelsResponse{}, fmt.Errorf("decode group %d models: %w", group.ID, err)
	}
	used := make(map[string]struct{}, len(groupModels))
	for _, model := range groupModels {
		if model.EntryID != "" {
			used[model.EntryID] = struct{}{}
		}
	}
	changed := false
	for index := range groupModels {
		if groupModels[index].EntryID != "" {
			continue
		}
		entryID, genErr := newEntryID(used)
		if genErr != nil {
			return GroupModelsResponse{}, app_errors.ErrInternalServer
		}
		groupModels[index].EntryID = entryID
		changed = true
	}
	if changed {
		encoded, encodeErr := json.Marshal(groupModels)
		if encodeErr != nil {
			return GroupModelsResponse{}, app_errors.ErrInternalServer
		}
		if err := s.db.WithContext(ctx).Model(&models.Group{}).Where("id = ?", groupID).Update("models", models.JSON(encoded)).Error; err != nil {
			return GroupModelsResponse{}, app_errors.ParseDBError(err)
		}
		input, buildErr := stateloader.BuildCompileInputWithProxy(ctx, s.db, s.encryption, s.environmentProxy, s.channelRegistry)
		if buildErr != nil {
			return GroupModelsResponse{}, app_errors.ErrInternalServer
		}
		if _, publishErr := s.publishSnapshot(input); publishErr != nil {
			return GroupModelsResponse{}, app_errors.ErrInternalServer
		}
	}
	rows, err := loadModelPriceRows(ctx, s.db)
	if err != nil {
		return GroupModelsResponse{}, app_errors.ParseDBError(err)
	}

	return mapGroupModelsResponse(group.ChannelID, groupModels, rows)
}

func mapGroupModelsResponse(
	channelID string,
	groupModels []groupModelEntry,
	rows modelPriceRows,
) (GroupModelsResponse, error) {
	result := GroupModelsResponse{Items: make([]GroupModelResponse, 0, len(groupModels))}
	for _, model := range groupModels {
		item := GroupModelResponse{
			ID:             model.ID,
			Alias:          model.Alias,
			AliasEnabled:   model.Alias != "",
			ClientModel:    model.ID,
			EntryID:        model.EntryID,
			Weight:         cloneInt(model.Weight),
			Priority:       cloneInt(model.Priority),
			CircuitBreaker: cloneEntryCircuitBreaker(model.CircuitBreaker),
			PricingStatus:  PricingStatusPending,
		}
		if item.AliasEnabled {
			item.ClientModel = model.Alias
		}
		if price, priceExists := rows[pricing.Identity{ChannelID: channelID, ModelID: model.ID}]; priceExists {
			item.PricingStatus = resolvePricingStatus(price)
			item.PriceID = &price.ID
		}
		if item.PricingStatus == PricingStatusPending {
			result.Pending++
		}
		result.Items = append(result.Items, item)
	}
	result.Total = len(result.Items)
	return result, nil
}

func (s *Service) UpdateGroupModels(
	ctx context.Context,
	groupID uint,
	request GroupModelsUpdateRequest,
) (GroupModelsResponse, error) {
	if groupID == 0 {
		return GroupModelsResponse{}, app_errors.ErrBadRequest
	}
	if !request.Models.Set {
		return GroupModelsResponse{}, app_errors.ErrValidation
	}
	normalized, err := normalizeGroupModels(request.Models.Values)
	if err != nil {
		return GroupModelsResponse{}, err
	}
	modelIDsChanged := false
	_, err = s.writeGroupConfig(ctx, func(tx *gorm.DB) error {
		group, err := loadGroupRow(tx, groupID)
		if err != nil {
			return err
		}
		if err := validateGroupRowCandidate(ctx, tx, group, s.channelRegistry); err != nil {
			return fmt.Errorf("validate existing group %d: %w", groupID, app_errors.ErrInternalServer)
		}
		var previous []groupModelEntry
		if err := decodeGroupDiscoveryJSON(group.Models, &previous); err != nil {
			return fmt.Errorf("decode group %d models: %w", groupID, app_errors.ErrInternalServer)
		}
		modelIDsChanged = !sameGroupModelIDs(previous, normalized)
		preserved, preserveErr := preserveGroupModelFields(previous, normalized)
		if preserveErr != nil {
			return preserveErr
		}
		encoded, err := json.Marshal(preserved)
		if err != nil {
			return fmt.Errorf("encode group models: %w", err)
		}

		group.Models = models.JSON(encoded)
		if err := validateGroupRowCandidate(ctx, tx, group, s.channelRegistry); err != nil {
			return app_errors.ErrValidation
		}
		if err := tx.Model(&models.Group{}).
			Where("id = ?", groupID).
			Update("models", group.Models).Error; err != nil {
			return app_errors.ParseDBError(err)
		}
		return nil
	}, nil)
	if err != nil {
		return GroupModelsResponse{}, withControlOperationContext(err, groupID, 0)
	}
	if modelIDsChanged && s.catalogSync != nil {
		s.catalogSync.RequestGroupSync()
	}
	result, err := s.GetGroupModels(ctx, groupID)
	if err != nil {
		return GroupModelsResponse{}, fmt.Errorf(
			"load group %d models after update: %w",
			groupID,
			app_errors.ErrInternalServer,
		)
	}
	return result, nil
}

func preserveGroupModelFields(previous []groupModelEntry, requested []GroupModel) ([]GroupModel, error) {
	previousByEntryID := make(map[string]groupModelEntry, len(previous))
	previousByModel := make(map[string][]groupModelEntry, len(previous))
	usedEntryIDs := make(map[string]struct{}, len(previous))
	for _, model := range previous {
		if model.EntryID != "" {
			previousByEntryID[model.EntryID] = model
			usedEntryIDs[model.EntryID] = struct{}{}
		}
		key := model.ID + "\x00" + model.Alias
		previousByModel[key] = append(previousByModel[key], model)
	}
	// 同一组中 ID 相同的多条新行无法共享同一个已保存条目。
	// 每个已保存条目仅归首次匹配到它的新行所有；
	// 后续新行再次命中同一个已保存条目时视为不存在，以便分配新的 EntryID。
	assigned := make(map[string]struct{}, len(previous))
	result := make([]GroupModel, 0, len(requested))
	for _, model := range requested {
		var preserved groupModelEntry
		exists := false
		if model.EntryID != "" {
			preserved, exists = previousByEntryID[model.EntryID]
		}
		if !exists {
			// 精确匹配（含别名）优先，无精确匹配时回退到同 ID 的无别名条目。
			matches := previousByModel[model.ID+"\x00"+model.Alias]
			if len(matches) == 0 {
				matches = previousByModel[model.ID+"\x00"]
			}
			if len(matches) == 1 {
				if _, alreadyAssigned := assigned[matches[0].EntryID]; !alreadyAssigned {
					preserved, exists = matches[0], true
				}
			}
		}
		if exists {
			assigned[preserved.EntryID] = struct{}{}
		}
		if model.EntryID == "" && exists {
			model.EntryID = preserved.EntryID
		}
		if model.EntryID == "" {
			entryID, err := newEntryID(usedEntryIDs)
			if err != nil {
				return nil, app_errors.ErrInternalServer
			}
			model.EntryID = entryID
		}
		if !model.weightSet && exists {
			model.Weight = cloneInt(preserved.Weight)
		}
		if !model.prioritySet && exists {
			model.Priority = cloneInt(preserved.Priority)
		}
		if !model.circuitBreakerSet && exists {
			model.CircuitBreaker = cloneEntryCircuitBreaker(preserved.CircuitBreaker)
		}
		result = append(result, model)
	}
	return result, nil
}

func sameGroupModelIDs(left []groupModelEntry, right []GroupModel) bool {
	if len(left) != len(right) {
		return false
	}
	ids := make(map[string]struct{}, len(left))
	for _, model := range left {
		ids[model.ID] = struct{}{}
	}
	if len(ids) != len(right) {
		return false
	}
	for _, model := range right {
		if _, exists := ids[model.ID]; !exists {
			return false
		}
	}
	return true
}
