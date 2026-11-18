package control

import (
	"context"
	"encoding/json"
	"slices"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/agent"
	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

func (s *Service) ListAgentModelRoutes(ctx context.Context) ([]agent.ModelRouteGroup, error) {
	var groups []models.Group
	if err := s.db.WithContext(ctx).Select("id", "enabled", "models").Order("id ASC").Find(&groups).Error; err != nil {
		return nil, app_errors.ParseDBError(err)
	}
	result := make([]agent.ModelRouteGroup, 0, len(groups))
	for _, group := range groups {
		var entries []groupModelEntry
		if err := decodeGroupDiscoveryJSON(group.Models, &entries); err != nil {
			return nil, err
		}
		view := agent.ModelRouteGroup{GroupID: group.ID, Enabled: group.Enabled, Entries: make([]agent.ModelRoute, 0, len(entries))}
		for _, entry := range entries {
			view.Entries = append(view.Entries, agentModelRouteView(entry))
		}
		result = append(result, view)
	}
	return result, nil
}

func agentModelRouteID(entry groupModelEntry) string {
	if entry.EntryID != "" {
		return entry.EntryID
	}
	// The loader has already backfilled test aliases, even for entries
	// predating the persisted entry ID. Unlike the compiled fallback route
	// ID, the test alias survives an upstream model or alias rename.
	return entry.TestAlias
}

func agentModelRouteView(entry groupModelEntry) agent.ModelRoute {
	return agent.ModelRoute{
		EntryID: agentModelRouteID(entry), ID: entry.ID, Alias: entry.Alias,
		ReasoningEffort: entry.ReasoningEffort, Priority: entry.Priority, Weight: entry.Weight,
		Enabled: entry.Enabled == nil || *entry.Enabled,
	}
}

func (s *Service) SaveAgentModelRoute(ctx context.Context, groupID uint, entryID string, input agent.ModelRouteInput) (agent.ModelRoute, error) {
	return s.mutateAgentModelRoute(ctx, groupID, entryID, &input)
}

func (s *Service) DeleteAgentModelRoute(ctx context.Context, groupID uint, entryID string) error {
	_, err := s.mutateAgentModelRoute(ctx, groupID, entryID, nil)
	return err
}

func (s *Service) mutateAgentModelRoute(ctx context.Context, groupID uint, entryID string, input *agent.ModelRouteInput) (agent.ModelRoute, error) {
	var result agent.ModelRoute
	catalogChanged := false
	if _, err := s.writeGroupConfig(ctx, func(tx *gorm.DB) error {
		var group models.Group
		if err := tx.Select("id", "models", "channel_id").First(&group, groupID).Error; err != nil {
			return app_errors.ParseDBError(err)
		}
		var entries []groupModelEntry
		if err := decodeGroupDiscoveryJSON(group.Models, &entries); err != nil {
			return err
		}
		// Preserve the exact fields of unrelated entries and unknown fields
		// of the edited entry while validating against the typed model config.
		rows := make([]json.RawMessage, 0, len(entries))
		if err := decodeGroupDiscoveryJSON(group.Models, &rows); err != nil {
			return err
		}
		index := -1
		if entryID != "" {
			index = slices.IndexFunc(entries, func(entry groupModelEntry) bool { return entry.EntryID == entryID })
			if index < 0 {
				// Keep an inventory handle obtained before an administrator read
				// backfilled the entry_id usable.
				index = slices.IndexFunc(entries, func(entry groupModelEntry) bool { return entry.TestAlias == entryID })
			}
		}
		switch {
		case entryID == "" && input != nil:
			entry, err := newAgentModelRouteEntry(tx, entries)
			if err != nil {
				return err
			}
			index = len(entries)
			entries = append(entries, entry)
			seed, err := json.Marshal(entry)
			if err != nil {
				return err
			}
			rows = append(rows, seed)
		case entryID == "" || index < 0:
			return app_errors.ErrResourceNotFound
		}
		if input == nil {
			entries = slices.Delete(entries, index, index+1)
			rows = slices.Delete(rows, index, index+1)
			catalogChanged = true
		} else {
			entry := &entries[index]
			oldID := entry.ID
			kind, ok := s.channelRegistry.ProviderKind(channel.ID(group.ChannelID))
			if !ok {
				return app_errors.ErrValidation
			}
			updated, err := patchAgentModelRoute(rows[index], entry, *input, kind)
			if err != nil {
				return err
			}
			rows[index] = updated
			catalogChanged = oldID != entry.ID
			result = agentModelRouteView(*entry)
		}
		configs := make([]state.ModelConfig, 0, len(entries))
		for _, entry := range entries {
			configs = append(configs, entry.toModelConfig())
		}
		if err := state.ValidateModelRouteEntries("group", configs); err != nil {
			return app_errors.ErrValidation
		}
		encoded, err := json.Marshal(rows)
		if err != nil {
			return err
		}
		if err := tx.Model(&models.Group{}).Where("id = ?", groupID).Update("models", models.JSON(encoded)).Error; err != nil {
			return app_errors.ParseDBError(err)
		}
		return nil
	}, nil); err != nil {
		return agent.ModelRoute{}, err
	}
	if catalogChanged {
		s.catalogSync.RequestGroupSync()
	}
	return result, nil
}

func newAgentModelRouteEntry(tx *gorm.DB, entries []groupModelEntry) (groupModelEntry, error) {
	used := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		used[entry.EntryID] = struct{}{}
	}
	id, err := newEntryID(used)
	if err != nil {
		return groupModelEntry{}, err
	}
	model := []GroupModel{{EntryID: id}}
	if err := assignMissingTestAliases(tx, model); err != nil {
		return groupModelEntry{}, err
	}
	return groupModelEntry{EntryID: id, TestAlias: model[0].TestAlias}, nil
}

func patchAgentModelRoute(row json.RawMessage, entry *groupModelEntry, input agent.ModelRouteInput, kind channel.ProviderKind) (json.RawMessage, error) {
	var stored map[string]json.RawMessage
	if err := json.Unmarshal(row, &stored); err != nil {
		return nil, err
	}
	changes := make(map[string]any, 5)
	if input.ID.Set {
		if input.ID.Value == nil || strings.TrimSpace(*input.ID.Value) == "" {
			return nil, app_errors.ErrValidation
		}
		entry.ID = strings.TrimSpace(*input.ID.Value)
		changes["id"] = entry.ID
	}
	if input.Alias.Set {
		entry.Alias = ""
		if input.Alias.Value != nil {
			entry.Alias = strings.TrimSpace(*input.Alias.Value)
		}
		changes["alias"] = entry.Alias
	}
	if input.ReasoningEffort.Set {
		entry.ReasoningEffort = ""
		if input.ReasoningEffort.Value != nil {
			effort, err := canonicalScheduleReasoningEffort(*input.ReasoningEffort.Value)
			if err != nil {
				return nil, err
			}
			if (kind == channel.ProviderGemini || kind == channel.ProviderGoogleVertex) && effort == "none" {
				return nil, app_errors.ErrValidation
			}
			entry.ReasoningEffort = effort
		}
		changes["reasoning_effort"] = entry.ReasoningEffort
	}
	if input.Priority.Set {
		entry.Priority = input.Priority.Value
		changes["priority"] = entry.Priority
	}
	if input.Weight.Set {
		entry.Weight = input.Weight.Value
		changes["weight"] = entry.Weight
	}
	for name, value := range changes {
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		stored[name] = encoded
	}
	return json.Marshal(stored)
}
