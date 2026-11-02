package control

import (
	"encoding/json"
	"fmt"
	"strings"

	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

func scopeProjectModelGroups(
	groups []models.Group,
	snapshot *state.ConfigSnapshot,
	accessKeyID uint,
	registry *channel.Registry,
) ([]models.Group, map[uint][]protocol.Protocol, error) {
	if snapshot == nil || accessKeyID == 0 {
		return nil, nil, app_errors.ErrUnauthorized
	}
	accessKey, exists := snapshot.AccessKeysByID[accessKeyID]
	if !exists || accessKey.Status != state.AccessKeyStatusActive {
		return nil, nil, app_errors.ErrUnauthorized
	}
	result := make([]models.Group, 0, len(groups))
	protocolsByGroup := make(map[uint][]protocol.Protocol, len(groups))
	for _, group := range groups {
		if !group.Enabled {
			continue
		}
		if len(accessKey.Filters.Groups) > 0 {
			if _, allowed := accessKey.Filters.Groups[group.ID]; !allowed {
				continue
			}
		}
		protocols, err := projectModelGroupProtocols(registry, group)
		if err != nil {
			return nil, nil, err
		}
		filteredProtocols := make([]protocol.Protocol, 0, len(protocols))
		for _, value := range protocols {
			if len(accessKey.Filters.Protocols) > 0 {
				if _, allowed := accessKey.Filters.Protocols[value]; !allowed {
					continue
				}
			}
			filteredProtocols = append(filteredProtocols, value)
		}
		if len(filteredProtocols) == 0 {
			continue
		}
		var groupModels []GroupModel
		if err := decodeGroupDiscoveryJSON(group.Models, &groupModels); err != nil {
			return nil, nil, fmt.Errorf(
				"decode group %d models for access scope: %w",
				group.ID,
				app_errors.ErrInternalServer,
			)
		}
		filteredModels := make([]GroupModel, 0, len(groupModels))
		for _, model := range groupModels {
			clientModel := strings.TrimSpace(model.Alias)
			if clientModel == "" {
				clientModel = strings.TrimSpace(model.ID)
			}
			if len(accessKey.Filters.Models) > 0 {
				if _, allowed := accessKey.Filters.Models[clientModel]; !allowed {
					continue
				}
			}
			filteredModels = append(filteredModels, model)
		}
		if len(filteredModels) == 0 {
			continue
		}
		modelJSON, err := json.Marshal(filteredModels)
		if err != nil {
			return nil, nil, fmt.Errorf("encode scoped group models: %w", err)
		}
		group.Models = models.JSON(modelJSON)
		protocolsByGroup[group.ID] = append([]protocol.Protocol(nil), filteredProtocols...)
		result = append(result, group)
	}
	return result, protocolsByGroup, nil
}
