package control

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/channel"
	"gpt-load/internal/channel/spec"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/encryption"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

type GroupSettingsResponse struct {
	PriceMultiplier string                       `json:"price_multiplier"`
	ChannelID       channel.ID                   `json:"channel_id"`
	ConnectionType  models.ConnectionType        `json:"connection_type"`
	Params          json.RawMessage              `json:"params"`
	Name            string                       `json:"name"`
	ValidationModel *string                      `json:"validation_model"`
	ProviderURL     *string                      `json:"provider_url"`
	Enabled         bool                         `json:"enabled"`
	Overrides       config.Settings              `json:"overrides"`
	Effective       GroupEffectiveConfigResponse `json:"effective"`
	Proxy           outboundproxy.View           `json:"proxy"`
}

type GroupSettingsUpdateRequest struct {
	PriceMultiplier optionalField[string]               `json:"price_multiplier"`
	Name            optionalField[string]               `json:"name"`
	Params          optionalField[json.RawMessage]      `json:"params"`
	ValidationModel optionalField[string]               `json:"validation_model"`
	ProviderURL     optionalField[string]               `json:"provider_url"`
	Enabled         optionalField[bool]                 `json:"enabled"`
	Overrides       optionalField[config.Settings]      `json:"overrides"`
	Proxy           optionalField[outboundproxy.Config] `json:"proxy"`
}

type normalizedGroupSettingsUpdate struct {
	priceMultiplierMicros *int64
	name                  *string
	params                json.RawMessage
	paramsSet             bool
	validationModel       *string
	validationModelSet    bool
	providerURL           *string
	providerURLSet        bool
	enabled               *bool
	encodedOverrides      models.JSON
	overridesSet          bool
	proxyConfig           *string
	proxySet              bool
}

func (s *Service) GetGroupSettings(ctx context.Context, groupID uint) (GroupSettingsResponse, error) {
	if groupID == 0 {
		return GroupSettingsResponse{}, app_errors.ErrBadRequest
	}

	s.writeMu.RLock()
	defer s.writeMu.RUnlock()

	group, err := loadGroupRow(s.db.WithContext(ctx), groupID)
	if err != nil {
		return GroupSettingsResponse{}, err
	}
	snapshot := s.manager.Current()
	if snapshot == nil {
		return GroupSettingsResponse{}, fmt.Errorf(
			"runtime snapshot unavailable: %w", app_errors.ErrInternalServer,
		)
	}
	response, err := groupSettingsResponse(group, snapshot.Settings, s.channelRegistry)
	if err != nil {
		return GroupSettingsResponse{}, err
	}
	response.Proxy, err = s.groupProxyView(ctx, s.db, group)
	return response, err
}

// dropRetiredGroupOverrides 删除分组 overrides 里已退役的 retry_count，返回清理后的 JSON 与是否需要回写。
// 存量行 JSON 损坏时不在这里报错：分组设置读取路径会把损坏如实报出来。
func dropRetiredGroupOverrides(overrides models.JSON) (models.JSON, bool) {
	if len(overrides) == 0 {
		return overrides, false
	}
	settings := make(config.Settings)
	if err := decodeGroupDiscoveryJSON(overrides, &settings); err != nil {
		return overrides, false
	}
	if _, exists := settings[state.SettingRetryCount]; !exists {
		return overrides, false
	}
	delete(settings, state.SettingRetryCount)
	for key, value := range settings {
		if key == state.SettingParameterOverrides {
			continue
		}
		settings[key] = canonicalizeGroupSettingNumbers(value)
	}
	encoded, err := json.Marshal(settings)
	if err != nil {
		return overrides, false
	}
	return models.JSON(encoded), true
}

func groupSettingsResponse(
	group models.Group,
	system state.RuntimeSettings,
	registry *channel.Registry,
) (GroupSettingsResponse, error) {
	if registry == nil || group.ChannelID == "" {
		return GroupSettingsResponse{}, fmt.Errorf(
			"resolve group %d channel: %w", group.ID, app_errors.ErrInternalServer,
		)
	}
	channelID := channel.ID(group.ChannelID)
	validated, err := registry.ValidateParams(channelID, json.RawMessage(group.Params))
	if err != nil {
		return GroupSettingsResponse{}, fmt.Errorf(
			"validate group %d params: %w", group.ID, app_errors.ErrInternalServer,
		)
	}
	overrides := make(config.Settings)
	if len(group.Overrides) > 0 {
		if err := decodeGroupDiscoveryJSON(group.Overrides, &overrides); err != nil {
			return GroupSettingsResponse{}, fmt.Errorf("decode group %d config: %w", group.ID, err)
		}
	}
	if overrides == nil {
		overrides = make(config.Settings)
	}
	// retry_count 是系统级预算；历史分组配置不再作为覆盖展示。
	delete(overrides, state.SettingRetryCount)
	effective, err := effectiveGroupConfig(system, overrides)
	if err != nil {
		return GroupSettingsResponse{}, fmt.Errorf(
			"resolve group %d effective config: %w", group.ID, app_errors.ErrInternalServer,
		)
	}
	return GroupSettingsResponse{
		PriceMultiplier: priceMultiplierResponse(group.PriceMultiplierMicros),
		ChannelID:       channelID,
		ConnectionType:  normalizeGroupConnectionType(group.ConnectionType),
		Params:          validated.CanonicalJSON(),
		Name:            group.Name,
		ValidationModel: cloneString(group.ValidationModel),
		ProviderURL:     cloneString(group.ProviderURL),
		Enabled:         group.Enabled,
		Overrides:       overrides,
		Effective:       effective,
	}, nil
}

func cloneString(value *string) *string {
	if value == nil {
		return nil
	}
	cloned := *value
	return &cloned
}

func normalizeGroupSettingsUpdate(
	request GroupSettingsUpdateRequest,
	encryptionService encryption.Service,
) (normalizedGroupSettingsUpdate, error) {
	for _, nullable := range []bool{
		request.Name.Set && request.Name.Null,
		request.Params.Set && request.Params.Null,
		request.Enabled.Set && request.Enabled.Null,
		request.Overrides.Set && request.Overrides.Null,
	} {
		if nullable {
			return normalizedGroupSettingsUpdate{}, app_errors.ErrValidation
		}
	}
	if !request.Name.Set && !request.Params.Set && !request.ValidationModel.Set &&
		!request.ProviderURL.Set && !request.Enabled.Set && !request.Overrides.Set &&
		!request.Proxy.Set && !request.PriceMultiplier.Set {
		return normalizedGroupSettingsUpdate{}, app_errors.ErrBadRequest
	}

	result := normalizedGroupSettingsUpdate{}
	if request.PriceMultiplier.Set {
		value, err := normalizePriceMultiplier(request.PriceMultiplier)
		if err != nil {
			return normalizedGroupSettingsUpdate{}, err
		}
		result.priceMultiplierMicros = priceMultiplierStorage(value)
	}
	if request.Name.Set {
		value, err := normalizeGroupName(&request.Name.Value)
		if err != nil {
			return normalizedGroupSettingsUpdate{}, err
		}
		result.name = value
	}
	if request.Params.Set {
		result.paramsSet = true
		result.params = append(json.RawMessage(nil), request.Params.Value...)
	}
	if request.ValidationModel.Set {
		result.validationModelSet = true
		if !request.ValidationModel.Null {
			value, err := normalizeValidationModel(request.ValidationModel.Value)
			if err != nil {
				return normalizedGroupSettingsUpdate{}, err
			}
			result.validationModel = &value
		}
	}
	if request.ProviderURL.Set {
		result.providerURLSet = true
		if !request.ProviderURL.Null {
			value := strings.TrimSpace(request.ProviderURL.Value)
			if value == "" {
				result.providerURL = nil
			} else {
				normalized, err := spec.NormalizeBaseURL(value)
				if err != nil {
					return normalizedGroupSettingsUpdate{}, app_errors.ErrValidation
				}
				result.providerURL = &normalized
			}
		}
	}
	if request.Enabled.Set {
		value := request.Enabled.Value
		result.enabled = &value
	}
	if request.Overrides.Set {
		_, encoded, err := normalizeGroupSettings(request.Overrides.Value)
		if err != nil {
			return normalizedGroupSettingsUpdate{}, err
		}
		result.encodedOverrides = encoded
		result.overridesSet = true
	}
	proxyConfig, proxySet, err := normalizeProxyOverride(request.Proxy, encryptionService)
	if err != nil {
		return normalizedGroupSettingsUpdate{}, err
	}
	result.proxyConfig = proxyConfig
	result.proxySet = proxySet
	return result, nil
}

func (s *Service) UpdateGroupSettings(
	ctx context.Context,
	groupID uint,
	request GroupSettingsUpdateRequest,
) (GroupSettingsResponse, error) {
	if groupID == 0 {
		return GroupSettingsResponse{}, app_errors.ErrBadRequest
	}
	normalized, err := normalizeGroupSettingsUpdate(request, s.encryption)
	if err != nil {
		return GroupSettingsResponse{}, err
	}

	var committed models.Group
	var targetEntries []state.CredentialEntry
	targetChanged := false
	snapshot, err := s.writeGroupConfig(ctx, func(tx *gorm.DB) error {
		group, err := loadGroupRow(tx, groupID)
		if err != nil {
			return err
		}
		if err := validateGroupRowCandidate(ctx, tx, group, s.channelRegistry); err != nil {
			return fmt.Errorf("validate existing group %d: %w", groupID, app_errors.ErrInternalServer)
		}
		if request.Proxy.Set && !request.Proxy.Null &&
			!s.channelRegistry.SupportsOutboundProxy(channel.ID(group.ChannelID)) {
			return app_errors.ErrValidation
		}

		updates := make(map[string]any, 8)
		if normalized.priceMultiplierMicros != nil {
			group.PriceMultiplierMicros = normalized.priceMultiplierMicros
			updates["price_multiplier_micros"] = *normalized.priceMultiplierMicros
		}
		if normalized.name != nil {
			group.Name = *normalized.name
			updates["name"] = group.Name
		}
		if normalized.paramsSet {
			previousParams := append([]byte(nil), group.Params...)
			params, validateErr := s.channelRegistry.ValidateParams(
				channel.ID(group.ChannelID), normalized.params,
			)
			if validateErr != nil {
				return app_errors.ErrValidation
			}
			if normalizeGroupConnectionType(group.ConnectionType) == models.ConnectionTypeSubscription &&
				string(params.CanonicalJSON()) != "{}" {
				return app_errors.ErrValidation
			}
			group.Params = models.JSON(params.CanonicalJSON())
			targetChanged = !bytes.Equal(bytes.TrimSpace(previousParams), bytes.TrimSpace(group.Params))
			updates["params"] = append(models.JSON(nil), group.Params...)
		}
		if normalized.validationModelSet {
			group.ValidationModel = normalized.validationModel
			updates["validation_model"] = normalized.validationModel
		}
		if normalized.providerURLSet {
			group.ProviderURL = normalized.providerURL
			updates["provider_url"] = normalized.providerURL
		}
		if normalized.enabled != nil {
			group.Enabled = *normalized.enabled
			updates["enabled"] = group.Enabled
		}
		if normalized.overridesSet {
			group.Overrides = normalized.encodedOverrides
			updates["overrides"] = group.Overrides
		} else if purged, changed := dropRetiredGroupOverrides(group.Overrides); changed {
			// 分组 retry_count 已退役：任何一次保存都顺手把存量残留从库里删掉。
			group.Overrides = purged
			updates["overrides"] = purged
		}
		if normalized.proxySet {
			group.ProxyConfig = normalized.proxyConfig
			updates["proxy_config"] = normalized.proxyConfig
		}
		if err := validateGroupRowCandidate(ctx, tx, group, s.channelRegistry); err != nil {
			return app_errors.ErrValidation
		}
		if len(updates) > 0 {
			if err := tx.Model(&models.Group{}).Where("id = ?", groupID).Updates(updates).Error; err != nil {
				return app_errors.ParseDBError(err)
			}
		}
		if targetChanged {
			targetEntries, err = stateloader.BuildGroupCredentialEntries(ctx, tx, groupID)
			if err != nil {
				return err
			}
		}
		committed = group
		return nil
	}, func() error {
		if !targetChanged {
			return nil
		}
		if s.stats != nil {
			for _, entry := range targetEntries {
				s.stats.Reset(entry.ID)
			}
		}
		_, err := s.reconcileRegistryGroup(groupID, targetEntries)
		return err
	})
	if err != nil {
		return GroupSettingsResponse{}, withControlOperationContext(err, groupID, 0)
	}
	response, err := groupSettingsResponse(committed, snapshot.Settings, s.channelRegistry)
	if err != nil {
		return GroupSettingsResponse{}, err
	}
	response.Proxy, err = s.groupProxyView(ctx, s.db, committed)
	return response, err
}
