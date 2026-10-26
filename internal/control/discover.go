package control

import (
	"context"
	"encoding/json"
	"fmt"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/outboundproxy"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
)

type ModelDiscoveryRequest struct {
	ChannelID   channel.ID            `json:"channel_id"`
	Params      json.RawMessage       `json:"params"`
	Credentials string                `json:"credentials"`
	Proxy       *outboundproxy.Config `json:"proxy"`
}

type ModelDiscoveryResult struct {
	Models []ModelCandidate `json:"models"`
}

func (s *Service) DiscoverModels(
	ctx context.Context,
	request ModelDiscoveryRequest,
) (ModelDiscoveryResult, error) {
	if err := ctx.Err(); err != nil {
		return ModelDiscoveryResult{}, err
	}

	if s == nil || s.channelRegistry == nil || request.ChannelID == "" {
		return ModelDiscoveryResult{}, app_errors.ErrValidation
	}
	if _, ok := s.channelRegistry.Get(request.ChannelID); !ok {
		return ModelDiscoveryResult{}, app_errors.ErrValidation
	}
	if request.Proxy != nil && !s.channelRegistry.SupportsOutboundProxy(request.ChannelID) {
		return ModelDiscoveryResult{}, app_errors.ErrValidation
	}
	resolvedTarget, err := s.channelRegistry.Resolve(request.ChannelID, request.Params)
	if err != nil {
		return ModelDiscoveryResult{}, app_errors.ErrValidation
	}
	var discoveryCredentials []discoveryCredential
	credentials, normalizeErr := s.normalizeCredentials(request.ChannelID, request.Credentials)
	if normalizeErr != nil {
		return ModelDiscoveryResult{}, normalizeErr
	}
	discoveryCredentials = make([]discoveryCredential, 0, len(credentials.candidates))
	for index, candidate := range credentials.candidates {
		credential, validateErr := s.channelRegistry.ValidateCredential(request.ChannelID, candidate.canonical)
		if validateErr != nil {
			return ModelDiscoveryResult{}, app_errors.ErrValidation
		}
		apiKey, _ := credential.Value("api_key")
		discoveryCredentials = append(discoveryCredentials, discoveryCredential{
			snapshot: execution.NewCredentialSnapshot(
				uint(index+1), 1, uint64(index+1), credential.CanonicalJSON(),
			),
			apiKey: apiKey,
		})
	}
	systemSettings, globalProxy, err := stateloader.LoadSystemSettingsAndProxy(ctx, s.db, s.encryption)
	if parentErr := ctx.Err(); parentErr != nil {
		return ModelDiscoveryResult{}, parentErr
	}
	if err != nil {
		return ModelDiscoveryResult{}, fmt.Errorf("load model discovery settings: %w", err)
	}
	var draftProxy *outboundproxy.Config
	if request.Proxy != nil {
		normalized, normalizeErr := outboundproxy.Normalize(*request.Proxy)
		if normalizeErr != nil || normalized.Mode == outboundproxy.ModeInherit {
			return ModelDiscoveryResult{}, app_errors.ErrValidation
		}
		draftProxy = &normalized
	}
	snapshot, err := state.Compile(state.CompileInput{
		SystemSettings: systemSettings, ChannelRegistry: s.channelRegistry,
		GlobalProxy: globalProxy, EnvironmentProxy: s.environmentProxy,
		Groups: []state.GroupConfig{{
			ID: 1, Name: "draft", ChannelID: request.ChannelID,
			Params: append(json.RawMessage(nil), request.Params...), Proxy: draftProxy, Enabled: true,
		}},
	})
	if err != nil {
		return ModelDiscoveryResult{}, app_errors.ErrValidation
	}
	group, ok := snapshot.Groups[1]
	if !ok {
		return ModelDiscoveryResult{}, fmt.Errorf("compiled model discovery draft is missing")
	}
	network, err := s.proxyNetworkContext(group.Proxy)
	if err != nil {
		return ModelDiscoveryResult{}, err
	}
	for index := range discoveryCredentials {
		discoveryCredentials[index].proxy = network.Proxy
		discoveryCredentials[index].proxyFingerprint = network.Fingerprint
	}

	return s.executeModelDiscovery(ctx, discoveryTarget{
		channelID: request.ChannelID, resolvedTarget: resolvedTarget,
		credentials: discoveryCredentials, headerRules: group.HeaderRules,
		timeouts: group.Timeouts, catalogProviderID: resolvedTarget.CatalogProviderID,
	})
}
