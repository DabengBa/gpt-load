package bifrost

import (
	"fmt"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/protocol"
	"gpt-load/internal/provideradapter"
)

// Reconcile marks the canonical configs referenced by the latest state snapshot as active.
func (manager *RuntimeManager) Reconcile(targets []provideradapter.RuntimeTarget) error {
	if manager == nil || manager.pool == nil {
		return fmt.Errorf("reconcile provider runtimes: manager is unavailable")
	}
	configs := make([]effectiveProviderConfig, 0, len(targets)*6)
	for _, runtimeTarget := range targets {
		target := runtimeTarget.Target
		if target.ProviderKind == channel.ProviderMultiProtocolGateway {
			for _, clientProtocol := range []protocol.Protocol{
				protocol.OpenAICompletions,
				protocol.Anthropic,
				protocol.Gemini,
			} {
				baseSpec := execution.AttemptSpec{
					ClientProtocol: clientProtocol,
					RouteMode:      execution.RouteNative,
				}
				baseConfig, baseErr := buildEffectiveProviderConfigForAttempt(
					target,
					baseSpec,
					manager.options.allowPrivateNetwork,
				)
				if baseErr != nil {
					return fmt.Errorf("reconcile provider runtime %q profile %q: %w", target.ChannelID, clientProtocol, baseErr)
				}
				configs = append(configs, baseConfig)
				baseSpec.Proxy = runtimeTarget.Proxy
				effectiveConfig, effectiveErr := buildEffectiveProviderConfigForAttempt(
					target,
					baseSpec,
					manager.options.allowPrivateNetwork,
				)
				if effectiveErr != nil {
					return fmt.Errorf("reconcile provider runtime %q profile %q proxy: %w", target.ChannelID, clientProtocol, effectiveErr)
				}
				configs = append(configs, effectiveConfig)
			}
			continue
		}
		config, err := buildEffectiveProviderConfig(target, manager.options.allowPrivateNetwork)
		if err != nil {
			return fmt.Errorf("reconcile provider runtime %q: %w", target.ChannelID, err)
		}
		configs = append(configs, config)
		effectiveConfig, effectiveErr := buildEffectiveProviderConfigForAttempt(
			target,
			execution.AttemptSpec{Proxy: runtimeTarget.Proxy},
			manager.options.allowPrivateNetwork,
		)
		if effectiveErr != nil {
			return fmt.Errorf("reconcile provider runtime %q proxy: %w", target.ChannelID, effectiveErr)
		}
		configs = append(configs, effectiveConfig)
		for _, route := range []struct {
			protocol  protocol.Protocol
			operation execution.Operation
		}{
			{protocol.OpenAICompletions, execution.OperationChatCompletion},
			{protocol.OpenAIResponses, execution.OperationResponsesCreate},
			{protocol.Anthropic, execution.OperationChatCompletion},
		} {
			spec := execution.AttemptSpec{ClientProtocol: route.protocol, Operation: route.operation, RouteMode: execution.RouteNative}
			if _, ok := nativeMessageProvider(target.ProviderKind, spec); !ok {
				continue
			}
			if mode, ok := target.Mode(route.protocol, route.operation); !ok || mode != channel.RouteNative {
				continue
			}
			for _, proxy := range []outboundproxy.Effective{{}, runtimeTarget.Proxy} {
				spec.Proxy = proxy
				nativeConfig, err := buildEffectiveProviderConfigForAttempt(target, spec, manager.options.allowPrivateNetwork)
				if err != nil {
					return fmt.Errorf("reconcile provider runtime %q native %q: %w", target.ChannelID, route.protocol, err)
				}
				configs = append(configs, nativeConfig)
			}
		}
		if target.ProviderKind == channel.ProviderDeepSeek {
			responsesMode, ok := target.Mode(protocol.OpenAIResponses, execution.OperationResponsesCreate)
			if ok && responsesMode == channel.RouteNative {
				responsesConfig, responsesErr := buildDeepSeekResponsesProbeConfig(target, manager.options.allowPrivateNetwork)
				if responsesErr != nil {
					return fmt.Errorf("reconcile provider runtime %q Responses: %w", target.ChannelID, responsesErr)
				}
				configs = append(configs, responsesConfig)
				effectiveResponsesConfig, effectiveResponsesErr := buildEffectiveProviderConfigForAttempt(
					target,
					execution.AttemptSpec{
						ClientProtocol: protocol.OpenAIResponses,
						Operation:      execution.OperationProbe,
						RouteMode:      execution.RouteNative,
						Proxy:          runtimeTarget.Proxy,
					},
					manager.options.allowPrivateNetwork,
				)
				if effectiveResponsesErr != nil {
					return fmt.Errorf("reconcile provider runtime %q Responses proxy: %w", target.ChannelID, effectiveResponsesErr)
				}
				configs = append(configs, effectiveResponsesConfig)
			}
		}
	}
	manager.pool.reconcile(configs)
	return nil
}
