package bifrost

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net"
	"net/url"

	"github.com/maximhq/bifrost/core/schemas"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/protocol"
)

type effectiveProviderConfig struct {
	provider       schemas.ModelProvider
	providerConfig *schemas.ProviderConfig
	canonical      []byte
	fingerprint    string
	// base identifies the unpartitioned provider config accepted by a
	// credential-partitioned Runtime.
	baseCanonical   []byte
	baseFingerprint string
	// target identifies the code-owned provider target that keeps a
	// credential-partitioned Runtime eligible until that credential retires.
	targetCanonical       []byte
	targetFingerprint     string
	credentialPartitionID uint
	targetBaseURL         string
	custom                bool
}

func buildEffectiveProviderConfig(
	resolved channel.ResolvedTarget,
	allowPrivateNetwork bool,
) (effectiveProviderConfig, error) {
	if resolved.ProviderKind == channel.ProviderMultiProtocolGateway {
		return buildMultiProtocolGatewayProviderConfig(resolved, schemas.OpenAI, allowPrivateNetwork)
	}
	provider, baseURL, custom, err := resolveSDKProviderConfig(resolved)
	if err != nil {
		return effectiveProviderConfig{}, err
	}
	config := buildProviderConfig(provider, baseURL, custom, schemas.OpenAI, allowPrivateNetwork)
	return newEffectiveProviderConfig(provider, baseURL, custom, config)
}

func buildProviderConfig(
	provider schemas.ModelProvider,
	baseURL string,
	custom bool,
	baseProvider schemas.ModelProvider,
	allowPrivateNetwork bool,
) *schemas.ProviderConfig {
	config := providerConfig(baseURL, custom, baseProvider, allowPrivateNetwork)
	if !custom {
		config = providerConfig(baseURL, false, provider, allowPrivateNetwork)
	}
	config.CheckAndSetDefaults()
	return config
}

func newEffectiveProviderConfig(
	provider schemas.ModelProvider,
	baseURL string,
	custom bool,
	config *schemas.ProviderConfig,
) (effectiveProviderConfig, error) {
	canonical, err := json.Marshal(struct {
		Provider schemas.ModelProvider   `json:"provider"`
		Config   *schemas.ProviderConfig `json:"config"`
	}{Provider: provider, Config: config})
	if err != nil {
		return effectiveProviderConfig{}, fmt.Errorf("encode provider runtime config: %w", err)
	}
	digest := sha256.Sum256(canonical)
	return effectiveProviderConfig{
		provider:       provider,
		providerConfig: cloneProviderConfig(config),
		canonical:      canonical,
		fingerprint:    hex.EncodeToString(digest[:]),
		targetBaseURL:  baseURL,
		custom:         custom,
	}, nil
}

func buildEffectiveProviderConfigForAttempt(
	resolved channel.ResolvedTarget,
	spec execution.AttemptSpec,
	allowPrivateNetwork bool,
) (effectiveProviderConfig, error) {
	var base effectiveProviderConfig
	var err error
	if resolved.ProviderKind == channel.ProviderMultiProtocolGateway {
		provider, profileErr := multiProtocolGatewayProviderProfile(spec.ClientProtocol)
		if profileErr != nil {
			return effectiveProviderConfig{}, profileErr
		}
		base, err = buildMultiProtocolGatewayProviderConfig(resolved, provider, allowPrivateNetwork)
	} else {
		base, err = buildEffectiveProviderConfig(resolved, allowPrivateNetwork)
	}
	if err != nil {
		return base, err
	}
	if nativeProvider, ok := nativeMessageProvider(resolved.ProviderKind, spec); ok {
		provider := customProviderKey(nativeProvider, base.targetBaseURL)
		config := buildProviderConfig(provider, base.targetBaseURL, true, nativeProvider, allowPrivateNetwork)
		base, err = newEffectiveProviderConfig(provider, base.targetBaseURL, true, config)
		if err != nil {
			return effectiveProviderConfig{}, err
		}
		return applyAttemptProxy(base, spec.Proxy)
	}
	if !resolved.ProviderKind.SupportsOutboundProxy() {
		return base, nil
	}
	if resolved.ProviderKind != channel.ProviderDeepSeek ||
		spec.ClientProtocol != protocol.OpenAIResponses ||
		(spec.Operation != execution.OperationProbe && spec.Operation != execution.OperationModelTrace) ||
		channel.RouteMode(spec.RouteMode) != channel.RouteNative {
		return applyAttemptProxy(base, spec.Proxy)
	}

	provider := customProviderKey(schemas.OpenAI, base.targetBaseURL)
	config := buildProviderConfig(provider, base.targetBaseURL, true, schemas.OpenAI, allowPrivateNetwork)
	config.CustomProviderConfig.AllowedRequests.Responses = true
	config.CustomProviderConfig.AllowedRequests.ResponsesStream = true
	config.CustomProviderConfig.AllowedRequests.ChatCompletion = false
	config.CustomProviderConfig.AllowedRequests.ChatCompletionStream = false
	config.CustomProviderConfig.RequestPathOverrides[schemas.ResponsesRequest] = base.targetBaseURL + "/responses"
	config.CustomProviderConfig.RequestPathOverrides[schemas.ResponsesStreamRequest] = base.targetBaseURL + "/responses"
	config.CheckAndSetDefaults()
	base, err = newEffectiveProviderConfig(provider, base.targetBaseURL, true, config)
	if err != nil {
		return effectiveProviderConfig{}, err
	}
	return applyAttemptProxy(base, spec.Proxy)
}

func multiProtocolGatewayProviderProfile(clientProtocol protocol.Protocol) (schemas.ModelProvider, error) {
	switch clientProtocol {
	case "", protocol.OpenAICompletions, protocol.OpenAIResponses,
		protocol.OpenAIImages, protocol.OpenAIEmbeddings, protocol.Rerank:
		return schemas.OpenAI, nil
	case protocol.Anthropic:
		return schemas.Anthropic, nil
	case protocol.Gemini, protocol.GeminiEmbeddings:
		return schemas.Gemini, nil
	default:
		return "", fmt.Errorf("unsupported multi-protocol gateway client protocol %q", clientProtocol)
	}
}

func buildMultiProtocolGatewayProviderConfig(
	resolved channel.ResolvedTarget,
	provider schemas.ModelProvider,
	allowPrivateNetwork bool,
) (effectiveProviderConfig, error) {
	baseURL, configured, err := targetBaseURL(resolved.TargetConfig)
	if err != nil || !configured {
		return effectiveProviderConfig{}, fmt.Errorf("multi-protocol gateway root is required")
	}
	config := buildProviderConfig(provider, baseURL, false, provider, allowPrivateNetwork)
	return newEffectiveProviderConfig(provider, baseURL, false, config)
}

func applyAttemptProxy(
	base effectiveProviderConfig,
	effective outboundproxy.Effective,
) (effectiveProviderConfig, error) {
	effective, err := outboundproxy.NormalizeEffective(effective)
	if err != nil {
		return effectiveProviderConfig{}, fmt.Errorf("normalize attempt proxy: %w", err)
	}
	if effective.Config.Mode == outboundproxy.ModeDirect {
		return base, nil
	}
	config := cloneProviderConfig(base.providerConfig)
	proxy := &schemas.ProxyConfig{}
	switch effective.Config.Mode {
	case outboundproxy.ModeEnvironment:
		proxy.Type = schemas.EnvProxy
	case outboundproxy.ModeCustom:
		endpoint, parseErr := url.Parse(effective.Config.URL)
		if parseErr != nil || endpoint.Host == "" {
			return effectiveProviderConfig{}, fmt.Errorf("normalize attempt proxy: invalid endpoint")
		}
		switch endpoint.Scheme {
		case "http":
			proxy.Type = schemas.HTTPProxy
			if endpoint.Port() == "" {
				endpoint.Host = net.JoinHostPort(endpoint.Hostname(), "80")
			}
		case "socks5":
			proxy.Type = schemas.Socks5Proxy
		default:
			return effectiveProviderConfig{}, fmt.Errorf("normalize attempt proxy: unsupported transport")
		}
		username, password := "", ""
		if endpoint.User != nil {
			username = endpoint.User.Username()
			password, _ = endpoint.User.Password()
		}
		endpoint.User = nil
		proxy.URL = &schemas.SecretVar{Val: endpoint.String(), SecretType: schemas.SecretTypePlainText}
		proxy.Username = &schemas.SecretVar{Val: username, SecretType: schemas.SecretTypePlainText}
		proxy.Password = &schemas.SecretVar{Val: password, SecretType: schemas.SecretTypePlainText}
	default:
		return effectiveProviderConfig{}, fmt.Errorf("normalize attempt proxy: unsupported mode")
	}
	config.ProxyConfig = proxy
	partitioned, err := newEffectiveProviderConfig(base.provider, base.targetBaseURL, base.custom, config)
	if err != nil {
		return effectiveProviderConfig{}, err
	}
	partitioned.targetCanonical = append([]byte(nil), base.targetCanonical...)
	partitioned.targetFingerprint = base.targetFingerprint
	if partitioned.targetFingerprint == "" {
		partitioned.targetCanonical = append([]byte(nil), base.canonical...)
		partitioned.targetFingerprint = base.fingerprint
	}
	return partitioned, nil
}

func buildDeepSeekResponsesProbeConfig(
	resolved channel.ResolvedTarget,
	allowPrivateNetwork bool,
) (effectiveProviderConfig, error) {
	return buildEffectiveProviderConfigForAttempt(
		resolved,
		execution.AttemptSpec{
			ClientProtocol: protocol.OpenAIResponses,
			Operation:      execution.OperationProbe,
			RouteMode:      execution.RouteNative,
		},
		allowPrivateNetwork,
	)
}

func resolveSDKProviderConfig(resolved channel.ResolvedTarget) (schemas.ModelProvider, string, bool, error) {
	if preset, ok := sdkProviderSpecFor(resolved.ProviderKind); ok {
		baseURL, configured, err := targetBaseURL(resolved.TargetConfig)
		if err != nil {
			return "", "", false, err
		}
		if !configured {
			baseURL, _, err = sdkDefaultBaseURL(resolved.ProviderKind)
			if err != nil {
				return "", "", false, err
			}
		}
		return preset.provider, baseURL, false, nil
	}
	if resolved.ProviderKind != channel.ProviderOpenAICompatible {
		return "", "", false, fmt.Errorf("unsupported provider kind %q", resolved.ProviderKind)
	}
	baseURL, configured, err := targetBaseURL(resolved.TargetConfig)
	if err != nil || !configured {
		return "", "", false, fmt.Errorf("compatible provider base URL is required")
	}
	return customProviderKey(schemas.OpenAI, baseURL), baseURL, true, nil
}

func targetBaseURL(raw json.RawMessage) (string, bool, error) {
	var target struct {
		BaseURL string `json:"base_url"`
	}
	if err := json.Unmarshal(raw, &target); err != nil {
		return "", false, fmt.Errorf("decode provider target: %w", err)
	}
	if target.BaseURL == "" {
		return "", false, nil
	}
	return target.BaseURL, true, nil
}

func cloneEffectiveProviderConfig(source effectiveProviderConfig) effectiveProviderConfig {
	clone := source
	clone.providerConfig = cloneProviderConfig(source.providerConfig)
	clone.canonical = append([]byte(nil), source.canonical...)
	clone.baseCanonical = append([]byte(nil), source.baseCanonical...)
	clone.targetCanonical = append([]byte(nil), source.targetCanonical...)
	return clone
}

func partitionProviderRuntime(
	config effectiveProviderConfig,
	credential execution.CredentialSnapshot,
) (effectiveProviderConfig, error) {
	baseCanonical := append([]byte(nil), config.canonical...)
	baseFingerprint := config.fingerprint
	canonical, err := json.Marshal(struct {
		Base               json.RawMessage `json:"base"`
		CredentialID       uint            `json:"credential_id"`
		IdentityGeneration uint64          `json:"identity_generation"`
	}{
		Base:               baseCanonical,
		CredentialID:       credential.ID,
		IdentityGeneration: credential.IdentityGeneration,
	})
	if err != nil {
		return effectiveProviderConfig{}, fmt.Errorf("encode provider runtime partition: %w", err)
	}
	digest := sha256.Sum256(canonical)
	config.baseCanonical = baseCanonical
	config.baseFingerprint = baseFingerprint
	config.credentialPartitionID = credential.ID
	config.canonical = canonical
	config.fingerprint = hex.EncodeToString(digest[:])
	return config, nil
}
