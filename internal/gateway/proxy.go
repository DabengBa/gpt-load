package gateway

import (
	"fmt"

	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/platform/encryption"
)

// resolveAttemptProxy resolves the Group proxy. Credential-level proxy data is
// intentionally not part of the runtime credential identity anymore.
func resolveAttemptProxy(
	encryptionService encryption.Service,
	groupProxy outboundproxy.Effective,
) (outboundproxy.Effective, string, error) {
	if encryptionService == nil {
		return outboundproxy.Effective{}, "", fmt.Errorf("proxy encryption service is unavailable")
	}
	effective, err := outboundproxy.NormalizeEffective(groupProxy)
	if err != nil {
		return outboundproxy.Effective{}, "", fmt.Errorf("group proxy config is invalid")
	}
	fingerprint, err := effectiveProxyFingerprint(encryptionService, effective)
	if err != nil {
		return outboundproxy.Effective{}, "", err
	}
	return effective, fingerprint, nil
}

func effectiveProxyFingerprint(
	encryptionService encryption.Service,
	effective outboundproxy.Effective,
) (string, error) {
	effective, err := outboundproxy.NormalizeEffective(effective)
	if err != nil {
		return "", fmt.Errorf("proxy config is invalid")
	}
	if effective.Config.Mode == outboundproxy.ModeEnvironment {
		return encryptionService.Hash(`{"mode":"environment"}`), nil
	}
	encoded, err := outboundproxy.Encode(effective.Config)
	if err != nil {
		return "", fmt.Errorf("proxy config is invalid")
	}
	return encryptionService.Hash(encoded), nil
}
