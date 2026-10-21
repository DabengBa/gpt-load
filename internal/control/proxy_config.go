package control

import (
	"context"
	"errors"

	"gorm.io/gorm"

	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/platform/encryption"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func normalizeProxyOverride(
	field optionalField[outboundproxy.Config],
	encryptionService encryption.Service,
) (*string, bool, error) {
	if !field.Set {
		return nil, false, nil
	}
	if field.Null {
		return nil, true, nil
	}
	config, err := outboundproxy.Normalize(field.Value)
	if err != nil || config.Mode == outboundproxy.ModeInherit {
		return nil, false, app_errors.ErrValidation
	}
	if encryptionService == nil {
		return nil, false, app_errors.ErrInternalServer
	}
	encoded, err := outboundproxy.Encode(config)
	if err != nil {
		return nil, false, app_errors.ErrValidation
	}
	ciphertext, err := encryptionService.Encrypt(encoded)
	if err != nil {
		return nil, false, app_errors.ErrInternalServer
	}
	return &ciphertext, true, nil
}

func (s *Service) globalNetworkContext(
	ctx context.Context,
	db *gorm.DB,
) (proxyNetwork, error) {
	global, err := s.loadGlobalProxyConfig(ctx, db)
	if err != nil {
		return proxyNetwork{}, err
	}
	effective, err := outboundproxy.Resolve(nil, nil, global, s.environmentProxy)
	if err != nil {
		return proxyNetwork{}, app_errors.ErrInternalServer
	}
	return s.proxyNetworkContext(effective)
}

func (s *Service) draftNetworkContext(
	ctx context.Context,
	config *outboundproxy.Config,
) (proxyNetwork, error) {
	if config == nil {
		return s.globalNetworkContext(ctx, s.db)
	}
	normalized, err := outboundproxy.Normalize(*config)
	if err != nil || normalized.Mode == outboundproxy.ModeInherit {
		return proxyNetwork{}, app_errors.ErrValidation
	}
	effective, err := outboundproxy.Resolve(nil, &normalized, nil, nil)
	if err != nil {
		return proxyNetwork{}, app_errors.ErrValidation
	}
	return s.proxyNetworkContext(effective)
}

func (s *Service) groupNetworkContext(
	ctx context.Context,
	db *gorm.DB,
	group models.Group,
) (proxyNetwork, error) {
	_, effective, err := s.resolveGroupProxy(ctx, db, group)
	if err != nil {
		return proxyNetwork{}, err
	}
	return s.proxyNetworkContext(effective)
}

func (s *Service) proxyNetworkContext(
	effective outboundproxy.Effective,
) (proxyNetwork, error) {
	effective, err := outboundproxy.NormalizeEffective(effective)
	if err != nil || s.encryption == nil {
		return proxyNetwork{}, app_errors.ErrInternalServer
	}
	identity := ""
	if effective.Config.Mode == outboundproxy.ModeEnvironment {
		identity = `{"mode":"environment"}`
	} else {
		identity, err = outboundproxy.Encode(effective.Config)
		if err != nil {
			return proxyNetwork{}, app_errors.ErrInternalServer
		}
	}
	return proxyNetwork{
		Proxy: effective, Fingerprint: s.encryption.Hash(identity),
	}, nil
}

func decryptProxyOverride(
	encryptionService encryption.Service,
	ciphertext *string,
) (*outboundproxy.Config, error) {
	if ciphertext == nil {
		return nil, nil
	}
	if encryptionService == nil || *ciphertext == "" {
		return nil, app_errors.ErrInternalServer
	}
	plaintext, err := encryptionService.Decrypt(*ciphertext)
	if err != nil {
		return nil, app_errors.ErrInternalServer
	}
	config, err := outboundproxy.Decode(plaintext)
	plaintext = ""
	if err != nil || config.Mode == outboundproxy.ModeInherit {
		return nil, app_errors.ErrInternalServer
	}
	return &config, nil
}

func storedProxyIdentity(
	encryptionService encryption.Service,
	ciphertext *string,
) (string, string, error) {
	if ciphertext == nil {
		return "", "", nil
	}
	if encryptionService == nil || *ciphertext == "" {
		return "", "", app_errors.ErrInternalServer
	}
	plaintext, err := encryptionService.Decrypt(*ciphertext)
	if err != nil {
		return "", "", app_errors.ErrInternalServer
	}
	config, err := outboundproxy.Decode(plaintext)
	if err != nil || config.Mode == outboundproxy.ModeInherit {
		plaintext = ""
		return "", "", app_errors.ErrInternalServer
	}
	fingerprint := encryptionService.Hash(plaintext)
	plaintext = ""
	return *ciphertext, fingerprint, nil
}

func (s *Service) loadGlobalProxyConfig(
	ctx context.Context,
	db *gorm.DB,
) (*outboundproxy.Config, error) {
	var row models.SystemSetting
	err := db.WithContext(ctx).
		Select("key", "value").
		Where("key = ?", outboundproxy.SystemSettingKey).
		Take(&row).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, app_errors.ParseDBError(err)
	}
	return decryptProxyOverride(s.encryption, &row.Value)
}

func (s *Service) resolveGroupProxy(
	ctx context.Context,
	db *gorm.DB,
	group models.Group,
) (*outboundproxy.Config, outboundproxy.Effective, error) {
	configured, err := decryptProxyOverride(s.encryption, group.ProxyConfig)
	if err != nil {
		return nil, outboundproxy.Effective{}, err
	}
	global, err := s.loadGlobalProxyConfig(ctx, db)
	if err != nil {
		return nil, outboundproxy.Effective{}, err
	}
	effective, err := outboundproxy.Resolve(nil, configured, global, s.environmentProxy)
	if err != nil {
		return nil, outboundproxy.Effective{}, app_errors.ErrInternalServer
	}
	return configured, effective, nil
}

func (s *Service) groupProxyView(
	ctx context.Context,
	db *gorm.DB,
	group models.Group,
) (outboundproxy.View, error) {
	configured, effective, err := s.resolveGroupProxy(ctx, db, group)
	if err != nil {
		return outboundproxy.View{}, err
	}
	view, err := outboundproxy.NewView(configured, effective)
	if err != nil {
		return outboundproxy.View{}, app_errors.ErrInternalServer
	}
	return view, nil
}

type proxyNetwork struct {
	Proxy       outboundproxy.Effective
	Fingerprint string
}
