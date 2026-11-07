package control

import (
	"context"
	"errors"
	"math"
	"net/http"
	"strings"

	"gorm.io/gorm"

	"gpt-load/internal/channel/spec"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func (s *Service) RetryGroupSiteInfo(ctx context.Context, groupID uint) error {
	result := s.db.WithContext(ctx).Model(&models.GroupSiteConfig{}).
		Where("group_id = ? AND access_token <> '' AND user_id <> '' AND site_root_url <> ''", groupID).
		Updates(map[string]any{"status": "pending", "last_error": ""})
	if result.Error != nil {
		return app_errors.ParseDBError(result.Error)
	}
	if result.RowsAffected == 0 {
		return app_errors.ErrBadRequest
	}
	if s.siteInfoSync != nil {
		s.siteInfoSync.RequestGroupSync(groupID)
	}
	return nil
}

type siteToken struct {
	Key    string `json:"key"`
	Group  string `json:"group"`
	Status int    `json:"status"`
}

// siteInfoTokenGroup identifies a group only when all matching keys agree.
func siteInfoTokenGroup(rawKey, userGroup string, tokens []siteToken) (string, string) {
	rawKey = strings.TrimPrefix(rawKey, "sk-")
	if len(rawKey) < 8 {
		return "", ""
	}
	userGroup = strings.TrimSpace(userGroup)
	if userGroup == "" {
		userGroup = "default"
	}
	candidate := ""
	allExplicit := true
	for _, token := range tokens {
		if token.Status != 0 && token.Status != 1 {
			continue
		}
		key := strings.TrimPrefix(token.Key, "sk-")
		if len(key) < 8 || key[:4] != rawKey[:4] || key[len(key)-4:] != rawKey[len(rawKey)-4:] {
			continue
		}
		group := strings.TrimSpace(token.Group)
		if group == "" {
			group = userGroup
			allExplicit = false
		}
		if candidate != "" && candidate != group {
			return "", ""
		}
		candidate = group
	}
	if candidate == "auto" {
		return "auto", "auto"
	}
	if allExplicit {
		return candidate, candidate
	}
	return "", candidate
}

type GroupSiteAccessResponse struct {
	UserID          string                 `json:"user_id"`
	TokenConfigured bool                   `json:"token_configured"`
	SiteRootURL     string                 `json:"site_root_url"`
	Snapshot        *GroupSiteInfoSnapshot `json:"snapshot"`
}
type GroupSiteInfoSnapshot struct {
	Status            string   `json:"status"`
	SiteRootURL       string   `json:"site_root_url"`
	SiteName          string   `json:"site_name"`
	RemainingAmount   *float64 `json:"remaining_amount"`
	CurrencySymbol    string   `json:"currency_symbol"`
	EffectiveGroup    string   `json:"effective_group"`
	GroupRatio        *float64 `json:"group_ratio"`
	AnnouncementCount int      `json:"announcement_count"`
	FetchedAtMS       *int64   `json:"fetched_at_ms"`
	LastError         string   `json:"last_error"`
}
type GroupSiteAccessUpdateInput struct {
	UserID                optionalField[string] `json:"user_id"`
	AccessToken           optionalField[string] `json:"access_token"`
	ClearAccessToken      bool                  `json:"clear_access_token"`
	ConfirmSiteRootChange bool                  `json:"confirm_site_root_change"`
}

func finiteSiteInfoFloat(value *float64) *float64 {
	if value == nil || math.IsNaN(*value) || math.IsInf(*value, 0) {
		return nil
	}
	return value
}

var siteInvalidGroupSettings = &app_errors.APIError{HTTPStatus: http.StatusBadRequest, Code: "INVALID_GROUP_SETTINGS", Message: "bad_request"}
var siteInvalidGroupCreate = &app_errors.APIError{HTTPStatus: http.StatusBadRequest, Code: "INVALID_GROUP_CREATE", Message: "bad_request"}
var siteRootChangeConfirmationRequired = &app_errors.APIError{HTTPStatus: http.StatusBadRequest, Code: "SITE_ROOT_CHANGE_CONFIRMATION_REQUIRED", Message: "bad_request"}

func (s *Service) readSiteAccess(tx *gorm.DB, groupID uint) (GroupSiteAccessResponse, error) {
	var cfg models.GroupSiteConfig
	err := tx.Where("group_id = ?", groupID).Take(&cfg).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return GroupSiteAccessResponse{}, nil
	}
	if err != nil {
		return GroupSiteAccessResponse{}, app_errors.ParseDBError(err)
	}
	result := GroupSiteAccessResponse{UserID: cfg.UserID, TokenConfigured: cfg.AccessToken != "", SiteRootURL: cfg.SiteRootURL}
	if cfg.AccessToken != "" {
		var count int64
		if err := tx.Model(&models.SiteAnnouncement{}).Where("group_id = ?", groupID).Count(&count).Error; err != nil {
			return result, app_errors.ParseDBError(err)
		}
		result.Snapshot = &GroupSiteInfoSnapshot{Status: cfg.Status, SiteRootURL: cfg.SiteRootURL, SiteName: cfg.SiteName, RemainingAmount: finiteSiteInfoFloat(cfg.RemainingAmount), CurrencySymbol: cfg.CurrencySymbol, EffectiveGroup: cfg.EffectiveGroup, GroupRatio: finiteSiteInfoFloat(cfg.GroupRatio), AnnouncementCount: int(count), FetchedAtMS: cfg.FetchedAtMS, LastError: cfg.LastError}
	}
	return result, nil
}

// invalidateSiteCredential prevents in-flight fetches publishing an old key's ratio.
func (s *Service) invalidateSiteCredential(tx *gorm.DB, groupID uint) (bool, error) {
	result := tx.Model(&models.GroupSiteConfig{}).Where("group_id = ? AND access_token <> ''", groupID).Updates(map[string]any{
		"config_version": gorm.Expr("config_version + 1"), "token_group": "", "effective_group": "", "group_ratio": nil,
		"status": "pending", "last_error": "", "last_attempt_at_ms": nil, "updated_at_ms": s.now().UTC().UnixMilli(),
	})
	if result.Error != nil {
		return false, app_errors.ParseDBError(result.Error)
	}
	return result.RowsAffected != 0, nil
}

func (s *Service) updateSiteAccess(tx *gorm.DB, group models.Group, input optionalField[GroupSiteAccessUpdateInput]) (bool, error) {
	var cfg models.GroupSiteConfig
	err := tx.Where("group_id = ?", group.ID).Take(&cfg).Error
	exists := err == nil
	if err != nil && !errors.Is(err, gorm.ErrRecordNotFound) {
		return false, app_errors.ParseDBError(err)
	}
	if !exists {
		cfg = models.GroupSiteConfig{GroupID: group.ID, ConfigVersion: 1, CurrencySymbol: "$", Status: "unconfigured"}
	}
	if !input.Set && cfg.AccessToken == "" {
		return false, nil
	}
	if input.Set && input.Null {
		return false, siteInvalidGroupSettings
	}
	in := input.Value
	if in.UserID.Set && in.UserID.Null || in.AccessToken.Set && in.AccessToken.Null {
		return false, siteInvalidGroupSettings
	}
	userID := cfg.UserID
	if in.UserID.Set {
		userID = strings.TrimSpace(in.UserID.Value)
	}
	token := cfg.AccessToken
	newToken := in.AccessToken.Set && strings.TrimSpace(in.AccessToken.Value) != ""
	if in.ClearAccessToken {
		token = ""
		newToken = false
	} else if newToken {
		token, err = s.encryption.Encrypt(in.AccessToken.Value)
		if err != nil {
			return false, app_errors.ErrInternalServer
		}
	}
	root := ""
	if token != "" && group.ProviderURL != nil {
		if newToken || userID != cfg.UserID {
			if _, err := spec.NormalizeProviderURL(*group.ProviderURL); err != nil {
				return false, siteInvalidGroupSettings
			}
		}
		root, err = spec.ExtractSiteRootURL(*group.ProviderURL)
		if err != nil {
			return false, siteInvalidGroupSettings
		}
	}
	if token != "" && (userID == "" || root == "") {
		return false, siteInvalidGroupSettings
	}
	if cfg.AccessToken != "" && !in.ClearAccessToken && !newToken && root != cfg.SiteRootURL && !in.ConfirmSiteRootChange {
		return false, siteRootChangeConfirmationRequired
	}
	changed := userID != cfg.UserID || token != cfg.AccessToken || (token != "" && root != cfg.SiteRootURL) || in.ClearAccessToken
	if !exists && !input.Set {
		return false, nil
	}
	if !changed && exists {
		return false, nil
	}
	cfg.UserID = userID
	cfg.AccessToken = token
	if token != "" {
		cfg.SiteRootURL = root
	} else {
		cfg.SiteRootURL = ""
	}
	if exists {
		cfg.ConfigVersion++
	}
	cfg.SiteName = ""
	cfg.BalanceQuota = nil
	cfg.QuotaPerUnit = nil
	cfg.RemainingAmount = nil
	cfg.CurrencySymbol = "$"
	cfg.UserGroup = ""
	cfg.TokenGroup = ""
	cfg.EffectiveGroup = ""
	cfg.GroupRatio = nil
	cfg.FetchedAtMS = nil
	cfg.LastAttemptAtMS = nil
	cfg.LastError = ""
	cfg.AnnouncementStatus, cfg.NoticeStatus = "pending", "pending"
	cfg.AnnouncementFetchedAtMS, cfg.NoticeFetchedAtMS = nil, nil
	cfg.Status = "unconfigured"
	if token != "" {
		cfg.Status = "pending"
	}
	cfg.UpdatedAtMS = s.now().UTC().UnixMilli()
	if err := tx.Where("group_id = ?", group.ID).Delete(&models.SiteAnnouncement{}).Error; err != nil {
		return false, app_errors.ParseDBError(err)
	}
	if err := tx.Save(&cfg).Error; err != nil {
		return false, app_errors.ParseDBError(err)
	}
	return token != "", nil
}
