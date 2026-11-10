package control

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"gorm.io/gorm"

	"gpt-load/internal/platform/httpclient"
	"gpt-load/internal/storage/models"
)

const siteInfoRefreshInterval = 60 * time.Minute
const siteInfoSyncTimeout = 45 * time.Second
const siteInfoMaxConcurrentSyncs = 8

// fetchSiteJSON never follows redirects, so site credentials cannot be forwarded
// to a different origin. Error text intentionally excludes response bodies and URLs.
func fetchSiteJSON(ctx context.Context, client *http.Client, root, path, userID, token string, result any) error {
	safeClient := *client
	safeClient.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	for attempt := 0; attempt < 2; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, root+path, nil)
		if err != nil {
			return fmt.Errorf("invalid site request")
		}
		if token != "" {
			auth := token
			if attempt == 1 {
				auth = "Bearer " + token
			}
			req.Header.Set("Authorization", auth)
			req.Header.Set("New-Api-User", userID)
		}
		resp, err := safeClient.Do(req)
		if err != nil {
			return fmt.Errorf("site request failed")
		}
		if resp.StatusCode == http.StatusUnauthorized && attempt == 0 && token != "" && !strings.HasPrefix(token, "Bearer ") {
			resp.Body.Close()
			continue
		}
		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			resp.Body.Close()
			return fmt.Errorf("site returned HTTP %d", resp.StatusCode)
		}
		var envelope struct {
			Success bool            `json:"success"`
			Data    json.RawMessage `json:"data"`
		}
		err = json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&envelope)
		resp.Body.Close()
		if err != nil || !envelope.Success || len(envelope.Data) == 0 || bytes.Equal(bytes.TrimSpace(envelope.Data), []byte("null")) {
			return fmt.Errorf("invalid site response")
		}
		if err = json.Unmarshal(envelope.Data, result); err != nil {
			return fmt.Errorf("invalid site response data")
		}
		return nil
	}
	return fmt.Errorf("site unauthorized")
}

func parseSiteTokens(raw json.RawMessage) ([]siteToken, error) {
	var tokens []siteToken
	if bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, fmt.Errorf("invalid token list")
	}
	if err := json.Unmarshal(raw, &tokens); err == nil {
		return tokens, nil
	}
	var page struct {
		Items json.RawMessage `json:"items"`
		Data  json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(raw, &page); err != nil {
		return nil, err
	}
	if len(page.Items) != 0 {
		raw = page.Items
	} else {
		raw = page.Data
	}
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, fmt.Errorf("missing token list")
	}
	if err := json.Unmarshal(raw, &tokens); err != nil {
		return nil, err
	}
	return tokens, nil
}

func sanitizeSiteError(err error, token string) string {
	if err == nil {
		return ""
	}
	message := err.Error()
	if token != "" {
		message = strings.ReplaceAll(message, token, "[REDACTED]")
	}
	runes := []rune(message)
	if len(runes) > 240 {
		runes = runes[:240]
	}
	return string(runes)
}

func siteInfoAmount(quota int64, quotaPerUnit float64, currencyType, customSymbol string, customRate, usdRate float64) (string, float64) {
	if quotaPerUnit <= 0 {
		quotaPerUnit = 500000
	}
	amount := float64(quota) / quotaPerUnit
	if currencyType == "CUSTOM" && customSymbol != "" && customRate > 0 {
		return customSymbol, amount * customRate
	}
	if currencyType == "CNY" && usdRate > 0 {
		return "¥", amount * usdRate
	}
	return "$", amount
}

type SiteInfoSyncCoordinator struct {
	service *Service
	clients *httpclient.HTTPClientManager
	wake    chan struct{}
	mu      sync.Mutex
	active  map[uint]struct{}
	workers sync.WaitGroup
	timeout time.Duration
}

func NewSiteInfoSyncCoordinator(service *Service, clients *httpclient.HTTPClientManager) *SiteInfoSyncCoordinator {
	c := &SiteInfoSyncCoordinator{service: service, clients: clients, wake: make(chan struct{}, 1), active: make(map[uint]struct{}), timeout: siteInfoSyncTimeout}
	service.siteInfoSync = c
	return c
}
func (c *SiteInfoSyncCoordinator) RequestGroupSync(_ uint) {
	select {
	case c.wake <- struct{}{}:
	default:
	}
}
func (c *SiteInfoSyncCoordinator) Run(ctx context.Context) {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	defer c.workers.Wait()
	c.scan(ctx)
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.scan(ctx)
		case <-c.wake:
			c.scan(ctx)
		}
	}
}
func (c *SiteInfoSyncCoordinator) scan(ctx context.Context) {
	var configs []models.GroupSiteConfig
	now := c.service.now().UTC().UnixMilli()
	if err := c.service.db.WithContext(ctx).Where("access_token <> '' AND user_id <> '' AND site_root_url <> '' AND (status = ? OR last_attempt_at_ms IS NULL OR last_attempt_at_ms <= ?)", "pending", now-siteInfoRefreshInterval.Milliseconds()).Order("CASE WHEN last_attempt_at_ms IS NULL THEN 0 ELSE 1 END").Order("last_attempt_at_ms ASC").Order("group_id ASC").Find(&configs).Error; err != nil {
		return
	}
	for _, cfg := range configs {
		if ctx.Err() != nil {
			return
		}
		c.mu.Lock()
		if _, exists := c.active[cfg.GroupID]; exists {
			c.mu.Unlock()
			continue
		}
		if len(c.active) >= siteInfoMaxConcurrentSyncs {
			c.mu.Unlock()
			return
		}
		id := cfg.GroupID
		c.active[id] = struct{}{}
		c.workers.Add(1)
		c.mu.Unlock()
		go func() {
			defer c.workers.Done()
			_ = c.syncGroup(ctx, id)
			c.mu.Lock()
			delete(c.active, id)
			c.mu.Unlock()
			if ctx.Err() == nil {
				c.RequestGroupSync(id)
			}
		}()
	}
}

type siteStatus struct {
	SystemName    json.RawMessage `json:"system_name"`
	QuotaPerUnit  json.RawMessage `json:"quota_per_unit"`
	DisplayType   json.RawMessage `json:"quota_display_type"`
	CurrencyType  json.RawMessage `json:"currency_type"`
	CustomSymbol  json.RawMessage `json:"custom_currency_symbol"`
	CustomRate    json.RawMessage `json:"custom_currency_exchange_rate"`
	USDRate       json.RawMessage `json:"usd_exchange_rate"`
	Announcements json.RawMessage `json:"announcements"`
}

type siteAnnouncement struct {
	Content     string          `json:"content"`
	PublishDate json.RawMessage `json:"publishDate"`
	Type        json.RawMessage `json:"type"`
	Extra       json.RawMessage `json:"extra"`
}

func siteDedup(source string, values ...string) string {
	sum := sha256.Sum256([]byte(strings.Join(append([]string{source}, values...), "\x00")))
	return source + ":" + hex.EncodeToString(sum[:])
}
func siteEventAt(value string, fallback int64) int64 {
	if t, err := time.Parse(time.RFC3339, value); err == nil {
		return t.UnixMilli()
	}
	return fallback
}

func (c *SiteInfoSyncCoordinator) syncGroup(ctx context.Context, id uint) error {
	parent := ctx
	ctx, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()
	started := time.Now()
	s := c.service
	var cfg models.GroupSiteConfig
	if err := s.db.WithContext(ctx).Where("group_id = ?", id).Take(&cfg).Error; err != nil {
		return err
	}
	defer func() {
		if ctx.Err() == context.DeadlineExceeded && parent.Err() == nil {
			_ = c.storeSiteFailure(parent, cfg, "site sync timed out")
		}
	}()
	var group models.Group
	if err := s.db.WithContext(ctx).First(&group, id).Error; err != nil {
		return err
	}
	token, err := s.encryption.Decrypt(cfg.AccessToken)
	if err != nil {
		return c.storeSiteFailure(ctx, cfg, "site credential decryption failed")
	}
	_, proxy, err := s.resolveGroupProxy(ctx, s.db, group)
	if err != nil {
		return c.storeSiteFailure(ctx, cfg, "site proxy resolution failed")
	}
	client, err := c.clients.NewClientForOutboundProxy(&httpclient.Config{RequestTimeout: 30 * time.Second, DisableRedirects: true}, proxy)
	if err != nil {
		return c.storeSiteFailure(ctx, cfg, "site client configuration failed")
	}
	var user struct {
		Quota *int64 `json:"quota"`
		Group string `json:"group"`
	}
	// Reserve time for optional enrichment and committing the core balance.
	coreCtx, coreCancel := context.WithDeadline(ctx, started.Add(c.timeout*7/10))
	err = fetchSiteJSON(coreCtx, client, cfg.SiteRootURL, "/api/user/self", cfg.UserID, token, &user)
	coreTimedOut := coreCtx.Err() == context.DeadlineExceeded
	coreCancel()
	if err != nil {
		if coreTimedOut && ctx.Err() == nil {
			return c.storeSiteFailure(ctx, cfg, "site sync timed out")
		}
		return c.storeSiteFailure(ctx, cfg, sanitizeSiteError(err, token))
	}
	if user.Quota == nil {
		return c.storeSiteFailure(ctx, cfg, "invalid site response data")
	}
	// Optional endpoints share a deadline and each has a shorter cap, leaving
	// the final 15% of the total budget for the version-checked transaction.
	optionalCtx, optionalCancel := context.WithDeadline(ctx, started.Add(c.timeout*85/100))
	defer optionalCancel()
	optionalFetch := func(path, userID, token string, result any) error {
		limit := c.timeout / 8
		if limit > 5*time.Second {
			limit = 5 * time.Second
		}
		requestCtx, requestCancel := context.WithTimeout(optionalCtx, limit)
		defer requestCancel()
		return fetchSiteJSON(requestCtx, client, cfg.SiteRootURL, path, userID, token, result)
	}
	var status siteStatus
	statusErr := optionalFetch("/api/status", "", "", &status)
	statusOK := statusErr == nil
	conversionOK := statusOK
	var quotaPerUnit float64
	if conversionOK && len(status.QuotaPerUnit) != 0 {
		if bytes.Equal(bytes.TrimSpace(status.QuotaPerUnit), []byte("null")) || json.Unmarshal(status.QuotaPerUnit, &quotaPerUnit) != nil {
			conversionOK = false
		}
	}
	siteName := cfg.SiteName
	if statusOK && len(status.SystemName) != 0 {
		var name string
		if json.Unmarshal(status.SystemName, &name) == nil && !bytes.Equal(bytes.TrimSpace(status.SystemName), []byte("null")) {
			siteName = name
		}
	}
	var currencyType, customSymbol string
	var customRate, usdRate float64
	parseOptional := func(raw json.RawMessage, target any) bool {
		return len(raw) == 0 || (!bytes.Equal(bytes.TrimSpace(raw), []byte("null")) && json.Unmarshal(raw, target) == nil)
	}
	if conversionOK {
		if len(status.DisplayType) != 0 {
			conversionOK = parseOptional(status.DisplayType, &currencyType)
		} else {
			conversionOK = parseOptional(status.CurrencyType, &currencyType)
		}
		if conversionOK && currencyType == "CNY" {
			conversionOK = parseOptional(status.USDRate, &usdRate)
		}
		if conversionOK && currencyType == "CUSTOM" {
			conversionOK = parseOptional(status.CustomSymbol, &customSymbol) && parseOptional(status.CustomRate, &customRate)
		}
	}
	var announcements []json.RawMessage
	announcementsOK := statusOK
	if announcementsOK && len(status.Announcements) != 0 {
		announcementsOK = json.Unmarshal(status.Announcements, &announcements) == nil
		if !announcementsOK {
			statusErr = fmt.Errorf("invalid site announcement data")
		}
	}
	var notice string
	noticeErr := optionalFetch("/api/notice", "", "", &notice)
	noticeOK := noticeErr == nil
	user.Group = strings.TrimSpace(user.Group)
	if user.Group == "" {
		user.Group = "default"
	}
	tokenGroup, effectiveGroup := "", ""
	var enrichmentErr error
	var credential models.Credential
	if s.db.WithContext(ctx).Where("group_id = ?", id).First(&credential).Error == nil {
		_, apiKey, decodeErr := s.decodeCredential(group, credential)
		if decodeErr == nil && len(strings.TrimPrefix(apiKey, "sk-")) >= 8 {
			var tokens []siteToken
			complete := false
			for page := 1; page <= 100; page++ {
				var raw json.RawMessage
				if e := optionalFetch(fmt.Sprintf("/api/token/?p=%d&size=100", page), cfg.UserID, token, &raw); e != nil {
					enrichmentErr = e
					break
				}
				items, e := parseSiteTokens(raw)
				if e != nil {
					enrichmentErr = fmt.Errorf("invalid site token data")
					break
				}
				tokens = append(tokens, items...)
				if len(items) < 100 {
					complete = true
					break
				}
			}
			if complete {
				tokenGroup, effectiveGroup = siteInfoTokenGroup(apiKey, user.Group, tokens)
			} else if enrichmentErr == nil {
				enrichmentErr = fmt.Errorf("site token pagination limit exceeded")
			}
		}
	}
	var ratio *float64
	if effectiveGroup != "" && effectiveGroup != "auto" {
		var raw map[string]json.RawMessage
		if e := optionalFetch("/api/user/self/groups", cfg.UserID, token, &raw); e == nil {
			if value, ok := raw[effectiveGroup]; ok {
				var number *float64
				if json.Unmarshal(value, &number) == nil && number != nil {
					ratio = number
				} else {
					var object struct {
						Ratio *float64 `json:"ratio"`
					}
					if json.Unmarshal(value, &object) == nil && object.Ratio != nil {
						ratio = object.Ratio
					}
				}
			}
			if ratio == nil || math.IsNaN(*ratio) || math.IsInf(*ratio, 0) {
				enrichmentErr = fmt.Errorf("invalid site group ratio data")
			}
		} else {
			enrichmentErr = e
		}
	}
	now := s.now().UTC().UnixMilli()
	var storedAmount, storedQuotaPerUnit any
	symbol := ""
	if conversionOK {
		if quotaPerUnit <= 0 {
			quotaPerUnit = 500000
		}
		var amount float64
		symbol, amount = siteInfoAmount(*user.Quota, quotaPerUnit, currencyType, customSymbol, customRate, usdRate)
		if !math.IsInf(amount, 0) && !math.IsNaN(amount) {
			storedAmount = amount
			storedQuotaPerUnit = quotaPerUnit
		} else {
			conversionOK = false
		}
	}
	return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
		updates := map[string]any{"status": "ok", "last_error": "", "fetched_at_ms": now, "last_attempt_at_ms": now, "updated_at_ms": now, "site_name": siteName, "balance_quota": user.Quota, "quota_per_unit": storedQuotaPerUnit, "remaining_amount": storedAmount, "currency_symbol": symbol, "user_group": user.Group, "token_group": tokenGroup, "effective_group": effectiveGroup, "group_ratio": ratio, "announcement_status": "error", "notice_status": "error", "announcement_last_error": sanitizeSiteError(statusErr, token), "notice_last_error": sanitizeSiteError(noticeErr, token)}
		if enrichmentErr != nil {
			delete(updates, "token_group")
			delete(updates, "effective_group")
			delete(updates, "group_ratio")
			updates["status"] = "error"
			updates["last_error"] = sanitizeSiteError(enrichmentErr, token)
		}
		if !conversionOK {
			delete(updates, "quota_per_unit")
			delete(updates, "remaining_amount")
			delete(updates, "currency_symbol")
			updates["status"] = "error"
			updates["last_error"] = "site status unavailable or invalid"
		}
		if announcementsOK {
			updates["announcement_status"] = "ok"
			updates["announcement_fetched_at_ms"] = now
		}
		if noticeOK {
			updates["notice_status"] = "ok"
			updates["notice_fetched_at_ms"] = now
		}
		result := tx.Model(&models.GroupSiteConfig{}).Where("group_id = ? AND config_version = ?", id, cfg.ConfigVersion).Updates(updates)
		if result.Error != nil {
			return result.Error
		}
		if result.RowsAffected == 0 {
			return nil
		}
		var old []models.SiteAnnouncement
		if err := tx.Where("group_id = ?", id).Find(&old).Error; err != nil {
			return err
		}
		previous := map[string]int64{}
		for _, a := range old {
			previous[a.DedupKey] = a.EventAtMS
		}
		insert := func(source string, records []models.SiteAnnouncement) error {
			if err := tx.Where("group_id = ? AND source = ?", id, source).Delete(&models.SiteAnnouncement{}).Error; err != nil {
				return err
			}
			for i := range records {
				if err := tx.Create(&records[i]).Error; err != nil {
					return err
				}
			}
			return nil
		}
		if announcementsOK {
			records := []models.SiteAnnouncement{}
			seen := map[string]bool{}
			for _, raw := range announcements {
				var a siteAnnouncement
				if json.Unmarshal(raw, &a) != nil {
					continue
				}
				var extra string
				_ = json.Unmarshal(a.Extra, &extra)
				var publishDate, announcementType string
				_ = json.Unmarshal(a.PublishDate, &publishDate)
				_ = json.Unmarshal(a.Type, &announcementType)
				key := siteDedup("announcement", a.Content, extra, publishDate, announcementType)
				if seen[key] {
					continue
				}
				seen[key] = true
				fallback := previous[key]
				if fallback == 0 {
					fallback = now
				}
				level := announcementType
				if level == "" {
					level = "default"
				}
				records = append(records, models.SiteAnnouncement{GroupID: id, SiteRootURL: cfg.SiteRootURL, SiteName: siteName, DedupKey: key, Content: a.Content, Extra: extra, Level: level, Source: "announcement", EventAtMS: siteEventAt(publishDate, fallback), FetchedAtMS: now})
			}
			if err := insert("announcement", records); err != nil {
				return err
			}
		}
		if noticeOK {
			records := []models.SiteAnnouncement{}
			notice = strings.TrimSpace(notice)
			if notice != "" {
				key := siteDedup("notice", notice)
				event := previous[key]
				if event == 0 {
					event = now
				}
				records = append(records, models.SiteAnnouncement{GroupID: id, SiteRootURL: cfg.SiteRootURL, SiteName: siteName, DedupKey: key, Content: notice, Level: "default", Source: "notice", EventAtMS: event, FetchedAtMS: now})
			}
			if err := insert("notice", records); err != nil {
				return err
			}
		}
		return nil
	})
}
func (c *SiteInfoSyncCoordinator) storeSiteFailure(ctx context.Context, cfg models.GroupSiteConfig, message string) error {
	now := c.service.now().UTC().UnixMilli()
	return c.service.db.WithContext(ctx).Model(&models.GroupSiteConfig{}).Where("group_id = ? AND config_version = ?", cfg.GroupID, cfg.ConfigVersion).Updates(map[string]any{"status": "error", "last_error": message, "last_attempt_at_ms": now, "updated_at_ms": now, "announcement_status": "error", "notice_status": "error", "announcement_last_error": message, "notice_last_error": message}).Error
}
