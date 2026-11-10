package control

import (
	"context"
	"sort"
	"strings"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/storage/dbtx"
)

type AnnouncementsResponse struct {
	Items   []AnnouncementItem       `json:"items"`
	Sites   []AnnouncementSiteOption `json:"sites"`
	Sources []AnnouncementSource     `json:"sources,omitempty"`
}
type AnnouncementSourceState struct {
	Status      string `json:"status"`
	FetchedAtMS *int64 `json:"fetched_at_ms"`
	LastError   string `json:"last_error,omitempty"`
}
type AnnouncementSource struct {
	Group        AnnouncementGroupRef    `json:"group"`
	SiteRootURL  string                  `json:"site_root_url"`
	SiteName     string                  `json:"site_name"`
	Announcement AnnouncementSourceState `json:"announcement"`
	Notice       AnnouncementSourceState `json:"notice"`
}
type AnnouncementGroupRef struct {
	ID   uint   `json:"id"`
	Name string `json:"name"`
}
type AnnouncementItem struct {
	ID          string                 `json:"id"`
	SiteRootURL string                 `json:"site_root_url"`
	SiteName    string                 `json:"site_name"`
	Content     string                 `json:"content"`
	Extra       string                 `json:"extra"`
	Level       string                 `json:"level"`
	Source      string                 `json:"source"`
	EventAtMS   int64                  `json:"event_at_ms"`
	FetchedAtMS int64                  `json:"fetched_at_ms"`
	Groups      []AnnouncementGroupRef `json:"groups"`
}
type AnnouncementSiteOption struct {
	SiteRootURL string `json:"site_root_url"`
	SiteName    string `json:"site_name"`
	Count       int    `json:"count"`
}
type announcementRow struct {
	ID          uint
	GroupID     uint
	GroupName   string
	SiteRootURL string
	SiteName    string
	DedupKey    string
	Content     string
	Extra       string
	Level       string
	Source      string
	EventAtMS   int64
	FetchedAtMS int64
}

func aggregateAnnouncements(rows []announcementRow, site, query string) AnnouncementsResponse {
	result := AnnouncementsResponse{Items: []AnnouncementItem{}, Sites: []AnnouncementSiteOption{}, Sources: []AnnouncementSource{}}
	groups := make(map[[2]string]*AnnouncementItem)
	for _, row := range rows {
		key := [2]string{row.SiteRootURL, row.DedupKey}
		item := groups[key]
		if item == nil {
			item = &AnnouncementItem{ID: row.SiteRootURL + "#" + row.DedupKey, SiteRootURL: row.SiteRootURL, SiteName: row.SiteName, Content: row.Content, Extra: row.Extra, Level: row.Level, Source: row.Source, EventAtMS: row.EventAtMS, FetchedAtMS: row.FetchedAtMS, Groups: []AnnouncementGroupRef{}}
			groups[key] = item
		}
		if item.SiteName == "" && row.SiteName != "" {
			item.SiteName = row.SiteName
		}
		if row.EventAtMS > item.EventAtMS {
			item.EventAtMS = row.EventAtMS
		}
		if row.FetchedAtMS > item.FetchedAtMS {
			item.FetchedAtMS = row.FetchedAtMS
		}
		found := false
		for _, group := range item.Groups {
			if group.ID == row.GroupID {
				found = true
				break
			}
		}
		if !found {
			item.Groups = append(item.Groups, AnnouncementGroupRef{ID: row.GroupID, Name: row.GroupName})
		}
	}
	sites := make(map[string]*AnnouncementSiteOption)
	query = strings.ToLower(strings.TrimSpace(query))
	for _, item := range groups {
		sort.Slice(item.Groups, func(i, j int) bool { return item.Groups[i].ID < item.Groups[j].ID })
		option := sites[item.SiteRootURL]
		if option == nil {
			option = &AnnouncementSiteOption{SiteRootURL: item.SiteRootURL}
			sites[item.SiteRootURL] = option
		}
		option.Count++
		if item.SiteName != "" && (option.SiteName == "" || item.SiteName < option.SiteName) {
			option.SiteName = item.SiteName
		}
		if site != "" && item.SiteRootURL != site {
			continue
		}
		match := query == "" || strings.Contains(strings.ToLower(item.Content), query) || strings.Contains(strings.ToLower(item.Extra), query) || strings.Contains(strings.ToLower(item.SiteName), query) || strings.Contains(strings.ToLower(item.SiteRootURL), query)
		if !match {
			for _, group := range item.Groups {
				if strings.Contains(strings.ToLower(group.Name), query) {
					match = true
					break
				}
			}
		}
		if match {
			result.Items = append(result.Items, *item)
		}
	}
	for _, option := range sites {
		result.Sites = append(result.Sites, *option)
	}
	sort.Slice(result.Sites, func(i, j int) bool { return result.Sites[i].SiteRootURL < result.Sites[j].SiteRootURL })
	sort.Slice(result.Items, func(i, j int) bool {
		if result.Items[i].EventAtMS != result.Items[j].EventAtMS {
			return result.Items[i].EventAtMS > result.Items[j].EventAtMS
		}
		return result.Items[i].ID < result.Items[j].ID
	})
	return result
}

func (s *Service) GetAnnouncements(ctx context.Context, site, query string) (AnnouncementsResponse, error) {
	var rows []announcementRow
	var configs []struct {
		GroupID                 uint
		GroupName               string
		SiteRootURL             string
		SiteName                string
		AnnouncementStatus      string
		NoticeStatus            string
		AnnouncementFetchedAtMS *int64
		NoticeFetchedAtMS       *int64
		AnnouncementLastError   string
		NoticeLastError         string
	}
	err := dbtx.Run(ctx, s.db, dbtx.Options{Mode: dbtx.ReadSnapshot, Operation: "list announcements"}, func(tx *gorm.DB) error {
		if err := tx.Table("site_announcements").Select("site_announcements.*, groups.name AS group_name").Joins("JOIN groups ON groups.id = site_announcements.group_id").Order("site_announcements.id ASC").Scan(&rows).Error; err != nil {
			return err
		}
		return tx.Table("group_site_configs").Select("group_site_configs.group_id, groups.name AS group_name, site_root_url, site_name, announcement_status, notice_status, announcement_fetched_at_ms, notice_fetched_at_ms, announcement_last_error, notice_last_error").Joins("JOIN groups ON groups.id = group_site_configs.group_id").Where("access_token <> '' AND site_root_url <> ''").Order("group_site_configs.group_id ASC").Scan(&configs).Error
	})
	if err != nil {
		return AnnouncementsResponse{}, app_errors.ParseDBError(err)
	}
	result := aggregateAnnouncements(rows, site, query)
	state := func(status string, fetched *int64, lastError string) AnnouncementSourceState {
		if status == "" {
			status = "pending"
		}
		if status == "ok" && fetched != nil && s.now().UTC().UnixMilli()-*fetched > 2*siteInfoRefreshInterval.Milliseconds() {
			status = "stale"
		}
		return AnnouncementSourceState{Status: status, FetchedAtMS: fetched, LastError: lastError}
	}
	for _, cfg := range configs {
		result.Sources = append(result.Sources, AnnouncementSource{Group: AnnouncementGroupRef{ID: cfg.GroupID, Name: cfg.GroupName}, SiteRootURL: cfg.SiteRootURL, SiteName: cfg.SiteName, Announcement: state(cfg.AnnouncementStatus, cfg.AnnouncementFetchedAtMS, cfg.AnnouncementLastError), Notice: state(cfg.NoticeStatus, cfg.NoticeFetchedAtMS, cfg.NoticeLastError)})
	}
	return result, nil
}

func (s *Server) handleGetAnnouncements(c *gin.Context) {
	result, err := s.service.GetAnnouncements(c.Request.Context(), c.Query("site"), c.Query("q"))
	if err != nil {
		writeServiceError(c, "list_announcements", err)
		return
	}
	if c.Query("include") != "source_status" {
		result.Sources = nil
	}
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRetryGroupSiteInfo(c *gin.Context) {
	id, ok := groupID(c, "retry_group_site_info")
	if !ok {
		return
	}
	if err := s.service.RetryGroupSiteInfo(c.Request.Context(), id); err != nil {
		writeServiceError(c, "retry_group_site_info", err)
		return
	}
	response.SuccessI18n(c, "common.success", nil)
}
