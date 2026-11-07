package models

type GroupSiteConfig struct {
	AnnouncementLastError   string   `gorm:"type:text;not null;default:''"`
	NoticeLastError         string   `gorm:"type:text;not null;default:''"`
	AnnouncementStatus      string   `gorm:"not null;default:'pending'"`
	NoticeStatus            string   `gorm:"not null;default:'pending'"`
	AnnouncementFetchedAtMS *int64   `gorm:"column:announcement_fetched_at_ms"`
	NoticeFetchedAtMS       *int64   `gorm:"column:notice_fetched_at_ms"`
	GroupID                 uint     `gorm:"primaryKey;autoIncrement:false"`
	ConfigVersion           int64    `gorm:"column:config_version;not null;default:1"`
	UserID                  string   `gorm:"type:varchar(128);not null;default:''"`
	AccessToken             string   `gorm:"type:text;not null;default:''"`
	SiteRootURL             string   `gorm:"type:varchar(512);not null;default:''"`
	SiteName                string   `gorm:"type:varchar(255);not null;default:''"`
	BalanceQuota            *int64   `gorm:"column:balance_quota"`
	QuotaPerUnit            *float64 `gorm:"column:quota_per_unit"`
	RemainingAmount         *float64 `gorm:"column:remaining_amount"`
	CurrencySymbol          string   `gorm:"type:varchar(16);not null;default:'$'"`
	UserGroup               string   `gorm:"type:varchar(128);not null;default:''"`
	TokenGroup              string   `gorm:"type:varchar(128);not null;default:''"`
	EffectiveGroup          string   `gorm:"type:varchar(128);not null;default:''"`
	GroupRatio              *float64 `gorm:"column:group_ratio"`
	Status                  string   `gorm:"type:varchar(32);not null;default:'unconfigured'"`
	LastError               string   `gorm:"type:text;not null;default:''"`
	FetchedAtMS             *int64   `gorm:"column:fetched_at_ms;index"`
	LastAttemptAtMS         *int64   `gorm:"column:last_attempt_at_ms;index"`
	UpdatedAtMS             int64    `gorm:"column:updated_at_ms;not null"`
}

type SiteAnnouncement struct {
	ID          uint   `gorm:"primaryKey;autoIncrement"`
	GroupID     uint   `gorm:"not null;index;uniqueIndex:idx_site_announcements_group_dedup,priority:1"`
	SiteRootURL string `gorm:"type:varchar(512);not null;index"`
	SiteName    string `gorm:"type:varchar(255);not null;default:''"`
	DedupKey    string `gorm:"type:varchar(191);not null;uniqueIndex:idx_site_announcements_group_dedup,priority:2"`
	Content     string `gorm:"type:text;not null"`
	Extra       string `gorm:"type:text;not null;default:''"`
	Level       string `gorm:"type:varchar(32);not null;default:'default'"`
	Source      string `gorm:"type:varchar(32);not null;default:'announcement'"`
	EventAtMS   int64  `gorm:"column:event_at_ms;not null;index"`
	FetchedAtMS int64  `gorm:"column:fetched_at_ms;not null"`
}
