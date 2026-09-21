package models

// AgentCredential is an independently issued machine identity for the
// control-owned Agent surface. Only the HMAC-SHA-256 digest of the Bearer
// secret is persisted; the plaintext secret is returned exactly once at
// creation and can never be recovered from storage.
type AgentCredential struct {
	ID           uint   `gorm:"primaryKey;autoIncrement"`
	Name         string `gorm:"type:varchar(255);not null"`
	SecretHash   string `gorm:"column:secret_hash;type:varchar(128);not null;uniqueIndex"`
	Scopes       JSON   `gorm:"type:json;not null"`
	Status       string `gorm:"type:varchar(16);not null;default:'active';check:chk_agent_credentials_status,status IN ('active','disabled')"`
	ExpiresAtMS  *int64 `gorm:"column:expires_at_ms;check:chk_agent_credentials_expires_at,expires_at_ms IS NULL OR expires_at_ms >= 0"`
	DisabledAtMS *int64 `gorm:"column:disabled_at_ms;check:chk_agent_credentials_disabled_at,disabled_at_ms IS NULL OR disabled_at_ms >= 0"`
	CreatedAtMS  int64  `gorm:"column:created_at_ms;not null;autoCreateTime:milli;check:chk_agent_credentials_created_at,created_at_ms >= 0"`
	UpdatedAtMS  int64  `gorm:"column:updated_at_ms;not null;autoUpdateTime:milli;check:chk_agent_credentials_updated_at,updated_at_ms >= 0"`
}

// TableName pins the reviewed storage name for the Agent credential ledger.
func (AgentCredential) TableName() string { return "agent_credentials" }
