package migrations

import "gorm.io/gorm"

// retiredBaselineDefinitions pins the schema that the supported historical
// release declared for the retired subscription tables, so offline conversion
// fixtures (and anything else that must know what a real legacy database looks
// like) stop guessing. The models are byte-for-byte the definitions that
// migration 0001 of that release declared; the current release no longer
// creates these tables and must never migrate them.
//
// Nothing here re-enables the retired feature: the models exist only to state
// the historical DDL.

type retiredBaselineJSON []byte

type retiredBaselineCredentialStage struct {
	ID                   string              `gorm:"type:varchar(36);primaryKey;not null"`
	ChannelID            string              `gorm:"type:varchar(64);not null"`
	ConnectionType       string              `gorm:"type:varchar(32);not null;check:chk_credential_stage_connection_type,connection_type = 'subscription'"`
	AuthorizationMethod  string              `gorm:"type:varchar(32);not null;check:chk_credential_stage_authorization_method,authorization_method IN ('browser_oauth','device_oauth','oauth_file')"`
	Status               string              `gorm:"type:varchar(32);not null;check:chk_credential_stage_status,status IN ('pending_authorization','exchanging','ready','consumed','failed','cancelled','expired','outcome_unknown');index:idx_credential_stages_status_expires,priority:1"`
	EncryptedPayload     string              `gorm:"type:text;not null"`
	PayloadSchemaVersion uint                `gorm:"not null;default:1;check:chk_credential_stage_payload_schema,payload_schema_version > 0"`
	SafeSummaryJSON      retiredBaselineJSON `gorm:"column:safe_summary_json;type:json;not null"`
	IdentityFingerprint  string              `gorm:"type:varchar(128);not null;default:''"`
	OAuthStateHash       *string             `gorm:"column:oauth_state_hash;type:varchar(128);uniqueIndex:idx_credential_stages_oauth_state"`
	ExpiresAtMS          int64               `gorm:"column:expires_at_ms;not null;check:chk_credential_stage_expires_at,expires_at_ms >= 0;index:idx_credential_stages_status_expires,priority:2"`
	ConsumedAtMS         *int64              `gorm:"column:consumed_at_ms;check:chk_credential_stage_consumed_at,consumed_at_ms IS NULL OR consumed_at_ms >= 0"`
	ConsumedGroupID      *uint
	ErrorCode            string `gorm:"type:varchar(64);not null;default:''"`
	CreatedAtMS          int64  `gorm:"column:created_at_ms;not null;check:chk_credential_stage_created_at,created_at_ms >= 0"`
	UpdatedAtMS          int64  `gorm:"column:updated_at_ms;not null;check:chk_credential_stage_updated_at,updated_at_ms >= 0;index:idx_credential_stages_status_expires,priority:3"`
}

func (retiredBaselineCredentialStage) TableName() string { return "credential_stages" }

type retiredBaselineCredentialObservation struct {
	CredentialID                 uint                `gorm:"primaryKey;not null"`
	IdentityFingerprint          string              `gorm:"type:varchar(128);not null"`
	SchemaVersion                uint                `gorm:"not null;default:1;check:chk_credential_observation_schema,schema_version > 0"`
	ObservationVersion           uint64              `gorm:"not null;default:1;check:chk_credential_observation_version,observation_version > 0"`
	SnapshotJSON                 retiredBaselineJSON `gorm:"column:snapshot_json;type:json;not null"`
	State                        string              `gorm:"type:varchar(32);not null;check:chk_credential_observation_state,state IN ('fresh','stale','refreshing','error','unavailable')"`
	ObservedAtMS                 *int64              `gorm:"column:observed_at_ms;check:chk_credential_observation_observed_at,observed_at_ms IS NULL OR observed_at_ms >= 0"`
	FreshUntilMS                 *int64              `gorm:"column:fresh_until_ms;check:chk_credential_observation_fresh_until,fresh_until_ms IS NULL OR fresh_until_ms >= 0"`
	LastAttemptAtMS              *int64              `gorm:"column:last_attempt_at_ms;check:chk_credential_observation_last_attempt,last_attempt_at_ms IS NULL OR last_attempt_at_ms >= 0"`
	NextAllowedAtMS              *int64              `gorm:"column:next_allowed_at_ms;check:chk_credential_observation_next_allowed,next_allowed_at_ms IS NULL OR next_allowed_at_ms >= 0"`
	LastAuthRefreshSecretVersion *uint64             `gorm:"column:last_auth_refresh_secret_version"`
	LastErrorCode                string              `gorm:"type:varchar(64);not null;default:''"`
	UpdatedAtMS                  int64               `gorm:"column:updated_at_ms;not null;check:chk_credential_observation_updated_at,updated_at_ms >= 0"`
}

func (retiredBaselineCredentialObservation) TableName() string {
	return "credential_observations"
}

type retiredBaselineCredentialResetOperation struct {
	IdempotencyKey  string              `gorm:"column:idempotency_key;type:char(36);primaryKey;not null"`
	RequestDigest   []byte              `gorm:"column:request_digest;not null"`
	GroupID         uint                `gorm:"column:group_id;not null;index:idx_credential_reset_operations_credential,priority:1"`
	CredentialID    uint                `gorm:"column:credential_id;not null;index:idx_credential_reset_operations_credential,priority:2"`
	RedeemRequestID string              `gorm:"column:redeem_request_id;type:char(36);not null;uniqueIndex"`
	State           string              `gorm:"type:varchar(32);not null;check:chk_credential_reset_operation_state,state IN ('prepared','succeeded','rejected','outcome_unknown')"`
	ResultJSON      retiredBaselineJSON `gorm:"column:result_json;type:json"`
	ErrorCode       string              `gorm:"column:error_code;type:varchar(64);not null;default:''"`
	CreatedAtMS     int64               `gorm:"column:created_at_ms;not null;check:chk_credential_reset_operation_created_at,created_at_ms >= 0"`
	UpdatedAtMS     int64               `gorm:"column:updated_at_ms;not null;check:chk_credential_reset_operation_updated_at,updated_at_ms >= 0"`
	CompletedAtMS   *int64              `gorm:"column:completed_at_ms;check:chk_credential_reset_operation_completed_at,completed_at_ms IS NULL OR completed_at_ms >= 0"`
}

func (retiredBaselineCredentialResetOperation) TableName() string {
	return "credential_reset_operations"
}

// RetiredBaselineModels returns the historical models of the retired
// subscription schema. Callers may AutoMigrate them onto an otherwise current
// database to reproduce exactly what the supported legacy release stored.
func RetiredBaselineModels() []any {
	return []any{
		&retiredBaselineCredentialStage{},
		&retiredBaselineCredentialObservation{},
		&retiredBaselineCredentialResetOperation{},
	}
}

// RetiredBaselineColumns returns the historical column set of each retired
// table, derived from the pinned models so a fixture cannot silently drift from
// the schema it claims to reproduce.
func RetiredBaselineColumns(db *gorm.DB) (map[string][]string, error) {
	columns := make(map[string][]string, len(RetiredBaselineModels()))
	for _, model := range RetiredBaselineModels() {
		statement := &gorm.Statement{DB: db}
		if err := statement.Parse(model); err != nil {
			return nil, err
		}
		names := make([]string, 0, len(statement.Schema.Fields))
		for _, field := range statement.Schema.Fields {
			if field.DBName != "" {
				names = append(names, field.DBName)
			}
		}
		columns[statement.Schema.Table] = names
	}
	return columns, nil
}

// RetiredBaselineTables lists the retired table names.
func RetiredBaselineTables() []string {
	return []string{
		"credential_observations",
		"credential_reset_operations",
		"credential_stages",
	}
}
