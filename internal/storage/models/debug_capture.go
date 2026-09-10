package models

// DebugCapture is the queryable metadata and lifecycle row for one observed
// application-layer communication capture. Raw payloads live only in chunks.
type DebugCapture struct {
	ID           string                `gorm:"type:varchar(64);primaryKey;not null"`
	RequestID    string                `gorm:"column:request_id;type:varchar(255);not null;default:'';index:idx_debug_captures_request_id"`
	AccessKeyID  uint                  `gorm:"not null;default:0;index:idx_debug_captures_access_key_id"`
	Protocol     string                `gorm:"type:varchar(32);not null;default:'';index:idx_debug_captures_protocol"`
	Operation    string                `gorm:"type:varchar(64);not null;default:'';index:idx_debug_captures_operation"`
	Metadata     JSON                  `gorm:"type:json"`
	CreatedAtMS  int64                 `gorm:"column:created_at_ms;not null;index:idx_debug_captures_created_at_ms"`
	ExpiresAtMS  int64                 `gorm:"column:expires_at_ms;not null;index:idx_debug_captures_expires_at_ms"`
	State        string                `gorm:"type:varchar(16);not null;default:'active';index:idx_debug_captures_state;check:chk_debug_captures_state,state IN ('active','completed','failed','expired')"`
	Error        string                `gorm:"type:text;not null"`
	TerminalAtMS *int64                `gorm:"column:terminal_at_ms"`
	Attempts     []DebugCaptureAttempt `gorm:"foreignKey:CaptureID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
}

func (DebugCapture) TableName() string { return "debug_captures" }

// DebugCaptureAttempt records an upstream attempt and its terminal outcome.
type DebugCaptureAttempt struct {
	ID            string              `gorm:"type:varchar(64);primaryKey;not null;uniqueIndex:uidx_debug_capture_attempts_capture_id_id,priority:2"`
	CaptureID     string              `gorm:"column:capture_id;type:varchar(64);not null;index:idx_debug_capture_attempts_capture_id;uniqueIndex:uidx_debug_capture_attempts_capture_id_id,priority:1"`
	Sequence      int                 `gorm:"not null;check:chk_debug_capture_attempts_sequence,sequence > 0"`
	Metadata      JSON                `gorm:"type:json"`
	StartedAtMS   int64               `gorm:"column:started_at_ms;not null;index:idx_debug_capture_attempts_started_at_ms"`
	CompletedAtMS *int64              `gorm:"column:completed_at_ms"`
	State         string              `gorm:"type:varchar(16);not null;default:'active';index:idx_debug_capture_attempts_state;check:chk_debug_capture_attempts_state,state IN ('active','completed','failed')"`
	Error         string              `gorm:"type:text;not null"`
	Capture       *DebugCapture       `gorm:"foreignKey:CaptureID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
	Chunks        []DebugCaptureChunk `gorm:"foreignKey:CaptureID,AttemptID;references:CaptureID,ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
}

func (DebugCaptureAttempt) TableName() string { return "debug_capture_attempts" }

// DebugCaptureChunk stores one append operation (or one bounded body slice).
// The auto-increment ID is the append order within the database's serialized
// write boundary and is used for deterministic reconstruction and export.
type DebugCaptureChunk struct {
	ID          uint64               `gorm:"primaryKey;autoIncrement"`
	CaptureID   string               `gorm:"column:capture_id;type:varchar(64);not null;index:idx_debug_capture_chunks_capture_id"`
	AttemptID   string               `gorm:"column:attempt_id;type:varchar(64);not null;index:idx_debug_capture_chunks_attempt_part_direction_id,priority:1"`
	Part        string               `gorm:"type:varchar(16);not null;index:idx_debug_capture_chunks_attempt_part_direction_id,priority:2"`
	Direction   string               `gorm:"type:varchar(16);not null;index:idx_debug_capture_chunks_attempt_part_direction_id,priority:3"`
	Data        []byte               `gorm:"not null"`
	CreatedAtMS int64                `gorm:"column:created_at_ms;not null"`
	Capture     *DebugCapture        `gorm:"foreignKey:CaptureID;references:ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
	Attempt     *DebugCaptureAttempt `gorm:"foreignKey:CaptureID,AttemptID;references:CaptureID,ID;constraint:OnUpdate:CASCADE,OnDelete:CASCADE"`
}

func (DebugCaptureChunk) TableName() string { return "debug_capture_chunks" }
