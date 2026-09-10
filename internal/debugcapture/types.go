package debugcapture

import (
	"errors"
	"time"
)

var (
	ErrExpired  = errors.New("debug capture expired")
	ErrClosed   = errors.New("debug capture is closed")
	ErrNotFound = errors.New("debug capture not found")
)

type State string

const (
	StateActive    State = "active"
	StateCompleted State = "completed"
	StateFailed    State = "failed"
	StateExpired   State = "expired"
)

type Direction string

const (
	DirectionRequest  Direction = "request"
	DirectionResponse Direction = "response"
)

type Part string

const (
	PartHeaders Part = "headers"
	PartBody    Part = "body"
)

type SessionMetadata struct {
	RequestID   string         `json:"request_id,omitempty"`
	AccessKeyID uint           `json:"access_key_id,omitempty"`
	Protocol    string         `json:"protocol,omitempty"`
	Operation   string         `json:"operation,omitempty"`
	Fields      map[string]any `json:"fields,omitempty"`
}

type AttemptMetadata struct {
	Fields map[string]any `json:"fields,omitempty"`
	Events []EventRecord  `json:"events,omitempty"`
}

type EventRecord struct {
	Kind      string    `json:"kind"`
	Outcome   string    `json:"outcome,omitempty"`
	Written   int       `json:"written,omitempty"`
	Requested int       `json:"requested,omitempty"`
	Error     string    `json:"error,omitempty"`
	At        time.Time `json:"at"`
}

type SessionRecord struct {
	ID          string          `json:"id"`
	RequestID   string          `json:"request_id,omitempty"`
	AccessKeyID uint            `json:"access_key_id,omitempty"`
	Protocol    string          `json:"protocol,omitempty"`
	Operation   string          `json:"operation,omitempty"`
	Metadata    SessionMetadata `json:"metadata"`
	CreatedAt   time.Time       `json:"created_at"`
	ExpiresAt   time.Time       `json:"expires_at"`
	State       State           `json:"state"`
	Error       string          `json:"error,omitempty"`
	Attempts    []AttemptRecord `json:"attempts"`
}

type AttemptRecord struct {
	ID          string          `json:"id"`
	Sequence    int             `json:"sequence"`
	Metadata    AttemptMetadata `json:"metadata"`
	StartedAt   time.Time       `json:"started_at"`
	CompletedAt time.Time       `json:"completed_at,omitempty"`
	State       State           `json:"state"`
	Error       string          `json:"error,omitempty"`
}

type SessionQuery struct {
	RequestID     string
	AccessKeyID   uint
	Protocol      string
	Operation     string
	State         State
	CreatedAfter  time.Time
	CreatedBefore time.Time
	Offset        int
	Limit         int
}

type Health struct {
	Enabled           bool       `json:"enabled"`
	Running           bool       `json:"running"`
	RetentionSeconds  int64      `json:"retention_seconds"`
	Active            int64      `json:"active"`
	Completed         int64      `json:"completed"`
	Failed            int64      `json:"failed"`
	SweepTotal        uint64     `json:"sweep_total"`
	RemovedTotal      uint64     `json:"removed_total"`
	SweepFailureTotal uint64     `json:"sweep_failure_total"`
	Error             string     `json:"error,omitempty"`
	LastSweepAt       *time.Time `json:"last_sweep_at,omitempty"`
	LastFailureAt     *time.Time `json:"last_failure_at,omitempty"`
}

type Counts struct {
	Active    int64
	Completed int64
	Failed    int64
}
