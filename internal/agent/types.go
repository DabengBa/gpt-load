package agent

// SchemaVersion is the stable machine schema version for every Agent response.
// Additive fields may be introduced without a version bump; incompatible
// changes require a new version and a new namespace generation.
const SchemaVersion = 1

const (
	// DefaultPageLimit matches the existing request-log read contract.
	DefaultPageLimit = 50
	// MaxPageLimit bounds one Agent page.
	MaxPageLimit = 200
)

// Evidence state values deliberately separate "we can see there is no
// retained capture" from "we cannot tell", because debug capture retention is
// bounded and expired rows are removed.
const (
	EvidenceStateCaptured    = "captured"
	EvidenceStateNotRetained = "not_retained"
	EvidenceStateUnavailable = "unavailable"
)

// Capture attempt link state values separate a capture row that carries an
// explicit logical attempt reference from one that does not.
const (
	AttemptLinkStateLinked   = "linked"
	AttemptLinkStateUnlinked = "unlinked"
)

// RoutesView discriminator values.
const (
	RoutesViewIndex  = "index"
	RoutesViewDetail = "detail"
)

// CredentialStatus values for the Agent credential ledger.
const (
	CredentialStatusActive   = "active"
	CredentialStatusDisabled = "disabled"
)

// CredentialMetadata is the safe, persistable view of one Agent credential.
// It never contains the plaintext secret or its digest.
type CredentialMetadata struct {
	ID           uint    `json:"id"`
	Name         string  `json:"name"`
	Scopes       []Scope `json:"scopes"`
	Status       string  `json:"status"`
	ExpiresAtMS  *int64  `json:"expires_at_ms"`
	DisabledAtMS *int64  `json:"disabled_at_ms"`
	CreatedAtMS  int64   `json:"created_at_ms"`
	UpdatedAtMS  int64   `json:"updated_at_ms"`
}

// Capabilities describes the Agent surface and the calling identity.
type Capabilities struct {
	SchemaVersion            int                  `json:"schema_version"`
	Server                   ServerIdentity       `json:"server"`
	Caller                   CallerIdentity       `json:"caller"`
	Endpoints                []EndpointCapability `json:"endpoints"`
	Limits                   Limits               `json:"limits"`
	Features                 Features             `json:"features"`
	EvidenceRetentionSeconds int64                `json:"evidence_retention_seconds"`
	DecimalStringFields      []string             `json:"decimal_string_fields"`
}

// ServerIdentity identifies the serving process without exposing internals.
type ServerIdentity struct {
	UptimeSeconds    int64  `json:"uptime_seconds"`
	SnapshotRevision uint64 `json:"snapshot_revision"`
}

// CallerIdentity is the resolved identity of the current Agent credential.
type CallerIdentity struct {
	CredentialID uint    `json:"credential_id"`
	Name         string  `json:"name"`
	Scopes       []Scope `json:"scopes"`
	ExpiresAtMS  *int64  `json:"expires_at_ms"`
}

// EndpointCapability describes one Agent HTTP endpoint.
type EndpointCapability struct {
	Name          string `json:"name"`
	Method        string `json:"method"`
	Path          string `json:"path"`
	RequiredScope Scope  `json:"required_scope"`
	SideEffects   bool   `json:"side_effects"`
}

// Limits bounds Agent reads.
type Limits struct {
	DefaultPageSize int `json:"default_page_size"`
	MaxPageSize     int `json:"max_page_size"`
}

// Features reports optional surfaces. Raw evidence remains unavailable to
// ordinary Agent credentials; change proposals are reported when the
// control-owned lifecycle service is wired.
type Features struct {
	EvidenceRaw     bool `json:"evidence_raw"`
	ChangeProposals bool `json:"change_proposals"`
}

// PageInfo describes pagination without inventing a total count.
type PageInfo struct {
	Limit    int  `json:"limit"`
	Returned int  `json:"returned"`
	HasMore  bool `json:"has_more"`
}

// RequestList is the paginated request-log read model.
type RequestList struct {
	SchemaVersion int              `json:"schema_version"`
	Items         []RequestSummary `json:"items"`
	NextCursor    *string          `json:"next_cursor"`
	Page          PageInfo         `json:"page"`
}

// UsageTotalsView carries token accounting as decimal strings.
type UsageTotalsView struct {
	State                   string `json:"state"`
	UncachedInputTokens     string `json:"uncached_input_tokens"`
	CacheReadTokens         string `json:"cache_read_tokens"`
	CacheWrite5MTokens      string `json:"cache_write_5m_tokens"`
	CacheWrite1HTokens      string `json:"cache_write_1h_tokens"`
	CacheWriteUnknownTokens string `json:"cache_write_unknown_tokens"`
	OutputTokens            string `json:"output_tokens"`
	TotalTokens             string `json:"total_tokens"`
}

// CostView carries cost accounting as a decimal string.
type CostView struct {
	State            string `json:"state"`
	Completeness     string `json:"completeness"`
	EstimatedNanoUSD string `json:"estimated_nano_usd"`
}

// RequestSummary is the stable request-level projection. SemanticStatus and
// HTTPStatusCode are deliberately separate fields.
type RequestSummary struct {
	RequestID        string          `json:"request_id"`
	CompletedAtMS    int64           `json:"completed_at_ms"`
	SemanticStatus   string          `json:"semantic_status"`
	HTTPStatusCode   int             `json:"http_status_code"`
	Protocol         *string         `json:"protocol"`
	Operation        *string         `json:"operation"`
	ClientModel      *string         `json:"client_model"`
	UpstreamModel    *string         `json:"upstream_model"`
	Stream           bool            `json:"stream"`
	FirstResponseMs  *int64          `json:"first_response_ms"`
	DurationMs       int64           `json:"duration_ms"`
	AttemptCount     int             `json:"attempt_count"`
	ErrorCode        string          `json:"error_code"`
	ErrorSummary     string          `json:"error_summary"`
	AccessKeyID      *uint           `json:"access_key_id"`
	Usage            UsageTotalsView `json:"usage"`
	Cost             CostView        `json:"cost"`
	ProviderAttempts int             `json:"provider_attempts"`
}

// AttemptView is one provider attempt, keyed by its logical attempt reference.
type AttemptView struct {
	AttemptRef        string  `json:"attempt_ref"`
	Sequence          int     `json:"sequence"`
	GroupID           *uint   `json:"group_id"`
	CredentialID      *uint   `json:"credential_id"`
	ChannelID         *string `json:"channel_id"`
	Operation         *string `json:"operation"`
	RouteMode         *string `json:"route_mode"`
	UpstreamModel     *string `json:"upstream_model"`
	UpstreamRequestID *string `json:"upstream_request_id"`
	DispatchState     *string `json:"dispatch_state"`
	ResponseStarted   bool    `json:"response_started"`
	Protocol          *string `json:"protocol"`
	StatusCode        int     `json:"status_code"`
	DurationMs        int64   `json:"duration_ms"`
	FailureCategory   string  `json:"failure_category"`
	FailureOrigin     *string `json:"failure_origin"`
	FailureScope      *string `json:"failure_scope"`
	RetryDirective    *string `json:"retry_directive"`
	Effect            *string `json:"effect"`
	Action            string  `json:"action"`
	WillRetry         bool    `json:"will_retry"`
	ErrorCode         string  `json:"error_code"`
	ErrorSummary      string  `json:"error_summary"`
	Committed         bool    `json:"committed"`
}

// RequestDetail couples one request with its ordered provider attempts.
type RequestDetail struct {
	SchemaVersion int            `json:"schema_version"`
	Request       RequestSummary `json:"request"`
	Attempts      []AttemptView  `json:"attempts"`
	Evidence      EvidenceLink   `json:"evidence"`
}

// EvidenceLink is the non-raw evidence pointer carried by a request detail.
type EvidenceLink struct {
	State       string   `json:"state"`
	CaptureRefs []string `json:"capture_refs"`
	Path        string   `json:"path"`
}

// EvidenceView is the redacted evidence metadata projection. Raw captured
// headers and bodies are never exposed here.
type EvidenceView struct {
	SchemaVersion     int           `json:"schema_version"`
	RequestID         string        `json:"request_id"`
	State             string        `json:"state"`
	CapturesEnabled   bool          `json:"captures_enabled"`
	RetentionSeconds  int64         `json:"retention_seconds"`
	RawContentExposed bool          `json:"raw_content_exposed"`
	Captures          []CaptureView `json:"captures"`
}

// CaptureView is redacted capture session metadata.
type CaptureView struct {
	CaptureID    string               `json:"capture_id"`
	State        string               `json:"state"`
	Complete     bool                 `json:"complete"`
	CreatedAtMS  int64                `json:"created_at_ms"`
	ExpiresAtMS  int64                `json:"expires_at_ms"`
	Protocol     *string              `json:"protocol"`
	Operation    *string              `json:"operation"`
	ErrorPresent bool                 `json:"error_present"`
	Attempts     []CaptureAttemptView `json:"attempts"`
}

// CaptureAttemptView is redacted capture attempt metadata. StorageSequence is
// the capture store ordering and must never be read as the request attempt
// sequence; LogicalAttemptRef/LogicalAttemptSequence carry the explicit link.
type CaptureAttemptView struct {
	CaptureAttemptID       string  `json:"capture_attempt_id"`
	StorageSequence        int     `json:"storage_sequence"`
	State                  string  `json:"state"`
	StartedAtMS            int64   `json:"started_at_ms"`
	CompletedAtMS          *int64  `json:"completed_at_ms"`
	LogicalAttemptRef      *string `json:"logical_attempt_ref"`
	LogicalAttemptSequence *int    `json:"logical_attempt_sequence"`
	LinkState              string  `json:"link_state"`
}

// RoutesView carries either the route index or one route detail. The View
// field discriminates which of Items/Detail is meaningful.
type RoutesView struct {
	SchemaVersion    int                  `json:"schema_version"`
	View             string               `json:"view"`
	SnapshotRevision uint64               `json:"snapshot_revision"`
	Items            []RouteIndexItemView `json:"items"`
	Detail           *RouteDetailView     `json:"detail"`
}

// RouteIndexItemView summarizes one external model route candidate set.
type RouteIndexItemView struct {
	ExternalModel   string `json:"external_model"`
	Protocol        string `json:"protocol"`
	Operation       string `json:"operation"`
	CandidateCount int    `json:"candidate_count"`
	GroupIDs       []uint `json:"group_ids"`
}

// RouteDetailView lists the configured entries for one routed model.
type RouteDetailView struct {
	ExternalModel string           `json:"external_model"`
	Protocol      string           `json:"protocol"`
	Operation     string           `json:"operation"`
	Entries       []RouteEntryView `json:"entries"`
}

// RouteEntryView is one configured routing entry.
type RouteEntryView struct {
	GroupID         uint   `json:"group_id"`
	GroupName       string `json:"group_name"`
	EntryID         string `json:"entry_id"`
	UpstreamModelID string `json:"upstream_model_id"`
	RouteMode       string `json:"route_mode"`
	Weight          int    `json:"weight"`
	Priority        int    `json:"priority"`
}

// HealthView is the redacted Agent runtime health projection. It contains no
// credential identity, group name, or access key material.
type HealthView struct {
	SchemaVersion    int                    `json:"schema_version"`
	ObservedAtMS     int64                  `json:"observed_at_ms"`
	Version          string                 `json:"version"`
	UptimeSeconds    int64                  `json:"uptime_seconds"`
	SnapshotRevision uint64                 `json:"snapshot_revision"`
	Degraded         bool                   `json:"degraded"`
	Groups           GroupCountsView        `json:"groups"`
	RequestLog       RequestLogHealthView   `json:"request_log"`
	DebugCapture     DebugCaptureHealthView `json:"debug_capture"`
}

// GroupCountsView aggregates group availability without exposing names.
type GroupCountsView struct {
	Total   int `json:"total"`
	Enabled int `json:"enabled"`
}

// RequestLogHealthView is the bounded request-log persistence health view.
type RequestLogHealthView struct {
	QueueDepth                    int    `json:"queue_depth"`
	QueueCapacity                 int    `json:"queue_capacity"`
	PersistedTotal                uint64 `json:"persisted_total"`
	DroppedTotal                  uint64 `json:"dropped_total"`
	WriteFailureTotal             uint64 `json:"write_failure_total"`
	AccessQuotaCheckpointDegraded bool   `json:"access_quota_checkpoint_degraded"`
	Degraded                      bool   `json:"degraded"`
}

// DebugCaptureHealthView is the bounded evidence-store health view.
type DebugCaptureHealthView struct {
	Enabled           bool   `json:"enabled"`
	Running           bool   `json:"running"`
	RetentionSeconds  int64  `json:"retention_seconds"`
	Active            int64  `json:"active"`
	Completed         int64  `json:"completed"`
	Failed            int64  `json:"failed"`
	RemovedTotal      uint64 `json:"removed_total"`
	SweepFailureTotal uint64 `json:"sweep_failure_total"`
	ErrorPresent      bool   `json:"error_present"`
}

// UsageView is the Agent usage report for one aligned window.
type UsageView struct {
	SchemaVersion int                    `json:"schema_version"`
	ObservedAtMS  int64                  `json:"observed_at_ms"`
	FromMS        int64                  `json:"from_ms"`
	ToMS          int64                  `json:"to_ms"`
	Granularity   string                 `json:"granularity"`
	BucketWidthMS int64                  `json:"bucket_width_ms"`
	Summary       UsageAggregateView     `json:"summary"`
	Series        []UsageSeriesPointView `json:"series"`
	Breakdown     UsageBreakdownView     `json:"breakdown"`
}

// UsageAggregateView is a token and cost aggregate with decimal-string cost.
type UsageAggregateView struct {
	RequestCount             int64  `json:"request_count"`
	SuccessCount             int64  `json:"success_count"`
	FailureCount             int64  `json:"failure_count"`
	UncachedInputTokens      int64  `json:"uncached_input_tokens"`
	CacheReadTokens          int64  `json:"cache_read_tokens"`
	CacheWrite5MTokens       int64  `json:"cache_write_5m_tokens"`
	CacheWrite1HTokens       int64  `json:"cache_write_1h_tokens"`
	CacheWriteUnknownTokens  int64  `json:"cache_write_unknown_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	TotalTokens              int64  `json:"total_tokens"`
	EstimatedNanoUSD         string `json:"estimated_nano_usd"`
	DurationMsTotal          int64  `json:"duration_ms_total"`
	DurationSampleCount      int64  `json:"duration_sample_count"`
	FirstResponseMsTotal     int64  `json:"first_response_ms_total"`
	FirstResponseSampleCount int64  `json:"first_response_sample_count"`
	UsageMissingCount        int64  `json:"usage_missing_count"`
	PartialCount             int64  `json:"partial_count"`
	UnpricedRequestCount     int64  `json:"unpriced_request_count"`
	PricingPartialCount      int64  `json:"pricing_partial_count"`
}

// UsageSeriesPointView is one aligned usage bucket.
type UsageSeriesPointView struct {
	BucketStartMS int64              `json:"bucket_start_ms"`
	BucketEndMS   int64              `json:"bucket_end_ms"`
	Aggregate     UsageAggregateView `json:"aggregate"`
}

// UsageBreakdownView is the model-scoped usage breakdown.
type UsageBreakdownView struct {
	Scope      string                  `json:"scope"`
	Rows       []UsageBreakdownRowView `json:"rows"`
	Total      UsageAggregateView      `json:"total"`
	Pagination UsagePaginationView     `json:"pagination"`
}

// UsageBreakdownRowView is one model row in the usage breakdown.
type UsageBreakdownRowView struct {
	Model     string             `json:"model"`
	GroupID   *uint              `json:"group_id"`
	Aggregate UsageAggregateView `json:"aggregate"`
}

// UsagePaginationView mirrors the usage breakdown pagination contract.
type UsagePaginationView struct {
	Page       int `json:"page"`
	PageSize   int `json:"page_size"`
	TotalItems int `json:"total_items"`
	TotalPages int `json:"total_pages"`
}
