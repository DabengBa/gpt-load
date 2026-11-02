package control

import (
	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

type requestLogCursorPayload struct {
	Version       int    `json:"v"`
	CompletedAtMS int64  `json:"completed_at_ms"`
	RequestID     string `json:"request_id"`
}

type requestLogAccessKeyResponse struct {
	ID      uint    `json:"id"`
	Name    *string `json:"name"`
	Deleted bool    `json:"deleted"`
}

type requestLogReasoningResponse struct {
	Mode         *string `json:"mode"`
	Effort       *string `json:"effort"`
	BudgetTokens *string `json:"budget_tokens"`
}

type requestLogFeedbackResponse struct {
	FeedbackStatus          *health.FeedbackStatus `json:"feedback_status"`
	FeedbackReason          *string                `json:"feedback_reason"`
	ProviderFirstResponseMs *int64                 `json:"provider_first_response_ms"`
	ProviderTokensPerSecond *float64               `json:"provider_tokens_per_second"`
}

type requestLogAttemptResponse struct {
	Sequence          int                          `json:"sequence"`
	GroupID           uint                         `json:"group_id"`
	GroupName         string                       `json:"group_name"`
	ChannelID         *channel.ID                  `json:"channel_id"`
	CredentialID      *uint                        `json:"credential_id"`
	CredentialName    string                       `json:"credential_name"`
	Operation         *execution.Operation         `json:"operation"`
	RouteMode         *channel.RouteMode           `json:"route_mode"`
	UpstreamModel     *string                      `json:"upstream_model"`
	UpstreamRequestID *string                      `json:"upstream_request_id"`
	DispatchState     *execution.DispatchState     `json:"dispatch_state"`
	ResponseStarted   bool                         `json:"response_started"`
	UpstreamProtocol  *protocol.Protocol           `json:"upstream_protocol"`
	Reasoning         *requestLogReasoningResponse `json:"reasoning"`
	StatusCode        int                          `json:"status_code"`
	DurationMs        int64                        `json:"duration_ms"`
	requestLogFeedbackResponse
	FailureCategory telemetry.FailureCategory         `json:"failure_category"`
	FailureOrigin   *execution.ErrorOrigin            `json:"failure_origin"`
	FailureScope    *execution.ErrorScope             `json:"failure_scope"`
	RetryDirective  *telemetry.RetryDirective         `json:"retry_directive"`
	Effect          *telemetry.Effect                 `json:"effect"`
	RuleID          *string                           `json:"rule_id"`
	Action          string                            `json:"action"`
	WillRetry       bool                              `json:"will_retry"`
	ErrorCode       string                            `json:"error_code"`
	ErrorSummary    string                            `json:"error_summary"`
	Committed       bool                              `json:"committed"`
	PricingReceipt  *requestLogPricingReceiptResponse `json:"pricing_receipt"`
}

type requestLogPricingIdentityResponse struct {
	ChannelID *string `json:"channel_id,omitempty"`
	ModelID   string  `json:"model_id"`
}

type requestLogPricingMultiplierResponse struct {
	Numerator   string `json:"numerator"`
	Denominator string `json:"denominator"`
}

type requestLogPricingLineResponse struct {
	Code                  string                              `json:"code"`
	Quantity              string                              `json:"quantity"`
	RateNanoUSDPerMillion *string                             `json:"rate_nano_usd_per_million"`
	Multiplier            requestLogPricingMultiplierResponse `json:"multiplier"`
	State                 pricing.ReceiptLineState            `json:"state"`
	AmountNanoUSD         *string                             `json:"amount_nano_usd"`
}

type requestLogPricingReceiptResponse struct {
	SchemaVersion          int                               `json:"schema_version"`
	Method                 string                            `json:"method"`
	MethodVersion          int                               `json:"method_version"`
	Currency               string                            `json:"currency"`
	PricingMode            pricing.Mode                      `json:"pricing_mode"`
	Rule                   requestLogPricingIdentityResponse `json:"rule"`
	ContextThresholdTokens *string                           `json:"context_threshold_tokens"`
	LineItems              []requestLogPricingLineResponse   `json:"line_items"`
	TotalNanoUSD           string                            `json:"total_nano_usd"`
}

type requestLogItemResponse struct {
	RequestID              string                       `json:"request_id"`
	CompletedAtMS          int64                        `json:"completed_at_ms"`
	AccessKey              requestLogAccessKeyResponse  `json:"access_key"`
	Protocol               string                       `json:"protocol"`
	Operation              *execution.Operation         `json:"operation"`
	UpstreamProtocol       *protocol.Protocol           `json:"upstream_protocol"`
	ClientModel            *string                      `json:"client_model"`
	UpstreamModel          *string                      `json:"upstream_model"`
	UpstreamReportedModel  *string                      `json:"upstream_reported_model"`
	ModelConsistency       telemetry.ModelConsistency   `json:"model_consistency"`
	Reasoning              *requestLogReasoningResponse `json:"reasoning"`
	Status                 telemetry.RequestStatus      `json:"status"`
	StatusCode             int                          `json:"status_code"`
	Stream                 bool                         `json:"stream"`
	FirstResponseMs        *int64                       `json:"first_response_ms"`
	DurationMs             int64                        `json:"duration_ms"`
	FinalAttemptDurationMs *int64                       `json:"final_attempt_duration_ms"`
	AttemptCount           int                          `json:"attempt_count"`
	requestLogFeedbackResponse
	ErrorCode                    string               `json:"error_code"`
	ErrorSummary                 string               `json:"error_summary"`
	AffinityHit                  bool                 `json:"affinity_hit"`
	ResolvedAffinityCredentialID uint                 `json:"resolved_affinity_credential_id"`
	FirstAttemptCredentialID     uint                 `json:"first_attempt_credential_id"`
	AffinityHitAssignmentReached bool                 `json:"affinity_hit_assignment_reached"`
	ContinuityHit                bool                 `json:"continuity_hit"`
	AffinityKey                  *string              `json:"affinity_key"`
	AffinitySource               string               `json:"affinity_source"`
	AffinityState                string               `json:"affinity_state"`
	GroupID                      *uint                `json:"group_id"`
	ChannelID                    *channel.ID          `json:"channel_id"`
	CredentialID                 *uint                `json:"credential_id"`
	CredentialName               string               `json:"credential_name"`
	RouteMode                    *channel.RouteMode   `json:"route_mode"`
	UsageState                   usage.State          `json:"usage_state"`
	CostState                    pricing.CostState    `json:"cost_state"`
	PricingCompleteness          pricing.Completeness `json:"pricing_completeness"`
	PricingMode                  *pricing.Mode        `json:"pricing_mode"`
	ContextThresholdTokens       *string              `json:"context_threshold_tokens"`
	InputTokens                  string               `json:"input_tokens"`
	CacheReadTokens              string               `json:"cache_read_tokens"`
	CacheWrite5MTokens           string               `json:"cache_write_5m_tokens"`
	CacheWrite1HTokens           string               `json:"cache_write_1h_tokens"`
	CacheWriteUnknownTokens      string               `json:"cache_write_unknown_tokens"`
	OutputTokens                 string               `json:"output_tokens"`
	EstimatedCostNanoUSD         string               `json:"estimated_cost_nano_usd"`
}

type requestLogDetailResponse struct {
	requestLogItemResponse
	Attempts []requestLogAttemptResponse `json:"attempts"`
}

type requestLogListResponse struct {
	Items      []requestLogItemResponse `json:"items"`
	NextCursor *string                  `json:"next_cursor"`
}
type requestLogUsageCostResponse struct {
	groupID              *uint
	channelID            *channel.ID
	credentialID         *uint
	estimatedCostNanoUSD string
}
