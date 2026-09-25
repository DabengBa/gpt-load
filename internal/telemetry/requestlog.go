package telemetry

import (
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/usage"
)

type RequestStatus string

const (
	RequestStatusSuccess    RequestStatus = "success"
	RequestStatusError      RequestStatus = "error"
	RequestStatusIncomplete RequestStatus = "incomplete"
	RequestStatusCanceled   RequestStatus = "canceled"
)

type ModelConsistency string

const (
	ModelConsistencyNotApplicable ModelConsistency = "not_applicable"
	ModelConsistencyMatch         ModelConsistency = "match"
	ModelConsistencyUnknown       ModelConsistency = "unknown"
	ModelConsistencyMismatch      ModelConsistency = "mismatch"
)

type FailureCategory string

const (
	FailureCategoryOK                     FailureCategory = "ok"
	FailureCategoryRateLimited            FailureCategory = "rate_limited"
	FailureCategoryModelUnavailable       FailureCategory = "model_unavailable"
	FailureCategoryInvalidKey             FailureCategory = "invalid_key"
	FailureCategoryUpstreamHost           FailureCategory = "upstream_host_error"
	FailureCategoryClientError            FailureCategory = "client_error"
	FailureCategoryConversionUnsupported  FailureCategory = "conversion_unsupported"
	FailureCategoryDownstreamCancel       FailureCategory = "downstream_cancel"
	FailureCategoryAuthenticationRequired FailureCategory = "authentication_required"
	FailureCategoryBilling                FailureCategory = "billing"
	FailureCategoryAmbiguous              FailureCategory = "ambiguous"
)

// FailureCategoryFromHealth is the single source of truth for the
// health → request-log failure category mapping. Control-plane observations and
// gateway traffic must agree, so both call this instead of keeping private copies.
func FailureCategoryFromHealth(value health.FailureCategory) FailureCategory {
	switch value {
	case health.FailureCategoryOK:
		return FailureCategoryOK
	case health.FailureCategoryRateLimited:
		return FailureCategoryRateLimited
	case health.FailureCategoryModelUnavailable:
		return FailureCategoryModelUnavailable
	case health.FailureCategoryInvalidKey:
		return FailureCategoryInvalidKey
	case health.FailureCategoryUpstreamHostError:
		return FailureCategoryUpstreamHost
	case health.FailureCategoryClientError:
		return FailureCategoryClientError
	case health.FailureCategoryConversionUnsupported:
		return FailureCategoryConversionUnsupported
	case health.FailureCategoryDownstreamCancel:
		return FailureCategoryDownstreamCancel
	case health.FailureCategoryAuthenticationRequired:
		return FailureCategoryAuthenticationRequired
	case health.FailureCategoryBilling:
		return FailureCategoryBilling
	default:
		return FailureCategoryAmbiguous
	}
}

type RetryDirective string

const (
	RetryNone              RetryDirective = "none"
	RetryRefreshCredential RetryDirective = "refresh_credential"
	RetryNextCandidate     RetryDirective = "next_candidate"
)

func (value RetryDirective) Valid() bool {
	return value == RetryNone || value == RetryRefreshCredential || value == RetryNextCandidate
}

type Effect string

const (
	EffectNone                    Effect = "none"
	EffectCooldownCredential      Effect = "cooldown_credential"
	EffectRecordCredentialFailure Effect = "record_credential_failure"
	EffectSkipGroup               Effect = "skip_group"
)

func (value Effect) Valid() bool {
	return value == EffectNone || value == EffectCooldownCredential ||
		value == EffectRecordCredentialFailure || value == EffectSkipGroup
}

type Action string

const (
	ActionTerminate          Action = "terminate"
	ActionRetry              Action = "retry"
	ActionCooldownCredential Action = "cooldown_credential"
	ActionFailCredential     Action = "fail_credential"
	ActionSkipGroup          Action = "skip_group"
)

// AffinitySource 标识哪个可选请求提示驱动了亲和。
type AffinitySource string

const (
	// AffinitySourceNone 表示未评估任何亲和提示。
	AffinitySourceNone AffinitySource = ""
	// AffinitySourcePromptCacheKey 表示使用了显式 prompt_cache_key。
	AffinitySourcePromptCacheKey AffinitySource = "prompt_cache_key"
	// AffinitySourcePromptPrefix 表示使用了推断的稳定 prompt 前缀。
	AffinitySourcePromptPrefix AffinitySource = "prompt_prefix"
)

// AffinityState 是一次亲和绑定解析的 bounded 结果，不是热缓存命中率。
type AffinityState string

const (
	// AffinityStateNone 表示请求未评估亲和。
	AffinityStateNone AffinityState = ""
	// AffinityStateNoSignal 表示不存在显式或前缀亲和信号。
	AffinityStateNoSignal AffinityState = "no_signal"
	// AffinityStateCacheMiss 表示适用查找路径未找到绑定；生产亲和启用时包含 durable store 查询。
	// all-disabled 跳过持久查询，memory-only 测试仅查询热缓存，均沿用此既有状态。
	AffinityStateCacheMiss AffinityState = "cache_miss"
	// AffinityStateHit 表示热缓存或 durable read-through 得到的绑定通过了解析阶段的资格检查。
	// 实际首试仍受调度器当前的全部硬资格过滤约束。
	AffinityStateHit AffinityState = "hit"
	// AffinityStateGroupDisabled 表示绑定目标组已禁用亲和，保留 durable row。
	AffinityStateGroupDisabled AffinityState = "group_disabled"
	// AffinityStateTargetUnavailable 表示绑定目标未通过当前解析资格检查，保留 durable row。
	AffinityStateTargetUnavailable AffinityState = "target_unavailable"
	// AffinityStateCacheUnavailable 表示亲和解析不可用，涵盖本地缓存/配置/键条件及持久查询/解码错误。
	// 必需的持久查询失败在 provider dispatch 前 fail-closed，不普通 fallback；本地边界不由此枚举决定。
	// provider 成功后的持久写入失败另记 affinity_binding_persist_failed，不改已交付响应或此状态。
	AffinityStateCacheUnavailable AffinityState = "cache_unavailable"
)

type Attempt struct {
	Sequence                int
	CompletedAt             time.Time
	GroupID                 uint
	GroupName               string
	ChannelID               channel.ID
	CredentialID            uint
	Operation               execution.Operation
	RouteMode               channel.RouteMode
	UpstreamModel           string
	UpstreamRequestID       string
	DispatchState           execution.DispatchState
	ResponseStarted         bool
	UpstreamProtocol        protocol.Protocol
	Reasoning               reasoning.Config
	StatusCode              int
	DurationMs              int64
	FailureCategory         FailureCategory
	FailureOrigin           execution.ErrorOrigin
	FailureScope            execution.ErrorScope
	RetryDirective          RetryDirective
	Effect                  Effect
	RuleID                  string
	Action                  Action
	WillRetry               bool
	ErrorCode               string
	ErrorSummary            string
	Committed               bool
	HTTPCommitted           bool
	PayloadReleased         bool
	ClientVisibleBytes      int64
	BufferedPeakBytes       int64
	BufferedSpilled         bool
	BufferedStream          bool
	PayloadReleaseStartedMs int64
	Usage                   usage.Result
}

// PricingObservation is the frozen, dependency-neutral quote selected by the
// gateway for the attempt whose usage is attributed to the request outcome.
type PricingObservation struct {
	UpstreamModel        string
	CostState            string
	PricingCompleteness  string
	EstimatedCostNanoUSD int64
	ReceiptJSON          string
}

type UsageObservation struct {
	Result          usage.Result
	GroupID         uint
	ChannelID       channel.ID
	CredentialID    uint
	AttemptSequence int
	Pricing         PricingObservation
}

type RequestEvent struct {
	RequestID             string
	CompletedAt           time.Time
	AccessKeyID           uint
	Protocol              protocol.Protocol
	Operation             execution.Operation
	ClientModel           string
	UpstreamModel         string
	UpstreamReportedModel string
	ModelConsistency      ModelConsistency
	Status                RequestStatus
	StatusCode            int
	ErrorCode             string
	ErrorSummary          string
	Stream                bool
	FirstResponseMs       *int64
	DurationMs            int64
	AffinityHit           bool
	ContinuityHit         bool
	AffinityKey           string
	AffinitySource        AffinitySource
	AffinityState         AffinityState
	Reasoning             reasoning.Config
	Attempts              []Attempt
	Usage                 UsageObservation
	BufferedStream        bool
	BufferedPeakBytes     int64
	ReleaseStartedMs      *int64
}

type RequestLogSink interface {
	Emit(RequestEvent)
}

type NoopRequestLogSink struct{}

func (NoopRequestLogSink) Emit(RequestEvent) {}
