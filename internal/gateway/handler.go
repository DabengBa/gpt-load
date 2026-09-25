package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/affinity"
	"gpt-load/internal/channel"
	"gpt-load/internal/dialect"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/httplifecycle"
	"gpt-load/internal/platform/contentcoding"
	"gpt-load/internal/platform/encryption"
	"gpt-load/internal/platform/utils"
	"gpt-load/internal/pricing"
	"gpt-load/internal/ratelimit"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
	subscriptionproviders "gpt-load/internal/subscription/providers"
	subscriptionruntime "gpt-load/internal/subscription/runtime"
	"gpt-load/internal/telemetry"
)

const (
	maxRequestBodyBytes    = int64(128 << 20)
	maxDataPlaneModelBytes = 255
	fixedCooldown          = time.Minute
	debugHeaderGroup       = "X-GPTLoad-Group"
	// debugHeaderKey remains reserved so an upstream cannot inject it downstream.
	debugHeaderKey      = "X-GPTLoad-Key"
	debugHeaderAttempts = "X-GPTLoad-Attempts"
)

var debugHeaderNames = []string{
	debugHeaderGroup,
	debugHeaderKey,
	debugHeaderAttempts,
	requestIDHeader,
}

var errRequestTooLarge = errors.New("request body is too large")

type AttemptForwarder interface {
	Forward(context.Context, ForwardInput) UpstreamResult
	ForwardStream(context.Context, ForwardInput, http.ResponseWriter) UpstreamResult
}

type AccessKeyRPMLimiter interface {
	Allow(accessKeyID uint, limit int64) ratelimit.LimitDecision
}

// PriceTableProvider exposes the currently published immutable price table.
type PriceTableProvider interface {
	Load() *pricing.Table
}

type credentialMutationCoordinator interface {
	Do(uint, func())
}

type runtimeCredentialRegistry interface {
	scheduler.CredentialSource
	CaptureActiveCredentialRefs(groupIDs []uint) []state.CredentialRef
	CredentialRef(credentialID uint) (state.CredentialRef, bool)
	ActiveEncryptedCredentialDataIfMatch(ref state.CredentialRef) (string, bool)
	SetCooldownWithChange(credentialID uint, until time.Time) (exists bool, changed bool)
	SetCooldownWithChangeIfVersion(credentialID uint, expectedVersion uint64, until time.Time) (matched bool, changed bool)
	IncrFailure(credentialID uint) (int, bool)
	SetBlacklistedWithChange(credentialID uint) (exists bool, changed bool)
	ClearFailure(credentialID uint) bool
}

type entryRuntimeRegistry interface {
	SetEntryCooldownForEntry(groupID uint, entryID string, until time.Time) (exists bool, changed bool)
	IncrEntryFailureForEntry(groupID uint, entryID string) (int, bool)
	SetEntryBlacklistedForEntry(groupID uint, entryID string) (exists bool, changed bool)
	SetEntryBlacklistReleaseAt(groupID uint, entryID string, releaseAt time.Time) bool
	ClearEntryFailureForEntry(groupID uint, entryID string) bool
}

// blacklistReleaseRegistry schedules the local-only release of a blacklisted
// credential. It is resolved by assertion like the entry runtime boundary so
// registries that only need the compact mutation contract keep compiling.
type blacklistReleaseRegistry interface {
	SetBlacklistReleaseAt(credentialID uint, releaseAt time.Time) bool
}

type Handler struct {
	manager              *state.Manager
	channels             *channel.Registry
	subscriptions        *subscriptionruntime.Runtime
	registry             runtimeCredentialRegistry
	encryption           encryption.Service
	forwarder            AttemptForwarder
	dialects             dialect.Set
	stats                *health.StatsStore
	mutations            credentialMutationCoordinator
	limiter              AccessKeyRPMLimiter
	requestLogSink       telemetry.RequestLogSink
	priceTables          PriceTableProvider
	accessQuota          *accessquota.Runtime
	newRandom            func() *rand.Rand
	newRequestID         func() (string, error)
	captureFactory       CaptureFactory
	requestNow           func() time.Time
	now                  func() time.Time
	writeTimeout         time.Duration
	modelListLimit       int64
	logger               *logrus.Logger
	authFailureEvents    *utils.RateLimitedEventCounter
	routeNotFoundEvents  *utils.RateLimitedEventCounter
	lifecycle            *httplifecycle.Coordinator
	affinityCache        *affinity.Cache
	affinityStore        affinity.BindingStore
	affinityBindingLocks [affinityBindingLockStripes]sync.Mutex
	responseBindings     *state.ResponseBindings
	websocketLimits      websocketLimits
	websocketBudget      websocketBudget
}

func (handler *Handler) freezeAttemptPricing(
	selection scheduler.Selection,
	observations dialect.RequestMetadata,
	observationsAvailable bool,
	accessKeyMultiplier pricing.PriceMultiplier,
) frozenAttemptPricing {
	frozen := frozenAttemptPricing{
		channelID:     string(selection.ChannelID),
		groupID:       selection.GroupID,
		upstreamModel: optionalModelValue(selection.UpstreamModelID),
		applicable:    observations.ObserveUsage,
		metadataSet:   true,
		pricingMode:   observations.PricingMode,
		priceMultipliers: pricing.PriceMultipliers{
			Group:     selection.Group.PriceMultiplier,
			AccessKey: accessKeyMultiplier,
		},
		usageDiagnostics: observations.UsageDiagnostics,
		reasoning:        observations.Reasoning.Clone(),
	}
	if observations.Operation != execution.OperationWebSearch && observationsAvailable && handler != nil && handler.priceTables != nil {
		frozen.table = handler.priceTables.Load()
	}
	return frozen
}

func NewHandler(
	manager *state.Manager,
	registry *state.CredentialRegistry,
	encryptionService encryption.Service,
	forwarder AttemptForwarder,
	dialects dialect.Set,
	stats *health.StatsStore,
	mutations *health.MutationCoordinator,
	limiter AccessKeyRPMLimiter,
	requestLogSink telemetry.RequestLogSink,
	priceTables PriceTableProvider,
	accessQuotas ...*accessquota.Runtime,
) *Handler {
	if limiter == nil {
		limiter = unlimitedAccessKeyRPMLimiter{}
	}
	if requestLogSink == nil {
		requestLogSink = telemetry.NoopRequestLogSink{}
	}
	channels := channel.NewRegistry()
	subscriptions, _ := subscriptionruntime.NewRuntime(channels, subscriptionproviders.Implementations()...)
	handler := &Handler{
		manager: manager, channels: channels, subscriptions: subscriptions, registry: registry, encryption: encryptionService,
		forwarder: forwarder, dialects: dialects, stats: stats, mutations: mutations,
		limiter: limiter, requestLogSink: requestLogSink, priceTables: priceTables,
		affinityCache:    affinity.NewCache(),
		responseBindings: state.NewResponseBindings(),
		websocketLimits:  defaultWebsocketLimits(),
		newRandom:        func() *rand.Rand { return rand.New(rand.NewSource(rand.Int63())) },
		newRequestID:     newRequestID,
		requestNow:       time.Now,
		now:              time.Now,
		writeTimeout:     downstreamWriteTimeout,
		modelListLimit:   maxNonStreamingResponseBodyBytes,
		logger:           logrus.StandardLogger(),
		authFailureEvents: utils.NewRateLimitedEventCounter(
			time.Minute,
			time.Now,
		),
		routeNotFoundEvents: utils.NewRateLimitedEventCounter(
			time.Minute,
			time.Now,
		),
	}
	for _, runtime := range accessQuotas {
		if runtime != nil {
			handler.accessQuota = runtime
			break
		}
	}
	return handler
}

// NewHandlerWithLifecycle wires the process HTTP lifecycle coordinator into
// the production gateway while preserving the compact constructor used by
// focused gateway tests.
func NewHandlerWithLifecycle(
	manager *state.Manager,
	registry *state.CredentialRegistry,
	channelRegistry *channel.Registry,
	subscriptions *subscriptionruntime.Runtime,
	encryptionService encryption.Service,
	forwarder AttemptForwarder,
	dialects dialect.Set,
	stats *health.StatsStore,
	mutations *health.MutationCoordinator,
	limiter AccessKeyRPMLimiter,
	requestLogSink telemetry.RequestLogSink,
	priceTables PriceTableProvider,
	accessQuota *accessquota.Runtime,
	lifecycle *httplifecycle.Coordinator,
	responseBindings *state.ResponseBindings,
	affinityCache *affinity.Cache,
	affinityStore affinity.BindingStore,
) *Handler {
	handler := NewHandler(
		manager,
		registry,
		encryptionService,
		forwarder,
		dialects,
		stats,
		mutations,
		limiter,
		requestLogSink,
		priceTables,
		accessQuota,
	)
	if channelRegistry != nil {
		handler.channels = channelRegistry
	}
	if subscriptions != nil {
		handler.subscriptions = subscriptions
	}
	handler.lifecycle = lifecycle
	handler.responseBindings = responseBindings
	if affinityCache != nil {
		handler.affinityCache = affinityCache
	}
	handler.affinityStore = affinityStore
	return handler
}

type unlimitedAccessKeyRPMLimiter struct{}

func (unlimitedAccessKeyRPMLimiter) Allow(uint, int64) ratelimit.LimitDecision {
	return ratelimit.LimitDecision{Allowed: true}
}

type requestAccessQuotaAdmission struct {
	accessKeyID uint
	snapshot    *state.ConfigSnapshot
	ticket      accessquota.Ticket
	admitted    bool
}

func (handler *Handler) applyDecisionEffect(
	credentialID uint,
	decision health.Decision,
	statusCode int,
	attemptNow time.Time,
) {
	defaults := state.DefaultRuntimeSettings()
	handler.applyDecisionEffectWithBlacklistPolicy(
		credentialID,
		0,
		decision,
		statusCode,
		attemptNow,
		defaults.BlacklistThreshold,
		handler.blacklistReleaseDeadline(attemptNow),
	)
}

func (handler *Handler) applyGroupDecisionEffect(
	group state.GroupView,
	credentialID uint,
	credentialVersion uint64,
	decision health.Decision,
	statusCode int,
	attemptNow time.Time,
) {
	handler.applyGroupDecisionEffectForEntry(
		group,
		credentialID,
		credentialVersion,
		"",
		decision,
		statusCode,
		attemptNow,
	)
}

func (handler *Handler) applyGroupDecisionEffectForEntry(
	group state.GroupView,
	credentialID uint,
	credentialVersion uint64,
	entryID string,
	decision health.Decision,
	statusCode int,
	attemptNow time.Time,
) {
	entryID = strings.TrimSpace(entryID)
	releaseAt := handler.blacklistReleaseDeadline(attemptNow)
	if decision.Scope == execution.ErrorScopeModel && entryID != "" {
		if registry, ok := handler.registry.(entryRuntimeRegistry); ok {
			mutate := func() {
				if !isModelEntryFailure(decision) {
					// Compatibility C1: model-scoped decisions outside the
					// counted failure families (safety.replay_unknown and any
					// other uncounted rule) keep the legacy Effect-driven
					// behavior byte for byte.
					applyLegacyModelScopeEntryEffect(registry, group, entryID, decision, releaseAt)
					return
				}
				applyModelEntryBreakerEffect(registry, group, entryID, decision, attemptNow, releaseAt)
			}
			if handler.mutations == nil {
				mutate()
			} else {
				handler.mutations.Do(credentialID, mutate)
			}
			return
		}
	}
	handler.applyDecisionEffectWithBlacklistPolicy(
		credentialID,
		credentialVersion,
		decision,
		statusCode,
		attemptNow,
		group.BlacklistThreshold,
		releaseAt,
	)
}

// blacklistReleaseDeadline computes the local-only blacklist release deadline
// for one failure event from the current published system setting, falling back
// to the shipped default when no snapshot is available.
func (handler *Handler) blacklistReleaseDeadline(attemptNow time.Time) time.Time {
	seconds := state.DefaultRuntimeSettings().BlacklistReleaseSeconds
	if handler != nil && handler.manager != nil {
		if snapshot := handler.manager.Current(); snapshot != nil &&
			snapshot.Settings.BlacklistReleaseSeconds > 0 {
			seconds = snapshot.Settings.BlacklistReleaseSeconds
		}
	}
	return attemptNow.Add(time.Duration(seconds) * time.Second)
}

// isModelEntryFailure implements design §4.1: a model-level failure is defined
// by the decision features alone, independent of Effect. Every upstream-model
// failure family of the judge qualifies (model.unavailable,
// images.model_unavailable, embeddings.model_unavailable,
// candidate.unavailable); only safety.replay_unknown is excluded because the
// replay safety of the attempt is unknown.
func isModelEntryFailure(decision health.Decision) bool {
	return decision.Category == health.FailureCategoryModelUnavailable &&
		decision.Scope == execution.ErrorScopeModel &&
		decision.RuleID != "safety.replay_unknown"
}

// applyModelEntryBreakerEffect counts model-level failures and applies the
// per-entry circuit breaker configuration. Without an explicit per-entry
// threshold the failure is not counted at all (design §3.2: the group-level
// BlacklistThreshold is deliberately not inherited); without an explicit
// cooldown the decision's own cooldown applies.
func applyModelEntryBreakerEffect(
	registry entryRuntimeRegistry,
	group state.GroupView,
	entryID string,
	decision health.Decision,
	attemptNow time.Time,
	releaseAt time.Time,
) {
	breaker := group.ModelBreakerByEntry[group.ID][entryID]
	if breaker != nil && breaker.BlacklistThreshold != nil {
		if atomicRegistry, ok := registry.(*state.CredentialRegistry); ok {
			atomicRegistry.RecordEntryFailureWithBlacklist(
				group.ID, entryID, *breaker.BlacklistThreshold, releaseAt,
			)
		} else if count, exists := registry.IncrEntryFailureForEntry(group.ID, entryID); exists &&
			count >= *breaker.BlacklistThreshold {
			if _, changed := registry.SetEntryBlacklistedForEntry(group.ID, entryID); changed {
				registry.SetEntryBlacklistReleaseAt(group.ID, entryID, releaseAt)
			}
		}
	}
	if breaker != nil && breaker.CooldownSeconds != nil {
		if *breaker.CooldownSeconds > 0 {
			registry.SetEntryCooldownForEntry(
				group.ID,
				entryID,
				attemptNow.Add(time.Duration(*breaker.CooldownSeconds)*time.Second),
			)
		}
		// cooldown_seconds == 0 counts the failure without any cooldown;
		// no SetEntryCooldown call so no stale deadline can surface in
		// route inspection.
		return
	}
	if decision.Effect == health.EffectCooldownCredential && !decision.CooldownUntil.IsZero() {
		registry.SetEntryCooldownForEntry(group.ID, entryID, decision.CooldownUntil)
	}
}

// applyLegacyModelScopeEntryEffect preserves the pre-entry-breaker behavior
// for model-scoped decisions that are not counted as model-level failures.
func applyLegacyModelScopeEntryEffect(
	registry entryRuntimeRegistry,
	group state.GroupView,
	entryID string,
	decision health.Decision,
	releaseAt time.Time,
) {
	if decision.Effect == health.EffectCooldownCredential && !decision.CooldownUntil.IsZero() {
		registry.SetEntryCooldownForEntry(group.ID, entryID, decision.CooldownUntil)
	}
	if decision.Effect != health.EffectRecordCredentialFailure {
		return
	}
	if atomicRegistry, ok := registry.(*state.CredentialRegistry); ok {
		atomicRegistry.RecordEntryFailureWithBlacklist(
			group.ID, entryID, group.BlacklistThreshold, releaseAt,
		)
		return
	}
	count, exists := registry.IncrEntryFailureForEntry(group.ID, entryID)
	if !exists {
		return
	}
	if group.BlacklistThreshold > 0 && count >= group.BlacklistThreshold {
		if _, changed := registry.SetEntryBlacklistedForEntry(group.ID, entryID); changed {
			registry.SetEntryBlacklistReleaseAt(group.ID, entryID, releaseAt)
		}
	}
}

func refreshCooldownCredentialVersion(result UpstreamResult, credentialVersion uint64) uint64 {
	if result.DispatchState == execution.DispatchNotSent && result.ExecutionError != nil &&
		result.ExecutionError.Hint == execution.FailureHintRefreshUnavailable {
		return credentialVersion
	}
	return 0
}

func (handler *Handler) applyDecisionEffectWithBlacklistPolicy(
	credentialID uint,
	credentialVersion uint64,
	decision health.Decision,
	statusCode int,
	attemptNow time.Time,
	blacklistThreshold int,
	releaseAt time.Time,
) {
	switch decision.Effect {
	case health.EffectCooldownCredential:
		mutate := func() {
			until := decision.CooldownUntil
			exists, changed := false, false
			if credentialVersion == 0 {
				exists, changed = handler.registry.SetCooldownWithChange(credentialID, until)
			} else {
				exists, changed = handler.registry.SetCooldownWithChangeIfVersion(
					credentialID,
					credentialVersion,
					until,
				)
			}
			if !exists {
				return
			}
			handler.stats.RecordProblem(credentialID, decision.Category, statusCode, attemptNow)
			if changed {
				handler.logCredentialCooldown(credentialID, decision.Category, statusCode)
			}
		}
		if handler.mutations == nil {
			mutate()
		} else {
			handler.mutations.Do(credentialID, mutate)
		}
	case health.EffectRecordCredentialFailure:
		handler.mutations.Do(credentialID, func() {
			count := 0
			ok := false
			becameBlacklisted := false
			if atomicRegistry, atomic := handler.registry.(*state.CredentialRegistry); atomic {
				count, ok, becameBlacklisted = atomicRegistry.RecordFailureWithBlacklist(
					credentialID, blacklistThreshold, releaseAt,
				)
			} else {
				count, ok = handler.registry.IncrFailure(credentialID)
				if ok && blacklistThreshold > 0 && count >= blacklistThreshold {
					var exists bool
					exists, becameBlacklisted = handler.registry.SetBlacklistedWithChange(credentialID)
					if exists && becameBlacklisted {
						if releaser, releaserOK := handler.registry.(blacklistReleaseRegistry); releaserOK {
							releaser.SetBlacklistReleaseAt(credentialID, releaseAt)
						}
					}
				}
			}
			if !ok {
				return
			}
			handler.stats.RecordFailure(credentialID, decision.Category, statusCode, attemptNow)
			if becameBlacklisted {
				handler.logCredentialBlacklisted(credentialID, count, decision.Category, statusCode)
			}
		})
	}
}

func (handler *Handler) recordCredentialSuccess(credentialID uint, at time.Time) {
	handler.mutations.Do(credentialID, func() {
		if handler.registry.ClearFailure(credentialID) {
			handler.stats.RecordSuccess(credentialID, at)
		}
	})
}

// recordEntrySuccess mirrors the credential success path for route entries
// (design §4.2): a confirmed success clears the entry failure counter while
// leaving cooldown and blacklist untouched.
func (handler *Handler) recordEntrySuccess(groupID uint, entryID string, credentialID uint) {
	entryID = strings.TrimSpace(entryID)
	if groupID == 0 || entryID == "" {
		return
	}
	registry, ok := handler.registry.(entryRuntimeRegistry)
	if !ok {
		return
	}
	clear := func() {
		registry.ClearEntryFailureForEntry(groupID, entryID)
	}
	if handler.mutations == nil {
		clear()
	} else {
		handler.mutations.Do(credentialID, clear)
	}
}

// retryAttemptLimit converts the system retry_count into the total forward
// attempt budget of one request. retry_count is that budget itself: 0 and 1 both
// stop after the first attempt, 2 allows one candidate switch. The budget never
// decides whether a candidate switch is legal; replay safety owns that.
func retryAttemptLimit(retryCount int) int {
	if retryCount <= 1 {
		return 1
	}
	return retryCount
}

func (handler *Handler) Handle(ginContext *gin.Context) {
	requestContext, ok := dataPlaneRequestContextFrom(ginContext)
	if !ok ||
		!requestContext.authenticated ||
		requestContext.snapshot == nil {
		handler.dataPlaneRouteNotFound(ginContext)
		return
	}
	if requestContext.locallyRejected {
		handler.dataPlaneRouteNotFound(ginContext)
		return
	}
	if websocketIntent(ginContext.Request) {
		handler.handleWebsocket(ginContext, requestContext)
		return
	}
	requestStarted := requestContext.requestStarted
	snapshot := requestContext.snapshot
	accessKey := requestContext.accessKey
	selectedRoute := requestContext.selectedRoute

	requestID, err := handler.newRequestID()
	if err != nil {
		utils.LogPlaneBestEffort(
			logrus.StandardLogger(),
			logrus.WarnLevel,
			utils.LogPlaneData,
			logrus.Fields{"error": err},
			"Request telemetry disabled",
		)
		requestID = ""
	} else {
		ginContext.Writer.Header().Set(requestIDHeader, requestID)
	}
	if capture := captureFromContext(ginContext); capture != nil {
		capture.updateMetadata(CaptureSessionMetadata{
			RequestID:   requestID,
			AccessKeyID: accessKey.ID,
			Protocol:    string(selectedRoute.Protocol),
		})
	}

	var recorder *requestRecorder
	var dispatch *requestDispatch
	if selectedRoute.Kind == endpointForward {
		recorder = newRequestRecorder(
			handler.requestLogSink,
			requestID,
			requestStarted,
			accessKey.ID,
			selectedRoute.Protocol,
			handler.requestNow,
		)
		recorder.accessKeyMultiplier = accessKey.PriceMultiplier
		defer func() {
			recorder.completeMissingOutcome(
				ginContext.Writer.Written(),
				ginContext.Writer.Status(),
			)
			if dispatch != nil && dispatch.quotaAdmission != nil &&
				dispatch.quotaAdmission.admitted && handler.accessQuota != nil {
				completion := handler.accessQuota.Complete(
					dispatch.quotaAdmission.ticket,
					recorder.estimatedCostNanoUSD(),
				)
				handler.logAccessQuotaCompletionFault(accessKey.ID, completion)
			}
			recorder.emit()
		}()
	}

	outcome := handler.admitRequest(ginContext.Request.Context(), admissionInput{
		request:   ginContext.Request,
		route:     selectedRoute,
		snapshot:  snapshot,
		accessKey: accessKey,
		recorder:  recorder,
	})
	if outcome.inspected != nil {
		if capture := captureFromContext(ginContext); capture != nil {
			capture.updateMetadata(CaptureSessionMetadata{
				Operation: string(outcome.inspected.Operation),
			})
		}
	}
	if outcome.cancelled {
		return
	}
	if outcome.rejection != nil {
		handler.completeAdmissionRejection(ginContext, recorder, outcome.rejection)
		return
	}
	dispatch = outcome.dispatch
	if dispatch.modelsOnly {
		handler.writeVisibleModelList(ginContext, snapshot, accessKey, selectedRoute.Protocol)
		return
	}
	recorder.observeDispatch(dispatch)
	iterator := scheduler.New(snapshot, handler.registry, dispatch.query, handler.newRandom())
	handler.executeAttempts(
		ginContext,
		iterator,
		retryAttemptLimit(snapshot.Settings.RetryCount),
		dispatch,
		recorder,
	)
}

// completeAdmissionRejection maps an admission rejection onto the wire. The
// reason catalog in reason.go is the frozen HTTP contract; this mapping only
// applies the headers each rejection carries and delegates the body write.
func (handler *Handler) completeAdmissionRejection(
	ginContext *gin.Context,
	recorder *requestRecorder,
	rejection *admissionRejection,
) {
	if rejection.affinity != nil {
		recorder.setAffinityKey(rejection.affinity.displayKey)
		recorder.setAffinityObservations(rejection.affinity.source, rejection.affinity.state)
	}
	for name, values := range rejection.headers {
		for _, value := range values {
			ginContext.Writer.Header().Set(name, value)
		}
	}
	if rejection.retryAfter != nil {
		ginContext.Writer.Header().Set("Retry-After", strconv.Itoa(*rejection.retryAfter))
	}
	if rejection.quotaDecision != nil {
		handler.completeAccessQuotaReason(ginContext, recorder, *rejection.quotaDecision)
		return
	}
	handler.completeReason(ginContext, recorder, rejection.reason)
}

func (handler *Handler) quotaNow() time.Time {
	if handler != nil && handler.now != nil {
		return handler.now()
	}
	return time.Now()
}

func (handler *Handler) checkAccessQuotaForSnapshot(
	snapshot *state.ConfigSnapshot,
	accessKeyID uint,
	now time.Time,
) (accessquota.Decision, bool) {
	var decision accessquota.Decision
	if handler == nil || handler.manager == nil || handler.accessQuota == nil || snapshot == nil {
		return decision, false
	}
	current := handler.manager.WithCurrentSnapshotRead(func(currentSnapshot *state.ConfigSnapshot) bool {
		if currentSnapshot != snapshot {
			return false
		}
		decision = handler.accessQuota.Check(accessKeyID, now)
		return true
	})
	return decision, current
}

func (handler *Handler) admitAccessQuotaForSnapshot(
	snapshot *state.ConfigSnapshot,
	accessKeyID uint,
	now time.Time,
) (accessquota.Ticket, accessquota.Decision, bool) {
	var ticket accessquota.Ticket
	var decision accessquota.Decision
	if handler == nil || handler.manager == nil || handler.accessQuota == nil || snapshot == nil {
		return ticket, decision, false
	}
	current := handler.manager.WithCurrentSnapshotRead(func(currentSnapshot *state.ConfigSnapshot) bool {
		if currentSnapshot != snapshot {
			return false
		}
		ticket, decision = handler.accessQuota.Admit(accessKeyID, now)
		return true
	})
	return ticket, decision, current
}

func (handler *Handler) logAccessQuotaCompletionFault(
	accessKeyID uint,
	completion accessquota.CompletionResult,
) {
	if completion.Fault == "" {
		return
	}
	utils.LogPlaneBestEffort(
		handler.logger,
		logrus.ErrorLevel,
		utils.LogPlaneData,
		logrus.Fields{
			"access_key_id": accessKeyID,
			"failure_type":  string(completion.Fault),
		},
		"Access key cost limit accounting saturated",
	)
}

func optionalModelValue(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func retryAfterSeconds(duration time.Duration) int {
	seconds := int((duration + time.Second - 1) / time.Second)
	if seconds < 1 {
		return 1
	}
	if seconds > 60 {
		return 60
	}
	return seconds
}

func (handler *Handler) completeReason(
	ginContext *gin.Context,
	recorder *requestRecorder,
	value reason,
) {
	recorder.completeReason(value)
	if err := handler.writeReason(ginContext, value); err != nil {
		handler.completeWriteTerminal(ginContext, recorder, value.Status)
	}
}

func (handler *Handler) completeWriteTerminal(
	ginContext *gin.Context,
	recorder *requestRecorder,
	selectedStatus int,
) {
	if ginContext != nil && ginContext.Request != nil &&
		ginContext.Request.Context().Err() != nil {
		status := 0
		if ginContext.Writer != nil && ginContext.Writer.Written() {
			status = ginContext.Writer.Status()
		}
		recorder.completeCanceled(ginContext.Request.Context(), status, -1)
		return
	}
	recorder.completeDownstreamWrite(selectedStatus)
}

func readDecodedRequestBody(
	request *http.Request,
	encoding contentcoding.Encoding,
	encodedLimit int64,
	decodedLimit int64,
) ([]byte, error) {
	if request == nil || request.Body == nil {
		return nil, fmt.Errorf("request body is required")
	}
	if encodedLimit < 0 {
		return nil, fmt.Errorf("encoded request body limit must not be negative")
	}
	if decodedLimit < 0 {
		return nil, fmt.Errorf("decoded request body limit must not be negative")
	}
	if request.ContentLength > 0 && request.ContentLength > encodedLimit {
		return nil, errRequestTooLarge
	}
	reader := io.Reader(request.Body)
	if encodedLimit < math.MaxInt64 {
		reader = io.LimitReader(request.Body, encodedLimit+1)
	}
	encoded, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("read request body: %w", err)
	}
	if int64(len(encoded)) > encodedLimit {
		return nil, errRequestTooLarge
	}
	body, err := contentcoding.DecodeLimited(encoding, encoded, decodedLimit)
	if errors.Is(err, contentcoding.ErrDecodedBodyTooLarge) {
		return nil, errRequestTooLarge
	}
	return body, err
}

func headerFieldValues(headers http.Header, name string) []string {
	var values []string
	for actualName, fieldValues := range headers {
		if strings.EqualFold(actualName, name) {
			values = append(values, fieldValues...)
		}
	}
	return values
}


// executeAttempts runs the candidate loop. All admission-derived inputs come
// from the frozen dispatch; orchestration state lives on attemptLoop.
func (handler *Handler) executeAttempts(
	ginContext *gin.Context,
	iterator *scheduler.Iterator,
	forwardAttemptLimit int,
	dispatch *requestDispatch,
	recorder *requestRecorder,
) {
	newAttemptLoop(handler, ginContext, iterator, forwardAttemptLimit, dispatch, recorder).run()
}

func initializeDebugHeaders(headers http.Header) {
	headers.Set(debugHeaderGroup, "")
	headers.Set(debugHeaderAttempts, "0")
}

func updateDebugHeaders(headers http.Header, group string, attempts int) {
	headers.Set(debugHeaderGroup, group)
	headers.Set(debugHeaderAttempts, strconv.Itoa(attempts))
}

func transportReason(result UpstreamResult) reason {
	switch {
	case isConversionUnsupportedResult(result):
		return reasonProtocolConversionUnsupported
	case result.DispatchState == execution.DispatchNotSent && result.ExecutionError != nil &&
		result.ExecutionError.Kind == execution.ErrorKindInvalidRequest:
		return reasonInvalidProtocolRequest
	case errors.Is(result.Err, ErrUpstreamProtocol):
		return reasonUpstreamProtocol
	case isTimeoutError(result.Err):
		return reasonUpstreamTimeout
	default:
		return reasonUpstreamConnect
	}
}

func isConversionUnsupportedResult(result UpstreamResult) bool {
	return result.DispatchState == execution.DispatchNotSent && result.ExecutionError != nil &&
		result.ExecutionError.Kind == execution.ErrorKindConversionUnsupported
}

func (handler *Handler) writeUpstreamResponse(ginContext *gin.Context, result UpstreamResult) error {
	return handler.writeBufferedResponse(ginContext, result.StatusCode, result.Header, result.Body)
}

func (handler *Handler) writeBufferedResponse(
	ginContext *gin.Context,
	status int,
	headers http.Header,
	body []byte,
) (err error) {
	if handler == nil || ginContext == nil {
		return fmt.Errorf("downstream response writer is required")
	}
	controlled := newStreamWriteController(ginContext.Writer, handler.writeTimeout)
	defer func() {
		if clearErr := controlled.clear(); err == nil && clearErr != nil {
			err = fmt.Errorf("clear downstream write deadline: %w", clearErr)
		}
	}()

	method := ""
	if ginContext.Request != nil {
		method = ginContext.Request.Method
	}
	normalizedHeaders, writeBody := normalizeBufferedResponse(method, status, headers, body)
	for name, values := range normalizedHeaders {
		for _, value := range values {
			ginContext.Writer.Header().Add(name, value)
		}
	}
	if err := controlled.writeHeader(status); err != nil {
		return fmt.Errorf("write downstream response headers: %w", err)
	}
	ginContext.Writer.WriteHeaderNow()
	if writeBody && len(body) > 0 {
		written, writeErr := controlled.write(body)
		if writeErr != nil {
			return fmt.Errorf("write downstream response: %w", writeErr)
		}
		if written != len(body) {
			return fmt.Errorf("write downstream response: %w", io.ErrShortWrite)
		}
	}
	if err := controlled.flush(); err != nil {
		return fmt.Errorf("flush downstream response: %w", err)
	}
	return nil
}
