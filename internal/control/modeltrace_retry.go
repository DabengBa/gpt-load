package control

import (
	"bytes"
	"context"
	cryptorand "crypto/rand"
	"reflect"
	"slices"

	"sync"
	"time"

	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/state"
)

const (
	modelTraceRetryTTL      = 30 * time.Minute
	modelTraceRetryMaxRuns  = 64
	modelTraceRetryMaxBytes = 128 << 20
)

type ModelTraceRetryRequest struct {
	ModelTraceRequest
	RunID       string `json:"run_id"`
	ChallengeID string `json:"challenge_id"`
}

type modelTraceRetryRun struct {
	expiresAt  time.Time
	bytes      int
	executed   modelTraceExecutionContext
	challenges []modelTraceChallengeDTO
}

// Retry state is bounded and local to the control process. It is not request
// history and never stores the credential material used for dispatch.
type modelTraceRetryCache struct {
	mu   sync.Mutex
	runs map[string]modelTraceRetryRun
}

func (s *Service) keepModelTraceRun(id string, outcome *ModelTraceProbeResponse, executed modelTraceExecutionContext, challenges []modelTraceChallengeDTO) {
	cache := &s.modelTraceRetries
	cache.mu.Lock()
	defer cache.mu.Unlock()
	now := s.now()
	delete(cache.runs, id)
	totalBytes := 0
	for key, run := range cache.runs {
		if !now.Before(run.expiresAt) {
			delete(cache.runs, key)
		} else {
			totalBytes += run.bytes
		}
	}
	if !slices.ContainsFunc(outcome.Challenges, func(row ModelTraceChallengeResponse) bool { return row.Retryable }) {
		return
	}
	size := 0
	for i, challenge := range challenges {
		size += len(challenge.Prompt) + len(executed.outputs[i].Text)
	}
	if size > modelTraceRetryMaxBytes {
		for i := range outcome.Challenges {
			outcome.Challenges[i].Retryable = false
		}
		return
	}
	if id == "" {
		var err error
		id, err = newOperationID(cryptorand.Reader)
		if err != nil {
			for i := range outcome.Challenges {
				outcome.Challenges[i].Retryable = false
			}
			return
		}
	}
	for len(cache.runs) >= modelTraceRetryMaxRuns || totalBytes+size > modelTraceRetryMaxBytes {
		var oldest string
		for key, run := range cache.runs {
			if oldest == "" || run.expiresAt.Before(cache.runs[oldest].expiresAt) {
				oldest = key
			}
		}
		totalBytes -= cache.runs[oldest].bytes
		delete(cache.runs, oldest)
	}
	if cache.runs == nil {
		cache.runs = make(map[string]modelTraceRetryRun)
	}
	executed.credential.EncryptedValue = ""
	executed.priceTable = nil
	group := executed.group
	executed.group = state.GroupView{
		ID: group.ID, ChannelID: group.ChannelID, ResolvedTarget: group.ResolvedTarget,
		Proxy: group.Proxy, HeaderRules: group.HeaderRules,
	}
	executed.challenges = slices.Clone(outcome.Challenges)
	cache.runs[id] = modelTraceRetryRun{expiresAt: now.Add(modelTraceRetryTTL), bytes: size, executed: executed, challenges: challenges}
	outcome.RunID = &id
}

func (cache *modelTraceRetryCache) lookup(now time.Time, id string) (modelTraceRetryRun, bool) {
	cache.mu.Lock()
	defer cache.mu.Unlock()
	run, found := cache.runs[id]
	if found && !now.Before(run.expiresAt) {
		delete(cache.runs, id)
		return modelTraceRetryRun{}, false
	}
	return run, found
}

func (s *Service) RetryModelTraceChallenge(ctx context.Context, request ModelTraceRetryRequest) (ModelTraceProbeResponse, error) {
	if len(request.RunID) != 36 || request.ChallengeID == "" {
		return ModelTraceProbeResponse{}, app_errors.ErrBadRequest
	}
	ctx, done, err := s.beginModelTraceRun(ctx, request.ModelTraceRequest)
	if err != nil {
		return ModelTraceProbeResponse{}, err
	}
	defer done()
	run, found := s.modelTraceRetries.lookup(s.now(), request.RunID)
	executed := run.executed
	if !found || executed.group.ID != request.GroupID || executed.target.model != request.Model {
		return ModelTraceProbeResponse{}, app_errors.ErrBadRequest
	}
	index := slices.IndexFunc(executed.challenges, func(row ModelTraceChallengeResponse) bool {
		return row.ID == request.ChallengeID && row.Retryable
	})
	if index < 0 {
		return ModelTraceProbeResponse{}, app_errors.ErrBadRequest
	}
	observation, err := s.captureRuntimeObservation()
	if err != nil {
		return ModelTraceProbeResponse{}, err
	}
	group, found := observation.snapshot.LookupGroup(request.GroupID)
	target, supported := modelTraceTarget(group, request.Model)
	previous := executed.group
	// Do not mix samples from different destinations or request configurations.
	if !found || !supported || target != executed.target || previous.ChannelID != group.ChannelID ||
		previous.ResolvedTarget.ProviderKind != group.ResolvedTarget.ProviderKind ||
		!bytes.Equal(previous.ResolvedTarget.TargetConfig, group.ResolvedTarget.TargetConfig) ||
		!reflect.DeepEqual(previous.Proxy, group.Proxy) || !reflect.DeepEqual(previous.HeaderRules, group.HeaderRules) {
		return ModelTraceProbeResponse{}, app_errors.ErrBadRequest
	}
	entries, err := s.registry.SnapshotGroupCredentialEntriesExact(group.ID, []uint{executed.credential.ID})
	if err != nil || len(entries) != 1 || entries[0].Version != executed.credential.Version ||
		entries[0].IdentityGeneration != executed.credential.IdentityGeneration {
		return ModelTraceProbeResponse{}, app_errors.ErrBadRequest
	}
	detector, err := getModelTraceDetector(ctx)
	if err != nil {
		return ModelTraceProbeResponse{}, app_errors.ErrModelTraceUnavailable
	}
	if err := ctx.Err(); err != nil {
		return ModelTraceProbeResponse{}, err
	}
	executed.group = group
	executed.credential = credentialProbeRef(entries[0])
	executed.priceTable = s.frozenModelTracePriceTable()
	executed.challenges = slices.Clone(executed.challenges)
	executed.outputs = slices.Clone(executed.outputs)
	executed.outputs[index] = modelTraceOutput{}
	s.runModelTraceChallenge(ctx, &executed, index, run.challenges[index])
	outcome := s.finishModelTraceRun(detector, executed, run.challenges)
	s.keepModelTraceRun(request.RunID, &outcome, executed, run.challenges)
	return outcome, nil
}

func (server *Server) handleModelTraceRetry(c *gin.Context) {
	var request ModelTraceRetryRequest
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "modeltrace_probe_retry", mapControlJSONError(err))
		return
	}
	result, err := server.service.RetryModelTraceChallenge(c.Request.Context(), request)
	if err != nil {
		writeServiceError(c, "modeltrace_probe_retry", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}
