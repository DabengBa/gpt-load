package control

import (
	"fmt"
	"math"
	"strconv"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/health"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/reasoning"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func credentialLabelFor(labels map[uint]string, credentialID *uint) string {
	if credentialID == nil {
		return ""
	}
	// 保持既有响应结构：未知或不可读标识用占位，只有确认不存在才留空。
	if labels == nil {
		return "—"
	}
	label, exists := labels[*credentialID]
	if exists && label == "" {
		return "—"
	}
	return label
}

// 一页日志里的凭据数远小于条数：同一个号会被反复使用，去重后通常只剩几个。
func requestLogCredentialIDs(records []requestlog.Record) []uint {
	ids := make([]uint, 0, len(records))
	for _, record := range records {
		ids = append(ids, record.CredentialID)
		for _, attempt := range record.Attempts {
			ids = append(ids, attempt.CredentialID)
		}
	}
	return ids
}

func mapRequestLogListResponse(
	page requestlog.Page,
	credentialLabels map[uint]string,
) (requestLogListResponse, error) {
	result := requestLogListResponse{
		Items: make([]requestLogItemResponse, 0, len(page.Items)),
	}
	for _, record := range page.Items {
		if err := validateSafeMilliseconds(record.CompletedAtMS); err != nil {
			return requestLogListResponse{}, fmt.Errorf(
				"map request log completed_at_ms: %w",
				err,
			)
		}
		usageCost, err := mapRequestLogUsageCost(record)
		if err != nil {
			return requestLogListResponse{}, err
		}
		item, err := mapRequestLogItemResponse(record, usageCost, credentialLabels)
		if err != nil {
			return requestLogListResponse{}, err
		}
		result.Items = append(result.Items, item)
	}
	if page.NextCursor != nil {
		encoded, err := encodeRequestLogCursor(*page.NextCursor)
		if err != nil {
			return requestLogListResponse{}, err
		}
		result.NextCursor = &encoded
	}
	return result, nil
}

func mapRequestLogItemResponse(
	record requestlog.Record,
	usageCost requestLogUsageCostResponse,
	credentialLabels map[uint]string,
) (requestLogItemResponse, error) {
	inputTokens, ok := checkedRequestLogInputTokens(record)
	if !ok {
		return requestLogItemResponse{}, fmt.Errorf("map request log input tokens: overflow")
	}
	routeMode, err := nullableRequestLogRouteMode(record.RouteMode)
	if err != nil {
		return requestLogItemResponse{}, fmt.Errorf("map request log final route mode: %w", err)
	}
	operation, err := nullableRequestLogOperation(record.Operation)
	if err != nil {
		return requestLogItemResponse{}, fmt.Errorf("map request log operation: %w", err)
	}
	upstreamProtocol, err := nullableRequestLogUpstreamProtocol(record.UpstreamProtocol)
	if err != nil {
		return requestLogItemResponse{}, fmt.Errorf("map request log upstream protocol: %w", err)
	}
	pricingMode, err := nullableRequestLogPricingMode(record)
	if err != nil {
		return requestLogItemResponse{}, fmt.Errorf("map request log pricing mode: %w", err)
	}
	feedback, err := mapRequestLogFeedback(record.FinalAttemptFeedback)
	if err != nil {
		return requestLogItemResponse{}, err
	}
	var contextThresholdTokens *string
	if record.ContextThresholdTokens != nil {
		value := strconv.FormatInt(*record.ContextThresholdTokens, 10)
		contextThresholdTokens = &value
	}
	// 在 API 边界重新施加 bounded affinity 投影，确保没有
	// 未限定或原始信号能到达响应，包括由早于 0013 迁移的
	// 读取者产生的行。
	affinitySource, affinityState := requestlog.NormalizeAffinityObservation(
		record.AffinitySource, record.AffinityState,
	)
	return requestLogItemResponse{
		RequestID:     record.RequestID,
		CompletedAtMS: record.CompletedAtMS,
		AccessKey: requestLogAccessKeyResponse{
			ID:      record.AccessKey.ID,
			Name:    record.AccessKey.Name,
			Deleted: record.AccessKey.Deleted,
		},
		Protocol:                     string(record.Protocol),
		Operation:                    operation,
		UpstreamProtocol:             upstreamProtocol,
		ClientModel:                  nullableRequestLogModel(record.ClientModel),
		UpstreamModel:                nullableRequestLogModel(record.UpstreamModel),
		UpstreamReportedModel:        nullableRequestLogModel(record.UpstreamReportedModel),
		ModelConsistency:             record.ModelConsistency,
		Reasoning:                    mapRequestLogReasoning(record),
		Status:                       record.Status,
		StatusCode:                   record.StatusCode,
		Stream:                       record.Stream,
		FirstResponseMs:              record.FirstResponseMs,
		DurationMs:                   record.DurationMs,
		FinalAttemptDurationMs:       record.FinalAttemptDurationMs,
		AttemptCount:                 record.AttemptCount,
		FeedbackStatus:               feedback.FeedbackStatus,
		FeedbackReason:               feedback.FeedbackReason,
		ProviderFirstResponseMs:      feedback.ProviderFirstResponseMs,
		ProviderTokensPerSecond:      feedback.ProviderTokensPerSecond,
		ErrorCode:                    record.ErrorCode,
		ErrorSummary:                 record.ErrorSummary,
		AffinityHit:                  record.AffinityHit,
		ResolvedAffinityCredentialID: record.ResolvedAffinityCredentialID,
		FirstAttemptCredentialID:     record.FirstAttemptCredentialID,
		AffinityHitAssignmentReached: record.AffinityHitAssignmentReached,
		ContinuityHit:                record.ContinuityHit,
		AffinitySource:               affinitySource,
		AffinityState:                affinityState,
		AffinityKey:                  nullableRequestLogAffinityKey(record.AffinityKey),
		GroupID:                      usageCost.groupID,
		ChannelID:                    usageCost.channelID,
		CredentialID:                 usageCost.credentialID,
		CredentialName:               credentialLabelFor(credentialLabels, usageCost.credentialID),
		RouteMode:                    routeMode,
		UsageState:                   record.UsageState,
		CostState:                    record.CostState,
		PricingCompleteness:          record.PricingCompleteness,
		PricingMode:                  pricingMode,
		ContextThresholdTokens:       contextThresholdTokens,
		InputTokens:                  strconv.FormatInt(inputTokens, 10),
		CacheReadTokens:              strconv.FormatInt(record.CacheReadTokens, 10),
		CacheWrite5MTokens:           strconv.FormatInt(record.CacheWrite5MTokens, 10),
		CacheWrite1HTokens:           strconv.FormatInt(record.CacheWrite1HTokens, 10),
		CacheWriteUnknownTokens:      strconv.FormatInt(record.CacheWriteUnknownTokens, 10),
		OutputTokens:                 strconv.FormatInt(record.OutputTokens, 10),
		EstimatedCostNanoUSD:         usageCost.estimatedCostNanoUSD,
	}, nil
}

func mapRequestLogFeedback(feedback health.Feedback) (requestLogFeedbackResponse, error) {
	if !feedback.Status.Valid() {
		return requestLogFeedbackResponse{}, fmt.Errorf("map request log feedback: invalid status")
	}
	if feedback.FirstResponseMs != nil && (*feedback.FirstResponseMs < 0 || *feedback.FirstResponseMs > maxSafeInteger) {
		return requestLogFeedbackResponse{}, fmt.Errorf("map request log feedback: unsafe provider first response")
	}
	if feedback.TokensPerSecond != nil &&
		(math.IsNaN(*feedback.TokensPerSecond) || math.IsInf(*feedback.TokensPerSecond, 0) || *feedback.TokensPerSecond < 0) {
		return requestLogFeedbackResponse{}, fmt.Errorf("map request log feedback: invalid provider token rate")
	}
	switch feedback.Reason {
	case "", "upstream_failure", "first_response_slow", "output_rate_faulty", "output_rate_slow":
	default:
		return requestLogFeedbackResponse{}, fmt.Errorf("map request log feedback: invalid reason")
	}
	if (feedback.Status == health.FeedbackStatusUnassessed && feedback.Reason != "") ||
		(feedback.Status == health.FeedbackStatusNormal && feedback.Reason != "") ||
		(feedback.Status == health.FeedbackStatusSlow && feedback.Reason != "output_rate_slow") ||
		(feedback.Status == health.FeedbackStatusFaulty && feedback.Reason == "output_rate_slow") {
		return requestLogFeedbackResponse{}, fmt.Errorf("map request log feedback: status and reason do not match")
	}
	result := requestLogFeedbackResponse{
		ProviderFirstResponseMs: feedback.FirstResponseMs,
		ProviderTokensPerSecond: feedback.TokensPerSecond,
	}
	if feedback.Status != health.FeedbackStatusUnassessed {
		result.FeedbackStatus = new(feedback.Status)
	}
	if feedback.Reason != "" {
		result.FeedbackReason = new(feedback.Reason)
	}
	return result, nil
}

func nullableRequestLogPricingMode(record requestlog.Record) (*pricing.Mode, error) {
	mode := record.PricingMode
	if mode == "" {
		for index := len(record.Attempts) - 1; index >= 0; index-- {
			attempt := record.Attempts[index]
			if attempt.GroupID == record.GroupID &&
				attempt.ChannelID == record.ChannelID &&
				attempt.CredentialID == record.CredentialID &&
				attempt.PricingReceipt != nil {
				mode = attempt.PricingReceipt.PricingMode
				break
			}
		}
	}
	if mode == "" {
		return nil, nil
	}
	if !mode.Valid() {
		return nil, fmt.Errorf("invalid pricing mode")
	}
	return &mode, nil
}

func mapRequestLogReasoning(record requestlog.Record) *requestLogReasoningResponse {
	return mapRequestLogReasoningConfig(record.Reasoning)
}

func mapRequestLogReasoningConfig(config reasoning.Config) *requestLogReasoningResponse {
	if !config.Present() {
		return nil
	}
	result := &requestLogReasoningResponse{
		Mode:   nullableRequestLogReasoningValue(config.Mode),
		Effort: nullableRequestLogReasoningValue(config.Effort),
	}
	if config.BudgetTokens != nil {
		value := strconv.FormatInt(*config.BudgetTokens, 10)
		result.BudgetTokens = &value
	}
	return result
}

func nullableRequestLogReasoningValue(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func checkedRequestLogInputTokens(record requestlog.Record) (int64, bool) {
	total := int64(0)
	for _, value := range []int64{
		record.UncachedInputTokens,
		record.CacheReadTokens,
		record.CacheWrite5MTokens,
		record.CacheWrite1HTokens,
		record.CacheWriteUnknownTokens,
	} {
		var ok bool
		total, ok = usage.CheckedAdd(total, value)
		if !ok {
			return 0, false
		}
	}
	return total, true
}
func nullableRequestLogAffinityKey(value string) *string {
	value = requestlog.NormalizeAffinityKey(value)
	if value == "" {
		return nil
	}
	return &value
}

func nullableRequestLogModel(value string) *string {
	if value == "" {
		return nil
	}
	return &value
}

func nullableRequestLogChannelID(value channel.ID) (*channel.ID, error) {
	if value == "" {
		return nil, nil
	}
	if _, ok := requestLogChannelRegistry.Get(value); !ok {
		return nil, fmt.Errorf("unknown channel ID")
	}
	return &value, nil
}

func nullableRequestLogID(value uint, field string) (*uint, error) {
	if value == 0 {
		return nil, nil
	}
	if uint64(value) > uint64(maxSafeInteger) {
		return nil, fmt.Errorf("unsafe %s ID", field)
	}
	return &value, nil
}

func nullableRequestLogOperation(value execution.Operation) (*execution.Operation, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid operation")
	}
	return &value, nil
}

func nullableRequestLogUpstreamProtocol(value protocol.Protocol) (*protocol.Protocol, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid upstream protocol")
	}
	return &value, nil
}

func nullableRequestLogRouteMode(value channel.RouteMode) (*channel.RouteMode, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid route mode")
	}
	return &value, nil
}

func nullableRequestLogDispatchState(
	value execution.DispatchState,
) (*execution.DispatchState, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid dispatch state")
	}
	return &value, nil
}

func nullableRequestLogFailureOrigin(
	value execution.ErrorOrigin,
) (*execution.ErrorOrigin, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid failure origin")
	}
	return &value, nil
}

func nullableRequestLogFailureScope(
	value execution.ErrorScope,
) (*execution.ErrorScope, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid failure scope")
	}
	return &value, nil
}

func nullableRequestLogRetryDirective(
	value telemetry.RetryDirective,
) (*telemetry.RetryDirective, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid retry directive")
	}
	return &value, nil
}

func nullableRequestLogEffect(value telemetry.Effect) (*telemetry.Effect, error) {
	if value == "" {
		return nil, nil
	}
	if !value.Valid() {
		return nil, fmt.Errorf("invalid effect")
	}
	return &value, nil
}
func mapRequestLogUsageCost(record requestlog.Record) (requestLogUsageCostResponse, error) {
	if err := requestlog.ValidateUsageCostState(
		record.UsageState,
		record.CostState,
		record.PricingCompleteness,
		record.EstimatedCostNanoUSD,
	); err != nil {
		return requestLogUsageCostResponse{}, fmt.Errorf("map request log usage/cost: %w", err)
	}
	for _, value := range []int64{
		record.UncachedInputTokens,
		record.CacheReadTokens,
		record.CacheWrite5MTokens,
		record.CacheWrite1HTokens,
		record.CacheWriteUnknownTokens,
		record.OutputTokens,
	} {
		if value < 0 {
			return requestLogUsageCostResponse{}, fmt.Errorf("map request log usage tokens: negative value")
		}
	}
	if _, ok := usage.CheckedTotal(usage.Tokens{
		UncachedInput:     record.UncachedInputTokens,
		CacheRead:         record.CacheReadTokens,
		CacheWrite5M:      record.CacheWrite5MTokens,
		CacheWrite1H:      record.CacheWrite1HTokens,
		CacheWriteUnknown: record.CacheWriteUnknownTokens,
		Output:            record.OutputTokens,
	}); !ok {
		return requestLogUsageCostResponse{}, fmt.Errorf("map request log usage tokens: unsafe total")
	}
	if uint64(record.GroupID) > uint64(maxSafeInteger) {
		return requestLogUsageCostResponse{}, fmt.Errorf("map request log final Group ID: unsafe value")
	}
	channelID, err := nullableRequestLogChannelID(record.ChannelID)
	if err != nil {
		return requestLogUsageCostResponse{}, fmt.Errorf("map request log final channel: %w", err)
	}
	credentialID, err := nullableRequestLogID(record.CredentialID, "credential")
	if err != nil {
		return requestLogUsageCostResponse{}, fmt.Errorf("map request log final credential: %w", err)
	}
	if record.EstimatedCostNanoUSD < 0 {
		return requestLogUsageCostResponse{}, fmt.Errorf(
			"map request log usage/cost: negative estimated cost",
		)
	}
	result := requestLogUsageCostResponse{
		channelID:            channelID,
		credentialID:         credentialID,
		estimatedCostNanoUSD: strconv.FormatInt(record.EstimatedCostNanoUSD, 10),
	}
	if record.GroupID != 0 {
		groupID := record.GroupID
		result.groupID = &groupID
	}
	return result, nil
}

// CredentialLabels 把凭据 ID 翻译成脱敏的可读标识。
// 密文常驻凭据注册表，整个过程不读数据库；只按持久身份判断存在，不按运行状态过滤。
// 已存在但标识无法解析的凭据保留空值，只有注册表中不存在的凭据才省略。
// 入参允许重复，每个 ID 至多解密一次；nil 结果表示关联信息尚不可查询，而非删除。
func (s *Service) CredentialLabels(credentialIDs []uint) map[uint]string {
	if s == nil || s.manager == nil || s.registry == nil || s.encryption == nil || len(credentialIDs) == 0 {
		return nil
	}
	s.writeMu.RLock()
	snapshot := s.manager.Current()
	s.writeMu.RUnlock()
	if snapshot == nil {
		return nil
	}
	labels := make(map[uint]string, len(credentialIDs))
	seen := make(map[uint]struct{}, len(credentialIDs))
	for _, credentialID := range credentialIDs {
		if credentialID == 0 {
			continue
		}
		if _, done := seen[credentialID]; done {
			continue
		}
		seen[credentialID] = struct{}{}
		ref, known := s.registry.CredentialRef(credentialID)
		if !known {
			continue
		}
		labels[credentialID] = ""
		group, exists := snapshot.Groups[ref.GroupID]
		if !exists {
			group, exists = snapshot.DisabledGroups[ref.GroupID]
		}
		if !exists {
			continue
		}
		label, err := s.credentialLabel(ref.EncryptedValue, group.ChannelID)
		if err != nil || label == "" {
			continue
		}
		labels[credentialID] = label
	}
	return labels
}

// 与凭据列表一致：健康问题列表展示脱敏身份。
func (s *Service) credentialLabel(
	ciphertext string,
	channelID channel.ID,
) (string, error) {
	plaintext, err := s.encryption.Decrypt(ciphertext)
	if err != nil {
		return "", err
	}
	validated, err := normalizeStoredCredential(s.channelRegistry, channelID, plaintext)
	if err != nil {
		return "", err
	}
	return maskCanonicalCredential(validated.CanonicalJSON())
}
