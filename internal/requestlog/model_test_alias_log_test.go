package requestlog

import (
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/state"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func TestModelTestAliasDurableLogKeepsRouteAttributionAndUpstreamUsage(t *testing.T) {
	t.Parallel()
	const (
		requestID = "66666666-6666-4666-8666-666666666666"
		groupID   = uint(7)
		entryID   = "e000000000042"
		alias     = "a4g233"
		upstream  = "provider-model"
	)

	// This is the persisted group/model shape used to derive the route entry:
	// durable request-log rows intentionally store group/channel/attempt identity,
	// while the globally unique test alias resolves through the group model JSON.
	snapshot, err := state.Compile(state.CompileInput{
		ChannelRegistry: channel.NewRegistry(),
		Groups: []state.GroupConfig{{
			ConnectionType: "api_key",
			ID:             groupID,
			Name:           "primary",
			ChannelID:      channel.OpenAI,
			Params:         []byte(`{}`),
			Models:         []state.ModelConfig{{ID: upstream, TestAlias: alias, EntryID: entryID}},
			Settings:       config.Settings{},
			Enabled:        true,
		}},
	})
	if err != nil {
		t.Fatalf("Compile() error = %v", err)
	}
	targets := snapshot.ExecutionRouteCatalog[protocol.OpenAICompletions][execution.OperationChatCompletion][alias]
	if len(targets) != 1 || targets[0].GroupID != groupID || targets[0].UpstreamModelID != upstream || targets[0].EntryID != entryID {
		t.Fatalf("test alias route targets = %#v, want group %d/upstream %q/entry %q", targets, groupID, upstream, entryID)
	}

	event := testEvent(requestID)
	event.CompletedAt = time.Date(2026, time.September, 22, 8, 30, 0, 0, time.UTC)
	event.ClientModel = alias
	event.UpstreamModel = upstream
	event.UpstreamReportedModel = upstream
	event.Attempts[0].GroupID = groupID
	event.Attempts[0].ChannelID = channel.OpenAI
	event.Attempts[0].CredentialID = 33
	event.Attempts[0].UpstreamModel = upstream
	event.Usage.GroupID = groupID
	event.Usage.ChannelID = channel.OpenAI
	event.Usage.CredentialID = 33
	event.Usage.AttemptSequence = 1
	event.Usage.Result = usage.Result{
		State:  usage.StateComplete,
		Tokens: usage.Tokens{UncachedInput: 10, Output: 5},
	}
	event.Usage.Pricing = telemetry.PricingObservation{
		UpstreamModel:        upstream,
		CostState:            string(pricing.CostStatePriced),
		PricingCompleteness:  string(pricing.CompletenessComplete),
		EstimatedCostNanoUSD: 250_000_000,
	}

	db, _ := openRequestLogFileDB(t)
	service := emitRequestLogEvents(t, db, event)
	row := mustProbeRow(t, db, requestID)
	if row.ClientModel != alias || row.UpstreamModel != upstream || row.GroupID != groupID || row.ChannelID != string(channel.OpenAI) ||
		row.CredentialID != 33 || row.AttemptCount != 1 || row.EstimatedCostNanoUSD != 250_000_000 {
		t.Fatalf("durable request log = %#v, want alias/upstream/group/channel/attempt/pricing attribution", row)
	}
	attempts := mustProbeAttempts(t, db, requestID)
	if len(attempts) != 1 || attempts[0].GroupID != groupID || attempts[0].ChannelID != string(channel.OpenAI) ||
		attempts[0].CredentialID != 33 || attempts[0].UpstreamModel != upstream {
		t.Fatalf("durable request-log attempts = %#v, want selected route attribution", attempts)
	}
	windowStart := event.CompletedAt.Truncate(time.Hour)
	report, err := service.QueryUsage(t.Context(), UsageQuery{
		FromMS:      windowStart.UnixMilli(),
		ToMS:        windowStart.Add(time.Hour).UnixMilli(),
		Granularity: UsageGranularityHour,
	})
	if err != nil {
		t.Fatalf("QueryUsage() error = %v", err)
	}
	if report.Summary.RequestCount != 1 || report.Summary.EstimatedCostNanoUSD != 250_000_000 {
		t.Fatalf("usage report = %#v, want one upstream-priced request", report.Summary)
	}
	modelDistribution, ok := report.Distributions.Get(UsageDistributionDimensionModel, UsageDistributionMetricCost)
	if !ok || len(modelDistribution.Items) != 1 || modelDistribution.Items[0].Model != upstream || modelDistribution.Items[0].EstimatedCostNanoUSD != 250_000_000 {
		t.Fatalf("model cost distribution = %#v, want upstream model %q", modelDistribution, upstream)
	}
}
