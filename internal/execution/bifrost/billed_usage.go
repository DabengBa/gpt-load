package bifrost

import (
	"github.com/maximhq/bifrost/core/schemas"
	"gpt-load/internal/execution"
	"gpt-load/internal/usage"
)

func canceledResponsesUsage(ctx *schemas.BifrostContext, observed *execution.UsageEvidence, response *schemas.BifrostResponsesResponse, bifrostError *schemas.BifrostError) *execution.UsageEvidence {
	invalid := false
	if observed == nil && response != nil {
		var err error
		observed, err = billedUsageEvidence(response.Usage.ToBifrostLLMUsage())
		invalid = err != nil
	}
	if observed == nil {
		observed = errorUsageEvidence(bifrostError, nil)
	}
	observed = canceledReportedUsage(ctx, observed)
	if invalid {
		if observed == nil {
			observed = &execution.UsageEvidence{Normalized: usage.Result{State: usage.StateMissing}}
		}
		observed.Normalized.Diagnostics.Add(usage.DiagnosticInvalidPayload)
	}
	return observed
}

// Completed injected turns are published by their SDK owner before starting
// tools or another turn. Cancellation reads that immutable snapshot immediately.
func canceledReportedUsage(ctx *schemas.BifrostContext, observed *execution.UsageEvidence) *execution.UsageEvidence {
	if snapshot := schemas.ReportedUsageSnapshot(ctx); snapshot != nil {
		observed = errorUsageEvidence(&schemas.BifrostError{ExtraFields: schemas.BifrostErrorExtraFields{BilledUsage: snapshot}}, observed)
	}
	result := cloneUsage(observed)
	if result != nil && result.Normalized.State == usage.StateComplete {
		result.Normalized.State = usage.StatePartial
	}
	return result
}
