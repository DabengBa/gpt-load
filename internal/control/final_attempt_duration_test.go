package control

import (
	"encoding/json/v2"
	"strings"
	"testing"

	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/telemetry"
	"gpt-load/internal/usage"
)

func TestFinalAttemptDurationListAndDetailWire(t *testing.T) {
	t.Parallel()
	for _, duration := range []*int64{nil, new(int64(0)), new(int64(17))} {
		record := requestlog.Record{
			Protocol:               protocol.OpenAIResponses,
			ModelConsistency:       telemetry.ModelConsistencyNotApplicable,
			Status:                 telemetry.RequestStatusSuccess,
			UsageState:             usage.StateNotApplicable,
			CostState:              pricing.CostStateNotApplicable,
			PricingCompleteness:    pricing.CompletenessNotApplicable,
			DurationMs:             100,
			FinalAttemptDurationMs: duration,
		}
		list, err := mapRequestLogListResponse(requestlog.Page{Items: []requestlog.Record{record}}, nil)
		if err != nil {
			t.Fatal(err)
		}
		detail, err := mapRequestLogDetailResponse(record, nil)
		if err != nil {
			t.Fatal(err)
		}
		want, err := json.Marshal(duration)
		if err != nil {
			t.Fatal(err)
		}
		for _, response := range []any{list, detail} {
			raw, err := json.Marshal(response)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(string(raw), `"final_attempt_duration_ms":`+string(want)) || !strings.Contains(string(raw), `"duration_ms":100`) {
				t.Fatalf("timing wire = %s", raw)
			}
		}
	}
}
