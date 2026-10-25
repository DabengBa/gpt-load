package schemas

import (
	"sync"
	"testing"
)

func TestReportedUsageSnapshotOwnedReplacement(t *testing.T) {
	ctx := NewBifrostContext(t.Context(), NoDeadline)
	defer ctx.Cancel()
	if ReportedUsageSnapshot(ctx) != nil {
		t.Fatal("invented usage")
	}
	source := &BifrostLLMUsage{PromptTokens: 10, PromptTokensDetails: &ChatPromptTokensDetails{CachedWriteTokenDetails: &ChatCachedWriteTokenDetails{CachedWriteTokens5m: 3}}}
	SetReportedUsageSnapshot(ctx, source)
	source.PromptTokens = 99
	source.PromptTokensDetails.CachedWriteTokenDetails.CachedWriteTokens5m = 99
	got := ReportedUsageSnapshot(ctx)
	if got.PromptTokens != 10 || got.PromptTokensDetails.CachedWriteTokenDetails.CachedWriteTokens5m != 3 {
		t.Fatalf("snapshot aliases source: %+v", got)
	}
	got.PromptTokens = 88
	if ReportedUsageSnapshot(ctx).PromptTokens != 10 {
		t.Fatal("snapshot getter aliases stored value")
	}
	SetReportedUsageSnapshot(ctx, &BifrostLLMUsage{PromptTokens: 20})
	if ReportedUsageSnapshot(ctx).PromptTokens != 20 {
		t.Fatal("cumulative snapshot must replace, not add")
	}
	var wg sync.WaitGroup
	for range 4 {
		wg.Go(func() {
			for i := range 100 {
				SetReportedUsageSnapshot(ctx, &BifrostLLMUsage{PromptTokens: i})
				_ = ReportedUsageSnapshot(ctx)
			}
		})
	}
	wg.Wait()
}
