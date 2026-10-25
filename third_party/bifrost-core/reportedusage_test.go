package bifrost

import (
	"context"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

func TestInjectedResponsesReportedUsageBeforeToolAndNextTurn(t *testing.T) {
	ctx := schemas.NewBifrostContext(t.Context(), schemas.NoDeadline)
	defer ctx.Cancel()
	set := testInjectedSet(t)
	var calls int
	dispatch := func(*schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
		calls++
		if calls == 1 {
			return &schemas.BifrostResponsesResponse{Output: []schemas.ResponsesMessage{responsesFunctionCall("c1", "web_search")}, Usage: &schemas.ResponsesResponseUsage{InputTokens: 10, OutputTokens: 1, TotalTokens: 11}}, nil
		}
		if got := schemas.ReportedUsageSnapshot(ctx); got == nil || got.PromptTokens != 10 {
			t.Errorf("next turn snapshot=%+v", got)
		}
		return &schemas.BifrostResponsesResponse{Output: []schemas.ResponsesMessage{responsesText("done")}, Usage: &schemas.ResponsesResponseUsage{InputTokens: 20, OutputTokens: 2, TotalTokens: 22}}, nil
	}
	exec := func(call schemas.ChatAssistantMessageToolCall) *schemas.ChatMessage {
		got := schemas.ReportedUsageSnapshot(ctx)
		if got == nil || got.PromptTokens != 10 {
			t.Errorf("tool snapshot=%+v", got)
		}
		return searchResult(call)
	}
	_, err := runInjectedResponsesLoop(&schemas.BifrostResponsesRequest{Model: "m"}, set, dispatch, exec, ctx)
	if err != nil {
		t.Fatal(err)
	}
	for range 2 {
		got := schemas.ReportedUsageSnapshot(ctx)
		if got == nil || got.PromptTokens != 30 || got.CompletionTokens != 3 {
			t.Fatalf("final cumulative snapshot=%+v, want 30/3 exactly once", got)
		}
	}
}

func TestInjectedStreamReportedUsageBeforeTerminalHook(t *testing.T) {
	ctx := schemas.NewBifrostContext(t.Context(), schemas.NoDeadline)
	defer ctx.Cancel()
	cumulative := &schemas.BifrostLLMUsage{PromptTokens: 30, CompletionTokens: 3, TotalTokens: 33}
	response := &schemas.BifrostResponse{ResponsesResponse: &schemas.BifrostResponsesResponse{Usage: cumulative.ToResponsesResponseUsage()}}
	stream, err := chainInjectedStream(ctx, injectedStreamTurns{
		process: func(*schemas.BifrostResponse, bool) ([]clientEmit, bool) {
			return []clientEmit{{result: response, terminal: true}}, true
		},
		usage:     func() *schemas.BifrostLLMUsage { return cumulative },
		continues: func() bool { return false },
		dispatch: func(runner schemas.PostHookRunner, _ func(context.Context)) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
			_, _ = runner(ctx, response, nil)
			done := make(chan *schemas.BifrostStreamChunk)
			close(done)
			return done, nil
		},
	}, func(_ *schemas.BifrostContext, r *schemas.BifrostResponse, e *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
		if got := schemas.ReportedUsageSnapshot(ctx); got == nil || got.PromptTokens != 30 {
			t.Errorf("terminal hook snapshot=%+v", got)
		}
		return r, e
	}, func(context.Context) {})
	if err != nil {
		t.Fatal(err)
	}
	for range stream {
	}
}
