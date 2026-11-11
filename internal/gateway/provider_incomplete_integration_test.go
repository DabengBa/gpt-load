package gateway

import (
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/usage"
)

const outputLimitTerminal = "event: response.incomplete\ndata: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_limit\",\"object\":\"response\",\"status\":\"incomplete\",\"model\":\"gpt-4o\",\"output\":[],\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":4,\"output_tokens\":2,\"total_tokens\":6}}}\n\n"

func TestBufferedResponsesOutputLimitExhaustionPreservesUsage(t *testing.T) {
	const terminal = outputLimitTerminal
	var calls atomic.Int32
	upstream := newBufferedResponsesFramingUpstream(t, func() string {
		calls.Add(1)
		return terminal
	})

	forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
	engine, _ := newBufferedResponsesFramingRuntime(t, forwarder, 2,
		dialectGatewayGroup{id: 1, name: "first", upstreamURL: upstream.URL, apiKeys: []string{"sk-first"}},
		dialectGatewayGroup{id: 2, name: "second", upstreamURL: upstream.URL, apiKeys: []string{"sk-second"}},
	)
	response := performBufferedResponsesFramingRequest(t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`)
	if calls.Load() != 2 || len(forwarder.results) != 2 {
		t.Errorf("upstream calls=%d attempts=%d, want two attempts", calls.Load(), len(forwarder.results))
	}
	if got := response.Body.String(); !strings.Contains(got, "buffered_stream_failed") || strings.Contains(got, "response.incomplete") {
		t.Errorf("downstream=%q, want existing buffered failure", got)
	}
	for index, result := range forwarder.results {
		if result.Usage.Tokens != (usage.Tokens{UncachedInput: 4, Output: 2}) {
			t.Errorf("attempt %d usage=%+v, want input=4 output=2", index+1, result.Usage)
		}
		if result.PayloadReleased || result.Stream.EndReason != StreamEndProviderIncomplete {
			t.Errorf("attempt %d released=%t end=%v, want unreleased provider incomplete", index+1, result.PayloadReleased, result.Stream.EndReason)
		}
	}
}

func TestBufferedResponsesOutputLimitRetriesAndSettlesUsage(t *testing.T) {
	var calls atomic.Int32
	provider := &mutableGatewayPriceTableProvider{table: mustGatewayPriceTable(t, 2_000_000_000, true)}
	success := strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(strings.ReplaceAll(outputLimitTerminal, "response.incomplete", "response.completed"), `"status":"incomplete"`, `"status":"completed"`), "resp_limit", "resp_success"), `"incomplete_details":{"reason":"max_output_tokens"},`, "")
	upstream := newBufferedResponsesFramingUpstream(t, func() string {
		if calls.Add(1) == 1 {
			provider.Publish(mustGatewayPriceTable(t, 3_000_000_000, true))
			return outputLimitTerminal
		}
		return success
	})
	forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
	engine, handler := newBufferedResponsesFramingRuntime(t, forwarder, 2,
		dialectGatewayGroup{id: 1, name: "first", upstreamURL: upstream.URL, apiKeys: []string{"sk-first"}},
		dialectGatewayGroup{id: 2, name: "second", upstreamURL: upstream.URL, apiKeys: []string{"sk-second"}},
	)
	sink := &recordingRequestLogSink{}
	handler.requestLogSink = sink
	handler.priceTables = provider
	quota := accessquota.NewRuntime()
	if err := quota.Reconcile(map[uint][]accessquota.Rule{1: {{ID: 103, Revision: 1, Kind: accessquota.KindTotal, LimitNanoUSD: 10_000_000_000}}}); err != nil {
		t.Fatal(err)
	}
	handler.accessQuota = quota
	response := performBufferedResponsesFramingRequest(t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`)
	if calls.Load() != 2 || len(forwarder.results) != 2 {
		t.Fatalf("calls=%d attempts=%d", calls.Load(), len(forwarder.results))
	}
	if got := response.Body.String(); got != bufferedStreamHeartbeat+success {
		t.Errorf("downstream=%q", got)
	}
	if forwarder.results[0].PayloadReleased || !forwarder.results[1].PayloadReleased {
		t.Errorf("release boundary=%+v", forwarder.results)
	}
	events := sink.snapshot()
	if len(events) != 1 || len(events[0].Attempts) != 2 {
		t.Fatalf("events=%+v", events)
	}
	if events[0].Attempts[0].GroupID == events[0].Attempts[1].GroupID || events[0].Attempts[0].CredentialID == events[0].Attempts[1].CredentialID {
		t.Errorf("retry did not advance candidate: %+v", events[0].Attempts)
	}
	for i, want := range []int64{12_000, 18_000} {
		if got := events[0].Attempts[i].Pricing.EstimatedCostNanoUSD; got != want {
			t.Errorf("attempt %d cost=%d want=%d", i+1, got, want)
		}
		if got := forwarder.results[i].Usage.Tokens; got != (usage.Tokens{UncachedInput: 4, Output: 2}) {
			t.Errorf("attempt %d usage=%+v", i+1, got)
		}
	}
	view := quota.Snapshot(1, time.Now())
	if len(view.Rules) != 1 || view.Rules[0].UsedNanoUSD != 30_000 {
		t.Errorf("quota=%+v, want 30000", view)
	}
}

func TestBufferedResponsesUnconfirmedOutputLimitStillRetries(t *testing.T) {
	for _, test := range []struct {
		name     string
		terminal string
	}{
		{"unknown reason", strings.ReplaceAll(outputLimitTerminal, "max_output_tokens", "unknown")},
		{"missing reason", strings.ReplaceAll(outputLimitTerminal, `"reason":"max_output_tokens"`, `"other":"max_output_tokens"`)},
		{"missing status", strings.ReplaceAll(outputLimitTerminal, `"status":"incomplete",`, "")},
		{"unfinished event", strings.TrimSuffix(outputLimitTerminal, "\n\n")},
	} {
		t.Run(test.name, func(t *testing.T) {
			var calls atomic.Int32
			upstream := newBufferedResponsesFramingUpstream(t, func() string {
				calls.Add(1)
				return test.terminal
			})
			forwarder := &capturingExecutionForwarder{delegate: newConcreteTestExecutionForwarder(t)}
			engine, _ := newBufferedResponsesFramingRuntime(t, forwarder, 2,
				dialectGatewayGroup{id: 1, name: "first", upstreamURL: upstream.URL, apiKeys: []string{"sk-first"}},
				dialectGatewayGroup{id: 2, name: "second", upstreamURL: upstream.URL, apiKeys: []string{"sk-second"}},
			)
			response := performBufferedResponsesFramingRequest(t, engine, `{"model":"gpt-4o","input":"hello","stream":true,"store":false}`)
			if calls.Load() != 2 || len(forwarder.results) != 2 || !strings.Contains(response.Body.String(), "buffered_stream_failed") {
				t.Fatalf("calls=%d attempts=%d downstream=%q", calls.Load(), len(forwarder.results), response.Body.String())
			}
			for _, result := range forwarder.results {
				if result.PayloadReleased {
					t.Fatalf("unconfirmed terminal released: %+v", result.Stream)
				}
			}
		})
	}
}
