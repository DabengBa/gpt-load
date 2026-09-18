package dialect

import (
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

func TestInspectCodexSearchRequestNative(t *testing.T) {
	request := &ParsedRequest{
		Method: http.MethodPost,
		Path:   "/v1/alpha/search",
		Body:   []byte(`{"id":"session","model":"gpt-5","commands":{"search_query":[{"q":"test"}]},"service_tier":"priority","reasoning":{"effort":"high"}}`),
	}

	metadata, err := NewOpenAIResponses().InspectRequest(request)
	if err != nil {
		t.Fatalf("InspectRequest() error = %v", err)
	}
	if metadata.Operation != execution.OperationWebSearch {
		t.Fatalf("metadata.Operation = %q, want %q", metadata.Operation, execution.OperationWebSearch)
	}
	if metadata.RouteRequirement != execution.RouteRequirementNative {
		t.Fatalf("metadata.RouteRequirement = %q, want %q", metadata.RouteRequirement, execution.RouteRequirementNative)
	}
	if metadata.ObserveUsage {
		t.Fatal("metadata.ObserveUsage = true, want false")
	}
	if metadata.Stream {
		t.Fatal("metadata.Stream = true, want false")
	}
	if metadata.Model == nil || *metadata.Model != "gpt-5" {
		t.Fatalf("metadata.Model = %v, want gpt-5", metadata.Model)
	}
	if string(metadata.AffinityPrefix) != "codex-search\x00session" {
		t.Fatalf("metadata.AffinityPrefix = %q", string(metadata.AffinityPrefix))
	}
}

func TestInspectCodexSearchRequestRejectsMissingModel(t *testing.T) {
	request := &ParsedRequest{
		Method: http.MethodPost,
		Path:   "/v1/alpha/search",
		Body:   []byte(`{"id":"session"}`),
	}

	if _, err := NewOpenAIResponses().InspectRequest(request); err == nil {
		t.Fatal("InspectRequest() error = nil")
	}
}

func TestInspectCodexSearchRequestRejectsMissingID(t *testing.T) {
	request := &ParsedRequest{
		Method: http.MethodPost,
		Path:   "/v1/alpha/search",
		Body:   []byte(`{"model":"gpt-5"}`),
	}

	if _, err := NewOpenAIResponses().InspectRequest(request); err == nil {
		t.Fatal("InspectRequest() error = nil")
	}
}

func TestInspectCodexSearchRequestRejectsNonStringID(t *testing.T) {
	request := &ParsedRequest{
		Method: http.MethodPost,
		Path:   "/v1/alpha/search",
		Body:   []byte(`{"id":42,"model":"gpt-5"}`),
	}

	if _, err := NewOpenAIResponses().InspectRequest(request); err == nil {
		t.Fatal("InspectRequest() error = nil")
	}
}

func TestInspectCodexSearchRequestRejectsStreaming(t *testing.T) {
	request := &ParsedRequest{
		Method: http.MethodPost,
		Path:   "/v1/alpha/search",
		Body:   []byte(`{"id":"session","model":"gpt-5","stream":true}`),
	}

	if _, err := NewOpenAIResponses().InspectRequest(request); err == nil {
		t.Fatal("InspectRequest() error = nil")
	}
}
