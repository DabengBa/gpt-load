package dialect

import (
	"net/http"
	"strings"
	"testing"

	"gpt-load/internal/pricing"
	"gpt-load/internal/protocol"
	"gpt-load/internal/usage"
)

func TestOpenAIResponsesProtocolAndRequestMetadata(t *testing.T) {
	selected := NewOpenAIResponses()
	if got := selected.Protocol(); got != protocol.OpenAIResponses {
		t.Fatalf("Protocol() = %q, want %q", got, protocol.OpenAIResponses)
	}

	tests := []struct {
		name        string
		request     *ParsedRequest
		wantModel   string
		wantStream  bool
		wantObserve bool
		wantErr     bool
	}{
		{name: "empty create", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses"}, wantObserve: true},
		{name: "create stream", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses", Body: []byte(`{"model":"gpt-5","stream":true}`)}, wantModel: "gpt-5", wantStream: true, wantObserve: true},
		{name: "compact", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses/compact", Body: []byte(`{"model":"gpt-5"}`)}, wantModel: "gpt-5", wantObserve: true},
		{name: "input tokens", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses/input_tokens", Body: []byte(`{"model":"gpt-5"}`)}, wantModel: "gpt-5"},
		{name: "retrieve stream", request: &ParsedRequest{Method: http.MethodGet, Path: "/v1/responses/resp_123", RawQuery: "stream=true"}, wantStream: true},
		{name: "encoded stream", request: &ParsedRequest{Method: http.MethodGet, Path: "/v1/responses/resp_123", RawQuery: "%73tream=%74rue"}, wantStream: true},
		{name: "nil", wantErr: true},
		{name: "non object", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses", Body: []byte(`[]`)}, wantErr: true},
		{name: "blank model", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses", Body: []byte(`{"model":""}`)}, wantErr: true},
		{name: "duplicate stream query", request: &ParsedRequest{Method: http.MethodGet, Path: "/v1/responses/resp_123", RawQuery: "stream=true&stream=false"}, wantErr: true},
		{name: "invalid stream query", request: &ParsedRequest{Method: http.MethodGet, Path: "/v1/responses/resp_123", RawQuery: "stream=1"}, wantErr: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metadata, err := selected.InspectRequest(test.request)
			if test.wantErr {
				if err == nil {
					t.Fatalf("InspectRequest() = %#v, nil", metadata)
				}
				return
			}
			if err != nil || metadata.Stream != test.wantStream || metadata.ObserveUsage != test.wantObserve {
				t.Fatalf("InspectRequest() = %#v, %v", metadata, err)
			}
			if test.wantModel == "" {
				if metadata.Model != nil {
					t.Fatalf("model = %v, want nil", metadata.Model)
				}
			} else if metadata.Model == nil || *metadata.Model != test.wantModel {
				t.Fatalf("model = %v, want %q", metadata.Model, test.wantModel)
			}
		})
	}
}

func TestOpenAIResponsesExtractsPromptCacheKeyForCreateOnly(t *testing.T) {
	responsesCreateBody := func(fields string) *ParsedRequest {
		return &ParsedRequest{
			Method: http.MethodPost, Path: "/v1/responses",
			Body: []byte(`{"model":"gpt-4o",` + fields + `}`),
		}
	}
	exactLimit := strings.Repeat("b", maxPromptCacheKeyBytes)
	oversized := strings.Repeat("a", maxPromptCacheKeyBytes+1)
	tests := []struct {
		name    string
		request *ParsedRequest
		wantKey string
		wantErr bool
	}{
		{name: "valid key", request: responsesCreateBody(`"prompt_cache_key":"Session-42"`), wantKey: "Session-42"},
		{name: "preserves case unicode and inner whitespace", request: responsesCreateBody(`"prompt_cache_key":"Key 内 部-A"`), wantKey: "Key 内 部-A"},
		{name: "exact byte limit", request: responsesCreateBody(`"prompt_cache_key":"` + exactLimit + `"`), wantKey: exactLimit},
		{name: "null", request: responsesCreateBody(`"prompt_cache_key":null`)},
		{name: "number", request: responsesCreateBody(`"prompt_cache_key":7`)},
		{name: "bool", request: responsesCreateBody(`"prompt_cache_key":true`)},
		{name: "object", request: responsesCreateBody(`"prompt_cache_key":{"key":"x"}`)},
		{name: "array", request: responsesCreateBody(`"prompt_cache_key":["x"]`)},
		{name: "empty", request: responsesCreateBody(`"prompt_cache_key":""`)},
		{name: "leading whitespace", request: responsesCreateBody(`"prompt_cache_key":" leading"`)},
		{name: "trailing whitespace", request: responsesCreateBody(`"prompt_cache_key":"trailing "`)},
		{name: "control character", request: responsesCreateBody(`"prompt_cache_key":"bad\u0000key"`)},
		{name: "oversized", request: responsesCreateBody(`"prompt_cache_key":"` + oversized + `"`)},
		{name: "nested only", request: responsesCreateBody(`"metadata":{"prompt_cache_key":"nested"}`)},
		{name: "case variant", request: responsesCreateBody(`"Prompt_Cache_Key":"other"`)},
		{name: "duplicate", request: responsesCreateBody(`"prompt_cache_key":"a","prompt_cache_key":"b"`), wantErr: true},
		{name: "compact ignores", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses/compact", Body: []byte(`{"model":"gpt-5","prompt_cache_key":"compact"}`)}},
		{name: "input tokens ignores", request: &ParsedRequest{Method: http.MethodPost, Path: "/v1/responses/input_tokens", Body: []byte(`{"model":"gpt-5","prompt_cache_key":"tokens"}`)}},
		{name: "retrieve ignores", request: &ParsedRequest{Method: http.MethodGet, Path: "/v1/responses/resp_1", Body: []byte(`{"model":"gpt-5","prompt_cache_key":"retrieve"}`)}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			metadata, err := NewOpenAIResponses().InspectRequest(test.request)
			if test.wantErr {
				if err == nil {
					t.Fatalf("InspectRequest() = %#v, nil", metadata)
				}
				return
			}
			if err != nil || metadata.PromptCacheKey != test.wantKey {
				t.Fatalf("PromptCacheKey = %q, %v; want %q", metadata.PromptCacheKey, err, test.wantKey)
			}
		})
	}
}

func TestOpenAIResponsesRequestSelectsSupportedPricingModes(t *testing.T) {
	selected := NewOpenAIResponses()
	for _, test := range []struct {
		body        string
		mode        pricing.Mode
		unsupported bool
	}{
		{body: `{"model":"gpt-5","service_tier":"priority"}`, mode: pricing.ModeFast},
		{body: `{"model":"gpt-5","service_tier":"fast"}`, mode: pricing.ModeFast},
		{body: `{"model":"gpt-5","service_tier":"default"}`, mode: pricing.ModeStandard},
		{body: `{"model":"gpt-5","speed":"fast"}`, unsupported: true},
		{body: `{"model":"gpt-5","reasoning":{"mode":"pro"}}`, unsupported: true},
	} {
		metadata, err := selected.InspectRequest(&ParsedRequest{Method: http.MethodPost, Path: "/v1/responses", Body: []byte(test.body)})
		if err != nil || metadata.PricingMode != test.mode ||
			metadata.UsageDiagnostics.Has(usage.DiagnosticUnsupportedBillableDetail) != test.unsupported {
			t.Fatalf("InspectRequest(%s) = %#v, %v", test.body, metadata, err)
		}
	}
}

func TestOpenAIResponsesContinuationSkipsPromptAffinityAndKeepsOpaqueIDs(t *testing.T) {
	for _, test := range []struct {
		field string
		id    string
		err   bool
	}{
		{field: `"previous_response_id":" custom/id "`, id: " custom/id "},
		{field: `"previous_response_id":null`},
		{field: `"previous_response_id":""`},
		{field: `"PREVIOUS_RESPONSE_ID":"unrelated"`},
		{field: `"previous_response_id":123`, err: true},
		{field: `"previous_response_id":false`, err: true},
		{field: `"previous_response_id":[]`, err: true},
		{field: `"previous_response_id":{}`, err: true},
		{field: `"previous_response_id":"first","previous_response_id":"second"`, err: true},
	} {
		t.Run(test.field, func(t *testing.T) {
			request := &ParsedRequest{
				Method: http.MethodPost, Path: "/v1/responses",
				Body: []byte(`{"model":"gpt-4o","input":"continue",` + test.field + `}`),
			}
			metadata, err := NewOpenAIResponses().InspectRequest(request)
			if (err != nil) != test.err {
				t.Fatalf("error = %v, want error %t", err, test.err)
			}
			if err == nil && (metadata.PreviousResponseID != test.id ||
				(len(metadata.AffinityPrefix) == 0) != (test.id != "")) {
				t.Fatalf("metadata = %#v, want ID %q and mutually exclusive affinity", metadata, test.id)
			}
		})
	}
}
