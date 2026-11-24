package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/state"
)

func TestResponsesResourceChecksOwnershipAndPinsCredential(t *testing.T) {
	forwarder := &scriptedForwarder{results: []UpstreamResult{
		storedResponse("resp_first"), storedResponse("resp_first"),
	}}
	handler, engine, _ := newContinuationFixture(t, forwarder)
	useAffinityRandomValues(handler, 0, 1)
	serveContinuation(t, engine, "gl-client", `{"model":"gpt-4o","input":"initial"}`, http.StatusOK)

	other := handler.manager.Current().AccessKeysByHash[handler.encryption.Hash("gl-client")]
	other.ID = 2
	handler.manager.Current().AccessKeysByHash[handler.encryption.Hash("gl-other")] = other
	handler.manager.Current().AccessKeysByID[other.ID] = other

	for _, tc := range []struct{ name, method, key, path string }{
		{"unknown", http.MethodGet, "gl-client", "/v1/responses/resp_unknown"},
		{"other key", http.MethodGet, "gl-other", "/v1/responses/resp_first"},
		{"delete", http.MethodDelete, "gl-other", "/v1/responses/resp_first"},
		{"cancel", http.MethodPost, "gl-other", "/v1/responses/resp_first/cancel"},
		{"input items", http.MethodGet, "gl-other", "/v1/responses/resp_first/input_items"},
		{"extension", http.MethodPatch, "gl-other", "/v1/responses/resp_first/extension"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response := serveResponseResource(engine, tc.method, tc.path, tc.key)
			if response.Code != http.StatusNotFound ||
				!strings.Contains(response.Body.String(), `"code":"response_resource_not_found"`) ||
				len(forwarder.inputs) != 1 {
				t.Fatalf("status/body/attempts = %d/%s/%d, want local resource 404 and only the create attempt",
					response.Code, response.Body.String(), len(forwarder.inputs))
			}
		})
	}
	if t.Failed() {
		return
	}

	const path = "/v1/responses/resp_first"
	const query = "stream=false&include=output"
	response := serveResponseResource(engine, http.MethodGet, path+"?"+query, "gl-client")
	if response.Code != http.StatusOK || response.Body.String() != string(storedResponse("resp_first").Body) || len(forwarder.inputs) != 2 {
		t.Fatalf("status/body/attempts = %d/%s/%d, want the upstream resource and two attempts",
			response.Code, response.Body.String(), len(forwarder.inputs))
	}
	assertAffinityAttemptKeys(t, forwarder.inputs, []string{"sk-one", "sk-one"})
	input := forwarder.inputs[1]
	if input.RouteMode != execution.RouteNative || input.Request.Method != http.MethodGet ||
		input.Request.Path != path || input.Request.RawQuery != query {
		t.Fatalf("resource request did not preserve native route, method, path and query: %+v", input.Request)
	}
}

func TestResponsesResourceDoesNotFallbackFromBoundCredential(t *testing.T) {
	for _, tc := range []struct {
		name   string
		change func(*testing.T, *Handler)
	}{
		{"blacklisted", func(t *testing.T, handler *Handler) {
			if !handler.registry.(*state.CredentialRegistry).SetBlacklisted(1) {
				t.Fatal("blacklisting the bound credential failed")
			}
		}},
		{"access key filter", func(_ *testing.T, handler *Handler) {
			snapshot := handler.manager.Current()
			hash := handler.encryption.Hash("gl-client")
			key := snapshot.AccessKeysByHash[hash]
			key.Filters.Groups = map[uint]struct{}{2: {}}
			snapshot.AccessKeysByHash[hash] = key
			snapshot.AccessKeysByID[key.ID] = key
		}},
		{"replaced identity", func(t *testing.T, handler *Handler) {
			ref, ok := handler.registry.CredentialRef(1)
			if !ok {
				t.Fatal("credential 1 not found")
			}
			if err := handler.registry.(*state.CredentialRegistry).ApplyCredentialImport(1, []state.CredentialEntry{{
				ID: ref.ID, GroupID: ref.GroupID, Version: ref.Version + 1,
				IdentityGeneration: ref.IdentityGeneration + 1, Fingerprint: ref.Fingerprint, EncryptedValue: ref.EncryptedValue,
			}}); err != nil {
				t.Fatal(err)
			}
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			forwarder := &scriptedForwarder{results: []UpstreamResult{storedResponse("resp_first"), storedResponse("resp_first")}}
			handler, engine, _ := newContinuationFixture(t, forwarder)
			useAffinityRandomValues(handler, 0, 1)
			serveContinuation(t, engine, "gl-client", `{"model":"gpt-4o","input":"initial"}`, http.StatusOK)
			tc.change(t, handler)
			response := serveResponseResource(engine, http.MethodGet, "/v1/responses/resp_first", "gl-client")
			if response.Code != http.StatusServiceUnavailable || len(forwarder.inputs) != 1 {
				t.Fatalf("status/body/attempts = %d/%s/%d, want 503 and only the create attempt",
					response.Code, response.Body.String(), len(forwarder.inputs))
			}
		})
	}
}

func serveResponseResource(engine http.Handler, method, path, key string) *httptest.ResponseRecorder {
	request := httptest.NewRequest(method, path, nil)
	request.Header.Set("Authorization", "Bearer "+key)
	response := httptest.NewRecorder()
	engine.ServeHTTP(response, request)
	return response
}

func seedResponseResourceBinding(t *testing.T, handler *Handler, resourceID string) {
	t.Helper()
	ref, ok := handler.registry.CredentialRef(1)
	if !ok || !handler.responseBindings.Record(1, resourceID, ref) {
		t.Fatalf("could not seed ownership for %q", resourceID)
	}
}
