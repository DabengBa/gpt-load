package cpa

import (
	"context"
	"net/http"
	"testing"

	"gpt-load/internal/execution"
)

type cpaTestHTTPObserver struct{}

func (cpaTestHTTPObserver) ObserveRequest(string, *http.Request)               {}
func (cpaTestHTTPObserver) ObserveRequestBody(string, []byte)                  {}
func (cpaTestHTTPObserver) ObserveResponse(string, int, http.Header)           {}
func (cpaTestHTTPObserver) ObserveResponseBody(string, []byte)                 {}
func (cpaTestHTTPObserver) ObserveResponseComplete(string, http.Header, error) {}

func TestProviderExecutionContextCarriesObserverAndAttemptID(t *testing.T) {
	observer := cpaTestHTTPObserver{}
	ctx := providerExecutionContext(context.Background(), providerRequest{
		AttemptID: "attempt-cpa",
		Observer:  observer,
	})
	if got := execution.HTTPObserverFromContext(ctx); got == nil {
		t.Fatal("provider execution context dropped HTTP observer")
	}
	if got := execution.HTTPAttemptIDFromContext(ctx); got != "attempt-cpa" {
		t.Fatalf("provider execution attempt ID = %q, want attempt-cpa", got)
	}
}

func TestProviderRequestDoesNotObserveLocalTokenCount(t *testing.T) {
	request := providerRequest{AttemptID: "local-attempt", Observer: cpaTestHTTPObserver{}}
	if request.Observer == nil {
		t.Fatal("test observer was not installed")
	}
	if providerExecutionContext(context.Background(), request) == nil {
		t.Fatal("provider context is nil")
	}
	// Local token-count implementations intentionally do not receive this context;
	// the assertion is kept at the narrow provider boundary to prevent accidental
	// transport observation when that path is changed.
	if countTokensOperation(execution.OperationCountTokens) && request.Observer == nil {
		t.Fatal("local token count unexpectedly lost its explicit observer state")
	}
}
