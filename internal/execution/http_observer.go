package execution

import (
	"context"
	"net/http"
)

// HTTPObserver receives ordered notifications for bytes consumed by one
// selected upstream HTTP attempt. Implementations may run callbacks
// asynchronously, but must not assume they are called inline with transport
// operations.
type HTTPObserver interface {
	ObserveRequest(string, *http.Request)
	ObserveRequestBody(string, []byte)
	ObserveResponse(string, int, http.Header)
	ObserveResponseBody(string, []byte)
	ObserveResponseComplete(string, http.Header, error)
}

// These string keys are shared with the nested CPA module, which cannot import
// this root module because it is a separate Go module.
const (
	HTTPObserverContextKey  = "gpt-load.http-observer"
	HTTPAttemptIDContextKey = "gpt-load.http-attempt-id"
)

// WithHTTPObserver attaches an optional observer to an execution context.
func WithHTTPObserver(ctx context.Context, observer HTTPObserver) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	if observer == nil {
		return ctx
	}
	return context.WithValue(ctx, HTTPObserverContextKey, observer)
}

// HTTPObserverFromContext returns the observer attached to ctx, if any.
func HTTPObserverFromContext(ctx context.Context) HTTPObserver {
	if ctx == nil {
		return nil
	}
	observer, _ := ctx.Value(HTTPObserverContextKey).(HTTPObserver)
	return observer
}

// WithHTTPAttemptID binds the selected attempt identity to downstream HTTP
// execution contexts where the provider request type has no attempt field.
func WithHTTPAttemptID(ctx context.Context, attemptID string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, HTTPAttemptIDContextKey, attemptID)
}

// HTTPAttemptIDFromContext returns the selected upstream attempt identity.
func HTTPAttemptIDFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	attemptID, _ := ctx.Value(HTTPAttemptIDContextKey).(string)
	return attemptID
}
