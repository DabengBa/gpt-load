package gateway

import (
	"context"
	"errors"
	"net/http"

	"gpt-load/internal/accessquota"
	"gpt-load/internal/dialect"
	"gpt-load/internal/platform/contentcoding"
	platformheader "gpt-load/internal/platform/httpheader"
	"gpt-load/internal/scheduler"
	"gpt-load/internal/state"
)

// admissionInput carries the authenticated request context into admission.
// It is deliberately gin-free so the whole admission decision can be tested
// without HTTP plumbing.
type admissionInput struct {
	request   *http.Request
	route     route
	snapshot  *state.ConfigSnapshot
	accessKey state.AccessKeyView
	recorder  *requestRecorder
}

// admissionRejection is a rejection expressed as a value. The handler maps it
// onto the wire; the reason catalog itself is the frozen HTTP contract.
type admissionRejection struct {
	reason        reason
	retryAfter    *int
	headers       http.Header
	quotaDecision *accessquota.Decision
	// affinity carries the observation the recorder needs when the rejection
	// happens after affinity resolution (matching the original ordering where
	// setAffinityKey/setAffinityObservations ran before the err check).
	affinity *requestAffinity
}

// requestDispatch is the frozen admission outcome consumed by the attempt
// loop: everything needed to forward, nothing about how it was decided.
type requestDispatch struct {
	dialect               dialect.Dialect
	parsed                *dialect.ParsedRequest
	metadata              dialect.RequestMetadata
	model                 string
	streamMode            streamDelivery
	query                 scheduler.Query
	allowedCredentialRefs map[uint]state.CredentialRef
	affinity              requestAffinity
	quotaAdmission        *requestAccessQuotaAdmission
	modelsOnly            bool
}

type admissionOutcome struct {
	dispatch  *requestDispatch
	rejection *admissionRejection
	cancelled bool
	// inspected carries the parsed request metadata once InspectRequest has
	// succeeded, even when the outcome is a rejection or cancellation, so the
	// handler can keep the capture operation observation identical to the
	// original ordering.
	inspected *dialect.RequestMetadata
}

func (outcome admissionOutcome) withInspected(
	metadata *dialect.RequestMetadata,
) admissionOutcome {
	outcome.inspected = metadata
	return outcome
}

func rejectAdmission(value reason) admissionOutcome {
	return admissionOutcome{rejection: &admissionRejection{reason: value}}
}

func rejectAdmissionRetryAfter(value reason, retryAfterSeconds int) admissionOutcome {
	return admissionOutcome{rejection: &admissionRejection{
		reason:     value,
		retryAfter: &retryAfterSeconds,
	}}
}

// admitRequest owns the whole "may this request run, and where" decision:
// quota, rate limit, protocol decode, route resolution, and affinity. It never
// writes to the client; rejections and cancellations are reported as values.
func (handler *Handler) admitRequest(
	ctx context.Context,
	in admissionInput,
) admissionOutcome {
	accessKey := in.accessKey
	snapshot := in.snapshot
	selectedRoute := in.route
	recorder := in.recorder

	quotaAdmission := &requestAccessQuotaAdmission{accessKeyID: accessKey.ID}
	if in.request.URL.Path == "/v1/alpha/search" {
		quotaAdmission = nil
	}
	if quotaAdmission != nil && len(accessKey.CostLimitRules) > 0 {
		quotaAdmission.snapshot = snapshot
	}
	if quotaAdmission != nil && handler.accessQuota != nil {
		quotaDecision := accessquota.Decision{}
		if quotaAdmission.snapshot == nil {
			quotaDecision = handler.accessQuota.Check(accessKey.ID, handler.quotaNow())
		} else {
			var current bool
			quotaDecision, current = handler.checkAccessQuotaForSnapshot(
				quotaAdmission.snapshot,
				accessKey.ID,
				handler.quotaNow(),
			)
			if !current {
				return rejectAdmissionRetryAfter(reasonConfigurationChanged, 1)
			}
		}
		if !quotaDecision.Allowed {
			return admissionOutcome{rejection: &admissionRejection{
				reason:        reasonAccessKeyCostLimitExceeded,
				quotaDecision: &quotaDecision,
			}}
		}
	}
	limitDecision := handler.limiter.Allow(accessKey.ID, accessKey.RPMLimit)
	if !limitDecision.Allowed {
		return rejectAdmissionRetryAfter(
			reasonAccessKeyRateLimited,
			retryAfterSeconds(limitDecision.RetryAfter),
		)
	}
	if selectedRoute.Kind == endpointModels {
		if !contentcoding.IdentityAcceptable(
			headerFieldValues(in.request.Header, "Accept-Encoding"),
		) {
			return rejectAdmission(reasonNotAcceptable)
		}
		return admissionOutcome{dispatch: &requestDispatch{modelsOnly: true}}
	}

	selectedDialect, dialectReady := handler.dialects[selectedRoute.Protocol]
	if !dialectReady || selectedRoute.Kind != endpointForward {
		handler.logDataPlaneRouteNotFound(in.request, accessKey.ID)
		return rejectAdmission(reasonEndpointNotFound)
	}
	encoding, err := contentcoding.ParseContentEncoding(
		headerFieldValues(in.request.Header, "Content-Encoding"),
	)
	if err != nil {
		return admissionOutcome{rejection: &admissionRejection{
			reason: reasonUnsupportedContentEncoding,
			headers: http.Header{
				"Accept-Encoding": {contentcoding.SupportedRequestEncodings},
			},
		}}
	}
	if !contentcoding.IdentityAcceptable(
		headerFieldValues(in.request.Header, "Accept-Encoding"),
	) {
		return rejectAdmission(reasonNotAcceptable)
	}

	body, err := readDecodedRequestBody(
		in.request,
		encoding,
		maxRequestBodyBytes,
		maxRequestBodyBytes,
	)
	if err != nil {
		if ctx.Err() != nil {
			recorder.completeCanceled(ctx, 0, -1)
			return admissionOutcome{cancelled: true}
		}
		switch {
		case errors.Is(err, errRequestTooLarge):
			return rejectAdmission(reasonRequestTooLarge)
		case errors.Is(err, contentcoding.ErrInvalidContentEncoding):
			return rejectAdmission(reasonInvalidContentEncoding)
		default:
			return rejectAdmission(reasonInvalidProtocolRequest)
		}
	}
	requestHeaders := in.request.Header.Clone()
	platformheader.StripRepresentationMetadata(requestHeaders)
	parsed := &dialect.ParsedRequest{
		Method:   in.request.Method,
		Path:     in.request.URL.Path,
		RawQuery: in.request.URL.RawQuery,
		Header:   requestHeaders,
		Body:     body,
	}
	metadata, err := selectedDialect.InspectRequest(parsed)
	if err != nil {
		if ctx.Err() != nil {
			recorder.completeCanceled(ctx, 0, -1)
			return admissionOutcome{cancelled: true}
		}
		return rejectAdmission(reasonInvalidProtocolRequest)
	}
	inspected := &metadata
	model := ""
	if metadata.Model != nil {
		model = *metadata.Model
	}
	if len(model) > maxDataPlaneModelBytes {
		return rejectAdmission(reasonInvalidProtocolRequest).withInspected(inspected)
	}
	if ctx.Err() != nil {
		recorder.completeCanceled(ctx, 0, -1)
		return admissionOutcome{cancelled: true}.withInspected(inspected)
	}
	streamMode, streamRejectReason := evaluateStreamDelivery(
		selectedRoute.Protocol,
		metadata.Operation,
		metadata.Stream,
	)
	if streamMode == streamDeliveryReject {
		return rejectAdmission(*streamRejectReason).withInspected(inspected)
	}
	query := scheduler.Query{
		ClientProtocol:           selectedRoute.Protocol,
		Operation:                metadata.Operation,
		RouteRequirement:         metadata.RouteRequirement,
		ResponsesStorePreference: metadata.ResponsesStorePreference,
		ExternalModel:            metadata.Model,
		AccessKey:                accessKey,
	}
	candidateGroupIDs := scheduler.CandidateGroupIDsForQuery(snapshot, query)
	capturedRefs := handler.registry.CaptureActiveCredentialRefs(candidateGroupIDs)
	allowedCredentialRefs := make(map[uint]state.CredentialRef, len(capturedRefs))
	for _, ref := range capturedRefs {
		allowedCredentialRefs[ref.ID] = ref
	}

	allowedCredentialIDs := make(map[uint]struct{}, len(allowedCredentialRefs))
	for credentialID := range allowedCredentialRefs {
		allowedCredentialIDs[credentialID] = struct{}{}
	}
	query.AllowedCredentialIDs = allowedCredentialIDs
	var requestAff requestAffinity
	if metadata.PreviousResponseID != "" {
		binding, found := handler.responseBindings.Lookup(accessKey.ID, metadata.PreviousResponseID)
		if !found {
			return rejectAdmission(reasonResponseBindingNotFound).withInspected(inspected)
		}
		owner, exists := handler.registry.CredentialRef(binding.CredentialID)
		if !exists || owner.GroupID != binding.GroupID ||
			owner.IdentityGeneration != binding.IdentityGeneration {
			// 响应归属仍存在，但它记录的凭据身份已被替换：续接不能静默换身份。
			query.AllowedCredentialIDs = map[uint]struct{}{}
		} else {
			query.AllowedCredentialIDs = map[uint]struct{}{binding.CredentialID: {}}
		}
	} else {
		requestAff = handler.resolveRequestAffinity(
			ctx,
			snapshot, accessKey.ID, selectedRoute.Protocol, model, metadata.Operation,
			metadata, allowedCredentialRefs,
		)
		query.PreferredCredentialID = requestAff.preferredCredentialID
	}
	if requestAff.err != nil {
		outcome := rejectAdmissionRetryAfter(reasonConfigurationChanged, 1).withInspected(inspected)
		outcome.rejection.affinity = &requestAff
		return outcome
	}
	return admissionOutcome{inspected: inspected, dispatch: &requestDispatch{
		dialect:               selectedDialect,
		parsed:                parsed,
		metadata:              metadata,
		model:                 model,
		streamMode:            streamMode,
		query:                 query,
		allowedCredentialRefs: allowedCredentialRefs,
		affinity:              requestAff,
		quotaAdmission:        quotaAdmission,
	}}
}
