package health

import "math"

// FeedbackStatus is the performance assessment for one provider attempt.
type FeedbackStatus string

const (
	FeedbackStatusUnassessed FeedbackStatus = ""
	FeedbackStatusNormal     FeedbackStatus = "normal"
	FeedbackStatusSlow       FeedbackStatus = "slow"
	FeedbackStatusFaulty     FeedbackStatus = "faulty"
)

func (status FeedbackStatus) Valid() bool {
	return status == FeedbackStatusUnassessed || status == FeedbackStatusNormal ||
		status == FeedbackStatusSlow || status == FeedbackStatusFaulty
}

// Feedback contains the classification and provider-side measurements for an attempt.
type Feedback struct {
	Status          FeedbackStatus
	Reason          string
	FirstResponseMs *int64
	TokensPerSecond *float64
}

// FeedbackObservation contains measurements eligible for provider feedback.
type FeedbackObservation struct {
	Eligible        bool
	ProviderFailed  bool
	FirstResponseMs *int64
	GenerationMs    *int64
	OutputTokens    *int64
}

const (
	feedbackReasonUpstreamFailure   = "upstream_failure"
	feedbackReasonFirstResponseSlow = "first_response_slow"
	feedbackReasonOutputRateFaulty  = "output_rate_faulty"
	feedbackReasonOutputRateSlow    = "output_rate_slow"
	feedbackFirstResponseLimitMs    = int64(30_000)
	feedbackOutputRateFaultyLimit   = float64(10)
	feedbackOutputRateSlowLimit     = float64(20)
)

func ClassifyFeedback(observation FeedbackObservation) Feedback {
	feedback := Feedback{}
	if value := observation.FirstResponseMs; value != nil && *value >= 0 {
		feedback.FirstResponseMs = new(*value)
	}
	if observation.OutputTokens != nil && *observation.OutputTokens > 0 &&
		observation.GenerationMs != nil && *observation.GenerationMs > 0 {
		rate := float64(*observation.OutputTokens) * 1_000 / float64(*observation.GenerationMs)
		if !math.IsNaN(rate) && !math.IsInf(rate, 0) {
			feedback.TokensPerSecond = new(rate)
		}
	}

	if observation.ProviderFailed {
		feedback.Status = FeedbackStatusFaulty
		feedback.Reason = feedbackReasonUpstreamFailure
		return feedback
	}
	if !observation.Eligible || feedback.FirstResponseMs == nil && feedback.TokensPerSecond == nil {
		return feedback
	}
	if feedback.FirstResponseMs != nil && *feedback.FirstResponseMs > feedbackFirstResponseLimitMs {
		feedback.Status = FeedbackStatusFaulty
		feedback.Reason = feedbackReasonFirstResponseSlow
		return feedback
	}
	if feedback.TokensPerSecond != nil && *feedback.TokensPerSecond < feedbackOutputRateFaultyLimit {
		feedback.Status = FeedbackStatusFaulty
		feedback.Reason = feedbackReasonOutputRateFaulty
		return feedback
	}
	if feedback.TokensPerSecond != nil && *feedback.TokensPerSecond < feedbackOutputRateSlowLimit {
		feedback.Status = FeedbackStatusSlow
		feedback.Reason = feedbackReasonOutputRateSlow
		return feedback
	}
	feedback.Status = FeedbackStatusNormal
	return feedback
}
