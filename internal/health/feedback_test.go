package health

import "testing"

func TestClassifyFeedback(t *testing.T) {
	tests := []struct {
		name          string
		observation   FeedbackObservation
		wantStatus    FeedbackStatus
		wantReason    string
		wantFirstMs   *int64
		wantTokensSec *float64
	}{
		{
			name:        "provider failure precedes performance eligibility",
			observation: FeedbackObservation{ProviderFailed: true},
			wantStatus:  FeedbackStatusFaulty,
			wantReason:  "upstream_failure",
		},
		{
			name: "ineligible performance remains unassessed",
			observation: FeedbackObservation{
				FirstResponseMs: new(int64(31_000)),
				GenerationMs:    new(int64(1_000)),
				OutputTokens:    new(int64(1)),
			},
			wantFirstMs:   new(int64(31_000)),
			wantTokensSec: new(float64(1)),
		},
		{
			name: "first response over thirty seconds is faulty",
			observation: FeedbackObservation{
				Eligible:        true,
				FirstResponseMs: new(int64(30_001)),
			},
			wantStatus:  FeedbackStatusFaulty,
			wantReason:  "first_response_slow",
			wantFirstMs: new(int64(30_001)),
		},
		{
			name: "first response at thirty seconds is normal",
			observation: FeedbackObservation{
				Eligible:        true,
				FirstResponseMs: new(int64(30_000)),
			},
			wantStatus:  FeedbackStatusNormal,
			wantFirstMs: new(int64(30_000)),
		},
		{
			name: "output rate below ten is faulty",
			observation: FeedbackObservation{
				Eligible:     true,
				GenerationMs: new(int64(1_000)),
				OutputTokens: new(int64(9)),
			},
			wantStatus:    FeedbackStatusFaulty,
			wantReason:    "output_rate_faulty",
			wantTokensSec: new(float64(9)),
		},
		{
			name: "output rate at ten is slow",
			observation: FeedbackObservation{
				Eligible:     true,
				GenerationMs: new(int64(100)),
				OutputTokens: new(int64(1)),
			},
			wantStatus:    FeedbackStatusSlow,
			wantReason:    "output_rate_slow",
			wantTokensSec: new(float64(10)),
		},
		{
			name: "output rate below twenty is slow",
			observation: FeedbackObservation{
				Eligible:     true,
				GenerationMs: new(int64(200)),
				OutputTokens: new(int64(3)),
			},
			wantStatus:    FeedbackStatusSlow,
			wantReason:    "output_rate_slow",
			wantTokensSec: new(float64(15)),
		},
		{
			name: "output rate at twenty is normal",
			observation: FeedbackObservation{
				Eligible:     true,
				GenerationMs: new(int64(50)),
				OutputTokens: new(int64(1)),
			},
			wantStatus:    FeedbackStatusNormal,
			wantTokensSec: new(float64(20)),
		},
		{
			name: "first response reason wins over faulty output rate",
			observation: FeedbackObservation{
				Eligible:        true,
				FirstResponseMs: new(int64(30_001)),
				GenerationMs:    new(int64(1_000)),
				OutputTokens:    new(int64(9)),
			},
			wantStatus:    FeedbackStatusFaulty,
			wantReason:    "first_response_slow",
			wantFirstMs:   new(int64(30_001)),
			wantTokensSec: new(float64(9)),
		},
		{
			name: "valid first response alone is assessable",
			observation: FeedbackObservation{
				Eligible:        true,
				FirstResponseMs: new(int64(100)),
			},
			wantStatus:  FeedbackStatusNormal,
			wantFirstMs: new(int64(100)),
		},
		{
			name:        "missing measurements are unassessed",
			observation: FeedbackObservation{Eligible: true},
		},
		{
			name: "negative measurements are ignored",
			observation: FeedbackObservation{
				Eligible:        true,
				FirstResponseMs: new(int64(-1)),
				GenerationMs:    new(int64(1_000)),
				OutputTokens:    new(int64(-1)),
			},
		},
		{
			name: "negative generation duration produces no rate",
			observation: FeedbackObservation{
				Eligible:     true,
				GenerationMs: new(int64(-1)),
				OutputTokens: new(int64(1)),
			},
		},
		{
			name: "zero generation duration produces no rate",
			observation: FeedbackObservation{
				Eligible:     true,
				GenerationMs: new(int64(0)),
				OutputTokens: new(int64(1)),
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := ClassifyFeedback(test.observation)
			if got.Status != test.wantStatus {
				t.Errorf("Status = %q, want %q", got.Status, test.wantStatus)
			}
			if got.Reason != test.wantReason {
				t.Errorf("Reason = %q, want %q", got.Reason, test.wantReason)
			}
			assertInt64PointerEqual(t, got.FirstResponseMs, test.wantFirstMs)
			assertFloat64PointerEqual(t, got.TokensPerSecond, test.wantTokensSec)
		})
	}
}

func TestFeedbackStatusValid(t *testing.T) {
	for _, status := range []FeedbackStatus{"", FeedbackStatusNormal, FeedbackStatusSlow, FeedbackStatusFaulty} {
		if !status.Valid() {
			t.Errorf("%q.Valid() = false, want true", status)
		}
	}
	if FeedbackStatus("unknown").Valid() {
		t.Fatal("unknown status is valid")
	}
}

func assertInt64PointerEqual(t *testing.T, got, want *int64) {
	t.Helper()
	if (got == nil) != (want == nil) {
		t.Errorf("pointer = %v, want %v", got, want)
		return
	}
	if got != nil && *got != *want {
		t.Errorf("value = %d, want %d", *got, *want)
	}
}

func assertFloat64PointerEqual(t *testing.T, got, want *float64) {
	t.Helper()
	if (got == nil) != (want == nil) {
		t.Errorf("pointer = %v, want %v", got, want)
		return
	}
	if got != nil && *got != *want {
		t.Errorf("value = %v, want %v", *got, *want)
	}
}
