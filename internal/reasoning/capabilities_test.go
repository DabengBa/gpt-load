package reasoning

import (
	"errors"
	"testing"
)

func TestValidateEffortRejectsUnknownModelWithoutDowngrade(t *testing.T) {
	err := ValidateEffort("openai", "unlisted-model", "high")
	if !errors.Is(err, ErrUnsupportedEffort) {
		t.Fatalf("ValidateEffort() error = %v, want ErrUnsupportedEffort", err)
	}
}

func TestProviderFallbackEffortIsExact(t *testing.T) {
	cases := []struct {
		provider, model, accept, reject string
	}{
		{"openai", "gpt-5.4", "xhigh", "max"},
		{"gemini", "gemini-3.1-flash-lite-image", "minimal", "low"},
		{"anthropic", "claude-opus-4-7-20260501", "high", "minimal"},
	}
	for _, tc := range cases {
		t.Run(tc.provider+"/"+tc.model, func(t *testing.T) {
			cap := ProjectEffortCapability(tc.provider, tc.model)
			if !cap.Supported || cap.Reason == "" {
				t.Fatalf("projection = %#v, want supported fallback with reason", cap)
			}
			if err := ValidateEffort(tc.provider, tc.model, tc.accept); err != nil {
				t.Fatalf("ValidateEffort(%q) = %v, want nil", tc.accept, err)
			}
			if err := ValidateEffort(tc.provider, tc.model, tc.reject); !errors.Is(err, ErrUnsupportedEffort) {
				t.Fatalf("ValidateEffort(%q) = %v, want unsupported", tc.reject, err)
			}
		})
	}
}

func TestProjectEffortCapabilityIsSecretFree(t *testing.T) {
	projection := ProjectEffortCapability("openai", "unlisted-model")
	if projection.Known || projection.Supported || projection.Reason == "" {
		t.Fatalf("projection = %#v, want unknown unsupported capability with reason", projection)
	}
}
