package control

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"os"
	"path/filepath"
	"strings"
	"testing"

	lmfpd "github.com/Ikaleio/lm-detector/sdk/go"
	app_errors "gpt-load/internal/platform/errors"
)

func TestFingerprintInvalidEnabledEnvironment(t *testing.T) {
	previous := loadModelTraceConfig()
	t.Cleanup(func() { storeModelTraceConfig(previous) })
	storeModelTraceConfig(&ModelTraceConfig{Enabled: true})
	previousDetector := getModelTraceDetector
	t.Cleanup(func() { getModelTraceDetector = previousDetector })
	loads := 0
	getModelTraceDetector = func(ctx context.Context) (*lmfpd.Detector, error) {
		loads++
		return previousDetector(ctx)
	}
	err := loadModelTraceConfigFromEnvironment(func(key string) string {
		if key == "MODELTRACE_ENABLED" {
			return "flase"
		}
		return ""
	})
	if err == nil || err.Error() != "MODELTRACE_ENABLED must be a boolean" {
		t.Fatalf("error = %v", err)
	}
	if cfg := loadModelTraceConfig(); cfg == nil || cfg.Enabled {
		t.Errorf("invalid input should disable detection: %#v", cfg)
	}
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "invalid-enabled-secret")
	executor := &modelTraceChallengeExecutor{}
	fixture.service.executor = executor
	_, err = fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if !errors.Is(err, app_errors.ErrModelTraceDisabled) {
		t.Errorf("error = %v, want disabled", err)
	}
	if loads != 0 || len(executor.recordedPrompts()) != 0 || len(fixture.probeSink.Events()) != 0 {
		t.Errorf("invalid config caused detector loads=%d executions=%d logs=%d", loads, len(executor.recordedPrompts()), len(fixture.probeSink.Events()))
	}
}

func TestFingerprintExplicitDisablePreventsDispatch(t *testing.T) {
	previous := loadModelTraceConfig()
	t.Cleanup(func() { storeModelTraceConfig(previous) })
	previousDetector := getModelTraceDetector
	t.Cleanup(func() { getModelTraceDetector = previousDetector })
	getModelTraceDetector = func(context.Context) (*lmfpd.Detector, error) {
		t.Error("disabled probe initialized detector")
		return nil, errors.New("unexpected load")
	}
	for _, tc := range []struct {
		name    string
		disable func() error
	}{
		{"environment", func() error {
			return loadModelTraceConfigFromEnvironment(func(key string) string {
				if key == "MODELTRACE_ENABLED" {
					return "false"
				}
				return ""
			})
		}},
		{"api nil", func() error { SetProcessModelTraceConfig(nil); return nil }},
		{"api disabled", func() error { SetProcessModelTraceConfig(&ModelTraceConfig{Enabled: false}); return nil }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.disable(); err != nil {
				t.Fatal(err)
			}
			fixture := newServiceFixture(t)
			groupID := createGroupWithCredentials(t, fixture, "disabled-secret")
			executor := &modelTraceChallengeExecutor{}
			fixture.service.executor = executor
			_, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
			if !errors.Is(err, app_errors.ErrModelTraceDisabled) {
				t.Errorf("error = %v, want disabled", err)
			}
			if len(executor.recordedPrompts()) != 0 || len(fixture.probeSink.Events()) != 0 {
				t.Error("disabled probe dispatched or logged a challenge")
			}
		})
	}
}

func TestFingerprintLoadFailurePrecedesDispatch(t *testing.T) {
	previous := getModelTraceDetector
	t.Cleanup(func() { getModelTraceDetector = previous })
	getModelTraceDetector = func(context.Context) (*lmfpd.Detector, error) { return nil, errors.New("unreadable data") }
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "load-failure-secret")
	executor := &modelTraceChallengeExecutor{}
	fixture.service.executor = executor
	_, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if !errors.Is(err, app_errors.ErrModelTraceUnavailable) {
		t.Fatalf("error = %v, want unavailable", err)
	}
	if len(executor.recordedPrompts()) != 0 || len(fixture.probeSink.Events()) != 0 {
		t.Fatal("load failure dispatched or logged a challenge")
	}
}

func TestFingerprintEmbeddedDataLoadAndCleanup(t *testing.T) {
	t.Setenv("TMPDIR", t.TempDir())
	detector, err := loadModelTraceData(t.Context(), modelTraceBank, modelTraceDetectorData)
	if err != nil {
		t.Fatalf("load embedded data: %v", err)
	}
	entries, err := os.ReadDir(os.TempDir())
	if err != nil || len(entries) != 0 {
		t.Fatalf("staging directory remains: %v, %v", entries, err)
	}
	if detector.BuiltAt() == "" {
		t.Fatal("missing bank build time")
	}
	if _, err := detector.Analyze([]lmfpd.Output{{Text: "hello", ExpectedCount: 300}}, false); err != nil {
		t.Fatalf("analysis after cleanup: %v", err)
	}
	if _, err := loadModelTraceData(t.Context(), nil, nil); err == nil {
		t.Fatal("empty data loaded successfully")
	}
	entries, err = os.ReadDir(os.TempDir())
	if err != nil || len(entries) != 0 {
		t.Fatalf("failed load left files: %v, %v", entries, err)
	}
}

func TestFingerprintSDKStatesAndJSON(t *testing.T) {
	detector, err := loadModelTraceData(t.Context(), modelTraceBank, modelTraceDetectorData)
	if err != nil {
		t.Fatal(err)
	}
	numbers := make([]string, 300)
	for i := range numbers {
		n := i/2 + 1
		if i%2 == 1 {
			n = 300 - i/2
		}
		numbers[i] = fmt.Sprint(n)
	}
	modelLike := lmfpd.Output{Text: strings.Join(numbers, " "), ExpectedCount: 300}
	ordered := make([]string, 300)
	for i := range ordered {
		ordered[i] = fmt.Sprint(i + 1)
	}
	tests := []struct {
		name    string
		outputs []lmfpd.Output
	}{
		{"scored", []lmfpd.Output{modelLike, modelLike, modelLike}},
		{"partial", []lmfpd.Output{modelLike}},
		{"unscorable", []lmfpd.Output{{Text: "not numbers", ExpectedCount: 300}}},
		{"not_model", []lmfpd.Output{{Text: strings.Join(ordered, " "), ExpectedCount: 300}}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			analysis, err := detector.Analyze(tc.outputs, false)
			if err != nil {
				t.Fatal(err)
			}
			if analysis.Decision != tc.name {
				t.Fatalf("decision = %q, want %q", analysis.Decision, tc.name)
			}
			if tc.name != "scored" && analysis.Probability != nil {
				t.Fatalf("probability = %v, want nil", analysis.Probability)
			}
			response := ModelTraceProbeResponse{GroupID: 1, GroupName: "example", Model: "example", Challenges: []ModelTraceChallengeResponse{{ID: "example", ExpectedCount: 300, Outcome: modelTraceChallengeSucceeded}}, Analysis: analysis}
			encoded, err := json.Marshal(response)
			if err != nil {
				t.Fatal(err)
			}
			if path := os.Getenv("FINGERPRINT_EXAMPLES_DIR"); path != "" {
				if err := os.WriteFile(filepath.Join(path, tc.name+".json"), encoded, 0o600); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}

// The local detector does not require the retired external service settings.
func TestFingerprintLocalWithoutServiceConfiguration(t *testing.T) {
	previous := loadModelTraceConfig()
	t.Cleanup(func() { storeModelTraceConfig(previous) })
	if err := loadModelTraceConfigFromEnvironment(func(string) string { return "" }); err != nil {
		t.Fatal(err)
	}
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "local-fingerprint-secret")
	executor := &modelTraceChallengeExecutor{}
	fixture.service.executor = executor
	response, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
	if err != nil {
		t.Fatalf("local RunModelTraceProbe() = %v, want success without service configuration", err)
	}
	if len(executor.recordedPrompts()) != 3 || response.Analysis == nil {
		t.Fatalf("local response = %#v, want three challenges and SDK analysis", response)
	}
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]any
	if err := json.Unmarshal(encoded, &fields); err != nil {
		t.Fatal(err)
	}
	if _, ok := fields["service"]; ok {
		t.Fatalf("response contains retired service: %s", encoded)
	}
}
