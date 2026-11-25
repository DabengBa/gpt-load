package control

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	lmfpd "github.com/Ikaleio/lm-detector/sdk/go"
	"github.com/gin-gonic/gin"

	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
)

func TestModelTraceRetryReplacesOnlySelectedChallenge(t *testing.T) {
	initControlI18n(t)
	for _, unusable := range []bool{false, true} {
		name := "failed attempt"
		if unusable {
			name = "unusable sample"
		}
		t.Run(name, func(t *testing.T) {
			useModelTraceConfig(t, &ModelTraceConfig{Enabled: true})
			fixture := newServiceFixture(t)
			groupID := createGroupWithCredentials(t, fixture, "modeltrace-retry-secret")
			original := generateModelTraceChallenges
			t.Cleanup(func() { generateModelTraceChallenges = original })
			generateModelTraceChallenges = func(int) ([]lmfpd.Challenge, error) {
				return []lmfpd.Challenge{{ID: "first", Prompt: "first", ExpectedCount: 300}, {ID: "second", Prompt: "second", ExpectedCount: 300}, {ID: "third", Prompt: "third", ExpectedCount: 300}}, nil
			}
			var attemptStage int
			executor := &modelTraceChallengeExecutor{execute: func(spec execution.AttemptSpec) execution.AttemptResult {
				result := execution.AttemptResult{DispatchState: execution.DispatchMaybeSent, ResponseStarted: true,
					StatusCode: http.StatusOK, Header: http.Header{}, GeneratedText: strings.Repeat("42 7 21 ", 100)}
				if attemptStage < 2 && spec.PromptText == "second" {
					result.GeneratedText = "I did not generate a sequence."
					if !unusable || attemptStage == 1 {
						result.StatusCode = http.StatusTooManyRequests
						result.Error = &execution.ErrorEvidence{Kind: execution.ErrorKindHTTP, StatusCode: result.StatusCode}
						result.GeneratedText = ""
					}
				}
				return result
			}}
			fixture.service.executor = executor
			before, err := fixture.service.RunModelTraceProbe(t.Context(), modelTraceTestRequest(groupID))
			if err != nil || before.Analysis == nil || before.Analysis.UsedOutputs != 2 {
				t.Fatalf("initial run: used outputs should be 2; got %#v, %v", before.Analysis, err)
			}
			encoded, _ := json.Marshal(before)
			var run struct {
				ID         string `json:"run_id"`
				Challenges []struct {
					Retryable bool `json:"retryable"`
				} `json:"challenges"`
			}
			if err := json.Unmarshal(encoded, &run); err != nil || run.ID == "" || !run.Challenges[1].Retryable || run.Challenges[0].Retryable || run.Challenges[2].Retryable {
				t.Fatalf("initial result must offer to retry only the second challenge: %s", encoded)
			}
			router := gin.New()
			const authKey = "modeltrace-retry-test-admin"
			NewServer(&config.Config{AuthKey: authKey}, fixture.service).RegisterRoutes(router)
			post := func(challenge string) *httptest.ResponseRecorder {
				t.Helper()
				body, _ := json.Marshal(map[string]any{"group_id": groupID, "model": before.Model,
					"run_id": run.ID, "challenge_id": challenge})
				recorder := httptest.NewRecorder()
				request := httptest.NewRequestWithContext(t.Context(), http.MethodPost, "/api/modeltrace-probe/retry", bytes.NewReader(body))
				request.Header.Set("Content-Type", "application/json")
				request.Header.Set("Authorization", "Bearer "+authKey)
				router.ServeHTTP(recorder, request)
				return recorder
			}
			if got := post("first"); got.Code != http.StatusBadRequest {
				t.Fatalf("retry of accepted sample: %d %s", got.Code, got.Body)
			}
			attemptStage = 1
			failed := post("second")
			var envelope struct{ Data ModelTraceProbeResponse }
			if err := json.Unmarshal(failed.Body.Bytes(), &envelope); failed.Code != http.StatusOK || err != nil {
				t.Fatalf("failed upstream retry response: %d %s (%v)", failed.Code, failed.Body, err)
			}
			mid := envelope.Data
			if mid.RunID == nil || *mid.RunID != run.ID || !mid.Challenges[1].Retryable || mid.Challenges[1].Outcome != modelTraceChallengeFailed ||
				mid.Analysis == nil || mid.Analysis.UsedOutputs != 2 || len(mid.Analysis.Diagnostics) != 2 {
				t.Fatal("failed retry must replace the selected sample and keep the run retryable")
			}
			attemptStage = 2
			got := post("second")
			if got.Code != http.StatusOK {
				t.Fatalf("retry status: %d %s", got.Code, got.Body)
			}
			envelope.Data = ModelTraceProbeResponse{}
			if err := json.Unmarshal(got.Body.Bytes(), &envelope); err != nil {
				t.Fatal(err)
			}
			after := envelope.Data
			if after.Analysis == nil || after.Analysis.UsedOutputs != 3 {
				t.Fatalf("updated analysis = %#v, want all three samples", after.Analysis)
			}
			if !reflect.DeepEqual(before.Challenges[0], after.Challenges[0]) || !reflect.DeepEqual(before.Challenges[2], after.Challenges[2]) ||
				!reflect.DeepEqual(before.ContextCheck, after.ContextCheck) {
				t.Fatal("retry changed another challenge or the context check")
			}
			if after.Challenges[1].Outcome != modelTraceChallengeSucceeded || reflect.DeepEqual(before.Challenges[1].LogID, after.Challenges[1].LogID) {
				t.Fatal("retry must replace the selected outcome and link its new log")
			}
			if after.RunID != nil || after.Challenges[1].Retryable || post("second").Code != http.StatusBadRequest {
				t.Fatal("a run with all samples accepted must no longer accept retries")
			}
			prompts := executor.recordedPrompts()
			if len(prompts) != modelTraceChallengeCount+3 || prompts[len(prompts)-2] != "second" || prompts[len(prompts)-1] != "second" || len(fixture.probeSink.Events()) != len(prompts) {
				t.Fatalf("want one request using the original prompt and one log per retry; got prompts %q and %d logs", prompts, len(fixture.probeSink.Events()))
			}
		})
	}
}
