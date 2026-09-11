package control

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"gpt-load/internal/debugcapture"
	"gpt-load/internal/scheduler"
)

func TestAccessKeyMillisWireOmitsLegacyTimestampKeys(t *testing.T) {
	t.Parallel()
	encoded, err := json.Marshal(AccessKeyMetadata{
		ID:          7,
		CreatedAtMS: 1_784_894_400_000,
		UpdatedAtMS: 1_784_898_000_000,
	})
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	text := string(encoded)
	if !strings.Contains(text, `"created_at_ms":1784894400000`) ||
		!strings.Contains(text, `"updated_at_ms":1784898000000`) ||
		strings.Contains(text, `"created_at"`) ||
		strings.Contains(text, `"updated_at"`) {
		t.Fatalf("AccessKeyMetadata JSON = %s", text)
	}
}

func TestRevealMillisWireOmitsLegacyTimestampKey(t *testing.T) {
	t.Parallel()
	assertManagementWireObject(
		t,
		AccessKeyRevealResult{ID: 7, Key: "secret", RevealedAtMS: 1_784_894_400_000},
		[]string{"id", "key", "revealed_at_ms"},
	)
}

func TestHealthMillisWireUsesNullableEpochMilliseconds(t *testing.T) {
	t.Parallel()
	cooldownUntilMS := int64(1_784_894_460_000)
	assertManagementWireObject(
		t,
		runtimeHealthResponse{
			ObservedAtMS: 1_784_894_400_000,
			CooldownCredentials: []healthProblemCredentialResponse{{
				CooldownUntilMS: &cooldownUntilMS,
				Recovery: healthRecoveryResponse{
					Automatic: true,
					Mode:      "cooldown_expiry",
					AtMS:      &cooldownUntilMS,
				},
			}},
			RequestLog: requestLogHealthResponse{
				LastWriteFailureAtMS: &cooldownUntilMS,
			},
		},
		[]string{
			"observed_at_ms",
			"cooldown_credentials",
			"request_log",
		},
	)
}

func TestInspectMillisWireUsesEpochMilliseconds(t *testing.T) {
	t.Parallel()
	cooldownUntilMS := int64(1_784_894_460_000)
	assertManagementWireObject(
		t,
		routeInspectResponse{
			ObservedAtMS: 1_784_894_400_000,
			Groups: []routeInspectGroupResponse{{
				Credentials: []routeInspectCredentialResponse{{
					CooldownUntilMS: &cooldownUntilMS,
				}},
			}},
		},
		[]string{"observed_at_ms", "groups"},
	)
}

func TestHealthAndRouteInspectionUseCredentialWireNames(t *testing.T) {
	t.Parallel()
	healthBody, err := json.Marshal(runtimeHealthResponse{
		Counts:                 healthCountsResponse{Credentials: 1, Available: 1},
		CooldownCredentials:    []healthProblemCredentialResponse{{CredentialID: 7}},
		BlacklistedCredentials: []healthProblemCredentialResponse{},
	})
	if err != nil {
		t.Fatalf("json.Marshal(health) error = %v", err)
	}
	for _, token := range []string{
		`"counts":{"credentials":1,"available":1,"cooldown":0,"blacklisted":0}`,
		`"cooldown_credentials":[{"credential_id":7`,
		`"blacklisted_credentials":[]`,
	} {
		if !strings.Contains(string(healthBody), token) {
			t.Fatalf("health wire missing %s: %s", token, healthBody)
		}
	}
	for _, legacy := range []string{
		`"total"`, `"disabled"`, `"cooldown_keys"`, `"blacklisted_keys"`, `"key_id"`,
	} {
		if strings.Contains(string(healthBody), legacy) {
			t.Fatalf("health wire exposes legacy field %s: %s", legacy, healthBody)
		}
	}

	inspectBody, err := json.Marshal(routeInspectResponse{
		Groups: []routeInspectGroupResponse{{
			Credentials: []routeInspectCredentialResponse{{
				CredentialID: 9,
				ReasonCode:   optionalReason(scheduler.ReasonCredentialCooldown),
			}},
		}},
	})
	if err != nil {
		t.Fatalf("json.Marshal(route inspection) error = %v", err)
	}
	for _, token := range []string{
		`"credentials":[{"credential_id":9`,
		`"reason_code":"credential_cooldown"`,
	} {
		if !strings.Contains(string(inspectBody), token) {
			t.Fatalf("route inspection wire missing %s: %s", token, inspectBody)
		}
	}
	for _, legacy := range []string{`"keys"`, `"key_id"`, `"key_cooldown"`} {
		if strings.Contains(string(inspectBody), legacy) {
			t.Fatalf("route inspection wire exposes legacy field %s: %s", legacy, inspectBody)
		}
	}
}

func TestRequestLogMillisCostWireUsesCursorV2Fields(t *testing.T) {
	t.Parallel()
	assertManagementWireObject(
		t,
		requestLogCursorPayload{
			Version:       requestLogCursorV2,
			CompletedAtMS: 1_784_894_400_000,
			RequestID:     "00000000-0000-4000-8000-000000000001",
		},
		[]string{"v", "completed_at_ms", "request_id"},
	)
	assertManagementWireObject(
		t,
		requestLogItemResponse{
			CompletedAtMS:        1_784_894_400_000,
			EstimatedCostNanoUSD: "125000000",
		},
		[]string{"completed_at_ms", "estimated_cost_nano_usd"},
	)
}

func TestUsageMillisCostWireUsesIntegerBucketsAndStringCost(t *testing.T) {
	t.Parallel()
	assertManagementWireObject(
		t,
		usageResponse{
			FromMS:       1_784_894_400_000,
			ToMS:         1_784_898_000_000,
			ObservedAtMS: 1_784_895_000_000,
			Summary: usageAggregateResponse{
				EstimatedCostNanoUSD: "125000000",
			},
			Series: []usageSeriesResponse{{
				BucketStartMS: 1_784_894_400_000,
				BucketEndMS:   1_784_898_000_000,
			}},
		},
		[]string{"from_ms", "to_ms", "observed_at_ms", "summary", "series"},
	)
}

func TestIdempotencyOperationMillisWireOmitsLegacyTimestampKey(t *testing.T) {
	t.Parallel()
	assertManagementWireObject(
		t,
		operationExpiredData{
			OperationID:   "00000000-0000-4000-8000-000000000001",
			OperationKind: operationKindAccessKeyCreate,
			CompletedAtMS: 1_784_894_400_000,
		},
		[]string{"operation_id", "operation_kind", "resource_identity", "completed_at_ms"},
	)
}

func assertManagementWireObject(t *testing.T, value any, requiredKeys []string) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	for _, key := range requiredKeys {
		if _, exists := document[key]; !exists {
			t.Fatalf("management wire missing %q: %s", key, encoded)
		}
	}
	var decoded any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	assertNoLegacyManagementWireKeys(t, decoded, encoded)
}

func assertNoLegacyManagementWireKeys(t *testing.T, value any, encoded []byte) {
	t.Helper()
	legacyKeys := map[string]struct{}{}
	for _, key := range []string{
		"created_at",
		"updated_at",
		"completed_at",
		"observed_at",
		"cooldown_until",
		"bucket_start",
		"bucket_end",
		"revealed_at",
		"from",
		"to",
		"estimated_cost" + "_usd",
		"last_sweep_at",
		"last_failure_at",
	} {
		legacyKeys[key] = struct{}{}
	}
	var visit func(any)
	visit = func(current any) {
		switch typed := current.(type) {
		case map[string]any:
			for key, child := range typed {
				if _, legacy := legacyKeys[key]; legacy {
					t.Fatalf("management wire exposes legacy key %q: %s", key, encoded)
				}
				visit(child)
			}
		case []any:
			for _, child := range typed {
				visit(child)
			}
		}
	}
	visit(value)
}

func TestDebugCaptureHealthWireHasExactlyTwelveKeys(t *testing.T) {
	t.Parallel()
	expectedKeys := []string{
		"enabled",
		"running",
		"retention_seconds",
		"active",
		"completed",
		"failed",
		"sweep_total",
		"removed_total",
		"sweep_failure_total",
		"error",
		"last_sweep_at_ms",
		"last_failure_at_ms",
	}
	// 1) 零值 debugcapture.Health：功能未启用时经真实映射路径得到的契约对象必须恒定 12 键。
	assertDebugCaptureExactWireKeys(t, mustMapDebugCaptureHealth(t, debugcapture.Health{}), expectedKeys)
	// 2) Error: "counts_unavailable"（store 计数失败降级场景）同样恒定 12 键。
	assertDebugCaptureExactWireKeys(t, mustMapDebugCaptureHealth(t, debugcapture.Health{Error: "counts_unavailable"}), expectedKeys)
	// 3) 直接序列化零值响应对象：service.debugCaptureHealth == nil 时的序列化路径。
	assertDebugCaptureExactWireKeys(t, debugCaptureHealthResponse{}, expectedKeys)
	// 真实 marshal 证据：零值响应对象恰好 12 键且时间位为显式 null。
	if dump, err := json.Marshal(debugCaptureHealthResponse{}); err != nil {
		t.Fatalf("json.Marshal(zero debug_capture) error = %v", err)
	} else {
		t.Logf("debug_capture zero = %s", dump)
	}
	if dump, err := json.Marshal(mustMapDebugCaptureHealth(t, debugcapture.Health{Error: "counts_unavailable"})); err != nil {
		t.Fatalf("json.Marshal(counts_unavailable debug_capture) error = %v", err)
	} else {
		t.Logf("debug_capture counts_unavailable = %s", dump)
	}
}

func mustMapDebugCaptureHealth(t *testing.T, health debugcapture.Health) debugCaptureHealthResponse {
	t.Helper()
	response, err := mapDebugCaptureHealth(health)
	if err != nil {
		t.Fatalf("mapDebugCaptureHealth() error = %v", err)
	}
	return response
}

func assertDebugCaptureExactWireKeys(t *testing.T, value debugCaptureHealthResponse, expectedKeys []string) {
	t.Helper()
	encoded, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("json.Marshal(debug_capture) error = %v", err)
	}
	var document map[string]json.RawMessage
	if err := json.Unmarshal(encoded, &document); err != nil {
		t.Fatalf("json.Unmarshal(debug_capture) error = %v", err)
	}
	if len(document) != len(expectedKeys) {
		t.Fatalf("debug_capture wire has %d keys, want exactly %d: %s", len(document), len(expectedKeys), encoded)
	}
	for _, key := range expectedKeys {
		if _, exists := document[key]; !exists {
			t.Fatalf("debug_capture wire missing required key %q: %s", key, encoded)
		}
	}
	var decoded any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatalf("json.Unmarshal(debug_capture decoded) error = %v", err)
	}
	assertNoLegacyManagementWireKeys(t, decoded, encoded)
}

func TestDebugCaptureHealthWireOmitsLegacyTimestampKeys(t *testing.T) {
	t.Parallel()
	sweepAt := time.Date(2026, 9, 11, 5, 15, 56, 0, time.UTC)
	failureAt := time.Date(2026, 9, 11, 5, 16, 0, 0, time.UTC)
	encoded, err := json.Marshal(mustMapDebugCaptureHealth(t, debugcapture.Health{
		LastSweepAt:   &sweepAt,
		LastFailureAt: &failureAt,
	}))
	if err != nil {
		t.Fatalf("json.Marshal(debug_capture) error = %v", err)
	}
	text := string(encoded)
	if strings.Contains(text, `"last_sweep_at"`) || strings.Contains(text, `"last_failure_at"`) {
		t.Fatalf("debug_capture wire exposes legacy timestamp key: %s", text)
	}
	if !strings.Contains(text, `"last_sweep_at_ms"`) || !strings.Contains(text, `"last_failure_at_ms"`) {
		t.Fatalf("debug_capture wire missing _at_ms timestamp key: %s", text)
	}
}

func TestDebugCaptureHealthWireTimesAreEpochMilliseconds(t *testing.T) {
	t.Parallel()
	sweepAt := time.Date(2026, 9, 11, 5, 15, 56, 0, time.UTC)
	failureAt := time.Date(2026, 9, 11, 5, 16, 0, 0, time.UTC)
	response := mustMapDebugCaptureHealth(t, debugcapture.Health{
		LastSweepAt:   &sweepAt,
		LastFailureAt: &failureAt,
	})
	if response.LastSweepAtMS == nil || response.LastFailureAtMS == nil {
		t.Fatalf("debug_capture wire timestamps must be non-nil epoch milliseconds, got %+v", response)
	}
	wantSweepMS, err := optionalSafeEpochMilliseconds(sweepAt)
	if err != nil {
		t.Fatalf("optionalSafeEpochMilliseconds(sweep) error = %v", err)
	}
	wantFailureMS, err := optionalSafeEpochMilliseconds(failureAt)
	if err != nil {
		t.Fatalf("optionalSafeEpochMilliseconds(failure) error = %v", err)
	}
	if *response.LastSweepAtMS != *wantSweepMS || *response.LastFailureAtMS != *wantFailureMS {
		t.Fatalf("debug_capture wire timestamps = %v/%v, want %v/%v",
			response.LastSweepAtMS, response.LastFailureAtMS, wantSweepMS, wantFailureMS)
	}
	// 序列化后必须是 epoch 毫秒整数，而非 RFC3339 字符串。
	encoded, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("json.Marshal(debug_capture) error = %v", err)
	}
	text := string(encoded)
	if want := fmt.Sprintf(`"last_sweep_at_ms":%d`, *wantSweepMS); !strings.Contains(text, want) {
		t.Fatalf("debug_capture wire last_sweep_at_ms not integer %s: %s", want, text)
	}
	if want := fmt.Sprintf(`"last_failure_at_ms":%d`, *wantFailureMS); !strings.Contains(text, want) {
		t.Fatalf("debug_capture wire last_failure_at_ms not integer %s: %s", want, text)
	}
	if strings.Contains(text, "T") || strings.Contains(text, "Z") {
		t.Fatalf("debug_capture wire exposes RFC3339 timestamp: %s", text)
	}
}

func TestDebugCaptureHealthWireNullTimestampsWhenUnset(t *testing.T) {
	t.Parallel()
	encoded, err := json.Marshal(mustMapDebugCaptureHealth(t, debugcapture.Health{}))
	if err != nil {
		t.Fatalf("json.Marshal(debug_capture) error = %v", err)
	}
	text := string(encoded)
	if !strings.Contains(text, `"last_sweep_at_ms":null`) ||
		!strings.Contains(text, `"last_failure_at_ms":null`) {
		t.Fatalf("debug_capture wire timestamps must be explicit null when unset: %s", text)
	}
	t.Logf("debug_capture unset timestamps = %s", text)
}
