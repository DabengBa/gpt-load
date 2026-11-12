package control

import (
	"testing"

	"gpt-load/internal/execution"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/state"
)

func TestCredentialProbeIgnoresParameterOverridesForRequestAndRecovery(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "credential-override-compatibility")
	credential := takeGroupCredential(t, fixture, groupID)
	if _, ok := fixture.registry.IncrFailure(credential.ID); !ok || !fixture.registry.SetBlacklisted(credential.ID) {
		t.Fatal("failed to blacklist credential")
	}
	setOverride := func(value float64) {
		t.Helper()
		_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
			Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
				state.SettingParameterOverrides: []any{map[string]any{"set": map[string]any{"temperature": value}}},
			}},
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	setOverride(0.3)
	fixture.service.executor = &credentialProbeTestExecutor{execute: func(spec execution.AttemptSpec) execution.AttemptResult {
		if len(spec.ProbePreparedBody) != 0 || len(spec.ConfiguredParameters) != 0 || spec.ReasoningEffortOverridden {
			t.Errorf("credential probe unexpectedly carries model request settings: %#v", spec)
		}
		setOverride(0.7)
		return successfulCredentialProbeResult()
	}}
	response, err := fixture.service.TestGroupCredential(t.Context(), groupID, credential.ID)
	if err != nil || response.Outcome != ProbeOutcomePassed || !response.Recovered {
		t.Fatalf("credential probe after override change = %#v, %v", response, err)
	}
}
