package control

import (
	"testing"
	"time"

	"gpt-load/internal/execution"
	"gpt-load/internal/state"
)

func TestModelProbeDoesNotRecoverAfterReasoningPolicyChange(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createProbeGroup(t, fixture, []string{probeTestModel})
	group := fixture.manager.Current().Groups[groupID]
	key := state.RouteEntryKey{GroupID: groupID, EntryID: probeModelEntryID(group, probeTestModel)}
	if _, changed := fixture.registry.SetEntryBlacklistedWithChange(key); !changed {
		t.Fatal("failed to seed route blacklist")
	}
	started, release := make(chan struct{}), make(chan struct{})
	fixture.service.executor = &credentialProbeTestExecutor{execute: func(execution.AttemptSpec) execution.AttemptResult {
		close(started)
		<-release
		return successfulCredentialProbeResult()
	}}
	resultCh := make(chan ModelProbeResponse, 1)
	go func() {
		response, err := fixture.service.ProbeGroupModels(t.Context(), ModelProbeRequest{Targets: []ModelProbeTargetRequest{{GroupID: groupID, Model: probeTestModel}}})
		if err != nil {
			t.Errorf("ProbeGroupModels: %v", err)
		}
		resultCh <- response
	}()
	<-started
	models, err := fixture.service.GetGroupModels(t.Context(), groupID)
	if err != nil || len(models.Items) != 1 {
		close(release)
		t.Fatalf("GetGroupModels = %#v, %v", models, err)
	}
	revision := fixture.manager.Current().Revision
	_, err = fixture.service.UpdateModelRouteSchedule(t.Context(), modelRouteSchedulePatchRequest{
		SnapshotRevision: &revision,
		Updates: []modelRouteSchedulePatchUpdate{{GroupID: groupID, EntryID: models.Items[0].EntryID,
			ReasoningEffort: optionalField[string]{Set: true, Value: "high"}}},
	})
	close(release)
	response := <-resultCh
	if err != nil {
		t.Fatalf("UpdateModelRouteSchedule: %v", err)
	}
	if len(response.Results) != 1 || response.Results[0].Recovered {
		t.Fatalf("probe after reasoning change = %#v, want no recovery", response)
	}
	runtime, ok := fixture.registry.EntryRuntime(key, time.Now())
	if !ok || !runtime.Blacklisted {
		t.Fatalf("route runtime = %#v/%t, want blacklisted", runtime, ok)
	}
}
