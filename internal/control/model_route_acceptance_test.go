package control

import (
	"encoding/json"
	"testing"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

func TestModelRouteEntriesAcceptanceInspectionAndAPIRoundTrip(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	a, b, c, full, p2 := 30, 50, 20, 100, 2
	if _, err := fixture.manager.Publish(state.CompileInput{ChannelRegistry: fixture.channelRegistry, Groups: []state.GroupConfig{
		{ID: 1, Name: "openai", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: json.RawMessage(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "A", Alias: "A", Weight: &a}, {ID: "B", Alias: "A", Weight: &b}, {ID: "C", Alias: "A", Weight: &c, Priority: &p2}}},
		{ID: 2, Name: "claude", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: json.RawMessage(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "B", Alias: "A", Weight: &full}}},
		{ID: 3, Name: "gemini", ChannelID: channel.OpenAI, ConnectionType: "api_key", Params: json.RawMessage(`{}`), Enabled: true, Models: []state.ModelConfig{{ID: "D", Alias: "A", Weight: &full}}},
	}, AccessKeys: []state.AccessKeyConfig{{ID: 10, Name: "client", KeyHash: "hash", Status: state.AccessKeyStatusActive}}}); err != nil {
		t.Fatal(err)
	}
	if err := fixture.registry.ReplaceCredentials([]state.CredentialEntry{{ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "k1", AuthState: state.CredentialAuthStateReady, EncryptedValue: "k1"}, {ID: 21, GroupID: 2, Version: 1, IdentityGeneration: 2, Fingerprint: "k2", AuthState: state.CredentialAuthStateReady, EncryptedValue: "k2"}, {ID: 31, GroupID: 3, Version: 1, IdentityGeneration: 3, Fingerprint: "k3", AuthState: state.CredentialAuthStateReady, EncryptedValue: "k3"}}); err != nil {
		t.Fatal(err)
	}
	got, err := fixture.service.InspectRoute(routeInspectRequest{Protocol: "openai-completions", ExternalModel: "A", AccessKeyID: 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Groups) != 5 {
		t.Fatalf("inspection rows=%d, want 5", len(got.Groups))
	}
	// effective_share follows routable entry weights: priority-2 C is a fallback,
	// while all ready credentials keep A, B, B, and D available.
	want := []float64{30.0 / 280.0, 50.0 / 280.0, 100.0 / 280.0, 100.0 / 280.0, 0}
	for i, row := range got.Groups {
		if diff := row.EffectiveShare - want[i]; diff < -1e-9 || diff > 1e-9 {
			t.Fatalf("row %d share=%v want %v", i, row.EffectiveShare, want[i])
		}
	}
	if !got.Groups[4].Fallback {
		t.Fatalf("C row = %#v, want fallback", got.Groups[4])
	}

	roundTripGroupID := createModelRouteTestGroup(t, fixture)
	weightA, weightB, priorityB := 30, 70, 2
	if _, err := fixture.service.UpdateGroupModels(t.Context(), roundTripGroupID, GroupModelsUpdateRequest{Models: optionalGroupModels{Set: true, Values: []GroupModel{{ID: "A", Alias: "A", AliasEnabled: true, Weight: &weightA}, {ID: "B", Alias: "A", AliasEnabled: true, Weight: &weightB, Priority: &priorityB}}}}); err != nil {
		t.Fatal(err)
	}
	models, err := fixture.service.GetGroupModels(t.Context(), roundTripGroupID)
	if err != nil {
		t.Fatal(err)
	}
	if len(models.Items) != 2 || models.Items[0].Weight == nil || *models.Items[0].Weight != 30 || models.Items[1].Weight == nil || *models.Items[1].Weight != 70 || models.Items[1].Priority == nil || *models.Items[1].Priority != 2 {
		t.Fatalf("round trip = %#v", models.Items)
	}
}
