package control

import (
	"encoding/json"
	"errors"
	"testing"

	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func createModelRouteTestGroup(t *testing.T, fixture serviceFixture) uint {
	t.Helper()
	created, err := fixture.service.CreateGroup(t.Context(), GroupCreateRequest{
		ChannelID: channel.OpenAICompatible,
		Params:    json.RawMessage(`{"base_url":"https://model-route.example.com/v1"}`),
		Models: optionalGroupModels{
			Set: true,
			Values: []GroupModel{
				{ID: "provider-a", Alias: "public-a", AliasEnabled: true},
				{ID: "provider-b", Alias: "public-b", AliasEnabled: true},
			},
		},
		Credentials: "sk-model-route-a", ConnectionType: "api_key",
	})
	if err != nil {
		t.Fatalf("CreateGroup() error = %v", err)
	}
	return created.GroupID
}

func TestModelRouteEntriesSaveWeightsAndRoundTrip(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	groupID := createModelRouteTestGroup(t, fixture)

	weightA, weightB := 30, 70
	priorityB := 2
	if _, err := fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{
			Set: true,
			Values: []GroupModel{
				{ID: "provider-a", Alias: "public-a", AliasEnabled: true, Weight: &weightA},
				{ID: "provider-b", Alias: "public-b", AliasEnabled: true, Weight: &weightB, Priority: &priorityB},
			},
		},
	}); err != nil {
		t.Fatalf("UpdateGroupModels() error = %v", err)
	}

	got, err := fixture.service.GetGroupModels(t.Context(), groupID)
	if err != nil {
		t.Fatalf("GetGroupModels() error = %v", err)
	}
	if len(got.Items) != 2 {
		t.Fatalf("items = %#v", got.Items)
	}
	if got.Items[0].Weight == nil || *got.Items[0].Weight != 30 ||
		got.Items[1].Weight == nil || *got.Items[1].Weight != 70 ||
		got.Items[1].Priority == nil || *got.Items[1].Priority != 2 ||
		got.Items[0].Priority != nil {
		t.Fatalf("route entries lost weight/priority: %#v", got.Items)
	}

	var stored []groupModelEntry
	var group models.Group
	if err := fixture.db.First(&group, groupID).Error; err != nil {
		t.Fatalf("load group row: %v", err)
	}
	if err := json.Unmarshal(group.Models, &stored); err != nil {
		t.Fatalf("decode stored models: %v", err)
	}
	if len(stored) != 2 || stored[0].Weight == nil || *stored[0].Weight != 30 ||
		stored[1].Weight == nil || *stored[1].Weight != 70 ||
		stored[1].Priority == nil || *stored[1].Priority != 2 {
		t.Fatalf("stored models JSON lost route fields: %s", group.Models)
	}
}

func TestModelRouteEntriesLegacyPayloadWithoutRouteFieldsSaves(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	groupID := createModelRouteTestGroup(t, fixture)

	if _, err := fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{
			Set: true,
			Values: []GroupModel{
				{ID: "provider-c", Alias: "public-c", AliasEnabled: true},
			},
		},
	}); err != nil {
		t.Fatalf("UpdateGroupModels() legacy payload error = %v", err)
	}

	got, err := fixture.service.GetGroupModels(t.Context(), groupID)
	if err != nil {
		t.Fatalf("GetGroupModels() error = %v", err)
	}
	if len(got.Items) != 1 || got.Items[0].Weight != nil || got.Items[0].Priority != nil {
		t.Fatalf("legacy payload must default route fields to nil: %#v", got.Items)
	}
}

func TestModelRouteEntriesRejectsDuplicateAliasUpstreamPair(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	groupID := createModelRouteTestGroup(t, fixture)

	_, err := fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
		Models: optionalGroupModels{
			Set: true,
			Values: []GroupModel{
				{ID: "provider-a", Alias: "public-a", AliasEnabled: true},
				{ID: "provider-a", Alias: "public-a", AliasEnabled: true},
			},
		},
	})
	var apiErr *app_errors.APIError
	if !errors.As(err, &apiErr) || apiErr.Code != app_errors.ErrModelNameConflict.Code {
		t.Fatalf("duplicate pair error = %#v, want ErrModelNameConflict", err)
	}
}

func TestModelRouteEntriesRejectsZeroWeightSum(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	mustEnsureInitialPrices(t, fixture)
	groupID := createModelRouteTestGroup(t, fixture)

	zero := 0
	negative := -3
	for _, values := range [][]GroupModel{
		{
			{ID: "provider-a", Alias: "public-a", AliasEnabled: true, Weight: &zero},
			{ID: "provider-b", Alias: "public-a", AliasEnabled: true, Weight: &zero},
		},
		{
			{ID: "provider-a", Alias: "public-a", AliasEnabled: true, Weight: &negative},
		},
	} {
		if _, err := fixture.service.UpdateGroupModels(t.Context(), groupID, GroupModelsUpdateRequest{
			Models: optionalGroupModels{Set: true, Values: values},
		}); !errors.Is(err, app_errors.ErrValidation) {
			t.Fatalf("values %#v error = %v, want ErrValidation", values, err)
		}
	}
}
