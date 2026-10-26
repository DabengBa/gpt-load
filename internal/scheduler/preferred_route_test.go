package scheduler

import (
	"errors"
	"math/rand"
	"testing"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/state"
)

// preferredRouteQuery binds the preferred credential to one exact route entry
// and upstream model, which is the affinity target identity. The binding is
// published through the same wire contract the gateway uses.
func preferredRouteQuery(credentialID uint, entryID string, upstreamModel string) Query {
	query := preferredHealthQuery(credentialID)
	query.PreferredEntryID = entryID
	query.PreferredUpstreamModel = upstreamModel
	return query
}

// sameCredentialMultiModelSnapshot gives credential 11 two equally weighted
// entries under one public alias. Snapshot order puts the sibling model first,
// so a credential-only preference always selects "sibling-model".
func sameCredentialMultiModelSnapshot(t *testing.T) *state.ConfigSnapshot {
	t.Helper()
	return routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "sibling-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)},
				{ID: "bound-model", Alias: "public", EntryID: "e000000000002", Weight: intPointer(100)},
			}},
	})
}

// R1: the bound (entry, upstream model) is selected first even though the same
// credential's sibling model sorts earlier in the frozen snapshot order.
func TestIteratorPreferredRouteSelectsBoundModelOfSameCredential(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	iterator := New(
		sameCredentialMultiModelSnapshot(t),
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 11 || selection.EntryID != "e000000000002" ||
		selection.UpstreamModelID == nil || *selection.UpstreamModelID != "bound-model" {
		t.Fatalf("first Next() = %#v, want credential 11 entry e000000000002 model bound-model", selection)
	}
}

// R1/R3: with the bound route consumed, the same credential's sibling model is
// still reachable through ordinary scheduling.
func TestIteratorPreferredRouteFallsBackToSiblingModelAfterBoundAttempt(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	iterator := New(
		sameCredentialMultiModelSnapshot(t),
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	first, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if first.EntryID != "e000000000002" {
		t.Fatalf("first Next() entry = %q, want bound entry e000000000002", first.EntryID)
	}
	second, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if second.CredentialID != 11 || second.EntryID != "e000000000001" ||
		second.UpstreamModelID == nil || *second.UpstreamModelID != "sibling-model" {
		t.Fatalf("second Next() = %#v, want sibling model of the same credential", second)
	}
}

// R3: an unavailable bound route is not forced. The request proceeds with the
// same credential's healthy sibling model and reports that route-specific
// health bypass, which is the takeover permission for the exact route.
func TestIteratorPreferredRouteHealthBypassIsRouteScoped(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{
		{ID: 11, GroupID: 1, Version: 1, IdentityGeneration: 1, Fingerprint: "f11", EncryptedValue: "c11"},
	}); err != nil {
		t.Fatalf("ReplaceCredentials() error = %v", err)
	}
	registry.SetEntryCooldown(state.RouteEntryKey{GroupID: 1, EntryID: "e000000000002"}, now.Add(time.Minute))
	iterator := New(
		sameCredentialMultiModelSnapshot(t),
		registry,
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 11 || selection.EntryID != "e000000000001" {
		t.Fatalf("selection = %#v, want sibling route of credential 11", selection)
	}
	if !iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = false, want true for the cooled-down bound route")
	}
}

// R3: a healthy bound route never reports a bypass, even when the same
// credential has other healthy entries.
func TestIteratorPreferredRouteHealthySelectionIsNotBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	iterator := New(
		sameCredentialMultiModelSnapshot(t),
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.EntryID != "e000000000002" {
		t.Fatalf("selection entry = %q, want bound entry", selection.EntryID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true for a healthy bound route")
	}
}

// R3: a removed or renamed bound route cannot be selected. The stale binding
// degrades into an ordinary miss instead of blocking the request forever or
// stealing a different route as if it were the bound one.
func TestIteratorStalePreferredRouteFallsBackWithoutBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	iterator := New(
		sameCredentialMultiModelSnapshot(t),
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000099", "removed-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.CredentialID != 11 || selection.EntryID != "e000000000002" {
		t.Fatalf("selection entry = %q, want ordinary snapshot-ordered route e000000000002", selection.EntryID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true for a removed preferred route, want false")
	}
}

// R3: a consumed bound route is not re-selected; iteration drains every other
// route and then reports exhaustion instead of looping on the stale binding.
// R3: a filtered or disallowed bound route does not buy a health bypass. Static
// exclusions suppress the observation, so filtering alone never migrates the
// binding.
func TestIteratorPreferredRouteStaticExclusionIsNotBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	iterator := New(
		sameCredentialMultiModelSnapshot(t),
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}, {ID: 21, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }
	iterator.allowedCredentialIDs = map[uint]struct{}{21: {}}

	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("Next() error = %v, want ErrExhausted for the disallowed credential", err)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true for a static exclusion, want false")
	}
}

// R3: an all-tried bound route is request-scoped, not a health signal.
// R3: a remembered route that no longer exists in the snapshot is a deletion,
// not a health signal. Removing a route must not be reported as a health
// bypass, or the deleted route would be re-learned as a "migrated" binding.
func TestIteratorPreferredRouteDeletedRouteIsNotHealthBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "sibling-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)},
			}},
	})
	iterator := New(
		snapshot,
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.EntryID != "e000000000001" {
		t.Fatalf("selection entry = %q, want the surviving sibling route", selection.EntryID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true for a deleted route, want false")
	}
}

// R3: a permanently filtered-out bound route (zero weight) is a static
// exclusion as well, not a health signal.
func TestIteratorPreferredRouteZeroWeightRouteIsNotHealthBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "sibling-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)},
				{ID: "bound-model", Alias: "public", EntryID: "e000000000002", Weight: intPointer(0)},
			}},
	})
	iterator := New(
		snapshot,
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.EntryID != "e000000000001" {
		t.Fatalf("selection entry = %q, want the weighted sibling route", selection.EntryID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true for a zero-weight route, want false")
	}
}

func TestIteratorPreferredRouteTriedIsNotHealthBypass(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	iterator := New(
		sameCredentialMultiModelSnapshot(t),
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }
	iterator.tried[candidateKey{credentialID: 11, upstreamModel: "bound-model"}] = struct{}{}

	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.EntryID != "e000000000001" {
		t.Fatalf("selection entry = %q, want sibling route", selection.EntryID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("PreferredHealthBypassed() = true for an already-tried route, want false")
	}
}

func TestIteratorPreferredRouteExclusiveTargetExhausts(t *testing.T) {
	now := time.Date(2026, time.September, 14, 12, 0, 0, 0, time.UTC)
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{
		{ID: 1, Name: "preferred", ChannelID: channel.OpenAI, Params: []byte(`{}`), Enabled: true,
			Models: []state.ModelConfig{
				{ID: "sibling-model", Alias: "public", EntryID: "e000000000001", Weight: intPointer(100)},
				{ID: "bound-model", Alias: "public", EntryID: "e000000000002", Weight: intPointer(100)},
			}},
	})
	iterator := New(
		snapshot,
		fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}},
		preferredRouteQuery(11, "e000000000002", "bound-model"),
		rand.New(zeroRandSource{}),
	)
	iterator.now = func() time.Time { return now }

	seen := make([]string, 0, 2)
	for {
		selection, err := iterator.Next()
		if errors.Is(err, ErrExhausted) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		seen = append(seen, selection.EntryID)
	}
	if len(seen) != 2 || seen[0] != "e000000000002" || seen[1] != "e000000000001" {
		t.Fatalf("visited entries = %#v, want bound route first then sibling", seen)
	}
}

// Changing a model on an explicit entry must not retain the old preference.
func TestIteratorPreferredRouteRejectsModelEdit(t *testing.T) {
	snapshot := sameCredentialMultiModelSnapshot(t)
	query := preferredRouteQuery(11, "e000000000001", "old-model")
	iterator := New(snapshot, fakeCredentialSource{keys: []state.CredentialMeta{{ID: 11, GroupID: 1}}}, query, rand.New(zeroRandSource{}))
	selection, err := iterator.Next()
	if err != nil {
		t.Fatal(err)
	}
	if selection.EntryID != "e000000000002" {
		t.Fatalf("selected entry = %q, want ordinary bound-model entry after stale preference", selection.EntryID)
	}
	if iterator.PreferredHealthBypassed() {
		t.Fatal("model edit is not a health failure")
	}
}
