package scheduler

import (
	"errors"
	"math/rand"
	"testing"
	"time"

	"gpt-load/internal/state"
)

func TestIteratorFiltersOnlyUnavailableRouteEntry(t *testing.T) {
	now := inspectNow()
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "entries", ChannelID: "openai",
		Enabled: true,
		Models: []state.ModelConfig{
			{ID: "blocked", Alias: "public", Weight: intPointer(50), Priority: intPointer(1)},
			{ID: "sibling", Alias: "public", Weight: intPointer(50), Priority: intPointer(1)},
		},
	}})
	registry := state.NewCredentialRegistry()
	if err := registry.ReplaceCredentials([]state.CredentialEntry{{
		ID: 11, GroupID: 1, Status: state.CredentialStatusActive,
		Version: 1, IdentityGeneration: 1, Fingerprint: "test", EncryptedValue: "cipher",
	}}); err != nil {
		t.Fatal(err)
	}
	if _, changed := registry.SetEntryCooldownWithChange(
		state.RouteEntryKey{GroupID: 1, UpstreamModelID: "blocked"},
		now.Add(time.Minute),
	); !changed {
		t.Fatal("SetEntryCooldownWithChange() changed = false")
	}

	iterator := newWithClock(snapshot, registry, routeEntryQuery(), rand.New(zeroRandSource{}), func() time.Time {
		return now
	})
	selection, err := iterator.Next()
	if err != nil || selection.UpstreamModelID == nil || *selection.UpstreamModelID != "sibling" {
		t.Fatalf("Next() = %#v, %v, want sibling entry", selection, err)
	}
	if _, err := iterator.Next(); !errors.Is(err, ErrExhausted) {
		t.Fatalf("second Next() error = %v, want ErrExhausted", err)
	}
	if got := iterator.StaticReason(); got != ReasonEntryCooldown {
		t.Fatalf("StaticReason() = %q, want %q", got, ReasonEntryCooldown)
	}
}

func TestInspectReportsEntryRuntimeAndTierReasons(t *testing.T) {
	now := inspectNow()
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "entries", ChannelID: "openai",
		Enabled: true,
		Models: []state.ModelConfig{
			{ID: "primary", Alias: "public", Weight: intPointer(50), Priority: intPointer(1)},
			{ID: "fallback", Alias: "public", Weight: intPointer(50), Priority: intPointer(2)},
		},
	}})
	credentials := []state.CredentialRuntimeView{{
		ID: 11, GroupID: 1, Status: state.CredentialStatusActive, WeightAuto: state.DefaultWeight,
	}}

	inspection, err := InspectWithEntryRuntime(snapshot, credentials, []state.EntryRuntimeView{{
		Key:           state.RouteEntryKey{GroupID: 1, UpstreamModelID: "primary"},
		CooldownUntil: now.Add(time.Minute),
	}}, routeEntryQuery(), now)
	if err != nil {
		t.Fatalf("InspectWithEntryRuntime() error = %v", err)
	}
	if !inspection.Routable {
		t.Fatalf("inspection = %#v, want fallback routable", inspection)
	}
	var primary, fallback GroupInspection
	for _, group := range inspection.Groups {
		if group.UpstreamModelID == nil {
			continue
		}
		switch *group.UpstreamModelID {
		case "primary":
			primary = group
		case "fallback":
			fallback = group
		}
	}
	if primary.Routable || primary.Reason != ReasonEntryCooldown ||
		!primary.EntryCooldownUntil.Equal(now.Add(time.Minute)) {
		t.Fatalf("primary inspection = %#v", primary)
	}
	if !fallback.Routable || fallback.Reason != "" {
		t.Fatalf("fallback inspection = %#v", fallback)
	}

	blacklisted, err := InspectWithEntryRuntime(snapshot, credentials, []state.EntryRuntimeView{{
		Key:         state.RouteEntryKey{GroupID: 1, UpstreamModelID: "primary"},
		Blacklisted: true,
	}}, routeEntryQuery(), now)
	if err != nil {
		t.Fatalf("blacklist inspection error = %v", err)
	}
	for _, group := range blacklisted.Groups {
		if group.UpstreamModelID != nil && *group.UpstreamModelID == "primary" &&
			group.Reason != ReasonEntryBlacklisted {
			t.Fatalf("blacklisted primary inspection = %#v", group)
		}
	}
}

func TestInspectMarksRoutableLowerPriorityEntryDemoted(t *testing.T) {
	now := inspectNow()
	snapshot := routeEntrySnapshot(t, []state.GroupConfig{{
		ConnectionType: "api_key", ID: 1, Name: "entries", ChannelID: "openai",
		Enabled: true,
		Models: []state.ModelConfig{
			{ID: "primary", Alias: "public", Weight: intPointer(50), Priority: intPointer(1)},
			{ID: "fallback", Alias: "public", Weight: intPointer(50), Priority: intPointer(2)},
		},
	}})
	inspection, err := Inspect(snapshot, []state.CredentialRuntimeView{{
		ID: 11, GroupID: 1, Status: state.CredentialStatusActive, WeightAuto: state.DefaultWeight,
	}}, routeEntryQuery(), now)
	if err != nil {
		t.Fatalf("Inspect() error = %v", err)
	}
	for _, group := range inspection.Groups {
		if group.UpstreamModelID == nil {
			continue
		}
		if *group.UpstreamModelID == "fallback" && group.Reason != ReasonTierDemoted {
			t.Fatalf("fallback inspection = %#v, want %q", group, ReasonTierDemoted)
		}
	}
}
