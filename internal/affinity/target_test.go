package affinity

import "testing"

// R1: the remembered affinity identity is the exact route the successful
// provider attempt used, not only the credential that owned it.
func TestTargetCarriesRouteIdentity(t *testing.T) {
	target := Target{
		GroupID: 1, CredentialID: 2, IdentityGeneration: 3,
		EntryID: "e000000000001", UpstreamModelID: "bound-model",
	}
	if !target.Valid() {
		t.Fatal("Valid() = false, want true for a full route target")
	}
	sibling := target
	sibling.UpstreamModelID = "sibling-model"
	if sibling == target {
		t.Fatal("targets of two upstream models compare equal, want distinct route identity")
	}
	otherEntry := target
	otherEntry.EntryID = "e000000000002"
	if otherEntry == target {
		t.Fatal("targets of two entries compare equal, want distinct route identity")
	}
}

// R3: resource operations legitimately address no upstream model, so an empty
// upstream model stays valid while the entry identity remains mandatory.
func TestTargetValidityRequiresRouteIdentity(t *testing.T) {
	tests := []struct {
		name   string
		target Target
		want   bool
	}{
		{
			name:   "model route",
			target: Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3, EntryID: "e1", UpstreamModelID: "m"},
			want:   true,
		},
		{
			name:   "resource route without upstream model",
			target: Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3, EntryID: "e1"},
			want:   true,
		},
		{
			name:   "missing entry",
			target: Target{GroupID: 1, CredentialID: 2, IdentityGeneration: 3, UpstreamModelID: "m"},
			want:   false,
		},
		{
			name:   "missing credential",
			target: Target{GroupID: 1, IdentityGeneration: 3, EntryID: "e1"},
			want:   false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.target.Valid(); got != test.want {
				t.Fatalf("Valid() = %v, want %v for %#v", got, test.want, test.target)
			}
		})
	}
}
