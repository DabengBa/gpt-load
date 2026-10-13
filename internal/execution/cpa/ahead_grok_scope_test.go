package cpa

import "testing"

func TestGrokAheadRequestScope(t *testing.T) {
	base := providerRequest{Model: "grok-4.3", ContinuityKey: "tenant", AttemptID: "attempt-a"}
	first := grokRequest(base, "credential-1").ContinuityKey
	base.AttemptID = "attempt-b"
	if got := grokRequest(base, "credential-1").ContinuityKey; first == "" || got != first {
		t.Fatalf("continuity across attempts = %q / %q", first, got)
	}
	if got := grokRequest(base, "credential-2").ContinuityKey; got == first {
		t.Fatal("credential scope reused")
	}
	base.Model = "grok-other"
	if got := grokRequest(base, "credential-1").ContinuityKey; got == first {
		t.Fatal("model scope reused")
	}
	base.ContinuityKey = ""
	a := grokRequest(base, "credential-1").ContinuityKey
	base.AttemptID = "attempt-c"
	b := grokRequest(base, "credential-1").ContinuityKey
	if a == "" || b == "" || a == b {
		t.Fatalf("attempt isolation = %q / %q", a, b)
	}
	t.Logf("continuity=%q attempt-b=%q attempt-c=%q", first, a, b)
}
