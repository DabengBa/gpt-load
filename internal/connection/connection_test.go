package connection

import "testing"

func TestNormalizeOnlyTrimsExplicitConnectionTypes(t *testing.T) {
	for _, value := range []string{"", "  "} {
		if got := Normalize(value); got != "" {
			t.Fatalf("Normalize(%q) = %q, want empty", value, got)
		}
	}
	if got := Normalize(" api_key "); got != APIKey {
		t.Fatalf("Normalize(api_key) = %q, want %q", got, APIKey)
	}
}

func TestValidAcceptsOnlyProductConnectionTypes(t *testing.T) {
	for _, value := range []string{APIKey} {
		if !Valid(value) {
			t.Fatalf("Valid(%q) = false", value)
		}
	}
	if Valid("") {
		t.Fatal("Valid(empty) = true")
	}
	if Valid("oauth") {
		t.Fatal("Valid(oauth) = true")
	}
}
