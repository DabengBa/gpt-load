package control

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMultiplierRequestsRejectRetiredField(t *testing.T) {
	for _, target := range []any{&GroupCreateRequest{}, &GroupSettingsUpdateRequest{}, &AccessKeyCreateRequest{}, &AccessKeyUpdateRequest{}} {
		if err := decodeStrictControlJSONObject([]byte(`{"price_multiplier":"1"}`), target); err == nil || !strings.Contains(err.Error(), "unknown field") {
			t.Fatalf("%T retired field error = %v, want unknown field", target, err)
		}
	}
}

func TestMultiplierResponsesOmitRetiredField(t *testing.T) {
	for _, response := range []any{GroupSettingsResponse{}, GroupSummaryResponse{}, GroupCollectionItem{}, AccessKeyMetadata{}} {
		encoded, err := json.Marshal(response)
		if err != nil {
			t.Fatal(err)
		}
		var fields map[string]json.RawMessage
		if err := json.Unmarshal(encoded, &fields); err != nil {
			t.Fatal(err)
		}
		if _, exists := fields["price_multiplier"]; exists {
			t.Fatalf("%T contains retired field: %s", response, encoded)
		}
	}
}
