package codexrecovery

import "testing"

func TestClassifyDoesNotTreatSuccessfulOrMalformedJSONAsRejection(t *testing.T) {
	for _, body := range []string{
		`{"type":"response.completed","code":"invalid_encrypted_content","output":[]}`,
		`{"error":{"message":"encrypted content could not be verified"}`,
		`{"output":"No item found with id abc"}`,
	} {
		if got := Classify(body); got != "" {
			t.Errorf("Classify(%q) = %q, want no rejection", body, got)
		}
	}
}

func TestSanitizeKeepsReferencesCompactionAndToolAssociations(t *testing.T) {
	body := []byte(`{"previous_response_id":"owner","input":[{"type":"item_reference","id":"server"},{"type":"reasoning","encrypted_content":"opaque"}]}`)
	if cleaned, _, changed := Sanitize(body); changed {
		t.Fatalf("server history cannot be rebuilt: %s", cleaned)
	}
}
