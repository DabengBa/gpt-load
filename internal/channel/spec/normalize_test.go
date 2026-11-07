package spec

import (
	"encoding/json"
	"os"
	"testing"
)

func TestProviderURLSharedCases(t *testing.T) {
	data, err := os.ReadFile("testdata/provider_url_cases.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []struct {
		Input string `json:"input"`
		Valid bool   `json:"valid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Input, func(t *testing.T) {
			_, err := NormalizeProviderURL(tc.Input)
			if (err == nil) != tc.Valid {
				t.Errorf("valid = %v, error = %v, want %v", err == nil, err, tc.Valid)
			}
		})
	}
}

func TestExtractSiteRootURL(t *testing.T) {
	tests := []struct {
		input, want string
		invalid     bool
	}{
		{" https://Example.COM:443/path?q=1#section ", "https://example.com", false},
		{"http://example.com:8080/foo", "http://example.com:8080", false},
		{"https://[::1]:8443/a", "https://[::1]:8443", false},
		{"https://example.com:99999/path", "https://example.com:99999", false},
		{"https://[::1]:999999999999999999999999999999/path", "https://[::1]:999999999999999999999999999999", false},
		{"", "", true}, {"ftp://example.com", "", true}, {"https://u:p@example.com", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got, err := ExtractSiteRootURL(tt.input)
			if (err != nil) != tt.invalid || got != tt.want {
				t.Errorf("ExtractSiteRootURL(%q) = %q, %v; want %q, invalid %v", tt.input, got, err, tt.want, tt.invalid)
			}
		})
	}
}
