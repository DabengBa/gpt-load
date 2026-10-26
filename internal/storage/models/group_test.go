package models

import "testing"

func TestGroupBeforeSaveDefaultsParams(t *testing.T) {
	t.Parallel()

	for _, params := range []JSON{JSON(""), JSON("null"), JSON("  ")} {
		group := Group{Params: params}
		if err := group.BeforeSave(nil); err != nil {
			t.Fatalf("BeforeSave(%q) error = %v", params, err)
		}
		if string(group.Params) != `{}` {
			t.Fatalf("params = %s, want {}", group.Params)
		}
	}
}
