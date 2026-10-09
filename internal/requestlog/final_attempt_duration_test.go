package requestlog

import (
	"testing"
	"time"
)

func TestFinalAttemptDurationListAndDetail(t *testing.T) {
	t.Parallel()
	db := openRequestLogQueryDB(t)
	service := newRequestLogTestService(db)
	for _, test := range []struct {
		name     string
		id       string
		attempts []Attempt
		want     *int64
	}{
		{"retry", "00000000-0000-4000-8000-000000000971", []Attempt{{Sequence: 3, GroupID: 99, DurationMs: 17}, {Sequence: 1, DurationMs: 80}}, new(int64(17))},
		{"zero duration", "00000000-0000-4000-8000-000000000972", []Attempt{{Sequence: 1, DurationMs: 0}}, new(int64(0))},
		{"no attempts", "00000000-0000-4000-8000-000000000973", nil, nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			row := requestLogQueryRow(test.id, time.Now(), 0, "model", test.attempts)
			row.DurationMs = 100
			createRequestLogQueryRow(t, db, row)
			page, err := service.List(t.Context(), ListQuery{RequestID: test.id})
			if err != nil || len(page.Items) != 1 {
				t.Fatalf("List = %#v, %v", page, err)
			}
			detail, err := service.Get(t.Context(), test.id)
			if err != nil {
				t.Fatal(err)
			}
			for _, record := range []Record{page.Items[0], detail} {
				if record.DurationMs != 100 {
					t.Fatalf("request duration = %d", record.DurationMs)
				}
				got := record.FinalAttemptDurationMs
				if (got == nil) != (test.want == nil) || got != nil && *got != *test.want {
					t.Fatalf("final duration = %v, want %v", got, test.want)
				}
			}
		})
	}
}
