package requestlog

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

func TestFacetHistoricalCandidates(t *testing.T) {
	db := openRequestLogQueryDB(t)
	when := time.Date(2026, time.July, 24, 12, 0, 0, 0, time.UTC)
	for i := range 55 {
		model := "alpha"
		if i == 0 {
			model = "old-model"
		}
		row := requestLogQueryRow(fmt.Sprintf("00000000-0000-4000-8000-%012d", i), when.Add(time.Duration(i)*time.Second), 71, model, []Attempt{
			{GroupID: 12, GroupName: "deleted", StatusCode: 429},
			{GroupID: 13, GroupName: "final", StatusCode: 200},
		})
		createRequestLogQueryRow(t, db, row)
	}
	service := newRequestLogTestService(db)
	call := func(query ListQuery) FacetResult {
		t.Helper()
		result, err := service.Facets(t.Context(), query)
		if err != nil {
			t.Fatalf("Facets: %v", err)
		}
		return result
	}
	group, status := uint(12), 200
	result := call(ListQuery{GroupID: &group, ClientModel: "alpha", AttemptStatusCode: &status, Limit: 1})
	if !reflect.DeepEqual(result.Groups, []FacetGroup{{ID: 13, Name: "final"}}) {
		t.Fatalf("groups = %#v", result.Groups)
	}
	if !reflect.DeepEqual(result.ClientModels, []FacetClientModel{}) {
		t.Fatalf("cross-attempt models = %#v", result.ClientModels)
	}
	status = 429
	result = call(ListQuery{GroupID: &group, ClientModel: "alpha", AttemptStatusCode: &status, Limit: 1, Cursor: &Cursor{CompletedAtMS: 1, RequestID: "irrelevant"}})
	if !reflect.DeepEqual(result.Groups, []FacetGroup{{ID: 12, Name: "deleted"}}) {
		t.Fatalf("groups = %#v", result.Groups)
	}
	if !reflect.DeepEqual(result.ClientModels, []FacetClientModel{{Value: "alpha"}, {Value: "old-model"}}) {
		t.Fatalf("models = %#v", result.ClientModels)
	}
	from, to := when.UnixMilli(), when.Add(time.Second).UnixMilli()
	result = call(ListQuery{FromMS: &from, ToMS: &to, GroupID: &group})
	if !reflect.DeepEqual(result.ClientModels, []FacetClientModel{{Value: "old-model"}}) {
		t.Fatalf("time range models = %#v", result.ClientModels)
	}
	row := requestLogQueryRow("00000000-0000-4000-8000-000000001000", when, 71, "", []Attempt{{GroupID: 14}})
	createRequestLogQueryRow(t, db, row)
	noAttempt := requestLogQueryRow("00000000-0000-4000-8000-000000001001", when, 71, "without-attempt", nil)
	createRequestLogQueryRow(t, db, noAttempt)
	result = call(ListQuery{FromMS: &from, ToMS: &to})
	if !reflect.DeepEqual(result.ClientModels, []FacetClientModel{{Value: "old-model"}, {Value: "without-attempt"}}) {
		t.Fatalf("empty model was included: %#v", result.ClientModels)
	}
	resultWithGroup := call(ListQuery{FromMS: &from, ToMS: &to, GroupID: &group})
	if !reflect.DeepEqual(resultWithGroup.ClientModels, []FacetClientModel{{Value: "old-model"}}) {
		t.Fatalf("models with group = %#v", resultWithGroup.ClientModels)
	}
	if !reflect.DeepEqual(result.Groups, []FacetGroup{{ID: 12, Name: "deleted"}, {ID: 13, Name: "final"}, {ID: 14}}) {
		t.Fatalf("nameless group = %#v", result.Groups)
	}
	from = when.Add(2 * time.Hour).UnixMilli()
	result = call(ListQuery{FromMS: &from})
	if len(result.Groups) != 0 || result.Groups == nil || len(result.ClientModels) != 0 || result.ClientModels == nil {
		t.Fatalf("empty results = %#v", result)
	}
}

func TestFacetQueryPlanAndTiming(t *testing.T) {
	db := openRequestLogQueryDB(t)
	when := time.Date(2026, time.July, 24, 12, 0, 0, 0, time.UTC)
	for i := range 1000 {
		row := requestLogQueryRow(fmt.Sprintf("00000000-0000-4000-8000-%012d", i), when.Add(time.Duration(i)*time.Second), 71, fmt.Sprintf("model-%d", i%10), []Attempt{{GroupID: uint(i%5 + 1), StatusCode: 200}, {GroupID: uint(i%5 + 6), StatusCode: 429}})
		createRequestLogQueryRow(t, db, row)
	}
	type planRow struct{ Detail string }
	var plan []planRow
	if err := db.Raw(`EXPLAIN QUERY PLAN SELECT attempt.group_id, MAX(attempt.group_name) FROM request_log_attempts AS attempt WHERE attempt.request_id IN (SELECT id FROM request_logs WHERE completed_at_ms >= ?) AND attempt.group_id > 0 AND attempt.status_code = ? GROUP BY attempt.group_id ORDER BY attempt.group_id`, when.UnixMilli(), 429).Scan(&plan).Error; err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	status, from := 429, when.UnixMilli()
	for range 20 {
		if _, err := newRequestLogTestService(db).Facets(t.Context(), ListQuery{FromMS: &from, AttemptStatusCode: &status}); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("SQLite rows=1000 attempts=2000, 20 Facets calls=%s, average=%s", time.Since(start), time.Since(start)/20)
	for _, row := range plan {
		t.Logf("group query plan: %s", row.Detail)
	}
	plan = nil
	if err := db.Raw(`EXPLAIN QUERY PLAN SELECT DISTINCT client_model AS value FROM request_logs WHERE client_model <> '' AND completed_at_ms >= ? AND id IN (SELECT attempt.request_id FROM request_log_attempts AS attempt WHERE attempt.status_code = ?) ORDER BY client_model`, when.UnixMilli(), 429).Scan(&plan).Error; err != nil {
		t.Fatal(err)
	}
	for _, row := range plan {
		t.Logf("model query plan: %s", row.Detail)
	}
}
