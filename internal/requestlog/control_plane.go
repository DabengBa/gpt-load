package requestlog

import (
	"gorm.io/gorm"

	"gpt-load/internal/execution"
)

// withoutControlPlaneObservations keeps control-plane observations
// (operation = probe) out of a traffic read.
//
// Probe rows are durable request-log rows that really reached an upstream, so they
// belong in the log page, but they are not traffic: they must not move request
// counts, success rates, or per-credential activity. Excluding them on the
// aggregation-journal side alone is not enough, because a reader that queries
// request_logs or request_log_attempts directly bypasses that gate. Three such
// readers — QueryGroupUsage, queryCredentialBoundaryActivity and
// queryCredentialRequestLogUsage — each leaked probe rows before this helper
// existed. Every new direct read of either table must go through here.
func withoutControlPlaneObservations(db *gorm.DB) *gorm.DB {
	return db.Where("operation <> ?", string(execution.OperationProbe))
}
