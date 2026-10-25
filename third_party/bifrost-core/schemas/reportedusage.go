package schemas

type reportedUsageSnapshotKey struct{}

// SetReportedUsageSnapshot publishes an owned cumulative snapshot at the usage
// owner's synchronous merge point, before any next tool/turn or post-hook.
// The context lock synchronizes publication; the stored copy is immutable.
func SetReportedUsageSnapshot(ctx *BifrostContext, value *BifrostLLMUsage) {
	if ctx != nil && value != nil {
		ctx.SetValue(reportedUsageSnapshotKey{}, value.DeepCopy())
	}
}

// ReportedUsageSnapshot returns an independent copy, never a mutable provider
// accumulation handle. Reading it does not wait for cancellation/terminal hooks.
func ReportedUsageSnapshot(ctx *BifrostContext) *BifrostLLMUsage {
	if ctx == nil {
		return nil
	}
	value, _ := ctx.Value(reportedUsageSnapshotKey{}).(*BifrostLLMUsage)
	return value.DeepCopy()
}
