package agent

import (
	"github.com/gin-gonic/gin"

	"gpt-load/internal/debugcapture"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
)

func (server *Server) handleHealth(c *gin.Context) {
	if !requireScope(c, ScopeDiagnosticsRead) {
		return
	}
	if server.currentSnapshot() == nil {
		writeAgentServiceError(c, "agent_health", app_errors.ErrInternalServer)
		return
	}
	view := server.healthView()
	response.SuccessI18n(c, "common.success", view)
}

func (server *Server) healthView() HealthView {
	snapshot := server.currentSnapshot()
	view := HealthView{
		SchemaVersion: SchemaVersion,
		ObservedAtMS:  server.now().UTC().UnixMilli(),
		Version:       server.version,
		UptimeSeconds: server.uptimeSeconds(),
	}
	if snapshot != nil {
		view.SnapshotRevision = snapshot.Revision
	}
	if snapshot != nil {
		for _, catalog := range snapshot.GroupCatalog {
			view.Groups.Total++
			if catalog.Enabled {
				view.Groups.Enabled++
			}
		}
	}
	if server.requestLogStats != nil {
		stats := server.requestLogStats.Stats()
		view.RequestLog = RequestLogHealthView{
			QueueDepth:                    stats.QueueDepth,
			QueueCapacity:                 stats.QueueCapacity,
			PersistedTotal:                stats.PersistedTotal,
			DroppedTotal:                  stats.DroppedTotal,
			WriteFailureTotal:             stats.WriteFailureTotal,
			AccessQuotaCheckpointDegraded: stats.AccessQuotaCheckpointDegraded,
		}
		view.RequestLog.Degraded = requestLogDegraded(view.RequestLog)
	}
	if server.evidenceHealth != nil {
		health, err := server.evidenceHealth.Health()
		if err == nil {
			view.DebugCapture = DebugCaptureHealthView{
				Enabled:           health.Enabled,
				Running:           health.Running,
				RetentionSeconds:  health.RetentionSeconds,
				Active:            health.Active,
				Completed:         health.Completed,
				Failed:            health.Failed,
				RemovedTotal:      health.RemovedTotal,
				SweepFailureTotal: health.SweepFailureTotal,
				ErrorPresent:      health.Error != "",
			}
		} else {
			view.DebugCapture.ErrorPresent = true
		}
		if view.DebugCapture.RetentionSeconds == 0 {
			view.DebugCapture.RetentionSeconds = int64(debugcapture.RetentionPeriod().Seconds())
		}
	} else {
		view.DebugCapture.RetentionSeconds = int64(debugcapture.RetentionPeriod().Seconds())
	}
	view.Degraded = view.RequestLog.Degraded ||
		(view.DebugCapture.Enabled && !view.DebugCapture.Running) ||
		view.DebugCapture.ErrorPresent
	return view
}

func requestLogDegraded(view RequestLogHealthView) bool {
	if view.AccessQuotaCheckpointDegraded {
		return true
	}
	if view.WriteFailureTotal > 0 {
		return true
	}
	return view.QueueCapacity > 0 && view.QueueDepth >= view.QueueCapacity
}
