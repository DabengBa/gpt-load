package control

import (
	"context"
	"fmt"
	"time"

	"github.com/gin-gonic/gin"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/state"
	"gpt-load/internal/storage"
)

type sqliteMaintenanceHealth struct {
	storage.SQLiteMaintenanceStatus
	Error string `json:"error"`
}

type sqliteMaintenanceResponse struct {
	RetentionDays        int                     `json:"retention_days"`
	RetentionSource      string                  `json:"retention_source"`
	SweepIntervalSeconds int64                   `json:"sweep_interval_seconds"`
	SQLite               sqliteMaintenanceHealth `json:"sqlite"`
}

// SQLiteMaintenanceStatus exposes the effective snapshot and a read-only SQLite
// observation. Inspection failures are reported without exposing DSNs or paths.
func (service *Service) SQLiteMaintenanceStatus(ctx context.Context) (sqliteMaintenanceResponse, error) {
	if service == nil || service.db == nil || service.manager == nil {
		return sqliteMaintenanceResponse{}, app_errors.ErrInternalServer
	}
	service.writeMu.RLock()
	snapshot := service.manager.Current()
	if snapshot == nil {
		service.writeMu.RUnlock()
		return sqliteMaintenanceResponse{}, app_errors.ErrInternalServer
	}
	result := sqliteMaintenanceResponse{
		RetentionDays:        snapshot.Settings.RequestLogRetentionDays,
		RetentionSource:      "default",
		SweepIntervalSeconds: int64(retentionInterval / time.Second),
	}
	var overridden int64
	err := service.db.WithContext(ctx).Table("system_settings").Where("key = ?", state.SettingRequestLogRetentionDays).Count(&overridden).Error
	service.writeMu.RUnlock()
	if err != nil {
		return sqliteMaintenanceResponse{}, fmt.Errorf("read retention source: %w", app_errors.ErrInternalServer)
	}
	if overridden != 0 {
		result.RetentionSource = "system_setting"
	}
	inspectCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	status, err := storage.InspectSQLite(inspectCtx, service.db)
	result.SQLite = sqliteMaintenanceHealth{SQLiteMaintenanceStatus: status}
	if err != nil {
		result.SQLite = sqliteMaintenanceHealth{SQLiteMaintenanceStatus: storage.SQLiteMaintenanceStatus{MaintenanceMode: "offline_only"}, Error: "inspection_failed"}
	}
	return result, nil
}

func (server *Server) handleSQLiteMaintenanceStatus(c *gin.Context) {
	result, err := server.service.SQLiteMaintenanceStatus(c.Request.Context())
	if err != nil {
		writeServiceError(c, "sqlite_maintenance_status", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}
