package storage

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/glebarez/sqlite"
	"github.com/sirupsen/logrus"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"

	"gpt-load/internal/platform/config"
)

var databaseLogger = newDatabaseLogger(os.Stdout)

func newDatabaseLogger(output io.Writer) logger.Interface {
	base := logger.New(log.New(output, "\r\n", log.LstdFlags), logger.Config{
		SlowThreshold:             200 * time.Millisecond,
		LogLevel:                  logger.Warn,
		IgnoreRecordNotFoundError: true,
		ParameterizedQueries:      true,
		Colorful:                  true,
	})
	return databaseLogFilter{Interface: base}
}

type databaseLogFilter struct {
	logger.Interface
}

func (filter databaseLogFilter) LogMode(level logger.LogLevel) logger.Interface {
	return databaseLogFilter{Interface: filter.Interface.LogMode(level)}
}

func (filter databaseLogFilter) Trace(
	ctx context.Context,
	begin time.Time,
	query func() (string, int64),
	err error,
) {
	if ctx != nil &&
		errors.Is(ctx.Err(), context.Canceled) &&
		errors.Is(err, context.Canceled) {
		return
	}
	filter.Interface.Trace(ctx, begin, query, err)
}

func (filter databaseLogFilter) ParamsFilter(
	ctx context.Context,
	query string,
	params ...interface{},
) (string, []interface{}) {
	if paramsFilter, ok := filter.Interface.(gorm.ParamsFilter); ok {
		return paramsFilter.ParamsFilter(ctx, query, params...)
	}
	return query, nil
}

// Open opens a database using a fully resolved DSN.
// Resolving an empty DSN to DATA_DIR belongs to platform/config.
func Open(dsn string) (*gorm.DB, error) {
	return OpenWithSource(dsn, config.DatabaseSourceExternal)
}

// OpenWithSource opens a database and applies file controls only when the
// application owns the managed SQLite location.
func OpenWithSource(dsn string, source config.DatabaseSource) (*gorm.DB, error) {
	return openWithSource(dsn, source)
}

// OpenConfigured opens the database using the process configuration resolved
// by platform/config. Storage never reads environment variables directly.
func OpenConfigured(cfg *config.Config) (*gorm.DB, error) {
	if cfg == nil {
		return nil, fmt.Errorf("open database: configuration is unavailable")
	}
	return openWithSource(cfg.DatabaseDSN, cfg.DatabaseMetadata.Source)
}

func openWithSource(dsn string, source config.DatabaseSource) (*gorm.DB, error) {
	database, err := config.ParseDatabaseDSN(dsn)
	if err != nil {
		return nil, err
	}
	if database.Driver != config.DatabaseDriverSQLite {
		return nil, fmt.Errorf("open database: unsupported database driver %q", database.Driver)
	}
	switch source {
	case config.DatabaseSourceManaged, config.DatabaseSourceExternal:
	default:
		return nil, fmt.Errorf("open database: unsupported database source")
	}

	if source == config.DatabaseSourceExternal {
		logExternalDatabaseSource(database.Driver)
	}
	return openSQLite(database.DSN, source)
}

// openDatabase is the SQLite GORM/SQL lifecycle. SQLite requires one
// physical connection for its single-writer runtime and shared in-memory DSNs.
func openDatabase(
	driver config.DatabaseDriver,
	dialector gorm.Dialector,
) (*gorm.DB, error) {
	db, err := gorm.Open(dialector, &gorm.Config{
		Logger:         databaseLogger,
		TranslateError: true,
	})
	if err != nil {
		return nil, fmt.Errorf("open %s database: %w", databaseDisplayName(driver), err)
	}

	sqlDB, err := db.DB()
	if err != nil {
		return nil, fmt.Errorf("get %s connection pool: %w", databaseDisplayName(driver), err)
	}
	sqlDB.SetMaxOpenConns(1)
	sqlDB.SetMaxIdleConns(1)
	if err := sqlDB.PingContext(context.Background()); err != nil {
		_ = sqlDB.Close()
		return nil, fmt.Errorf("ping %s database: %w", databaseDisplayName(driver), err)
	}
	return db, nil
}

func newDatabaseDialector(database config.DatabaseConfig) (gorm.Dialector, error) {
	if database.Driver != config.DatabaseDriverSQLite {
		return nil, fmt.Errorf("unsupported database driver")
	}
	return sqlite.Open(database.DSN), nil
}

func logExternalDatabaseSource(driver config.DatabaseDriver) {
	logrus.WithFields(logrus.Fields{
		"database_source": config.DatabaseSourceExternal,
		"database_driver": driver,
	}).Info("Database storage is managed by the operator")
}

func databaseDisplayName(driver config.DatabaseDriver) string {
	switch driver {
	case config.DatabaseDriverSQLite:
		return "SQLite"
	default:
		return string(driver)
	}
}
