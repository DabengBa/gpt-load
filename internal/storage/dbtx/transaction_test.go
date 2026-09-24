package dbtx

import (
	"context"
	"path/filepath"
	"reflect"
	"runtime"
	"testing"
	"time"

	gormsqlite "github.com/glebarez/sqlite"
	"gorm.io/gorm"
	"gorm.io/gorm/logger"
)

func TestCapabilitiesForDriverPreserveDatabaseTransactionSemantics(t *testing.T) {
	tests := []struct {
		name     string
		driver   string
		writeSQL []string
		readSQL  []string
	}{
		{name: "sqlite", driver: "sqlite", writeSQL: []string{"BEGIN IMMEDIATE"}, readSQL: []string{"BEGIN"}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			capabilities, err := CapabilitiesForDriver(test.driver)
			if err != nil {
				t.Fatalf("CapabilitiesForDriver(%q) error = %v", test.driver, err)
			}
			writeSQL, err := capabilities.beginStatements(Write)
			if err != nil || !reflect.DeepEqual(writeSQL, test.writeSQL) {
				t.Fatalf("write begin SQL = %#v/%v, want %#v/nil", writeSQL, err, test.writeSQL)
			}
			readSQL, err := capabilities.beginStatements(ReadSnapshot)
			if err != nil || !reflect.DeepEqual(readSQL, test.readSQL) {
				t.Fatalf("read begin SQL = %#v/%v, want %#v/nil", readSQL, err, test.readSQL)
			}
		})
	}
}

func TestQueuedSQLiteWriterDoesNotStarveWALReaders(t *testing.T) {
	dsn := filepath.Join(t.TempDir(), "writer-gate.db") + "?_pragma=journal_mode(WAL)&_pragma=busy_timeout(5000)"
	db, err := gorm.Open(
		gormsqlite.Open(dsn),
		&gorm.Config{Logger: logger.Default.LogMode(logger.Silent)},
	)
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("get database: %v", err)
	}
	sqlDB.SetMaxOpenConns(2)
	t.Cleanup(func() { _ = sqlDB.Close() })
	if err := db.Exec("CREATE TABLE reader_probe (value INTEGER NOT NULL)").Error; err != nil {
		t.Fatalf("create reader probe: %v", err)
	}
	if err := db.Exec("INSERT INTO reader_probe (value) VALUES (1)").Error; err != nil {
		t.Fatalf("seed reader probe: %v", err)
	}

	releaseFirstWriter := make(chan struct{})
	firstWriterEntered := make(chan struct{})
	firstWriterDone := make(chan error, 1)
	go func() {
		firstWriterDone <- Run(context.Background(), db, Options{Mode: Write}, func(*gorm.DB) error {
			close(firstWriterEntered)
			<-releaseFirstWriter
			return nil
		})
	}()
	<-firstWriterEntered

	secondWriterStarted := make(chan struct{})
	secondWriterDone := make(chan error, 1)
	go func() {
		close(secondWriterStarted)
		secondWriterDone <- Run(context.Background(), db, Options{Mode: Write}, func(*gorm.DB) error {
			return nil
		})
	}()
	<-secondWriterStarted

	gate := sqliteWriteGateFor(sqlDB)
	deadline := time.Now().Add(time.Second)
	for gate.waiters.Load() == 0 && time.Now().Before(deadline) {
		runtime.Gosched()
	}
	if gate.waiters.Load() != 1 {
		close(releaseFirstWriter)
		<-firstWriterDone
		<-secondWriterDone
		t.Fatalf("queued SQLite writers = %d, want 1", gate.waiters.Load())
	}

	readCtx, cancelRead := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancelRead()
	var value int
	readErr := db.WithContext(readCtx).Raw("SELECT value FROM reader_probe").Scan(&value).Error

	close(releaseFirstWriter)
	if err := <-firstWriterDone; err != nil {
		t.Fatalf("first writer: %v", err)
	}
	if err := <-secondWriterDone; err != nil {
		t.Fatalf("queued writer: %v", err)
	}
	if readErr != nil {
		t.Fatalf("WAL reader was starved by a queued writer: %v", readErr)
	}
	if value != 1 {
		t.Fatalf("reader value = %d, want 1", value)
	}
}

func TestSQLiteWriteGatesAreScopedToTheirConnectionPool(t *testing.T) {
	open := func(name string) *gorm.DB {
		db, err := gorm.Open(
			gormsqlite.Open(filepath.Join(t.TempDir(), name)+"?_pragma=journal_mode(WAL)"),
			&gorm.Config{Logger: logger.Default.LogMode(logger.Silent)},
		)
		if err != nil {
			t.Fatalf("open %s: %v", name, err)
		}
		sqlDB, err := db.DB()
		if err != nil {
			t.Fatalf("get %s database: %v", name, err)
		}
		t.Cleanup(func() { _ = sqlDB.Close() })
		return db
	}

	first, second := open("first.db"), open("second.db")
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	err := Run(ctx, first, Options{Mode: Write}, func(*gorm.DB) error {
		return Run(ctx, second, Options{Mode: Write}, func(*gorm.DB) error { return nil })
	})
	if err != nil {
		t.Fatalf("nested writes to independent SQLite pools: %v", err)
	}
}

func TestDiscardConnectionTreatsBadConnectionAsSuccessfulCleanup(t *testing.T) {
	db, err := gorm.Open(
		gormsqlite.Open(":memory:"),
		&gorm.Config{Logger: logger.Default.LogMode(logger.Silent)},
	)
	if err != nil {
		t.Fatalf("open database: %v", err)
	}
	sqlDB, err := db.DB()
	if err != nil {
		t.Fatalf("get database: %v", err)
	}
	defer sqlDB.Close()

	connection, err := sqlDB.Conn(context.Background())
	if err != nil {
		t.Fatalf("get connection: %v", err)
	}
	if err := discardConnection("test", connection); err != nil {
		t.Fatalf("discardConnection() error = %v, want nil", err)
	}
}
