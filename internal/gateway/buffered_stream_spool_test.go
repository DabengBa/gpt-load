package gateway

import (
	"bytes"
	"errors"
	"os"
	"testing"
)

func TestBufferedStreamSpoolSpillsToUnlinkedPrivateUnixFile(t *testing.T) {
	spool, err := newBufferedStreamSpool(4, 32)
	if err != nil {
		t.Fatal(err)
	}
	directory := ""
	defer func() {
		if err := spool.Close(); err != nil {
			t.Fatal(err)
		}
		if directory != "" {
			if _, err := os.Stat(directory); !os.IsNotExist(err) {
				t.Fatalf("spool directory still exists: %v", err)
			}
		}
	}()
	if _, err := spool.Write([]byte("abcde")); err != nil {
		t.Fatal(err)
	}
	if spool.file == nil || spool.directory == "" {
		t.Fatal("spool did not spill after memory threshold")
	}
	directory = spool.directory
	if _, err := os.Stat(spool.file.Name()); !os.IsNotExist(err) {
		t.Fatalf("spool file is still named in the filesystem: %v", err)
	}
	info, err := os.Stat(directory)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0700 {
		t.Fatalf("spool directory permissions = %o, want 700", info.Mode().Perm())
	}
	var got bytes.Buffer
	if _, err := spool.ReplayTo(&got); err != nil {
		t.Fatal(err)
	}
	if got.String() != "abcde" || spool.Size() != 5 {
		t.Fatalf("replayed spool = %q, size = %d", got.String(), spool.Size())
	}
}

func TestBufferedStreamSpoolRejectsHardLimitWithoutAppending(t *testing.T) {
	spool, err := newBufferedStreamSpool(4, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer spool.Close()
	if _, err := spool.Write([]byte("12345")); err == nil {
		t.Fatal("hard-limit write unexpectedly succeeded")
	}
	if spool.Size() != 0 {
		t.Fatalf("spool size = %d after rejected write, want zero", spool.Size())
	}
}

func TestBufferedStreamSpoolReleasesBudgetAfterReservationFailure(t *testing.T) {
	before := bufferedStreamReservedBytes.Load()
	bufferedStreamReservedBytes.Store(bufferedStreamTotalBudget)
	t.Cleanup(func() { bufferedStreamReservedBytes.Store(before) })
	spool, err := newBufferedStreamSpool(4, 32)
	if err != nil {
		t.Fatal(err)
	}
	defer spool.Close()
	if _, err := spool.Write([]byte("budget")); err == nil {
		t.Fatal("budget-exhausted write unexpectedly succeeded")
	}
	if got := bufferedStreamReservedBytes.Load(); got != bufferedStreamTotalBudget {
		t.Fatalf("reserved budget changed after rejected write: %d", got)
	}
}

func TestBufferedStreamSpoolCleansUpAfterSpillWriteFailure(t *testing.T) {
	tempDir := t.TempDir()
	oldTemp := os.Getenv("TMPDIR")
	if err := os.Setenv("TMPDIR", tempDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Setenv("TMPDIR", oldTemp) })
	oldWrite := bufferedStreamSpoolFileWrite
	bufferedStreamSpoolFileWrite = func(*os.File, []byte) (int, error) {
		return 0, errors.New("injected spill write failure")
	}
	t.Cleanup(func() { bufferedStreamSpoolFileWrite = oldWrite })
	before := bufferedStreamReservedBytes.Load()
	spool, err := newBufferedStreamSpool(1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := spool.Write([]byte("spill")); err == nil {
		t.Fatal("spill write unexpectedly succeeded")
	}
	if err := spool.Close(); err != nil {
		t.Fatal(err)
	}
	if spool.file != nil || spool.directory != "" || spool.Size() != 0 {
		t.Fatalf("failed spill retained resources: %#v", spool)
	}
	if got := bufferedStreamReservedBytes.Load(); got != before {
		t.Fatalf("reserved budget after failed spill cleanup = %d, want %d", got, before)
	}
	entries, err := os.ReadDir(tempDir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("failed spill left temporary entries: %#v", entries)
	}
}

func TestBufferedStreamSpoolReadFailureStillCloses(t *testing.T) {
	before := bufferedStreamReservedBytes.Load()
	spool, err := newBufferedStreamSpool(1, 32)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := spool.Write([]byte("spill")); err != nil {
		t.Fatal(err)
	}
	if err := spool.file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := spool.ReplayTo(&bytes.Buffer{}); err == nil {
		t.Fatal("closed spool file replay unexpectedly succeeded")
	}
	if err := spool.Close(); err != nil {
		t.Logf("close after forced read failure: %v", err)
	}
	if spool.file != nil || spool.directory != "" {
		t.Fatalf("read failure did not clean spool: %#v", spool)
	}
	if got := bufferedStreamReservedBytes.Load(); got != before {
		t.Fatalf("reserved budget after read failure cleanup = %d, want %d", got, before)
	}
}
