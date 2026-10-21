package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDispatchCommandConvertLegacyDBRequiresBothPaths(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exitCode := dispatchCommand([]string{"convert-legacy-db", "--source", "legacy.db"}, &stdout, &stderr)
	if exitCode == 0 {
		t.Fatalf("missing target accepted: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "--target") {
		t.Fatalf("stderr = %q, want missing --target explanation", stderr.String())
	}
}

func TestDispatchCommandConvertLegacyDBRejectsUnknownFlags(t *testing.T) {
	var stdout, stderr bytes.Buffer
	exitCode := dispatchCommand([]string{"convert-legacy-db", "--key", "k"}, &stdout, &stderr)
	if exitCode == 0 {
		t.Fatalf("unknown flag accepted: %s", stdout.String())
	}
}

func TestDispatchCommandConvertLegacyDBReportsMissingSource(t *testing.T) {
	dir := t.TempDir()
	source := filepath.Join(dir, "missing.db")
	target := filepath.Join(dir, "out", "gpt-load.db")

	var stdout, stderr bytes.Buffer
	exitCode := dispatchCommand(
		[]string{"convert-legacy-db", "--source", source, "--target", target},
		&stdout, &stderr,
	)
	if exitCode == 0 {
		t.Fatalf("missing source accepted: %s", stdout.String())
	}
	if !strings.Contains(stderr.String(), "convert-legacy-db:") {
		t.Fatalf("stderr = %q, want command-prefixed failure", stderr.String())
	}
	if _, err := os.Lstat(target); !os.IsNotExist(err) {
		t.Fatalf("failed conversion published a target: %v", err)
	}
}

func TestPrintHelpListsConvertLegacyDB(t *testing.T) {
	var output bytes.Buffer
	printHelp(&output)
	if !strings.Contains(output.String(), "convert-legacy-db --source <path> --target <path>") {
		t.Fatalf("help does not document the converter:\n%s", output.String())
	}
}
