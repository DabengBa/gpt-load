package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestPrintHelpMarksKeyMigrationAsDeferred(t *testing.T) {
	var output bytes.Buffer
	printHelp(&output)

	help := output.String()
	if !strings.Contains(help, "migrate-keys") || !strings.Contains(help, "later release") {
		t.Fatalf("help does not mark migrate-keys as deferred:\n%s", help)
	}
}

func TestPrintHelpOmitsRemovedWindowsServiceManagement(t *testing.T) {
	var output bytes.Buffer
	printHelp(&output)

	help := output.String()
	if strings.Contains(help, "Windows Service Commands:") || strings.Contains(help, "service ") {
		t.Fatalf("help exposes removed Windows service management:\n%s", help)
	}
}

func TestDispatchCommandDoesNotRunLegacyKeyMigration(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := dispatchCommand([]string{"migrate-keys"}, &stdout, &stderr)

	if exitCode == 0 {
		t.Fatal("dispatchCommand(migrate-keys) exit code = 0, want non-zero")
	}
	if !strings.Contains(stderr.String(), "later release") {
		t.Fatalf("stderr does not explain deferred command: %s", stderr.String())
	}
}

func TestDispatchCommandDoesNotRecognizeRemovedServiceNamespace(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := dispatchCommand([]string{"service", "status"}, &stdout, &stderr)

	if exitCode == 0 {
		t.Fatal("dispatchCommand(service status) exit code = 0, want unknown command failure")
	}
	if !strings.Contains(stderr.String(), "Unknown command: service") {
		t.Fatalf("stderr = %q, want unknown service command", stderr.String())
	}
}
