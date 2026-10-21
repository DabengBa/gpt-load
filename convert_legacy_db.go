package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"slices"

	"gpt-load/internal/storage/offlineconvert"
)

// convertLegacyDatabase is the operator entry point for the offline converter.
// It only takes file paths: the encryption key file is intentionally not read
// or copied, and the operator keeps it next to the published database.
func convertLegacyDatabase(args []string, stdout, stderr io.Writer) error {
	flags := flag.NewFlagSet("convert-legacy-db", flag.ContinueOnError)
	flags.SetOutput(stderr)
	source := flags.String("source", "", "legacy SQLite database file (read-only)")
	target := flags.String("target", "", "new SQLite database file to publish (must not exist)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if leftover := flags.Args(); len(leftover) != 0 {
		return fmt.Errorf("unexpected arguments: %v", leftover)
	}
	if *source == "" || *target == "" {
		return errors.New("both --source and --target are required")
	}

	result, err := offlineconvert.Convert(context.Background(), offlineconvert.Options{
		SourceDSN: *source,
		TargetDSN: *target,
	})
	if err != nil {
		return err
	}

	fmt.Fprintf(stdout, "converted %s -> %s\n", *source, *target)
	tables := make([]string, 0, len(result.Tables))
	for table := range result.Tables {
		tables = append(tables, table)
	}
	slices.Sort(tables)
	for _, table := range tables {
		fmt.Fprintf(stdout, "  %-32s %d rows\n", table, result.Tables[table])
	}
	if len(result.RetiredColumns) != 0 {
		fmt.Fprintf(stdout, "dropped retired columns: %v\n", result.RetiredColumns)
	}
	fmt.Fprintln(stdout, "not migrated: request/usage logs and debug captures")
	fmt.Fprintln(stdout, "copy your encryption.key into the target DATA_DIR before starting the gateway")
	return nil
}
