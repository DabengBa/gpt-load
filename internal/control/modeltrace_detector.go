package control

import (
	"context"
	_ "embed"
	"os"
	"path/filepath"
	"sync"

	lmfpd "github.com/Ikaleio/lm-detector/sdk/go"
)

// Data and SDK are pinned to upstream commit f920cfe2ff0bab94989ce01ac061e21425b6e321.
//
//go:embed lmdata/unified_bank.json
var modelTraceBank []byte

//go:embed lmdata/shared_detector.bin
var modelTraceDetectorData []byte

var modelTraceDetectorMu sync.Mutex
var modelTraceDetector *lmfpd.Detector

// The SDK reads and decodes both files into memory during Load; it retains no
// path or file handles, so the private staging directory can be removed.
func loadModelTraceData(ctx context.Context, bank, detector []byte) (*lmfpd.Detector, error) {
	dir, err := os.MkdirTemp("", "gpt-load-fingerprint-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	if err = os.WriteFile(filepath.Join(dir, "unified_bank.json"), bank, 0o600); err != nil {
		return nil, err
	}
	if err = os.WriteFile(filepath.Join(dir, "shared_detector.bin"), detector, 0o600); err != nil {
		return nil, err
	}
	return lmfpd.Load(ctx, lmfpd.LoadOptions{Sources: []string{dir}})
}

var getModelTraceDetector = func(ctx context.Context) (*lmfpd.Detector, error) {
	modelTraceDetectorMu.Lock()
	defer modelTraceDetectorMu.Unlock()
	if modelTraceDetector != nil {
		return modelTraceDetector, nil
	}
	detector, err := loadModelTraceData(ctx, modelTraceBank, modelTraceDetectorData)
	if err != nil {
		return nil, err
	}
	modelTraceDetector = detector
	return detector, nil
}
