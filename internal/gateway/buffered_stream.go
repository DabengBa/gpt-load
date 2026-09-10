//go:build !windows

package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"

	"gpt-load/internal/dialect"
	"gpt-load/internal/protocol"
)

const (
	bufferedStreamMemoryLimit   = int64(1 << 20)
	bufferedStreamResponseLimit = int64(32 << 20)
	bufferedStreamTotalBudget   = int64(256 << 20)
	bufferedStreamHeartbeat     = ": keep-alive\n\n"
)

var bufferedStreamReservedBytes atomic.Int64

var bufferedStreamSpoolFileWrite = func(file *os.File, data []byte) (int, error) {
	return file.Write(data)
}

type bufferedStreamSessionContextKey struct{}

type bufferedStreamSession struct {
	HTTPCommitted   bool
	VisibleBytes    int64
	PayloadReleased bool
}

func bufferedStreamSessionFromContext(ctx context.Context) *bufferedStreamSession {
	if ctx != nil {
		if session, ok := ctx.Value(bufferedStreamSessionContextKey{}).(*bufferedStreamSession); ok && session != nil {
			return session
		}
	}
	return &bufferedStreamSession{}
}

// BufferedStreamOutput is the irreversible downstream state for one request.
// HTTP commitment can happen on a heartbeat while payload release remains false.
type BufferedStreamOutput struct {
	HTTPCommitted      bool
	PayloadReleased    bool
	ClientVisibleBytes int64
}

func (output *BufferedStreamOutput) markHTTPCommitted() {
	if output != nil {
		output.HTTPCommitted = true
	}
}

func (output *BufferedStreamOutput) markVisibleBytes(count int64) {
	if output == nil || count <= 0 {
		return
	}
	output.ClientVisibleBytes += count
}

func (output *BufferedStreamOutput) markPayloadReleased() error {
	if output == nil {
		return errors.New("buffered stream output is required")
	}
	if output.PayloadReleased {
		return errors.New("buffered stream payload was already released")
	}
	if !output.HTTPCommitted {
		return errors.New("buffered stream payload cannot release before HTTP commit")
	}
	output.PayloadReleased = true
	return nil
}

func (output BufferedStreamOutput) canRetry() bool {
	return output.HTTPCommitted && !output.PayloadReleased && output.ClientVisibleBytes > 0
}

type bufferedStreamSpool struct {
	memoryLimit int64
	maxBytes    int64
	memory      bytes.Buffer
	file        *os.File
	directory   string
	size        int64
	closed      bool
}

func newBufferedStreamSpool(memoryLimit, maxBytes int64) (*bufferedStreamSpool, error) {
	if memoryLimit < 0 || maxBytes <= 0 || memoryLimit > maxBytes {
		return nil, errors.New("invalid buffered stream spool limits")
	}
	return &bufferedStreamSpool{memoryLimit: memoryLimit, maxBytes: maxBytes}, nil
}

func (spool *bufferedStreamSpool) Write(data []byte) (int, error) {
	if spool == nil || spool.closed {
		return 0, errors.New("buffered stream spool is closed")
	}
	if int64(len(data)) > spool.maxBytes-spool.size {
		return 0, fmt.Errorf("buffered stream response exceeds %d bytes", spool.maxBytes)
	}
	if !reserveBufferedStreamBytes(int64(len(data))) {
		return 0, fmt.Errorf("buffered stream total budget exceeds %d bytes", bufferedStreamTotalBudget)
	}
	written, writeErr := spool.writeReserved(data)
	if written < len(data) {
		bufferedStreamReservedBytes.Add(-int64(len(data) - written))
	}
	return written, writeErr
}

func (spool *bufferedStreamSpool) writeReserved(data []byte) (int, error) {
	if spool.file == nil && spool.size+int64(len(data)) <= spool.memoryLimit {
		written, err := spool.memory.Write(data)
		spool.size += int64(written)
		return written, err
	}
	if err := spool.spill(); err != nil {
		return 0, err
	}
	written, err := spool.file.Write(data)
	spool.size += int64(written)
	return written, err
}

func reserveBufferedStreamBytes(count int64) bool {
	if count < 0 {
		return false
	}
	for {
		current := bufferedStreamReservedBytes.Load()
		if current > bufferedStreamTotalBudget-count {
			return false
		}
		if bufferedStreamReservedBytes.CompareAndSwap(current, current+count) {
			return true
		}
	}
}

func (spool *bufferedStreamSpool) spill() error {
	if spool.file != nil {
		return nil
	}
	directory, err := os.MkdirTemp("", "gpt-load-buffer-")
	if err != nil {
		return fmt.Errorf("create buffered stream spool directory: %w", err)
	}
	if err := os.Chmod(directory, 0700); err != nil {
		_ = os.RemoveAll(directory)
		return fmt.Errorf("protect buffered stream spool directory: %w", err)
	}
	file, err := os.OpenFile(filepath.Join(directory, "payload"), os.O_CREATE|os.O_EXCL|os.O_RDWR, 0600)
	if err != nil {
		_ = os.RemoveAll(directory)
		return fmt.Errorf("create buffered stream spool file: %w", err)
	}
	// Keep the descriptor, not a pathname. A failed request cannot leave the
	// payload name visible to backup or static-file scanners.
	if err := os.Remove(file.Name()); err != nil {
		_ = file.Close()
		_ = os.RemoveAll(directory)
		return fmt.Errorf("unlink buffered stream spool file: %w", err)
	}
	if _, err := bufferedStreamSpoolFileWrite(file, spool.memory.Bytes()); err != nil {
		_ = file.Close()
		_ = os.RemoveAll(directory)
		return fmt.Errorf("spill buffered stream payload: %w", err)
	}
	spool.file = file
	spool.directory = directory
	spool.memory.Reset()
	return nil
}

func (spool *bufferedStreamSpool) ReplayTo(writer io.Writer) (int64, error) {
	if spool == nil || spool.closed || writer == nil {
		return 0, errors.New("buffered stream spool is not readable")
	}
	if spool.file == nil {
		written, err := writer.Write(spool.memory.Bytes())
		return int64(written), err
	}
	if _, err := spool.file.Seek(0, io.SeekStart); err != nil {
		return 0, fmt.Errorf("seek buffered stream spool: %w", err)
	}
	written, err := io.CopyBuffer(writer, spool.file, make([]byte, 32*1024))
	return written, err
}

func (spool *bufferedStreamSpool) Close() error {
	if spool == nil || spool.closed {
		return nil
	}
	spool.closed = true
	bufferedStreamReservedBytes.Add(-spool.size)
	var closeErr error
	if spool.file != nil {
		if err := spool.file.Close(); closeErr == nil {
			closeErr = err
		}
	}
	if spool.directory != "" {
		if err := os.RemoveAll(spool.directory); closeErr == nil {
			closeErr = err
		}
	}
	spool.file = nil
	spool.directory = ""
	spool.size = 0
	spool.memory.Reset()
	return closeErr
}

func (spool *bufferedStreamSpool) Spilled() bool {
	return spool != nil && spool.file != nil
}

func (spool *bufferedStreamSpool) Size() int64 {
	if spool == nil {
		return 0
	}
	return spool.size
}

type bufferedStreamCaptureWriter struct {
	header http.Header
	status int
	spool  *bufferedStreamSpool
}

func newBufferedStreamCaptureWriter(spool *bufferedStreamSpool) *bufferedStreamCaptureWriter {
	return &bufferedStreamCaptureWriter{header: make(http.Header), spool: spool}
}

func (writer *bufferedStreamCaptureWriter) Header() http.Header    { return writer.header }
func (writer *bufferedStreamCaptureWriter) WriteHeader(status int) { writer.status = status }
func (writer *bufferedStreamCaptureWriter) Write(data []byte) (int, error) {
	return writer.spool.Write(data)
}
func (*bufferedStreamCaptureWriter) FlushError() error { return nil }

func supportsBufferedStreamProtocol(value protocol.Protocol) bool {
	switch value {
	case protocol.OpenAICompletions, protocol.OpenAIResponses, protocol.Anthropic:
		return true
	default:
		return false
	}
}

func bufferedStreamReplayEligible(request *dialect.ParsedRequest, selected dialect.Dialect) bool {
	if request == nil || selected == nil || request.Method != http.MethodPost ||
		(request.Path != "/v1/chat/completions" && request.Path != "/v1/responses" && request.Path != "/v1/messages") {
		return false
	}
	var object map[string]any
	if err := json.Unmarshal(request.Body, &object); err != nil || object == nil {
		return false
	}
	if selected.Protocol() == protocol.OpenAIResponses {
		return dialect.ResponsesReplayEligible(request.Body)
	}
	tools, exists := object["tools"]
	if !exists {
		return true
	}
	values, ok := tools.([]any)
	if !ok {
		return false
	}
	for _, raw := range values {
		tool, ok := raw.(map[string]any)
		if !ok {
			return false
		}
		if selected.Protocol() == protocol.Anthropic {
			if _, hasType := tool["type"]; hasType {
				return false
			}
			if _, hasName := tool["name"]; !hasName {
				return false
			}
			continue
		}
		typeName, _ := tool["type"].(string)
		if typeName != "function" && typeName != "custom" {
			return false
		}
	}
	return true
}

func meaningfulJSONField(object map[string]any, key string) bool {
	value, exists := object[key]
	if !exists || value == nil {
		return false
	}
	text, ok := value.(string)
	return !ok || strings.TrimSpace(text) != ""
}

func jsonBoolValue(value any, want bool) bool {
	parsed, ok := value.(bool)
	return ok && parsed == want
}
