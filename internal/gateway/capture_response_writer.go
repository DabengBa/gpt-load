package gateway

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"

	"github.com/gin-gonic/gin"
)

type captureResponseWriter struct {
	gin.ResponseWriter
	attempt   CaptureAttempt
	recordErr func(error)
	committed bool
	mu        sync.Mutex
}

func newCaptureResponseWriter(writer gin.ResponseWriter, attempt CaptureAttempt, recordErr ...func(error)) *captureResponseWriter {
	var onError func(error)
	if len(recordErr) > 0 {
		onError = recordErr[0]
	}
	return &captureResponseWriter{ResponseWriter: writer, attempt: attempt, recordErr: onError}
}

func (writer *captureResponseWriter) captureError(err error) {
	if writer != nil && writer.recordErr != nil && err != nil {
		writer.recordErr(err)
	}
}

func (writer *captureResponseWriter) commit(status int) {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	if writer.committed {
		return
	}
	writer.committed = true
	if writer.attempt == nil {
		return
	}
	data := append([]byte("HTTP "+strconv.Itoa(status)+"\r\n"), serializeHeaders(writer.Header())...)
	if err := writer.attempt.AppendResponseHeaders(data); err != nil {
		writer.captureError(fmt.Errorf("capture response headers: %w", err))
	}
}

func (writer *captureResponseWriter) WriteHeader(statusCode int) {
	writer.ResponseWriter.WriteHeader(statusCode)
	writer.commit(statusCode)
}

func (writer *captureResponseWriter) WriteHeaderNow() {
	// Gin's writer owns commit semantics; observe the committed state only
	// after delegating, without synthesizing a second WriteHeader call.
	writer.ResponseWriter.WriteHeaderNow()
	status := writer.ResponseWriter.Status()
	if status == 0 {
		status = http.StatusOK
	}
	writer.commit(status)
}

func (writer *captureResponseWriter) committedValue() bool {
	writer.mu.Lock()
	defer writer.mu.Unlock()
	return writer.committed
}

func (writer *captureResponseWriter) Write(data []byte) (int, error) {
	if !writer.committedValue() {
		writer.WriteHeader(http.StatusOK)
	}
	n, err := writer.ResponseWriter.Write(data)
	if n > 0 && writer.attempt != nil {
		written := n
		if written > len(data) {
			written = len(data)
		}
		if appendErr := writer.attempt.AppendResponseBody(append([]byte(nil), data[:written]...)); appendErr != nil {
			writer.captureError(fmt.Errorf("capture response body: %w", appendErr))
		}
	}
	if events, ok := writer.attempt.(captureResponseEvents); ok {
		if n != len(data) {
			if eventErr := events.RecordResponseShortWrite(n, len(data)); eventErr != nil {
				writer.captureError(eventErr)
			}
		}
		if err != nil {
			if eventErr := events.RecordResponseError(err); eventErr != nil {
				writer.captureError(eventErr)
			}
		}
	}
	if n != len(data) {
		writer.captureError(fmt.Errorf("capture response short write: %w", io.ErrShortWrite))
	}
	if err != nil {
		writer.captureError(fmt.Errorf("capture response write: %w", err))
	}
	return n, err
}

func (writer *captureResponseWriter) WriteString(value string) (int, error) {
	return writer.Write([]byte(value))
}

func (writer *captureResponseWriter) Flush() {
	_ = writer.FlushError()
}

func (writer *captureResponseWriter) FlushError() error {
	if !writer.committedValue() {
		writer.WriteHeader(http.StatusOK)
	}
	err := http.NewResponseController(writer.ResponseWriter).Flush()
	if events, ok := writer.attempt.(captureResponseEvents); ok {
		if err == nil {
			if eventErr := events.RecordResponseFlush(); eventErr != nil {
				writer.captureError(eventErr)
			}
		} else {
			if eventErr := events.RecordResponseError(err); eventErr != nil {
				writer.captureError(eventErr)
			}
			writer.captureError(fmt.Errorf("capture response flush: %w", err))
		}
	}
	return err
}

func (writer *captureResponseWriter) ReadFrom(reader io.Reader) (int64, error) {
	buffer := make([]byte, 32*1024)
	var total int64
	for {
		read, readErr := reader.Read(buffer)
		if read > 0 {
			written, writeErr := writer.Write(buffer[:read])
			total += int64(written)
			if writeErr != nil {
				writer.captureError(fmt.Errorf("capture response write: %w", writeErr))
				return total, writeErr
			}
			if written != read {
				writer.captureError(fmt.Errorf("capture response short write: %w", io.ErrShortWrite))
				return total, io.ErrShortWrite
			}
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				return total, nil
			}
			writer.captureError(fmt.Errorf("capture response read: %w", readErr))
			return total, readErr
		}
	}
}

func (writer *captureResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	hijacker, ok := writer.ResponseWriter.(http.Hijacker)
	if !ok {
		err := http.ErrNotSupported
		writer.recordHijackOutcome(err)
		return nil, nil, err
	}
	conn, buffered, err := hijacker.Hijack()
	writer.recordHijackOutcome(err)
	return conn, buffered, err
}

func (writer *captureResponseWriter) recordHijackOutcome(err error) {
	events, supported := writer.attempt.(captureResponseHijackEvents)
	if supported {
		if eventErr := events.RecordResponseHijack(err); eventErr != nil {
			writer.captureError(eventErr)
		}
	} else if err == nil {
		writer.captureError(errors.New("capture target cannot persist successful response hijack"))
	}
	if err != nil {
		if events, ok := writer.attempt.(captureResponseEvents); ok {
			if eventErr := events.RecordResponseError(err); eventErr != nil {
				writer.captureError(eventErr)
			}
		}
		writer.captureError(fmt.Errorf("capture response hijack: %w", err))
	}
}

func (writer *captureResponseWriter) CloseNotify() <-chan bool {
	if notifier, ok := writer.ResponseWriter.(http.CloseNotifier); ok {
		return notifier.CloseNotify()
	}
	return make(chan bool)
}

func (writer *captureResponseWriter) Pusher() http.Pusher {
	if pusher, ok := writer.ResponseWriter.(http.Pusher); ok {
		return pusher
	}
	return nil
}

func (writer *captureResponseWriter) Unwrap() http.ResponseWriter {
	return writer.ResponseWriter
}

var _ gin.ResponseWriter = (*captureResponseWriter)(nil)
var _ io.ReaderFrom = (*captureResponseWriter)(nil)
var _ http.Flusher = (*captureResponseWriter)(nil)
var _ http.Hijacker = (*captureResponseWriter)(nil)
