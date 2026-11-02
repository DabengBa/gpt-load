package gateway

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
)

type captureRequestBody struct {
	io.ReadCloser
	context          context.Context
	observe          func([]byte)
	recordReadError  func(error)
	recordCloseError func(error)
	recordFailure    func(error)
	recordOutcome    func(string, error)
	readOnce         sync.Once
	closeOnce        sync.Once
	cancelOnce       sync.Once
}

func (body *captureRequestBody) Read(p []byte) (int, error) {
	n, err := body.ReadCloser.Read(p)
	if n > 0 && body.observe != nil {
		body.observe(append([]byte(nil), p[:n]...))
	}
	if err != nil && !errors.Is(err, io.EOF) {
		body.readOnce.Do(func() {
			if body.recordReadError != nil {
				body.recordReadError(fmt.Errorf("request body read: %w", err))
			}
			if body.recordOutcome != nil {
				body.recordOutcome("read_error", err)
			}
		})
	}
	body.recordCancellation()
	return n, err
}

func (body *captureRequestBody) Close() error {
	err := body.ReadCloser.Close()
	if err != nil {
		body.closeOnce.Do(func() {
			if body.recordCloseError != nil {
				body.recordCloseError(fmt.Errorf("request body close: %w", err))
			}
			if body.recordOutcome != nil {
				body.recordOutcome("close_error", err)
			}
		})
	}
	body.recordCancellation()
	return err
}

func (body *captureRequestBody) recordCancellation() {
	if body.context == nil || body.context.Err() == nil {
		return
	}
	err := body.context.Err()
	body.cancelOnce.Do(func() {
		if body.recordFailure != nil {
			body.recordFailure(fmt.Errorf("request body context: %w", err))
		}
		if body.recordOutcome != nil {
			body.recordOutcome("context_canceled", err)
		}
	})
}
