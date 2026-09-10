package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/protocol"
)

func (handler *Handler) writeBufferedStreamError(context *gin.Context, value protocol.Protocol, responseID string) error {
	if context == nil {
		return errors.New("buffered stream context is required")
	}
	if context.Request == nil || context.Request.Context().Err() != nil {
		if context.Request == nil {
			return errors.New("buffered stream request is required")
		}
		return context.Request.Context().Err()
	}
	var payload struct {
		Type  string `json:"type"`
		Error struct {
			Type    string `json:"type"`
			Message string `json:"message"`
			Code    string `json:"code,omitempty"`
		} `json:"error"`
		Response *struct {
			ID string `json:"id"`
		} `json:"response,omitempty"`
	}
	payload.Type = "error"
	payload.Error.Type = "server_error"
	payload.Error.Message = "The buffered upstream stream could not be completed."
	payload.Error.Code = "buffered_stream_failed"
	if responseID != "" {
		payload.Response = &struct {
			ID string `json:"id"`
		}{ID: responseID}
		payload.Error.Message = "The buffered upstream stream could not be completed."
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	event := "event: error\ndata: " + string(encoded) + "\n\n"
	if value == protocol.OpenAICompletions {
		event = "data: {\"error\":{\"type\":\"server_error\",\"message\":\"The buffered upstream stream could not be completed.\",\"code\":\"buffered_stream_failed\"}}\n\n"
	}
	writer := newStreamWriteController(context.Writer, downstreamWriteTimeout)
	defer func() { _ = writer.clear() }()
	written, err := writer.write([]byte(event))
	if err != nil {
		return fmt.Errorf("write buffered stream error: %w", err)
	}
	if written != len(event) {
		return fmt.Errorf("write buffered stream error: %w", io.ErrShortWrite)
	}
	if err := writer.flush(); err != nil && !errors.Is(err, http.ErrNotSupported) {
		return fmt.Errorf("flush buffered stream error: %w", err)
	}
	return nil
}
