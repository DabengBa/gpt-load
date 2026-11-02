package control

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"

	app_errors "gpt-load/internal/platform/errors"
)

const maxControlJSONBodyBytes int64 = 32 << 20

func bindStrictJSON(c *gin.Context, target any) error {
	if c.Request.ContentLength > maxControlJSONBodyBytes {
		return &http.MaxBytesError{Limit: maxControlJSONBodyBytes}
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxControlJSONBodyBytes)
	raw, err := io.ReadAll(c.Request.Body)
	if err != nil {
		return err
	}
	return decodeStrictControlJSONObject(raw, target)
}

func bindOptionalEmptyJSONObject(c *gin.Context) error {
	decoder, err := newControlJSONDecoder(c)
	if err != nil {
		return err
	}
	var object map[string]json.RawMessage
	if err := decoder.Decode(&object); err != nil {
		if errors.Is(err, io.EOF) {
			return nil
		}
		return err
	}
	if object == nil || len(object) != 0 {
		return fmt.Errorf("request body must be an empty JSON object")
	}
	var extra json.RawMessage
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("decode JSON request: multiple values")
		}
		return err
	}
	return nil
}

func newControlJSONDecoder(c *gin.Context) (*json.Decoder, error) {
	if c.Request.ContentLength > maxControlJSONBodyBytes {
		return nil, &http.MaxBytesError{Limit: maxControlJSONBodyBytes}
	}
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxControlJSONBodyBytes)
	return json.NewDecoder(c.Request.Body), nil
}

func mapControlJSONError(err error) *app_errors.APIError {
	if errors.Is(err, app_errors.ErrValidation) {
		return app_errors.ErrValidation
	}
	var maxBytesError *http.MaxBytesError
	if errors.As(err, &maxBytesError) {
		return app_errors.ErrRequestTooLarge
	}
	return app_errors.ErrInvalidJSON
}

func requiredIdempotencyKey(c *gin.Context, operation string) (string, bool) {
	values := c.Request.Header.Values("Idempotency-Key")
	if len(values) == 0 || len(values) == 1 && values[0] == "" {
		writeServiceError(c, operation, app_errors.ErrIdempotencyKeyRequired)
		return "", false
	}
	if len(values) != 1 || validateIdempotencyKey(values[0]) != nil {
		writeServiceError(c, operation, app_errors.ErrInvalidIdempotencyKey)
		return "", false
	}
	return values[0], true
}

func setSecretResponseHeaders(c *gin.Context) {
	c.Header("Cache-Control", "no-store")
	c.Header("Pragma", "no-cache")
}

func positiveIDParam(c *gin.Context, param, operation string) (uint, bool) {
	parsed, err := strconv.ParseUint(c.Param(param), 10, strconv.IntSize)
	if err != nil || parsed == 0 {
		writeServiceError(c, operation, app_errors.ErrBadRequest)
		return 0, false
	}
	return uint(parsed), true
}

func accessKeyID(c *gin.Context) (uint, bool) { return positiveIDParam(c, "id", "access_key_id") }
func groupID(c *gin.Context, operation string) (uint, bool) {
	return positiveIDParam(c, "group_id", operation)
}
func credentialID(c *gin.Context, operation string) (uint, bool) {
	return positiveIDParam(c, "credential_id", operation)
}
