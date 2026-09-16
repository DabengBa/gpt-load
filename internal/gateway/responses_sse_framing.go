package gateway

import (
	"bytes"
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"
	"errors"
	"fmt"
	"io"
	"unicode/utf8"
)

var errResponsesSSEFraming = errors.New("invalid OpenAI Responses SSE framing")

type responsesSSEFramingNormalizer struct {
	pending                []byte
	candidate              []byte
	candidateName          []byte
	maxEventBytes          int
	peakPendingBytes       int
	scanOffset             int
	scanLineStart          int
	scanProbes             int
	candidateCommentsValid bool
	failed                 bool
	finished               bool
}

func newResponsesSSEFramingNormalizer(maxEventBytes int) *responsesSSEFramingNormalizer {
	return &responsesSSEFramingNormalizer{
		maxEventBytes: normalizedSSEEventLimit(maxEventBytes),
	}
}

// push returns only complete output blocks. An event-only block is held until
// its immediately following data block proves that it can be merged.
func (normalizer *responsesSSEFramingNormalizer) push(chunk []byte) ([]byte, error) {
	if normalizer == nil {
		return nil, fmt.Errorf("Responses SSE framing normalizer is required")
	}
	if normalizer.failed {
		return nil, fmt.Errorf("Responses SSE framing normalizer is failed")
	}
	if normalizer.finished {
		return nil, fmt.Errorf("Responses SSE framing normalizer is finished")
	}

	var output bytes.Buffer
	for len(chunk) > 0 || len(normalizer.pending) > 0 {
		part, err := normalizer.consume(false)
		if err != nil {
			return nil, err
		}
		_, _ = output.Write(part)
		if len(chunk) == 0 {
			return output.Bytes(), nil
		}

		available := normalizer.maxEventBytes + 1 - len(normalizer.pending)
		if available <= 0 {
			return normalizer.fail(errSSEEventTooLarge)
		}
		take := min(len(chunk), available)
		normalizer.pending = append(normalizer.pending, chunk[:take]...)
		normalizer.peakPendingBytes = max(normalizer.peakPendingBytes, len(normalizer.pending))
		chunk = chunk[take:]
	}
	return output.Bytes(), nil
}

func (normalizer *responsesSSEFramingNormalizer) finish() ([]byte, error) {
	if normalizer == nil {
		return nil, fmt.Errorf("Responses SSE framing normalizer is required")
	}
	if normalizer.failed {
		return nil, fmt.Errorf("Responses SSE framing normalizer is failed")
	}
	if normalizer.finished {
		return nil, fmt.Errorf("Responses SSE framing normalizer is finished")
	}
	output, err := normalizer.consume(true)
	if err != nil {
		return nil, err
	}
	normalizer.finished = true
	return output, nil
}

func (normalizer *responsesSSEFramingNormalizer) consume(final bool) ([]byte, error) {
	var output bytes.Buffer
	for {
		if len(normalizer.candidate) > 0 {
			end, complete := normalizer.findBlock(final)
			if !complete {
				if final {
					return normalizer.fail(errSSEEventIncomplete)
				}
				if len(normalizer.pending) > normalizer.maxEventBytes {
					return normalizer.fail(errSSEEventTooLarge)
				}
				return output.Bytes(), nil
			}
			if end > normalizer.maxEventBytes {
				return normalizer.fail(errSSEEventTooLarge)
			}
			block := normalizer.pending[:end]
			parsed := parseResponsesSSEBlock(block)
			if parsed.kind != responsesSSEDataOnly {
				return normalizer.fail(fmt.Errorf(
					"%w: event block is not followed by a data-only block",
					errResponsesSSEFraming,
				))
			}
			if !normalizer.candidateCommentsValid || !parsed.commentsValid {
				return normalizer.fail(fmt.Errorf(
					"%w: recovery block contains invalid UTF-8 comment data",
					errResponsesSSEFraming,
				))
			}
			if err := parsed.validateForEvent(normalizer.candidateName); err != nil {
				return normalizer.fail(err)
			}
			candidateBody := normalizer.candidate[:len(normalizer.candidate)-sseBlockDelimiterSize(normalizer.candidate)]
			mergedSize := len(candidateBody) + len(block)
			if mergedSize > normalizer.maxEventBytes {
				return normalizer.fail(errSSEEventTooLarge)
			}
			output.Grow(output.Len() + mergedSize)
			_, _ = output.Write(candidateBody)
			_, _ = output.Write(block)
			normalizer.discard(end)
			normalizer.candidate = nil
			normalizer.candidateName = nil
			normalizer.candidateCommentsValid = true
			continue
		}

		end, complete := normalizer.findBlock(final)
		if !complete {
			if final {
				if len(normalizer.pending) > 0 {
					return normalizer.fail(errSSEEventIncomplete)
				}
				return output.Bytes(), nil
			}
			if len(normalizer.pending) > normalizer.maxEventBytes {
				return normalizer.fail(errSSEEventTooLarge)
			}
			return output.Bytes(), nil
		}
		if end > normalizer.maxEventBytes {
			return normalizer.fail(errSSEEventTooLarge)
		}

		block := bytes.Clone(normalizer.pending[:end])
		parsed := parseResponsesSSEBlock(block)
		if parsed.kind == responsesSSEInvalidEventOnly {
			return normalizer.fail(fmt.Errorf(
				"%w: malformed event-only block",
				errResponsesSSEFraming,
			))
		}
		normalizer.discard(end)
		if parsed.kind == responsesSSEEventOnly {
			normalizer.candidate = block
			normalizer.candidateName = bytes.Clone(parsed.eventName)
			normalizer.candidateCommentsValid = parsed.commentsValid
			continue
		}
		_, _ = output.Write(block)
	}
}

func (normalizer *responsesSSEFramingNormalizer) fail(err error) ([]byte, error) {
	normalizer.failed = true
	normalizer.pending = nil
	normalizer.candidate = nil
	normalizer.candidateName = nil
	normalizer.candidateCommentsValid = true
	normalizer.scanOffset = 0
	normalizer.scanLineStart = 0
	return nil, err
}

func (normalizer *responsesSSEFramingNormalizer) discard(count int) {
	normalizer.pending = normalizer.pending[count:]
	if len(normalizer.pending) == 0 {
		normalizer.pending = nil
	}
	normalizer.scanOffset = max(0, normalizer.scanOffset-count)
	normalizer.scanLineStart = max(0, normalizer.scanLineStart-count)
}

type responsesSSEBlockKind uint8

const (
	responsesSSEOrdinaryBlock responsesSSEBlockKind = iota
	responsesSSEEventOnly
	responsesSSEInvalidEventOnly
	responsesSSEDataOnly
)

type responsesSSEBlock struct {
	kind          responsesSSEBlockKind
	eventName     []byte
	dataValues    [][]byte
	commentsValid bool
}

func parseResponsesSSEBlock(block []byte) responsesSSEBlock {
	parsed := responsesSSEBlock{commentsValid: true}
	eventCount := 0
	dataCount := 0
	otherField := false
	for _, line := range splitSSEEventLines(block) {
		if len(line.content) == 0 {
			continue
		}
		if line.content[0] == ':' {
			parsed.commentsValid = parsed.commentsValid && utf8.Valid(line.content)
			continue
		}
		field, value := parseResponsesSSEField(line.content)
		switch field {
		case "event":
			eventCount++
			parsed.eventName = value
		case "data":
			dataCount++
			parsed.dataValues = append(parsed.dataValues, value)
		default:
			otherField = true
		}
	}

	switch {
	case dataCount > 0:
		parsed.kind = responsesSSEOrdinaryBlock
		if eventCount == 0 && !otherField {
			parsed.kind = responsesSSEDataOnly
		}
	case eventCount > 0:
		parsed.kind = responsesSSEInvalidEventOnly
		if eventCount == 1 && !otherField && len(parsed.eventName) > 0 && utf8.Valid(parsed.eventName) {
			parsed.kind = responsesSSEEventOnly
		}
	}
	return parsed
}

func parseResponsesSSEField(line []byte) (string, []byte) {
	field, value, hasColon := bytes.Cut(line, []byte{':'})
	if !hasColon {
		return string(field), nil
	}
	if len(value) > 0 && value[0] == ' ' {
		value = value[1:]
	}
	return string(field), value
}

func (block responsesSSEBlock) validateForEvent(eventName []byte) error {
	if len(block.dataValues) == 0 {
		return fmt.Errorf("%w: data block has no data", errResponsesSSEFraming)
	}
	for _, value := range block.dataValues {
		if len(value) == 0 {
			return fmt.Errorf("%w: data block contains empty data", errResponsesSSEFraming)
		}
	}
	payload := bytes.Join(block.dataValues, []byte{'\n'})
	if !utf8.Valid(payload) {
		return fmt.Errorf("%w: data payload is not UTF-8", errResponsesSSEFraming)
	}
	if bytes.Equal(bytes.TrimSpace(payload), []byte("[DONE]")) {
		return fmt.Errorf("%w: data payload is [DONE]", errResponsesSSEFraming)
	}
	typeName, err := responsesSSEObjectType(payload)
	if err != nil {
		return fmt.Errorf("%w: invalid data payload: %v", errResponsesSSEFraming, err)
	}
	if typeName != string(eventName) {
		return fmt.Errorf("%w: event name and payload type differ", errResponsesSSEFraming)
	}
	return nil
}

func responsesSSEObjectType(payload []byte) (string, error) {
	decoder := jsontext.NewDecoder(bytes.NewReader(payload), jsontext.AllowDuplicateNames(true))
	token, err := decoder.ReadToken()
	if err != nil || token.Kind() != jsontext.Kind('{') {
		return "", fmt.Errorf("payload is not a JSON object")
	}

	typeCount := 0
	typeName := ""
	for {
		token, err = decoder.ReadToken()
		if err != nil {
			return "", fmt.Errorf("read object member: %w", err)
		}
		if token.Kind() == jsontext.Kind('}') {
			break
		}
		if token.Kind() != jsontext.Kind('"') {
			return "", fmt.Errorf("object member name is not a string")
		}
		memberName := token.String()
		value, err := decoder.ReadValue()
		if err != nil {
			return "", fmt.Errorf("read object member value: %w", err)
		}
		if memberName != "type" {
			continue
		}
		typeCount++
		if typeCount > 1 {
			return "", fmt.Errorf("object has duplicate type members")
		}
		if err := jsonv2.Unmarshal(value, &typeName); err != nil {
			return "", fmt.Errorf("type member is not a string: %w", err)
		}
	}
	if typeCount != 1 {
		return "", fmt.Errorf("object has no unique type member")
	}

	if _, err := decoder.ReadToken(); !errors.Is(err, io.EOF) {
		if err == nil {
			return "", fmt.Errorf("payload has multiple JSON values")
		}
		return "", fmt.Errorf("trailing JSON data: %w", err)
	}
	return typeName, nil
}

// findBlock locates the end of the next complete SSE block. It resumes from the
// cursor left by the previous call so each fed byte is probed at most once; the
// only exception is a '\r' held at the end of the currently available data,
// which must be re-examined when more data arrives to decide whether it opens a
// CRLF terminator. scanLineStart tracks the start of the in-progress line so the
// resumption stays byte-for-byte equivalent to a full rescan.
func (normalizer *responsesSSEFramingNormalizer) findBlock(final bool) (int, bool) {
	data := normalizer.pending
	index := min(normalizer.scanOffset, len(data))
	lineStart := min(normalizer.scanLineStart, index)
	for index < len(data) {
		normalizer.scanProbes++
		switch data[index] {
		case '\n':
			index++
			blankLine := index-1 == lineStart
			lineStart = index
			if blankLine {
				normalizer.setScanCursor(index, lineStart)
				return index, true
			}
		case '\r':
			if index+1 == len(data) {
				if !final {
					normalizer.setScanCursor(index, lineStart)
					return 0, false
				}
				if index == lineStart {
					normalizer.setScanCursor(index+1, index+1)
					return index + 1, true
				}
				normalizer.setScanCursor(index, lineStart)
				return 0, false
			}
			end := index + 1
			if data[end] == '\n' {
				end++
			}
			blankLine := index == lineStart
			index = end
			lineStart = index
			if blankLine {
				normalizer.setScanCursor(index, lineStart)
				return index, true
			}
		default:
			index++
		}
	}
	normalizer.setScanCursor(index, lineStart)
	return 0, false
}

func (normalizer *responsesSSEFramingNormalizer) setScanCursor(offset, lineStart int) {
	normalizer.scanOffset = offset
	normalizer.scanLineStart = lineStart
}

func sseBlockDelimiterSize(block []byte) int {
	if bytes.HasSuffix(block, []byte("\r\n")) {
		return 2
	}
	return 1
}
