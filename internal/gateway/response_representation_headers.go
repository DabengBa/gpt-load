package gateway

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

func hasSingleHeaderFieldValue(headers http.Header, name, want string) bool {
	values := headerFieldValues(headers, name)
	return len(values) == 1 && values[0] == want
}

func hasSafeContentRange(headers http.Header, bodyLength int) bool {
	if bodyLength <= 0 {
		return false
	}
	rangeValue, ok := parseSatisfiedContentRange(headers)
	if !ok {
		return false
	}
	return rangeValue.end-rangeValue.start == int64(bodyLength)-1
}

func hasSafeHeadContentRange(headers http.Header) bool {
	rangeValue, ok := parseSatisfiedContentRange(headers)
	if !ok {
		return false
	}
	values := headerFieldValues(headers, "Content-Length")
	if len(values) == 0 {
		return true
	}
	contentLength, ok := singleNonNegativeContentLength(headers)
	if !ok {
		return false
	}
	length, err := strconv.ParseInt(contentLength, 10, 64)
	if err != nil || length <= 0 {
		return false
	}
	return rangeValue.end-rangeValue.start == length-1
}

type satisfiedContentRange struct {
	start int64
	end   int64
}

func parseSatisfiedContentRange(headers http.Header) (satisfiedContentRange, bool) {
	values := headerFieldValues(headers, "Content-Range")
	if len(values) != 1 {
		return satisfiedContentRange{}, false
	}
	parts := strings.Fields(values[0])
	if len(parts) != 2 || !strings.EqualFold(parts[0], "bytes") {
		return satisfiedContentRange{}, false
	}
	rangeParts := strings.Split(parts[1], "/")
	if len(rangeParts) != 2 {
		return satisfiedContentRange{}, false
	}
	bounds := strings.Split(rangeParts[0], "-")
	if len(bounds) != 2 {
		return satisfiedContentRange{}, false
	}
	if !isDecimalContentRangeNumber(bounds[0]) || !isDecimalContentRangeNumber(bounds[1]) {
		return satisfiedContentRange{}, false
	}
	start, startErr := strconv.ParseInt(bounds[0], 10, 64)
	end, endErr := strconv.ParseInt(bounds[1], 10, 64)
	if startErr != nil || endErr != nil || start < 0 || end < start {
		return satisfiedContentRange{}, false
	}
	if rangeParts[1] == "*" {
		return satisfiedContentRange{start: start, end: end}, true
	}
	if !isDecimalContentRangeNumber(rangeParts[1]) {
		return satisfiedContentRange{}, false
	}
	total, totalErr := strconv.ParseInt(rangeParts[1], 10, 64)
	if totalErr != nil || total <= end {
		return satisfiedContentRange{}, false
	}
	return satisfiedContentRange{start: start, end: end}, true
}

func isDecimalContentRangeNumber(value string) bool {
	if value == "" {
		return false
	}
	for _, character := range value {
		if character < '0' || character > '9' {
			return false
		}
	}
	return true
}

func representationMetadataContainsSecrets(headers http.Header, secrets []string) bool {
	for name, values := range headers {
		if !isRepresentationMetadataHeader(name) {
			continue
		}
		for _, secret := range secrets {
			if headerValuesContainLiteral(values, secret) {
				return true
			}
		}
	}
	return false
}

func successRepresentationProtocolError(stage string) error {
	return fmt.Errorf("%w: invalid success response representation at %s", ErrUpstreamProtocol, stage)
}
