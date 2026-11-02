package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"gpt-load/internal/protocol"
)

func imagesCredentialLiteralsRemain(body []byte, secrets []string) bool {
	replacers, exists := newCredentialLiteralReplacers(secrets)
	if !exists {
		return false
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return credentialLiteralRemains(string(body), replacers.residual)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return credentialLiteralRemains(string(body), replacers.residual)
	}
	return imagesJSONCredentialLiteralsRemain(value, replacers.residual)
}

func embeddingsCredentialLiteralsRemain(body []byte, secrets []string) bool {
	replacers, exists := newCredentialLiteralReplacers(secrets)
	if !exists {
		return false
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	remains, err := scanEmbeddingsCredentialValue(
		decoder,
		replacers.residual,
		embeddingsJSONRoot,
	)
	if err != nil || remains {
		return true
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return true
	}
	return false
}

func opaqueCredentialLiteralsRemain(
	clientProtocol protocol.Protocol,
	body []byte,
	secrets []string,
) bool {
	if clientProtocol == protocol.Rerank {
		return rerankCredentialLiteralsRemain(body, secrets)
	}
	if clientProtocol == protocol.OpenAIEmbeddings {
		return embeddingsCredentialLiteralsRemain(body, secrets)
	}
	if clientProtocol == protocol.GeminiEmbeddings {
		// Gemini 的向量只有数字：只流式扫描字符串值和键，不为向量构建整棵值树。
		return rerankCredentialLiteralsRemain(body, secrets)
	}
	return imagesCredentialLiteralsRemain(body, secrets)
}

// embeddingsProtocol 标识响应体以大向量为主、需要零拷贝处理的 embedding 协议。
func embeddingsProtocol(value protocol.Protocol) bool {
	return value == protocol.OpenAIEmbeddings || value == protocol.GeminiEmbeddings
}

type embeddingsJSONLocation uint8

const (
	embeddingsJSONRoot embeddingsJSONLocation = iota
	embeddingsJSONDataArray
	embeddingsJSONDataItem
	embeddingsJSONOpaqueVector
	embeddingsJSONOther
)

// scanEmbeddingsCredentialValue walks one JSON value without materializing the
// potentially large vector. Root data[*].embedding stays opaque across scalar
// and array tokens; object subtrees and same-named extension fields remain
// subject to exact credential inspection.
func scanEmbeddingsCredentialValue(
	decoder *json.Decoder,
	residual *strings.Replacer,
	location embeddingsJSONLocation,
) (bool, error) {
	token, err := decoder.Token()
	if err != nil {
		return false, err
	}
	switch typed := token.(type) {
	case json.Delim:
		switch typed {
		case '{':
			if location == embeddingsJSONOpaqueVector {
				location = embeddingsJSONOther
			}
			dataSeen := false
			embeddingSeen := false
			for decoder.More() {
				keyToken, keyErr := decoder.Token()
				if keyErr != nil {
					return false, keyErr
				}
				key, ok := keyToken.(string)
				if !ok {
					return false, fmt.Errorf("JSON object key is not a string")
				}
				if location == embeddingsJSONRoot && key == "data" {
					if dataSeen {
						return false, fmt.Errorf("duplicate root data field")
					}
					dataSeen = true
				}
				valueIsVector := location == embeddingsJSONDataItem && key == "embedding"
				if valueIsVector {
					if embeddingSeen {
						return false, fmt.Errorf("duplicate embedding field")
					}
					embeddingSeen = true
				}
				if location != embeddingsJSONOpaqueVector && !valueIsVector &&
					credentialLiteralRemains(key, residual) {
					return true, nil
				}
				nextLocation := embeddingsJSONOther
				switch {
				case location == embeddingsJSONRoot && key == "data":
					nextLocation = embeddingsJSONDataArray
				case valueIsVector:
					nextLocation = embeddingsJSONOpaqueVector
				case location == embeddingsJSONOpaqueVector:
					nextLocation = embeddingsJSONOpaqueVector
				}
				remains, valueErr := scanEmbeddingsCredentialValue(
					decoder,
					residual,
					nextLocation,
				)
				if valueErr != nil || remains {
					return remains, valueErr
				}
			}
			end, endErr := decoder.Token()
			if endErr != nil || end != json.Delim('}') {
				return false, fmt.Errorf("close JSON object")
			}
		case '[':
			for decoder.More() {
				nextLocation := embeddingsJSONOther
				if location == embeddingsJSONDataArray {
					nextLocation = embeddingsJSONDataItem
				} else if location == embeddingsJSONOpaqueVector {
					nextLocation = embeddingsJSONOpaqueVector
				}
				remains, valueErr := scanEmbeddingsCredentialValue(decoder, residual, nextLocation)
				if valueErr != nil || remains {
					return remains, valueErr
				}
			}
			end, endErr := decoder.Token()
			if endErr != nil || end != json.Delim(']') {
				return false, fmt.Errorf("close JSON array")
			}
		default:
			return false, fmt.Errorf("unexpected JSON delimiter")
		}
	case string:
		if location != embeddingsJSONOpaqueVector && credentialLiteralRemains(typed, residual) {
			return true, nil
		}
	case json.Number, bool, nil:
		// Credentials are JSON strings. Numeric vector and usage values must not
		// collide with short compatible-provider secrets such as "1".
	default:
		return false, fmt.Errorf("unexpected JSON token")
	}
	return false, nil
}

func imagesJSONCredentialLiteralsRemain(value any, residual *strings.Replacer) bool {
	switch typed := value.(type) {
	case string:
		return credentialLiteralRemains(typed, residual)
	case []any:
		for _, item := range typed {
			if imagesJSONCredentialLiteralsRemain(item, residual) {
				return true
			}
		}
	case map[string]any:
		for key, item := range typed {
			if key == "b64_json" {
				if _, isString := item.(string); isString {
					continue
				}
			}
			if imagesJSONCredentialLiteralsRemain(item, residual) {
				return true
			}
		}
	}
	return false
}

func jsonCredentialLiteralsRemain(value any, residual *strings.Replacer) bool {
	switch typed := value.(type) {
	case string:
		return credentialLiteralRemains(typed, residual)
	case json.Number:
		return credentialLiteralRemains(typed.String(), residual)
	case bool:
		return credentialLiteralRemains(fmt.Sprintf("%t", typed), residual)
	case nil:
		return credentialLiteralRemains("null", residual)
	case []any:
		for _, item := range typed {
			if jsonCredentialLiteralsRemain(item, residual) {
				return true
			}
		}
		return false
	case map[string]any:
		for key, item := range typed {
			if credentialLiteralRemains(key, residual) ||
				jsonCredentialLiteralsRemain(item, residual) {
				return true
			}
		}
		return false
	default:
		return true
	}
}

type boundedCredentialBuffer struct {
	buffer bytes.Buffer
	limit  int64
}

func (writer *boundedCredentialBuffer) Write(value []byte) (int, error) {
	if writer == nil || writer.limit < 0 ||
		int64(writer.buffer.Len()) > writer.limit-int64(len(value)) {
		return 0, fmt.Errorf("credential replacement exceeds response limit")
	}
	return writer.buffer.Write(value)
}

// Rerank 文档保持原文，只检查已知凭据；逐 token 检查也覆盖重复字段的值。
func rerankCredentialLiteralsRemain(body []byte, secrets []string) bool {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 || trimmed[0] != '{' || !json.Valid(trimmed) {
		return true
	}
	replacers, exists := newCredentialLiteralReplacers(secrets)
	if !exists {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(trimmed))
	decoder.UseNumber()
	for {
		token, err := decoder.Token()
		if err == io.EOF {
			return false
		}
		if err != nil {
			return true
		}
		if value, ok := token.(string); ok && credentialLiteralRemains(value, replacers.residual) {
			return true
		}
	}
}
