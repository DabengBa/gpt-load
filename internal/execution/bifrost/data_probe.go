package bifrost

import (
	"bytes"
	"encoding/json"
	"gpt-load/internal/protocol"
	"math"
	"strconv"
)

// validDataProbeResponse verifies the one input's vector or ranking directly.
func validDataProbeResponse(selected protocol.Protocol, body []byte) bool {
	var root struct {
		Embedding struct {
			Values json.RawMessage `json:"values"`
		} `json:"embedding"`
		Data []struct {
			Index     int             `json:"index"`
			Embedding json.RawMessage `json:"embedding"`
		} `json:"data"`
		Results []struct {
			Index *int            `json:"index"`
			Score json.RawMessage `json:"relevance_score"`
		} `json:"results"`
		Error json.RawMessage `json:"error"`
	}
	if json.Unmarshal(body, &root) != nil || len(root.Error) > 0 && !bytes.Equal(root.Error, []byte("null")) {
		return false
	}
	if selected == protocol.Rerank {
		if len(root.Results) != 1 || root.Results[0].Index == nil || *root.Results[0].Index != 0 {
			return false
		}
		if !validProbeNumber(root.Results[0].Score) {
			return false
		}
		score, _ := strconv.ParseFloat(string(root.Results[0].Score), 64)
		return score >= 0 && score <= 1
	}
	vector := root.Embedding.Values
	if selected == protocol.OpenAIEmbeddings {
		if len(root.Data) != 1 || root.Data[0].Index != 0 {
			return false
		}
		vector = root.Data[0].Embedding
	}
	var values []json.RawMessage
	if json.Unmarshal(vector, &values) != nil || len(values) == 0 {
		return false
	}
	for _, value := range values {
		if !validProbeNumber(value) {
			return false
		}
	}
	return true
}

func validProbeNumber(raw []byte) bool {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || raw[0] != '-' && (raw[0] < '0' || raw[0] > '9') {
		return false
	}
	value, err := strconv.ParseFloat(string(raw), 64)
	return err == nil && !math.IsInf(value, 0) && !math.IsNaN(value)
}
