package catalog

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"gpt-load/internal/pricing"
)

const maxModelIDBytes = 255

var (
	providerCanonicalFields = []string{"id", "name", "api", "npm", "models"}
	modelCanonicalFields    = []string{
		"id", "name", "description", "family", "attachment", "reasoning",
		"tool_call", "structured_output", "temperature", "knowledge",
		"release_date", "last_updated", "modalities", "open_weights", "limit",
		"status", "cost", "experimental",
	}
	modalityCanonicalFields     = []string{"input", "output"}
	limitCanonicalFields        = []string{"context", "input", "output"}
	costCanonicalFields         = []string{"input", "output", "cache_read", "cache_write", "tiers"}
	experimentalCanonicalFields = []string{"modes"}
	modeCanonicalFields         = []string{"cost", "provider"}
	tierCanonicalFields         = []string{"tier", "input", "output", "cache_read", "cache_write"}
	tierRuleCanonicalFields     = []string{"type", "size"}
)

// Parse decodes, validates, and normalizes the retained Models.dev-compatible subset.
func Parse(reader io.Reader) (*Snapshot, error) {
	if reader == nil {
		return nil, fmt.Errorf("catalog reader is required")
	}
	decoder := json.NewDecoder(reader)
	decoder.UseNumber()
	providerValues, err := decodeUniqueObject(decoder, "provider")
	if err != nil {
		return nil, err
	}
	if err := requireJSONEOF(decoder); err != nil {
		return nil, err
	}

	snapshot := &Snapshot{Providers: make(map[string]Provider, len(providerValues))}
	for providerKey, rawProvider := range providerValues {
		provider, err := parseProvider(providerKey, rawProvider)
		if err != nil {
			return nil, fmt.Errorf("provider %q: %w", providerKey, err)
		}
		snapshot.Providers[providerKey] = provider
	}
	if err := validateSnapshot(snapshot); err != nil {
		return nil, err
	}
	return snapshot, nil
}

func parseProvider(key string, raw json.RawMessage) (Provider, error) {
	fields, err := decodeCanonicalObject(raw, "provider", providerCanonicalFields)
	if err != nil {
		return Provider{}, err
	}
	id, err := decodeStringField(fields, "id")
	if err != nil {
		return Provider{}, err
	}
	if id != key {
		return Provider{}, fmt.Errorf("map key and provider id must match")
	}
	if _, err := pricing.ProviderScopeKey(id); err != nil {
		return Provider{}, err
	}
	nameValue, err := decodeStringField(fields, "name")
	if err != nil {
		return Provider{}, err
	}
	name := strings.TrimSpace(nameValue)
	if name == "" {
		return Provider{}, fmt.Errorf("provider name is required")
	}
	modelsRaw, hasModels := fields["models"]
	if !hasModels || bytes.Equal(bytes.TrimSpace(modelsRaw), []byte("null")) {
		return Provider{}, fmt.Errorf("provider models object is required")
	}
	decoder := json.NewDecoder(bytes.NewReader(modelsRaw))
	decoder.UseNumber()
	modelValues, err := decodeUniqueObject(decoder, "model")
	if err != nil {
		return Provider{}, err
	}
	if err := requireJSONEOF(decoder); err != nil {
		return Provider{}, err
	}

	if _, err := decodeStringField(fields, "api"); err != nil {
		return Provider{}, err
	}
	if _, err := decodeStringField(fields, "npm"); err != nil {
		return Provider{}, err
	}
	provider := Provider{
		ID: id, Name: name, Models: make(map[string]Model, len(modelValues)),
	}
	for modelKey, rawModel := range modelValues {
		model, err := parseModel(modelKey, rawModel)
		if err != nil {
			return Provider{}, fmt.Errorf("model %q: %w", modelKey, err)
		}
		provider.Models[modelKey] = model
	}
	return provider, nil
}

func parseModel(key string, raw json.RawMessage) (Model, error) {
	fields, err := decodeCanonicalObject(raw, "model", modelCanonicalFields)
	if err != nil {
		return Model{}, err
	}
	id, err := decodeStringField(fields, "id")
	if err != nil {
		return Model{}, err
	}
	if id != key {
		return Model{}, fmt.Errorf("map key and model id must match")
	}
	if err := validateModelID(id); err != nil {
		return Model{}, err
	}
	nameValue, err := decodeStringField(fields, "name")
	if err != nil {
		return Model{}, err
	}
	name := strings.TrimSpace(nameValue)
	if name == "" {
		return Model{}, fmt.Errorf("model name is required")
	}
	metadata, err := parseModelMetadata(fields)
	if err != nil {
		return Model{}, err
	}
	model := Model{ID: id, Name: name, Metadata: metadata}
	var cost *ModelCost
	costRaw, hasCost := fields["cost"]
	if hasCost && !bytes.Equal(bytes.TrimSpace(costRaw), []byte("null")) {
		cost, err = parseCost(costRaw)
		if err != nil {
			return Model{}, err
		}
	}
	modePrices, err := parseExperimentalModePrices(fields["experimental"])
	if err != nil {
		return Model{}, err
	}
	if len(modePrices) > 0 {
		if cost == nil {
			cost = &ModelCost{}
		}
		cost.ModePrices = modePrices
	}
	model.Cost = cost
	return model, nil
}

func parseExperimentalModePrices(raw json.RawMessage) (map[pricing.Mode]pricing.Prices, error) {
	if len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
		return nil, nil
	}
	fields, err := decodeCanonicalObject(raw, "model experimental", experimentalCanonicalFields)
	if err != nil {
		return nil, err
	}
	modesRaw, exists := fields["modes"]
	if !exists || bytes.Equal(bytes.TrimSpace(modesRaw), []byte("null")) {
		return nil, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(modesRaw))
	decoder.UseNumber()
	modes, err := decodeUniqueObject(decoder, "experimental modes")
	if err != nil {
		return nil, err
	}
	if err := requireJSONEOF(decoder); err != nil {
		return nil, err
	}

	result := make(map[pricing.Mode]pricing.Prices)
	for key, rawMode := range modes {
		mode := pricing.Mode(key)
		if !mode.Valid() || mode == pricing.ModeStandard {
			return nil, fmt.Errorf("invalid experimental pricing mode %q", key)
		}
		modeFields, err := decodeCanonicalObject(rawMode, "experimental mode", modeCanonicalFields)
		if err != nil {
			return nil, fmt.Errorf("mode %q: %w", key, err)
		}
		rawCost, exists := modeFields["cost"]
		if !exists || bytes.Equal(bytes.TrimSpace(rawCost), []byte("null")) {
			continue
		}
		prices, err := parseModePrices(rawCost)
		if err != nil {
			return nil, fmt.Errorf("mode %q: %w", key, err)
		}
		if hasSetPrice(prices) {
			result[mode] = prices
		}
	}
	if len(result) == 0 {
		return nil, nil
	}
	return result, nil
}
