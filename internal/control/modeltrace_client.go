package control

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	app_errors "gpt-load/internal/platform/errors"
)

const (
	modelTraceChallengeCount = 3
	modelTraceMaxBodyBytes   = 4 << 20
	// modelTraceMinTokenLength mirrors the deployment's own token floor so a
	// mistyped configuration fails here instead of at the first request.
	modelTraceMinTokenLength   = 16
	defaultModelTraceTimeout   = 30 * time.Second
	maxModelTraceTimeout       = 5 * time.Minute
	modelTraceTruncatedMarker  = "...[truncated]"
	modelTraceMaxErrorFragment = 512
)

// ModelTraceConfig is the operator-supplied identity of one ModelTrace
// analysis deployment. A nil value means the feature is not configured.
type ModelTraceConfig struct {
	Enabled bool
	BaseURL string
	Token   string
	// RequestTimeout bounds one ModelTrace HTTP exchange. It never bounds the
	// upstream challenge requests, which keep the group's own timeouts.
	RequestTimeout time.Duration
}

// GenericModelTraceConfig is the process-level configuration holder. It is a
// single global value because ModelTrace is one operator-managed dependency,
// not per-group configuration.
type GenericModelTraceConfig struct {
	value atomic.Pointer[ModelTraceConfig]
}

// NewGenericModelTraceConfig creates the process configuration holder.
func NewGenericModelTraceConfig() *GenericModelTraceConfig {
	return &GenericModelTraceConfig{}
}

// Load returns the current configuration, or nil when the feature is absent.
func (holder *GenericModelTraceConfig) Load() *ModelTraceConfig {
	if holder == nil {
		return nil
	}
	return holder.value.Load()
}

// Publish replaces the process configuration. A nil value disables the feature.
func (holder *GenericModelTraceConfig) Publish(cfg *ModelTraceConfig) {
	if holder == nil {
		return
	}
	holder.value.Store(cfg)
}

var processModelTraceConfig = NewGenericModelTraceConfig()

// SetProcessModelTraceConfig installs the process-level ModelTrace deployment
// configuration. Passing nil disables the feature.
func SetProcessModelTraceConfig(cfg *ModelTraceConfig) {
	processModelTraceConfig.Publish(cfg)
}

func loadModelTraceConfig() *ModelTraceConfig {
	return processModelTraceConfig.Load()
}

func storeModelTraceConfig(cfg *ModelTraceConfig) {
	processModelTraceConfig.Publish(cfg)
}

// loadModelTraceConfigFromEnvironment builds the configuration from environment
// values. An explicit `MODELTRACE_ENABLED=false` disables the feature even when a
// URL and token are present; a configured URL and token imply enabled.
func loadModelTraceConfigFromEnvironment(getenv func(string) string) error {
	enabledValue := strings.TrimSpace(getenv("MODELTRACE_ENABLED"))
	baseURL := strings.TrimSpace(getenv("MODELTRACE_URL"))
	token := strings.TrimSpace(getenv("MODELTRACE_TOKEN"))
	timeoutValue := strings.TrimSpace(getenv("MODELTRACE_TIMEOUT_SECONDS"))

	enabled := baseURL != "" || token != ""
	if enabledValue != "" {
		parsed, err := strconv.ParseBool(enabledValue)
		if err != nil {
			return fmt.Errorf("MODELTRACE_ENABLED must be a boolean")
		}
		enabled = parsed
	}
	if !enabled {
		storeModelTraceConfig(nil)
		return nil
	}
	baseURL = strings.TrimRight(baseURL, "/")
	timeout := defaultModelTraceTimeout
	if timeoutValue != "" {
		seconds, err := strconv.Atoi(timeoutValue)
		if err != nil || seconds < 1 || seconds > int(maxModelTraceTimeout/time.Second) {
			return fmt.Errorf(
				"MODELTRACE_TIMEOUT_SECONDS must be an integer between 1 and %d",
				int(maxModelTraceTimeout/time.Second),
			)
		}
		timeout = time.Duration(seconds) * time.Second
	}
	storeModelTraceConfig(&ModelTraceConfig{
		Enabled: true, BaseURL: baseURL, Token: token, RequestTimeout: timeout,
	})
	return nil
}

// normalizedModelTraceConfig validates the deployment identity. It returns
// ErrModelTraceDisabled when the feature is off and ErrModelTraceMisconfigured
// when the operator supplied an unusable URL or token.
func normalizedModelTraceConfig(cfg *ModelTraceConfig) (*ModelTraceConfig, error) {
	if cfg == nil {
		return nil, app_errors.ErrModelTraceDisabled
	}
	if !cfg.Enabled {
		return nil, app_errors.ErrModelTraceDisabled
	}
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if baseURL == "" || strings.TrimSpace(cfg.Token) == "" {
		return nil, app_errors.ErrModelTraceMisconfigured
	}
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return nil, app_errors.ErrModelTraceMisconfigured
	}
	if parsed.RawQuery != "" || parsed.Fragment != "" || parsed.User != nil {
		return nil, app_errors.ErrModelTraceMisconfigured
	}
	if len(cfg.Token) < modelTraceMinTokenLength {
		return nil, app_errors.ErrModelTraceMisconfigured
	}
	timeout := cfg.RequestTimeout
	if timeout <= 0 {
		timeout = defaultModelTraceTimeout
	}
	if timeout > maxModelTraceTimeout {
		return nil, app_errors.ErrModelTraceMisconfigured
	}
	return &ModelTraceConfig{
		Enabled: true, BaseURL: baseURL, Token: cfg.Token, RequestTimeout: timeout,
	}, nil
}

type modelTraceChallengeDTO struct {
	ID            string `json:"id"`
	ExpectedCount int    `json:"expected_count"`
	Prompt        string `json:"prompt"`
}

type modelTraceChallengesResponse struct {
	Challenges []modelTraceChallengeDTO `json:"challenges"`
}

type modelTraceOutput struct {
	PromptID      string `json:"prompt_id"`
	ExpectedCount int    `json:"expected_count"`
	Text          string `json:"text"`
}

type modelTraceAnalysis struct {
	Prediction        string                    `json:"prediction"`
	PredictionName    string                    `json:"prediction_name"`
	Probability       float64                   `json:"probability"`
	FamilyPrediction  string                    `json:"family_prediction"`
	FamilyProbability float64                   `json:"family_probability"`
	UsedOutputs       int                       `json:"used_outputs"`
	Method            string                    `json:"method"`
	Candidates        []modelTraceCandidate     `json:"candidates"`
	Diagnostics       []modelTraceDiagnostic    `json:"diagnostics"`
	Bank              modelTraceBankSummary     `json:"bank"`
	Upstream          modelTraceUpstreamSummary `json:"upstream"`
}

type modelTraceCandidate struct {
	Model                  string  `json:"model"`
	Family                 string  `json:"family"`
	Probability            float64 `json:"probability"`
	ConditionalProbability float64 `json:"conditional_probability"`
}

type modelTraceDiagnostic struct {
	PromptID   string `json:"prompt_id"`
	Accepted   bool   `json:"accepted"`
	ParsedNum  int    `json:"parsed_numbers"`
	MinimumNum int    `json:"minimum_numbers"`
}

type modelTraceBankSummary struct {
	Models     int    `json:"models"`
	BuiltAt    string `json:"built_at"`
	BankSHA256 string `json:"bank_sha256"`
}

type modelTraceUpstreamSummary struct {
	Repository string `json:"repository"`
	Commit     string `json:"commit"`
}

// modelTraceClient talks to one ModelTrace deployment. It never receives
// upstream credentials: the deployment only sees challenge outputs.
type modelTraceClient struct {
	baseURL string
	token   string
	http    *http.Client
}

func newModelTraceClient(cfg *ModelTraceConfig) *modelTraceClient {
	return &modelTraceClient{
		baseURL: cfg.BaseURL,
		token:   cfg.Token,
		http:    &http.Client{Timeout: cfg.RequestTimeout},
	}
}

func (client *modelTraceClient) challenges(ctx context.Context) ([]modelTraceChallengeDTO, error) {
	var payload modelTraceChallengesResponse
	if err := client.call(ctx, http.MethodGet, "/api/challenges", nil, &payload); err != nil {
		return nil, err
	}
	if len(payload.Challenges) != modelTraceChallengeCount {
		return nil, fmt.Errorf("modeltrace returned %d challenges, want %d",
			len(payload.Challenges), modelTraceChallengeCount)
	}
	seen := make(map[string]struct{}, len(payload.Challenges))
	for _, challenge := range payload.Challenges {
		if strings.TrimSpace(challenge.ID) == "" || challenge.ExpectedCount <= 0 ||
			strings.TrimSpace(challenge.Prompt) == "" {
			return nil, fmt.Errorf("modeltrace returned a malformed challenge")
		}
		if _, duplicate := seen[challenge.ID]; duplicate {
			return nil, fmt.Errorf("modeltrace returned a duplicate challenge id")
		}
		seen[challenge.ID] = struct{}{}
	}
	return payload.Challenges, nil
}

func (client *modelTraceClient) analyze(
	ctx context.Context,
	outputs []modelTraceOutput,
) (*modelTraceAnalysis, error) {
	var payload modelTraceAnalysis
	if err := client.call(ctx, http.MethodPost, "/api/analyze",
		map[string]any{"outputs": outputs}, &payload); err != nil {
		return nil, err
	}
	if strings.TrimSpace(payload.Prediction) == "" {
		return nil, fmt.Errorf("modeltrace returned an empty prediction")
	}
	return &payload, nil
}

// call performs one authenticated ModelTrace exchange. Every transport,
// timeout, authentication, and protocol problem becomes ErrModelTraceUnavailable
// so the control plane answers with one stable failure and never retries.
func (client *modelTraceClient) call(
	ctx context.Context,
	method string,
	path string,
	requestBody any,
	target any,
) error {
	var reader io.Reader
	if requestBody != nil {
		encoded, err := json.Marshal(requestBody)
		if err != nil {
			return app_errors.ErrModelTraceUnavailable
		}
		reader = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, client.baseURL+path, reader)
	if err != nil {
		return app_errors.ErrModelTraceUnavailable
	}
	request.Header.Set("Authorization", "Bearer "+client.token)
	request.Header.Set("Accept", "application/json")
	if requestBody != nil {
		request.Header.Set("Content-Type", "application/json")
	}
	response, err := client.http.Do(request)
	if err != nil {
		return app_errors.ErrModelTraceUnavailable
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, modelTraceMaxBodyBytes))
	if err != nil {
		return app_errors.ErrModelTraceUnavailable
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return modelTraceStatusError(path, response.StatusCode, body)
	}
	if err := json.Unmarshal(body, target); err != nil {
		return app_errors.ErrModelTraceUnavailable
	}
	return nil
}

// modelTraceStatusError maps a non-2xx answer onto the stable failure set. A
// rejected analysis is the deployment's own verdict about the submitted
// outputs, so it stays distinguishable from an unavailable service.
func modelTraceStatusError(path string, status int, body []byte) error {
	if path == "/api/analyze" &&
		(status == http.StatusBadRequest || status == http.StatusUnprocessableEntity) {
		return app_errors.ErrModelTraceAnalysisRejected
	}
	var envelope struct {
		Error string `json:"error"`
	}
	code := ""
	if json.Unmarshal(body, &envelope) == nil {
		code = sanitizeModelTraceFragment(envelope.Error)
	}
	if code == "" {
		code = "status " + strconv.Itoa(status)
	}
	return app_errors.NewAPIErrorWithData(app_errors.ErrModelTraceUnavailable, map[string]any{
		"modeltrace_error": code,
	})
}

// sanitizeModelTraceFragment keeps a bounded, printable fragment of an upstream
// message. The deployment never echoes request bodies, but a truncated fragment
// still bounds what can reach a log line.
func sanitizeModelTraceFragment(value string) string {
	trimmed := strings.Join(strings.Fields(value), " ")
	if len(trimmed) <= modelTraceMaxErrorFragment {
		return trimmed
	}
	return trimmed[:modelTraceMaxErrorFragment] + modelTraceTruncatedMarker
}

// modelTraceAvailableError reports whether an error means "the deployment could
// not answer". It exists so callers can keep one failure vocabulary.
func modelTraceAvailableError(err error) bool {
	return errors.Is(err, app_errors.ErrModelTraceUnavailable)
}
