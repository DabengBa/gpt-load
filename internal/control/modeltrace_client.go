package control

import (
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	lmfpd "github.com/Ikaleio/lm-detector/sdk/go"
)

const modelTraceChallengeCount = 3

// ModelTraceConfig is retained for source compatibility with old callers.
// Only Enabled is used by the local detector; external service fields are retired.
type ModelTraceConfig struct {
	Enabled        bool
	BaseURL        string
	Token          string
	RequestTimeout time.Duration
}

type GenericModelTraceConfig struct {
	value atomic.Pointer[ModelTraceConfig]
}

func NewGenericModelTraceConfig() *GenericModelTraceConfig { return &GenericModelTraceConfig{} }
func (holder *GenericModelTraceConfig) Load() *ModelTraceConfig {
	if holder == nil {
		return nil
	}
	return holder.value.Load()
}
func (holder *GenericModelTraceConfig) Publish(cfg *ModelTraceConfig) {
	if holder != nil {
		holder.value.Store(cfg)
	}
}

var processModelTraceConfig = func() *GenericModelTraceConfig {
	holder := NewGenericModelTraceConfig()
	holder.Publish(&ModelTraceConfig{Enabled: true})
	return holder
}()

func SetProcessModelTraceConfig(cfg *ModelTraceConfig) { processModelTraceConfig.Publish(cfg) }
func loadModelTraceConfig() *ModelTraceConfig          { return processModelTraceConfig.Load() }
func storeModelTraceConfig(cfg *ModelTraceConfig)      { processModelTraceConfig.Publish(cfg) }
func loadModelTraceConfigFromEnvironment(getenv func(string) string) error {
	enabled := true
	if value := strings.TrimSpace(getenv("MODELTRACE_ENABLED")); value != "" {
		parsed, err := strconv.ParseBool(value)
		if err != nil {
			storeModelTraceConfig(&ModelTraceConfig{Enabled: false})
			return fmt.Errorf("MODELTRACE_ENABLED must be a boolean")
		}
		enabled = parsed
	}
	storeModelTraceConfig(&ModelTraceConfig{Enabled: enabled})
	return nil
}

type modelTraceChallengeDTO = lmfpd.Challenge
type modelTraceOutput struct {
	PromptID       string
	ExpectedCount  int
	Text           string
	ChallengeIndex int
}
