// Package proxy provides high-performance OpenAI multi-key proxy server with multi-format transformer support
package proxy

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"gpt-load/internal/channel"
	"gpt-load/internal/config"
	"gpt-load/internal/encryption"
	app_errors "gpt-load/internal/errors"
	"gpt-load/internal/keypool"
	"gpt-load/internal/models"
	"gpt-load/internal/response"
	"gpt-load/internal/services"
	"gpt-load/internal/transformer"
	"gpt-load/internal/transformer/inbound"
	"gpt-load/internal/transformer/model"
	"gpt-load/internal/transformer/outbound"
	"gpt-load/internal/utils"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// TransformerProxy represents the proxy server with multi-format transformer support
type TransformerProxy struct {
	keyProvider       *keypool.KeyProvider
	groupManager      *services.GroupManager
	subGroupManager   *services.SubGroupManager
	settingsManager   *config.SystemSettingsManager
	channelFactory    *channel.Factory
	requestLogService *services.RequestLogService
	encryptionSvc     encryption.Service
	detector          *transformer.FormatDetector
}

// NewTransformerProxy creates a new transformer proxy server
func NewTransformerProxy(
	keyProvider *keypool.KeyProvider,
	groupManager *services.GroupManager,
	subGroupManager *services.SubGroupManager,
	settingsManager *config.SystemSettingsManager,
	channelFactory *channel.Factory,
	requestLogService *services.RequestLogService,
	encryptionSvc encryption.Service,
) (*TransformerProxy, error) {
	return &TransformerProxy{
		keyProvider:       keyProvider,
		groupManager:      groupManager,
		subGroupManager:   subGroupManager,
		settingsManager:   settingsManager,
		channelFactory:    channelFactory,
		requestLogService: requestLogService,
		encryptionSvc:     encryptionSvc,
		detector:          transformer.NewFormatDetector(),
	}, nil
}


// HandleProxyWithTransform is the main entry point for proxy requests with format transformation
func (tp *TransformerProxy) HandleProxyWithTransform(c *gin.Context) {
	startTime := time.Now()
	groupName := c.Param("group_name")

	originalGroup, err := tp.groupManager.GetGroupByName(groupName)
	if err != nil {
		response.Error(c, app_errors.ParseDBError(err))
		return
	}

	// Select sub-group if this is an aggregate group
	subGroupName, err := tp.subGroupManager.SelectSubGroup(originalGroup)
	if err != nil {
		logrus.WithFields(logrus.Fields{
			"aggregate_group": originalGroup.Name,
			"error":           err,
		}).Error("Failed to select sub-group from aggregate")
		response.Error(c, app_errors.NewAPIError(app_errors.ErrNoKeysAvailable, "No available sub-groups"))
		return
	}

	group := originalGroup
	if subGroupName != "" {
		group, err = tp.groupManager.GetGroupByName(subGroupName)
		if err != nil {
			response.Error(c, app_errors.ParseDBError(err))
			return
		}
	}

	channelHandler, err := tp.channelFactory.GetChannel(group)
	if err != nil {
		response.Error(c, app_errors.NewAPIError(app_errors.ErrInternalServer, fmt.Sprintf("Failed to get channel for group '%s': %v", groupName, err)))
		return
	}

	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		logrus.Errorf("Failed to read request body: %v", err)
		response.Error(c, app_errors.NewAPIError(app_errors.ErrBadRequest, "Failed to read request body"))
		return
	}
	c.Request.Body.Close()

	// Detect client API format
	inboundType, err := tp.detector.DetectFormat(c.Request.URL.Path, bodyBytes)
	if err != nil {
		logrus.Warnf("Failed to detect format, using default OpenAI Chat: %v", err)
		inboundType = inbound.InboundTypeOpenAIChat
	}

	// Get inbound transformer
	inboundTransformer := inbound.GetInbound(inboundType)
	if inboundTransformer == nil {
		response.Error(c, app_errors.NewAPIError(app_errors.ErrBadRequest, fmt.Sprintf("Unsupported inbound format: %s", inboundType.String())))
		return
	}

	// Transform request to internal format
	internalReq, err := inboundTransformer.TransformRequest(c.Request.Context(), bodyBytes)
	if err != nil {
		logrus.Errorf("Failed to transform request: %v", err)
		response.Error(c, app_errors.NewAPIError(app_errors.ErrBadRequest, fmt.Sprintf("Failed to parse request: %v", err)))
		return
	}

	// Get outbound transformer based on group's API format
	outboundType := tp.getOutboundType(group.GetAPIFormat())
	outboundTransformer := outbound.GetOutbound(outboundType)
	if outboundTransformer == nil {
		response.Error(c, app_errors.NewAPIError(app_errors.ErrInternalServer, fmt.Sprintf("Unsupported outbound format: %s", outboundType.String())))
		return
	}

	isStream := internalReq.IsStreaming()

	tp.executeRequestWithTransform(c, channelHandler, originalGroup, group, internalReq, inboundTransformer, outboundTransformer, bodyBytes, isStream, startTime, 0)
}


// getOutboundType converts API format string to OutboundType
func (tp *TransformerProxy) getOutboundType(apiFormat string) outbound.OutboundType {
	switch apiFormat {
	case models.APIFormatOpenAIChat:
		return outbound.OutboundTypeOpenAIChat
	case models.APIFormatOpenAIResponse:
		return outbound.OutboundTypeOpenAIResponse
	case models.APIFormatAnthropic:
		return outbound.OutboundTypeAnthropic
	case models.APIFormatGemini:
		return outbound.OutboundTypeGemini
	default:
		return outbound.OutboundTypeOpenAIChat
	}
}

// executeRequestWithTransform is the core function for handling requests with transformation
func (tp *TransformerProxy) executeRequestWithTransform(
	c *gin.Context,
	channelHandler channel.ChannelProxy,
	originalGroup *models.Group,
	group *models.Group,
	internalReq *model.InternalLLMRequest,
	inboundTransformer model.Inbound,
	outboundTransformer model.Outbound,
	originalBodyBytes []byte,
	isStream bool,
	startTime time.Time,
	retryCount int,
) {
	cfg := group.EffectiveConfig

	apiKey, err := tp.keyProvider.SelectKey(group.ID)
	if err != nil {
		logrus.Errorf("Failed to select a key for group %s on attempt %d: %v", group.Name, retryCount+1, err)
		response.Error(c, app_errors.NewAPIError(app_errors.ErrNoKeysAvailable, err.Error()))
		tp.logRequest(c, originalGroup, group, nil, startTime, http.StatusServiceUnavailable, err, isStream, "", channelHandler, originalBodyBytes, models.RequestTypeFinal)
		return
	}

	// Get base URL using channel's smooth weighted round-robin selection
	baseURL := tp.getBaseURL(channelHandler, group)

	// Apply model redirect rules before building upstream request
	if len(group.ModelRedirectMap) > 0 && internalReq.Model != "" {
		if targetModel, found := group.ModelRedirectMap[internalReq.Model]; found {
			logrus.WithFields(logrus.Fields{
				"group":          group.Name,
				"original_model": internalReq.Model,
				"target_model":   targetModel,
			}).Debug("Model redirected in transformer flow")
			internalReq.Model = targetModel
		} else if group.ModelRedirectStrict {
			logrus.WithFields(logrus.Fields{
				"group": group.Name,
				"model": internalReq.Model,
			}).Warn("Model not in redirect rules and strict mode enabled")
			response.Error(c, app_errors.NewAPIError(app_errors.ErrBadRequest, fmt.Sprintf("model '%s' is not configured in redirect rules", internalReq.Model)))
			return
		}
	}

	var ctx context.Context
	var cancel context.CancelFunc
	if isStream {
		ctx, cancel = context.WithCancel(c.Request.Context())
	} else {
		timeout := time.Duration(cfg.RequestTimeout) * time.Second
		ctx, cancel = context.WithTimeout(c.Request.Context(), timeout)
	}
	defer cancel()

	// Decrypt key value if needed
	keyValue := apiKey.KeyValue

	// Transform internal request to upstream HTTP request
	upstreamReq, err := outboundTransformer.TransformRequest(ctx, internalReq, baseURL, keyValue)
	if err != nil {
		logrus.Errorf("Failed to transform request for upstream: %v", err)
		response.Error(c, app_errors.NewAPIError(app_errors.ErrInternalServer, fmt.Sprintf("Failed to build upstream request: %v", err)))
		return
	}

	upstreamURL := upstreamReq.URL.String()

	// Copy relevant headers from original request
	for key, values := range c.Request.Header {
		// Skip headers that are set by the transformer
		lowerKey := strings.ToLower(key)
		if lowerKey == "authorization" || lowerKey == "x-api-key" || lowerKey == "x-goog-api-key" ||
			lowerKey == "content-type" || lowerKey == "content-length" || lowerKey == "anthropic-version" {
			continue
		}
		for _, value := range values {
			upstreamReq.Header.Add(key, value)
		}
	}

	// Apply custom header rules
	if len(group.HeaderRuleList) > 0 {
		headerCtx := utils.NewHeaderVariableContextFromGin(c, group, apiKey)
		utils.ApplyHeaderRules(upstreamReq, group.HeaderRuleList, headerCtx)
	}

	var client *http.Client
	if isStream {
		client = channelHandler.GetStreamClient()
		upstreamReq.Header.Set("X-Accel-Buffering", "no")
	} else {
		client = channelHandler.GetHTTPClient()
	}

	resp, err := client.Do(upstreamReq)
	if resp != nil {
		defer resp.Body.Close()
	}

	// Handle errors and retries
	if err != nil || (resp != nil && resp.StatusCode >= 400 && resp.StatusCode != http.StatusNotFound) {
		if err != nil && app_errors.IsIgnorableError(err) {
			logrus.Debugf("Client-side ignorable error for key %s, aborting retries: %v", utils.MaskAPIKey(apiKey.KeyValue), err)
			tp.logRequest(c, originalGroup, group, apiKey, startTime, 499, err, isStream, upstreamURL, channelHandler, originalBodyBytes, models.RequestTypeFinal)
			return
		}

		var statusCode int
		var errorMessage string
		var parsedError string

		if err != nil {
			statusCode = 500
			errorMessage = err.Error()
			parsedError = errorMessage
			logrus.Debugf("Request failed (attempt %d/%d) for key %s: %v", retryCount+1, cfg.MaxRetries, utils.MaskAPIKey(apiKey.KeyValue), err)
		} else {
			statusCode = resp.StatusCode
			errorBody, readErr := io.ReadAll(resp.Body)
			if readErr != nil {
				logrus.Errorf("Failed to read error body: %v", readErr)
				errorBody = []byte("Failed to read error body")
			}

			errorBody = handleGzipCompression(resp, errorBody)
			errorMessage = string(errorBody)
			parsedError = app_errors.ParseUpstreamError(errorBody)
			logrus.Debugf("Request failed with status %d (attempt %d/%d) for key %s. Parsed Error: %s", statusCode, retryCount+1, cfg.MaxRetries, utils.MaskAPIKey(apiKey.KeyValue), parsedError)
		}

		tp.keyProvider.UpdateStatus(apiKey, group, false, parsedError)

		isLastAttempt := retryCount >= cfg.MaxRetries
		requestType := models.RequestTypeRetry
		if isLastAttempt {
			requestType = models.RequestTypeFinal
		}

		tp.logRequest(c, originalGroup, group, apiKey, startTime, statusCode, errors.New(parsedError), isStream, upstreamURL, channelHandler, originalBodyBytes, requestType)

		if isLastAttempt {
			// Transform error response to client format
			tp.handleErrorResponse(c, statusCode, errorMessage, inboundTransformer)
			return
		}

		tp.executeRequestWithTransform(c, channelHandler, originalGroup, group, internalReq, inboundTransformer, outboundTransformer, originalBodyBytes, isStream, startTime, retryCount+1)
		return
	}

	logrus.Debugf("Request for group %s succeeded on attempt %d with key %s", group.Name, retryCount+1, utils.MaskAPIKey(apiKey.KeyValue))

	// Handle response with transformation
	if isStream {
		tp.handleStreamingWithTransform(c, resp, inboundTransformer, outboundTransformer)
	} else {
		tp.handleNormalWithTransform(c, resp, inboundTransformer, outboundTransformer)
	}

	tp.logRequest(c, originalGroup, group, apiKey, startTime, resp.StatusCode, nil, isStream, upstreamURL, channelHandler, originalBodyBytes, models.RequestTypeFinal)
}


// getBaseURL uses the channel's smooth weighted round-robin selection to get an upstream URL
func (tp *TransformerProxy) getBaseURL(channelHandler channel.ChannelProxy, group *models.Group) string {
	// Use the channel's BuildUpstreamURL which internally uses smooth weighted round-robin
	upstreamURL, err := channelHandler.BuildUpstreamURL(&url.URL{Path: ""}, group.Name)
	if err != nil {
		logrus.Warnf("Failed to get upstream URL for group %s: %v", group.Name, err)
		return ""
	}
	// Validate the URL but keep the full path to honor configured upstream prefixes
	_, err = url.Parse(upstreamURL)
	if err != nil {
		logrus.Warnf("Failed to parse upstream URL %s: %v", upstreamURL, err)
		return ""
	}
	// Return the full upstream URL including any path prefix
	return strings.TrimSuffix(upstreamURL, "/")
}

// handleNormalWithTransform handles non-streaming response with transformation
func (tp *TransformerProxy) handleNormalWithTransform(
	c *gin.Context,
	resp *http.Response,
	inboundTransformer model.Inbound,
	outboundTransformer model.Outbound,
) {
	// Transform upstream response to internal format
	internalResp, err := outboundTransformer.TransformResponse(c.Request.Context(), resp)
	if err != nil {
		logrus.Errorf("Failed to transform upstream response: %v", err)
		response.Error(c, app_errors.NewAPIError(app_errors.ErrInternalServer, "Failed to process upstream response"))
		return
	}

	// Handle error response from upstream
	if internalResp.IsError() {
		tp.handleInternalErrorResponse(c, internalResp, inboundTransformer)
		return
	}

	// Transform internal response to client format
	clientBody, err := inboundTransformer.TransformResponse(c.Request.Context(), internalResp)
	if err != nil {
		logrus.Errorf("Failed to transform response for client: %v", err)
		response.Error(c, app_errors.NewAPIError(app_errors.ErrInternalServer, "Failed to format response"))
		return
	}

	c.Header("Content-Type", "application/json")
	c.Status(http.StatusOK)
	c.Writer.Write(clientBody)
}

// handleStreamingWithTransform handles streaming response with transformation
// It correctly processes SSE format conversion between different API formats
// and handles the [DONE] marker semantics across format conversions
func (tp *TransformerProxy) handleStreamingWithTransform(
	c *gin.Context,
	resp *http.Response,
	inboundTransformer model.Inbound,
	outboundTransformer model.Outbound,
) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")

	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		logrus.Error("Streaming unsupported by the writer, falling back to normal response")
		tp.handleNormalWithTransform(c, resp, inboundTransformer, outboundTransformer)
		return
	}

	scanner := bufio.NewScanner(resp.Body)
	// Increase buffer size for large chunks
	buf := make([]byte, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	// Track if we've sent the [DONE] marker
	doneSent := false

	// Buffer for accumulating multi-line SSE events (Anthropic style)
	var eventBuffer strings.Builder
	var currentEventType string

	for scanner.Scan() {
		line := scanner.Text()

		// Skip empty lines but check if we need to process accumulated event
		if strings.TrimSpace(line) == "" {
			if eventBuffer.Len() > 0 {
				// Process accumulated event
				tp.processStreamEvent(c, flusher, currentEventType, eventBuffer.String(), inboundTransformer, outboundTransformer, &doneSent)
				eventBuffer.Reset()
				currentEventType = ""
			}
			continue
		}

		// Handle SSE event type line
		if strings.HasPrefix(line, "event:") {
			currentEventType = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}

		// Handle SSE data line
		if strings.HasPrefix(line, "data:") {
			data := strings.TrimPrefix(line, "data:")
			data = strings.TrimPrefix(data, " ") // Remove optional space after "data:"

			// Handle [DONE] marker
			if strings.TrimSpace(data) == "[DONE]" {
				if !doneSent {
					c.Writer.Write([]byte("data: [DONE]\n\n"))
					flusher.Flush()
					doneSent = true
				}
				continue
			}

			// Accumulate data for multi-line events or process immediately
			if currentEventType != "" {
				// Anthropic-style event with type
				eventBuffer.WriteString(data)
			} else {
				// OpenAI-style data-only event
				tp.processStreamData(c, flusher, []byte(data), inboundTransformer, outboundTransformer, &doneSent)
			}
			continue
		}

		// Handle non-SSE format (Gemini JSON streaming)
		// Gemini returns JSON objects directly without SSE prefix
		if strings.HasPrefix(line, "{") || strings.HasPrefix(line, "[") {
			tp.processStreamData(c, flusher, []byte(line), inboundTransformer, outboundTransformer, &doneSent)
		}
	}

	// Process any remaining buffered event
	if eventBuffer.Len() > 0 {
		tp.processStreamEvent(c, flusher, currentEventType, eventBuffer.String(), inboundTransformer, outboundTransformer, &doneSent)
	}

	if err := scanner.Err(); err != nil {
		if !app_errors.IsIgnorableError(err) {
			logrus.Errorf("Error reading stream: %v", err)
		}
	}

	// Send [DONE] marker if not already sent
	if !doneSent {
		c.Writer.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}
}

// processStreamEvent processes an SSE event with type (Anthropic style)
func (tp *TransformerProxy) processStreamEvent(
	c *gin.Context,
	flusher http.Flusher,
	eventType string,
	data string,
	inboundTransformer model.Inbound,
	outboundTransformer model.Outbound,
	doneSent *bool,
) {
	// Build full event data for Anthropic-style parsing
	eventData := []byte(fmt.Sprintf("event: %s\ndata: %s", eventType, data))

	tp.processStreamData(c, flusher, eventData, inboundTransformer, outboundTransformer, doneSent)
}

// processStreamData processes stream data and writes transformed output to client
func (tp *TransformerProxy) processStreamData(
	c *gin.Context,
	flusher http.Flusher,
	eventData []byte,
	inboundTransformer model.Inbound,
	outboundTransformer model.Outbound,
	doneSent *bool,
) {
	// Transform upstream stream data to internal format
	internalChunk, err := outboundTransformer.TransformStream(c.Request.Context(), eventData)
	if err != nil {
		logrus.Debugf("Failed to transform stream chunk: %v", err)
		return
	}
	if internalChunk == nil {
		return
	}

	// Handle error in stream
	if internalChunk.IsError() {
		errorBody, _ := inboundTransformer.TransformResponse(c.Request.Context(), internalChunk)
		c.Writer.Write([]byte("data: "))
		c.Writer.Write(errorBody)
		c.Writer.Write([]byte("\n\n"))
		flusher.Flush()
		return
	}

	// Transform internal chunk to client format
	clientChunk, err := inboundTransformer.TransformStream(c.Request.Context(), internalChunk)
	if err != nil {
		logrus.Debugf("Failed to transform stream chunk for client: %v", err)
		return
	}
	if clientChunk == nil {
		return
	}

	// Write to client
	c.Writer.Write(clientChunk)
	flusher.Flush()
}


// handleErrorResponse handles error response transformation to client format
func (tp *TransformerProxy) handleErrorResponse(c *gin.Context, statusCode int, errorMessage string, inboundTransformer model.Inbound) {
	// Try to parse as JSON error
	var errorJSON map[string]any
	if err := json.Unmarshal([]byte(errorMessage), &errorJSON); err == nil {
		// Already valid JSON, return as-is
		c.JSON(statusCode, errorJSON)
		return
	}

	// Create internal error response
	internalResp := &model.InternalLLMResponse{
		Object:  "error",
		Created: time.Now().Unix(),
		Error: &model.ResponseError{
			StatusCode: statusCode,
			Detail: model.ErrorDetail{
				Message: errorMessage,
				Type:    "upstream_error",
			},
		},
	}

	// Transform to client format
	clientBody, err := inboundTransformer.TransformResponse(c.Request.Context(), internalResp)
	if err != nil {
		// Fallback to generic error
		c.JSON(statusCode, gin.H{
			"error": gin.H{
				"message": errorMessage,
				"type":    "upstream_error",
			},
		})
		return
	}

	c.Header("Content-Type", "application/json")
	c.Status(statusCode)
	c.Writer.Write(clientBody)
}

// handleInternalErrorResponse handles internal error response
func (tp *TransformerProxy) handleInternalErrorResponse(c *gin.Context, internalResp *model.InternalLLMResponse, inboundTransformer model.Inbound) {
	statusCode := http.StatusInternalServerError
	if internalResp.Error != nil && internalResp.Error.StatusCode > 0 {
		statusCode = internalResp.Error.StatusCode
	}

	clientBody, err := inboundTransformer.TransformResponse(c.Request.Context(), internalResp)
	if err != nil {
		// Fallback to generic error
		errorMsg := "Unknown error"
		if internalResp.Error != nil {
			errorMsg = internalResp.Error.Detail.Message
		}
		c.JSON(statusCode, gin.H{
			"error": gin.H{
				"message": errorMsg,
				"type":    "api_error",
			},
		})
		return
	}

	c.Header("Content-Type", "application/json")
	c.Status(statusCode)
	c.Writer.Write(clientBody)
}

// logRequest is a helper function to create and record a request log
func (tp *TransformerProxy) logRequest(
	c *gin.Context,
	originalGroup *models.Group,
	group *models.Group,
	apiKey *models.APIKey,
	startTime time.Time,
	statusCode int,
	finalError error,
	isStream bool,
	upstreamAddr string,
	channelHandler channel.ChannelProxy,
	bodyBytes []byte,
	requestType string,
) {
	if tp.requestLogService == nil {
		return
	}

	var requestBodyToLog, userAgent string

	if group.EffectiveConfig.EnableRequestBodyLogging {
		requestBodyToLog = utils.TruncateString(string(bodyBytes), 65000)
		userAgent = c.Request.UserAgent()
	}

	duration := time.Since(startTime).Milliseconds()

	logEntry := &models.RequestLog{
		GroupID:      group.ID,
		GroupName:    group.Name,
		IsSuccess:    finalError == nil && statusCode < 400,
		SourceIP:     c.ClientIP(),
		StatusCode:   statusCode,
		RequestPath:  utils.TruncateString(c.Request.URL.String(), 500),
		Duration:     duration,
		UserAgent:    userAgent,
		RequestType:  requestType,
		IsStream:     isStream,
		UpstreamAddr: utils.TruncateString(upstreamAddr, 500),
		RequestBody:  requestBodyToLog,
	}

	// Set parent group
	if originalGroup != nil && originalGroup.GroupType == "aggregate" && originalGroup.ID != group.ID {
		logEntry.ParentGroupID = originalGroup.ID
		logEntry.ParentGroupName = originalGroup.Name
	}

	if channelHandler != nil && bodyBytes != nil {
		logEntry.Model = channelHandler.ExtractModel(c, bodyBytes)
	}

	if apiKey != nil {
		// Encrypt key value for log storage
		encryptedKeyValue, err := tp.encryptionSvc.Encrypt(apiKey.KeyValue)
		if err != nil {
			logrus.WithError(err).Error("Failed to encrypt key value for logging")
			logEntry.KeyValue = "failed-to-encryption"
		} else {
			logEntry.KeyValue = encryptedKeyValue
		}
		// Add KeyHash for reverse lookup
		logEntry.KeyHash = tp.encryptionSvc.Hash(apiKey.KeyValue)
	}

	if finalError != nil {
		logEntry.ErrorMessage = finalError.Error()
	}

	if err := tp.requestLogService.Record(logEntry); err != nil {
		logrus.Errorf("Failed to record request log: %v", err)
	}
}
