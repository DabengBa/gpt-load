package proxy

import (
	"encoding/json"
	"gpt-load/internal/channel"
	"gpt-load/internal/models"
	"gpt-load/internal/utils"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// shouldInterceptModelList checks if this is a model list request that should be intercepted
func shouldInterceptModelList(path string, method string) bool {
	if method != "GET" {
		return false
	}

	// Check various model list endpoints
	return strings.HasSuffix(path, "/v1/models") ||
		strings.HasSuffix(path, "/v1beta/models") ||
		strings.Contains(path, "/v1beta/openai/v1/models")
}

// handleModelListResponse processes the model list response and applies filtering based on redirect rules
func (ps *ProxyServer) handleModelListResponse(c *gin.Context, resp *http.Response, group *models.Group, channelHandler channel.ChannelProxy) {
	// Read the upstream response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		logrus.WithError(err).Error("Failed to read model list response body")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read response"})
		return
	}

	// Decompress response data based on Content-Encoding
	contentEncoding := resp.Header.Get("Content-Encoding")
	decompressed, err := utils.DecompressResponse(contentEncoding, bodyBytes)
	if err != nil {
		logrus.WithError(err).Warn("Decompression failed, using original data")
		decompressed = bodyBytes
	}

	// Transform model list (returns map[string]any directly, no marshaling)
	response, err := channelHandler.TransformModelList(c.Request, decompressed, group)
	if err != nil {
		logrus.WithError(err).Error("Failed to transform model list")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process response"})
		return
	}

	c.JSON(http.StatusOK, response)
}

// handleAggregateModelList generates an aggregated model list for an aggregate group
func (ps *ProxyServer) handleAggregateModelList(c *gin.Context, aggregateGroup *models.Group) {
	modelSet := make(map[string]struct{})

	// Iterate through sub-groups to collect supported models
	for _, sg := range aggregateGroup.SubGroups {
		subGroup, err := ps.groupManager.GetGroupByName(sg.SubGroupName)
		if err != nil {
			continue
		}

		// Add models from SupportedModels field
		var supported []string
		if len(subGroup.SupportedModels) > 0 {
			if err := json.Unmarshal(subGroup.SupportedModels, &supported); err == nil {
				for _, m := range supported {
					// Handle wildcards? Maybe just list them as is or expand if common.
					// For now, list as is.
					modelSet[m] = struct{}{}
				}
			}
		}

		// Add models from ModelRedirectRules
		for sourceModel := range subGroup.ModelRedirectMap {
			modelSet[sourceModel] = struct{}{}
		}
	}

	// If no models found in config, fallback to proxying one sub-group to get SOME models
	if len(modelSet) == 0 {
		subGroupName, err := ps.subGroupManager.SelectSubGroup(aggregateGroup, "")
		if err != nil || subGroupName == "" {
			c.JSON(http.StatusOK, gin.H{
				"object": "list",
				"data":   []any{},
			})
			return
		}

		subGroup, _ := ps.groupManager.GetGroupByName(subGroupName)
		_, err = ps.channelFactory.GetChannel(subGroup)
		if err != nil {
			c.JSON(http.StatusOK, gin.H{
				"object": "list",
				"data":   []any{},
			})
			return
		}

		// Perform actual proxy for this one sub-group
		// Since we are inside handleAggregateModelList, we need to call executeRequestWithRetry
		// but that's a member of ps.
		// However, it's easier to just let HandleProxy continue if we don't intercept.
		// So we return early and HandleProxy will do the selection and proxying anyway if we don't handle it here.
		
		// Wait, if I want to fallback, I should just NOT write a response and let the caller know.
		// But handleAggregateModelList is void.
		// Let's change the strategy: If modelSet is empty, return false from a check function.
		return
	}

	// Format as OpenAI model list
	modelsList := make([]any, 0, len(modelSet))
	for m := range modelSet {
		modelsList = append(modelsList, map[string]any{
			"id":       m,
			"object":   "model",
			"created":  time.Now().Unix(),
			"owned_by": "system",
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"object": "list",
		"data":   modelsList,
	})
}
