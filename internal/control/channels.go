package control

import (
	"context"

	"github.com/gin-gonic/gin"

	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
)

const channelMaxQueryRunes = 200

type ChannelListResponse struct {
	Items []ChannelListItem `json:"items"`
	Total int               `json:"total"`
}

type ChannelListItem struct {
	channel.Descriptor
	DefaultBaseURL string `json:"default_base_url"`
}

type channelDefaultBaseURLProvider interface {
	DefaultBaseURL(channel.ID) (string, bool, error)
}

func (s *Service) ListChannels(ctx context.Context, query string) (ChannelListResponse, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return ChannelListResponse{}, err
	}
	if s == nil || s.channelRegistry == nil {
		return ChannelListResponse{}, app_errors.ErrInternalServer
	}
	descriptors := s.channelRegistry.Search(query)
	items := make([]ChannelListItem, 0, len(descriptors))
	for _, descriptor := range descriptors {
		item := ChannelListItem{Descriptor: descriptor}
		if s.channelDefaultBaseURLs != nil {
			baseURL, unique, err := s.channelDefaultBaseURLs.DefaultBaseURL(descriptor.ID)
			if err != nil {
				return ChannelListResponse{}, err
			}
			if unique {
				item.DefaultBaseURL = baseURL
			}
		}
		items = append(items, item)
	}
	return ChannelListResponse{Items: items, Total: len(items)}, nil
}

func parseChannelQuery(rawQuery string, forceQuery bool) (string, *app_errors.APIError) {
	values, apiErr := parseCollectionQueryValues(rawQuery, forceQuery, "q")
	if apiErr != nil {
		return "", apiErr
	}
	query, ok := collectionQueryText(values, "q", channelMaxQueryRunes)
	if !ok {
		return "", app_errors.ErrBadRequest
	}
	return query, nil
}

func (s *Server) handleListChannels(c *gin.Context) {
	query, apiErr := parseChannelQuery(c.Request.URL.RawQuery, c.Request.URL.ForceQuery)
	if apiErr != nil {
		writeServiceError(c, "list_channels", apiErr)
		return
	}
	result, err := s.service.ListChannels(c.Request.Context(), query)
	if err != nil {
		writeServiceError(c, "list_channels", err)
		return
	}
	response.SuccessI18n(c, "common.success", result)
}
