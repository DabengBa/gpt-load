// Package providers assembles the subscription provider implementations.
package providers

import (
	"gpt-load/internal/subscription/providers/antigravity"
	"gpt-load/internal/subscription/providers/claude"
	"gpt-load/internal/subscription/providers/codex"
	"gpt-load/internal/subscription/providers/grok"
	subscriptionruntime "gpt-load/internal/subscription/runtime"
)

func Implementations() []subscriptionruntime.Implementations {
	return []subscriptionruntime.Implementations{
		codex.Implementations(),
		claude.Implementations(),
		antigravity.Implementations(),
		grok.Implementations(),
	}
}
