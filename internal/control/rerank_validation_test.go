package control

import (
	"testing"

	"gpt-load/internal/channel"
)

func TestBuildGroupProbeTargetRequiresDeclaredGenerativeContract(t *testing.T) {
	t.Parallel()
	group := probeTestGroupView(t, channel.Claude, "api_key", "claude-3")
	if _, ok := buildGroupProbeTarget(group, "claude-3"); ok {
		t.Fatal("probe unexpectedly supported without a declared generative contract")
	}
}
