package control

import (
	"testing"
	"time"

	"gpt-load/internal/health"
	"gpt-load/internal/state"
)

func TestMapCredentialRuntimeItemBlacklistedRecoveryIsScheduledRelease(t *testing.T) {
	t.Parallel()

	item, err := mapCredentialRuntimeItem(
		"sk-****",
		1,
		state.CredentialRuntimeView{},
		healthBucketBlacklisted,
		health.CredentialStats{},
		time.Now(),
	)
	if err != nil {
		t.Fatalf("mapCredentialRuntimeItem() error = %v", err)
	}
	if !item.Recovery.Automatic || item.Recovery.Mode != "scheduled_release" || item.Recovery.AtMS != nil {
		t.Fatalf("blacklisted recovery = %#v, want automatic scheduled release without deadline", item.Recovery)
	}
}
