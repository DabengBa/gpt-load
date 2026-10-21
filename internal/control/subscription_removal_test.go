package control

import (
	"strings"
	"testing"
)

func TestSubscriptionRoutesRemoved(t *testing.T) {
	initControlI18n(t)
	fixture := newServiceFixture(t)
	server := &Server{service: fixture.service}
	for _, route := range server.HTTPModule().Routes {
		for _, retired := range []string{"credential-stages", "subscription-accounts", "observation", "reset-credit", "connect", "refresh", "download"} {
			if strings.Contains(route.Path, retired) && (strings.Contains(route.Path, "credential") || strings.Contains(route.Path, "subscription")) {
				t.Errorf("retired route remains registered: %s", route.Path)
			}
		}
	}
}
