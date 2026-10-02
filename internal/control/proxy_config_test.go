package control

import (
	"errors"
	"testing"

	"gpt-load/internal/outboundproxy"
	"gpt-load/internal/platform/config"
	app_errors "gpt-load/internal/platform/errors"
)

func TestGroupProxyIsSharedByCredentialNetworkContext(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-proxy-test")
	group, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		Proxy: optionalField[outboundproxy.Config]{Set: true, Value: outboundproxy.Config{
			Mode: outboundproxy.ModeCustom,
			URL:  "socks5://group-user:group-password@group-proxy.example.com:1080",
		}},
	})
	if err != nil {
		t.Fatalf("UpdateGroupSettings(proxy) error = %v", err)
	}
	if group.Proxy.EffectiveSource != outboundproxy.SourceGroup {
		t.Fatalf("group proxy = %#v", group.Proxy)
	}
	credentials, err := fixture.service.GetGroupCredential(t.Context(), groupID)
	if err != nil {
		t.Fatal(err)
	}
	if credentials.Credential == nil {
		t.Fatalf("credential = %#v", credentials)
	}
	if _, err := fixture.service.UpdateGroupCredential(t.Context(), groupID, credentials.Credential.CredentialID, CredentialUpdateRequest{
		Credentials: optionalField[string]{Set: true, Value: "replacement"},
	}); err != nil {
		t.Fatalf("replace credential = %v", err)
	}
}

func TestCredentialProxyFieldsAreRejected(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "sk-proxy-test")
	_, err := fixture.service.UpdateGroupSettings(t.Context(), groupID, GroupSettingsUpdateRequest{
		Overrides: optionalField[config.Settings]{Set: true, Value: config.Settings{
			"unknown": true,
		}},
	})
	if !errors.Is(err, app_errors.ErrValidation) {
		t.Fatalf("invalid settings error = %v, want validation", err)
	}
}
