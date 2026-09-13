package control

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"testing"

	"gpt-load/internal/channel"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
	subscriptionruntime "gpt-load/internal/subscription/runtime"
)

func subscriptionIdentityJSON(t *testing.T, channelID channel.ID, account, scope, token, expires string) []byte {
	t.Helper()
	values := map[string]string{"type": string(channelID), "access_token": "access-" + token, "refresh_token": "refresh-" + token, "email": token + "@example.com", "expired": expires}
	if channelID == channel.Codex {
		values["account_id"] = account
		if scope != "" {
			claims, err := json.Marshal(map[string]any{"https://api.openai.com/auth": map[string]string{"chatgpt_account_id": account, "chatgpt_user_id": scope}})
			if err != nil {
				t.Fatal(err)
			}
			values["id_token"] = "e30." + base64.RawURLEncoding.EncodeToString(claims) + ".signature"
		}
	} else {
		values["account_uuid"] = account
		if scope != "" {
			values["organization_uuid"] = scope
		}
	}
	raw, err := json.Marshal(values)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestSubscriptionRefreshIdentityAllowsEnrichmentWithoutDowngrade(t *testing.T) {
	t.Parallel()
	for _, channelID := range []channel.ID{channel.Codex, channel.Claude} {
		t.Run(string(channelID), func(t *testing.T) {
			for _, test := range []struct {
				name, oldScope, newAccount, newScope string
				allowed                              bool
			}{
				{"same identity", "scope-one", "account-one", "scope-one", true},
				{"legacy metadata still absent", "", "account-one", "", true},
				{"enriched metadata", "", "account-one", "scope-one", true},
				{"missing metadata", "scope-one", "account-one", "", false},
				{"different scope", "scope-one", "account-one", "scope-two", false},
				{"different account", "scope-one", "account-two", "scope-one", false},
			} {
				t.Run(test.name, func(t *testing.T) {
					for _, path := range []string{"transient", "ready stage"} {
						t.Run(path, func(t *testing.T) {
							f := newServiceFixture(t)
							driver, ok := subscriptionsDriver(f.service.subscriptions, channelID)
							if !ok {
								t.Fatal("driver missing")
							}
							current, err := driver.Parse(subscriptionIdentityJSON(t, channelID, "account-one", test.oldScope, "current", "2000-01-01T00:00:00Z"))
							if err != nil {
								t.Fatal(err)
							}
							refreshed, err := driver.Parse(subscriptionIdentityJSON(t, channelID, test.newAccount, test.newScope, "next", "2099-01-01T00:00:00Z"))
							if err != nil {
								t.Fatal(err)
							}
							f.service.refreshSubscriptionCredential = func(context.Context, channel.ID, subscriptionruntime.Credential) (subscriptionruntime.Credential, error) {
								return refreshed, nil
							}
							if path == "transient" {
								_, err = f.service.prepareTransientSubscriptionCredential(t.Context(), channelID, driver, current)
							} else {
								stage, stageErr := f.service.persistReadyCredentialStage(t.Context(), channelID, "oauth_file", current)
								if stageErr != nil {
									t.Fatal(stageErr)
								}
								row, loadErr := f.service.loadCredentialStage(t.Context(), stage.StageID)
								if loadErr != nil {
									t.Fatal(loadErr)
								}
								_, err = f.service.prepareReadySubscriptionStageCredential(t.Context(), row, driver, current, true)
								if test.allowed && err == nil {
									// 正常刷新后的短期暂存必须仍可消费，不能留下过期的身份指纹。
									_, err = f.service.CreateGroup(t.Context(), GroupCreateRequest{Name: stringPointer("refreshed identity"), ChannelID: channelID, ConnectionType: models.ConnectionTypeSubscription, Models: optionalGroupModels{Set: true, Values: []GroupModel{}}, StagedCredentialIDs: []string{stage.StageID}})
								}
							}
							if test.allowed && err != nil {
								t.Fatalf("same authorization with optional metadata change rejected: %v", err)
							}
							if !test.allowed && !errors.Is(err, app_errors.ErrCredentialReauthorizationRequired) {
								t.Fatalf("changed identity error = %v, want reauthorization required", err)
							}
						})
					}
				})
			}
		})
	}
}
