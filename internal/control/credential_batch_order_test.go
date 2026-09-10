package control

import (
	"errors"
	"reflect"
	"testing"

	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/storage/models"
)

func TestBatchDeleteRetiresCommittedCredentialRuntime(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	runtime := &recordingCredentialRuntimeExecutor{}
	fixture.service.executor = runtime
	groupID := createGroupWithCredentials(t, fixture, "first-secret")

	var row models.Credential
	if err := fixture.db.Where("group_id = ?", groupID).Take(&row).Error; err != nil {
		t.Fatal(err)
	}
	result, err := fixture.service.BatchGroupCredentials(t.Context(), groupID, CredentialBatchRequest{
		Action: CredentialBatchDelete, CredentialIDs: []uint{row.ID},
	})
	if err != nil {
		t.Fatalf("BatchGroupCredentials() error = %v", err)
	}
	if !reflect.DeepEqual(result.AffectedCredentialIDs, []uint{row.ID}) {
		t.Fatalf("affected credentials = %#v", result.AffectedCredentialIDs)
	}
	if got := runtime.retiredCredentialIDs(); !reflect.DeepEqual(got, []uint{row.ID}) {
		t.Fatalf("retired credential runtimes = %#v", got)
	}
}

func TestBatchCredentialMutationRejectsRemovedOperatorActions(t *testing.T) {
	t.Parallel()
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "first-secret")

	for _, action := range []CredentialBatchAction{"enable", "disable"} {
		_, err := fixture.service.BatchGroupCredentials(t.Context(), groupID, CredentialBatchRequest{
			Action: action, CredentialIDs: []uint{1},
		})
		if !errors.Is(err, app_errors.ErrValidation) {
			t.Fatalf("BatchGroupCredentials(%q) error = %v, want validation", action, err)
		}
	}
}
