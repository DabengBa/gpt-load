package control

import (
	"context"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"gpt-load/internal/agent"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/state"
	"gpt-load/internal/storage/models"
)

func TestChangeProposalLifecycleBindsOneOperation(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "proposal-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	if len(groupModels) != 1 || groupModels[0].EntryID == "" {
		t.Fatalf("created group models = %#v, want one persisted entry", groupModels)
	}

	principal := agent.Principal{
		CredentialID: 7,
		Scopes:       []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply},
	}
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{
		Updates: []agent.ChangeProposalUpdateInput{{
			GroupID:          groupID,
			EntryID:          groupModels[0].EntryID,
			Weight:           agent.ProposalInt{Set: true, Value: 2},
			Priority:         agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight:   agent.ProposalInt{Set: true, Value: 1},
			ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}},
	})
	if err != nil {
		t.Fatalf("CreateChangeProposal() error = %v", err)
	}
	if proposal.State != ProposalStatePending || proposal.Execution.State != agent.ProposalExecutionNotBound {
		t.Fatalf("created proposal = %#v, want pending and not_bound", proposal)
	}

	if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
		t.Fatalf("ApproveChangeProposal() error = %v", err)
	}

	const firstKey = "11111111-1111-4111-8111-111111111111"
	first, err := fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, firstKey)
	if err != nil {
		t.Fatalf("first ApplyChangeProposal() error = %v", err)
	}
	if first.Execution.State != agent.ProposalExecutionApplied || first.Execution.OperationID == nil {
		t.Fatalf("first apply = %#v, want applied operation", first)
	}
	sameKeyReplay, err := fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, firstKey)
	if err != nil {
		t.Fatalf("same-key replay error = %v", err)
	}
	if sameKeyReplay.Execution.OperationID == nil || *sameKeyReplay.Execution.OperationID != *first.Execution.OperationID {
		t.Fatalf("same-key replay operation = %#v, want %q", sameKeyReplay.Execution, *first.Execution.OperationID)
	}

	const secondKey = "22222222-2222-4222-8222-222222222222"
	second, err := fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, secondKey)
	if err != nil {
		t.Fatalf("second ApplyChangeProposal() error = %v", err)
	}
	if second.Execution.OperationID == nil || *second.Execution.OperationID != *first.Execution.OperationID {
		t.Fatalf("second operation = %#v, want replay of %q", second.Execution, *first.Execution.OperationID)
	}

	updated := loadCreatedGroupModels(t, fixture, groupID)
	if len(updated) != 1 || updated[0].Weight == nil || *updated[0].Weight != 2 {
		t.Fatalf("updated group models = %#v, want one weight mutation", updated)
	}

	secondProposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{
		Updates: []agent.ChangeProposalUpdateInput{{
			GroupID:          groupID,
			EntryID:          groupModels[0].EntryID,
			Weight:           agent.ProposalInt{Set: true, Value: 3},
			Priority:         agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight:   agent.ProposalInt{Set: true, Value: 2},
			ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}},
	})
	if err != nil {
		t.Fatalf("second CreateChangeProposal() error = %v", err)
	}
	_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, secondProposal.ProposalID, firstKey)
	var reused *app_errors.APIError
	if !errors.As(err, &reused) || reused.Code != app_errors.ErrIdempotencyKeyReused.Code {
		t.Fatalf("reused key for another proposal error = %v, want %s", err, app_errors.ErrIdempotencyKeyReused.Code)
	}
}

func TestChangeProposalCompactionPreservesAppliedProjectionAndExpiresReplay(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "proposal-compaction-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	principal := agent.Principal{
		CredentialID: 27,
		Scopes:       []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply},
	}
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{
		Updates: []agent.ChangeProposalUpdateInput{{
			GroupID:          groupID,
			EntryID:          groupModels[0].EntryID,
			Weight:           agent.ProposalInt{Set: true, Value: 2},
			Priority:         agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight:   agent.ProposalInt{Set: true, Value: 1},
			ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}},
	})
	if err != nil {
		t.Fatalf("CreateChangeProposal() error = %v", err)
	}
	if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
		t.Fatalf("ApproveChangeProposal() error = %v", err)
	}
	const firstKey = "99999999-9999-4999-8999-999999999999"
	if _, err := fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, firstKey); err != nil {
		t.Fatalf("ApplyChangeProposal() error = %v", err)
	}

	compacted, err := fixture.service.CompactCompletedOperations(
		t.Context(),
		time.Now().UTC().Add(8*24*time.Hour),
	)
	if err != nil {
		t.Fatalf("CompactCompletedOperations() error = %v", err)
	}
	if compacted != 1 {
		t.Fatalf("compacted rows = %d, want 1", compacted)
	}

	view, err := fixture.service.GetChangeProposal(t.Context(), proposal.ProposalID)
	if err != nil {
		t.Fatalf("GetChangeProposal() after compaction error = %v", err)
	}
	if view.Execution.State != agent.ProposalExecutionApplied || view.Execution.CanReconcile {
		t.Fatalf("compacted execution = %#v, want applied and non-reconcilable", view.Execution)
	}

	for _, key := range []string{
		firstKey,
		"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
	} {
		_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, key)
		assertAPIErrorCode(t, err, app_errors.ErrIdempotencyResultExpired.Code)
	}
}

func TestChangeProposalRevokeAllowsPendingProposal(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "proposal-pending-revoke-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), agent.Principal{
		CredentialID: 28,
		Scopes:       []agent.Scope{agent.ScopeChangesPropose},
	}, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
		GroupID:          groupID,
		EntryID:          groupModels[0].EntryID,
		Weight:           agent.ProposalInt{Set: true, Value: 2},
		Priority:         agent.ProposalInt{Set: true, Value: 1},
		ExpectedWeight:   agent.ProposalInt{Set: true, Value: 1},
		ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
	}}})
	if err != nil {
		t.Fatalf("CreateChangeProposal() error = %v", err)
	}
	revoked, err := fixture.service.RevokeChangeProposal(t.Context(), proposal.ProposalID)
	if err != nil {
		t.Fatalf("RevokeChangeProposal() error = %v", err)
	}
	if revoked.State != agent.ProposalStateRevoked {
		t.Fatalf("revoked state = %q, want %q", revoked.State, agent.ProposalStateRevoked)
	}
}

func TestChangeProposalRecoveryDoesNotMutateOnRead(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "proposal-recovery-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	principal := agent.Principal{
		CredentialID: 8,
		Scopes:       []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply},
	}
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{
		Updates: []agent.ChangeProposalUpdateInput{{
			GroupID:          groupID,
			EntryID:          groupModels[0].EntryID,
			Weight:           agent.ProposalInt{Set: true, Value: 3},
			Priority:         agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight:   agent.ProposalInt{Set: true, Value: 1},
			ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}},
	})
	if err != nil {
		t.Fatalf("CreateChangeProposal() error = %v", err)
	}
	if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
		t.Fatalf("ApproveChangeProposal() error = %v", err)
	}

	fixture.service.beforeAdvanceOperationStage = func(_ context.Context, operation *models.ControlOperation, stage operationStage) error {
		if operation.OperationKind == string(operationKindModelRouteScheduleApply) && stage == operationStageSnapshotPublished {
			return errors.New("test snapshot publication stage record failure")
		}
		return nil
	}
	const firstKey = "33333333-3333-4333-8333-333333333333"
	_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, firstKey)
	var incomplete *app_errors.APIError
	if !errors.As(err, &incomplete) || incomplete.Code != app_errors.ErrControlOperationIncomplete.Code {
		t.Fatalf("first apply error = %v, want control operation incomplete", err)
	}

	var operation models.ControlOperation
	if err := fixture.db.Where("proposal_id = ?", proposal.ProposalID).First(&operation).Error; err != nil {
		t.Fatalf("load incomplete operation: %v", err)
	}
	lastStage := operation.LastCompletedStage
	failedStage := operation.FailedStage
	if lastStage != string(operationStageDBCommitted) || failedStage != string(operationStageSnapshotPublished) {
		t.Fatalf("incomplete operation stages = %q/%q, want db_committed/snapshot_published", lastStage, failedStage)
	}

	view, err := fixture.service.GetChangeProposal(t.Context(), proposal.ProposalID)
	if err != nil {
		t.Fatalf("GetChangeProposal() error = %v", err)
	}
	if view.Execution.State != agent.ProposalExecutionFailed {
		t.Fatalf("read execution state = %q, want failed", view.Execution.State)
	}
	var afterRead models.ControlOperation
	if err := fixture.db.First(&afterRead, operation.CommitSequence).Error; err != nil {
		t.Fatalf("reload operation after read: %v", err)
	}
	if afterRead.LastCompletedStage != lastStage || afterRead.FailedStage != failedStage {
		t.Fatalf("read changed operation stages to %q/%q", afterRead.LastCompletedStage, afterRead.FailedStage)
	}

	fixture.service.beforeAdvanceOperationStage = nil
	const secondKey = "44444444-4444-4444-8444-444444444444"
	recovered, err := fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, secondKey)
	if err != nil {
		t.Fatalf("recovery apply error = %v", err)
	}
	if recovered.Execution.State != agent.ProposalExecutionApplied || recovered.Execution.OperationID == nil {
		t.Fatalf("recovered view = %#v, want applied bound operation", recovered)
	}
	if recovered.Execution.OperationKind == nil || *recovered.Execution.OperationKind != string(operationKindModelRouteScheduleApply) {
		t.Fatalf("recovered operation kind = %#v, want model route apply", recovered.Execution.OperationKind)
	}
	updated := loadCreatedGroupModels(t, fixture, groupID)
	if updated[0].Weight == nil || *updated[0].Weight != 3 {
		t.Fatalf("recovered group models = %#v, want one weight mutation", updated)
	}
}

func TestChangeProposalRecoveryAfterSnapshotPublicationFailure(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "proposal-publish-failure-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	principal := agent.Principal{CredentialID: 26, Scopes: []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply}}
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
		GroupID: groupID, EntryID: groupModels[0].EntryID,
		Weight: agent.ProposalInt{Set: true, Value: 7}, Priority: agent.ProposalInt{Set: true, Value: 1},
		ExpectedWeight: agent.ProposalInt{Set: true, Value: 1}, ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
		t.Fatal(err)
	}
	publishSnapshot := fixture.service.publishSnapshot
	failed := true
	fixture.service.publishSnapshot = func(input state.CompileInput) (*state.ConfigSnapshot, error) {
		if failed {
			return nil, errors.New("test snapshot publication failure")
		}
		return publishSnapshot(input)
	}
	_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "77777777-7777-4777-8777-777777777777")
	assertAPIErrorCode(t, err, app_errors.ErrControlOperationIncomplete.Code)
	var operation models.ControlOperation
	if err := fixture.db.Where("proposal_id = ?", proposal.ProposalID).First(&operation).Error; err != nil {
		t.Fatal(err)
	}
	if operation.LastCompletedStage != string(operationStageDBCommitted) || operation.FailedStage != string(operationStageSnapshotPublished) {
		t.Fatalf("operation stages = %q/%q, want db_committed/snapshot_published", operation.LastCompletedStage, operation.FailedStage)
	}
	failed = false
	view, err := fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "88888888-8888-4888-8888-888888888888")
	if err != nil {
		t.Fatal(err)
	}
	if view.Execution.State != agent.ProposalExecutionApplied {
		t.Fatalf("recovered execution state = %q, want applied", view.Execution.State)
	}
}

func TestChangeProposalApplyRejectsRuntimeRevisionAndExpectedCurrentConflicts(t *testing.T) {
	t.Run("runtime epoch", func(t *testing.T) {
		fixture := newServiceFixture(t)
		groupID := createGroupWithCredentials(t, fixture, "proposal-epoch-secret")
		groupModels := loadCreatedGroupModels(t, fixture, groupID)
		principal := agent.Principal{CredentialID: 21, Scopes: []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply}}
		proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
			GroupID: groupID, EntryID: groupModels[0].EntryID,
			Weight: agent.ProposalInt{Set: true, Value: 2}, Priority: agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight: agent.ProposalInt{Set: true, Value: 1}, ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
			t.Fatal(err)
		}
		if err := fixture.db.Model(&models.AgentChangeProposal{}).Where("id = ?", proposal.ProposalID).
			Update("approved_runtime_epoch", fixture.manager.RuntimeEpoch()+1).Error; err != nil {
			t.Fatal(err)
		}
		_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "11111111-1111-4111-8111-111111111111")
		assertAPIErrorCode(t, err, errAgentChangeProposalRuntimeConflict.Code)
	})

	t.Run("snapshot revision", func(t *testing.T) {
		fixture := newServiceFixture(t)
		groupID := createGroupWithCredentials(t, fixture, "proposal-revision-secret")
		groupModels := loadCreatedGroupModels(t, fixture, groupID)
		principal := agent.Principal{CredentialID: 22, Scopes: []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply}}
		proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
			GroupID: groupID, EntryID: groupModels[0].EntryID,
			Weight: agent.ProposalInt{Set: true, Value: 3}, Priority: agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight: agent.ProposalInt{Set: true, Value: 1}, ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
			t.Fatal(err)
		}
		if err := fixture.db.Model(&models.AgentChangeProposal{}).Where("id = ?", proposal.ProposalID).
			Update("base_snapshot_revision", fixture.manager.Current().Revision+1).Error; err != nil {
			t.Fatal(err)
		}
		_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "22222222-2222-4222-8222-222222222222")
		assertAPIErrorCode(t, err, errAgentChangeProposalRevisionConflict.Code)
	})

	t.Run("expected current", func(t *testing.T) {
		fixture := newServiceFixture(t)
		groupID := createGroupWithCredentials(t, fixture, "proposal-value-secret")
		groupModels := loadCreatedGroupModels(t, fixture, groupID)
		principal := agent.Principal{CredentialID: 23, Scopes: []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply}}
		proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
			GroupID: groupID, EntryID: groupModels[0].EntryID,
			Weight: agent.ProposalInt{Set: true, Value: 4}, Priority: agent.ProposalInt{Set: true, Value: 1},
			ExpectedWeight: agent.ProposalInt{Set: true, Value: 1}, ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
		}}})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
			t.Fatal(err)
		}
		changedWeight := 2
		groupModels[0].Weight = &changedWeight
		encoded, err := json.Marshal(groupModels)
		if err != nil {
			t.Fatal(err)
		}
		if err := fixture.db.Model(&models.Group{}).Where("id = ?", groupID).Update("models", models.JSON(encoded)).Error; err != nil {
			t.Fatal(err)
		}
		_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "33333333-3333-4333-8333-333333333333")
		assertAPIErrorCode(t, err, errAgentChangeProposalValueConflict.Code)
	})
}

func TestChangeProposalRecoveryAfterCompletedStageRecordFailure(t *testing.T) {
	fixture := newServiceFixture(t)
	groupID := createGroupWithCredentials(t, fixture, "proposal-completed-stage-secret")
	groupModels := loadCreatedGroupModels(t, fixture, groupID)
	principal := agent.Principal{CredentialID: 24, Scopes: []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply}}
	proposal, err := fixture.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
		GroupID: groupID, EntryID: groupModels[0].EntryID,
		Weight: agent.ProposalInt{Set: true, Value: 5}, Priority: agent.ProposalInt{Set: true, Value: 1},
		ExpectedWeight: agent.ProposalInt{Set: true, Value: 1}, ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fixture.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
		t.Fatal(err)
	}
	fixture.service.beforeAdvanceOperationStage = func(_ context.Context, operation *models.ControlOperation, stage operationStage) error {
		if operation.OperationKind == string(operationKindModelRouteScheduleApply) && stage == operationStageCompleted {
			return errors.New("test completed stage record failure")
		}
		return nil
	}
	_, err = fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "44444444-4444-4444-8444-444444444444")
	assertAPIErrorCode(t, err, app_errors.ErrControlOperationIncomplete.Code)

	var operation models.ControlOperation
	if err := fixture.db.Where("proposal_id = ?", proposal.ProposalID).First(&operation).Error; err != nil {
		t.Fatal(err)
	}
	if operation.LastCompletedStage != string(operationStageSnapshotPublished) || operation.FailedStage != string(operationStageCompleted) {
		t.Fatalf("operation stages = %q/%q, want snapshot_published/completed", operation.LastCompletedStage, operation.FailedStage)
	}
	fixture.service.beforeAdvanceOperationStage = nil
	view, err := fixture.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "55555555-5555-4555-8555-555555555555")
	if err != nil {
		t.Fatal(err)
	}
	if view.Execution.State != agent.ProposalExecutionApplied {
		t.Fatalf("recovered execution state = %q, want applied", view.Execution.State)
	}
	updated := loadCreatedGroupModels(t, fixture, groupID)
	if updated[0].Weight == nil || *updated[0].Weight != 5 {
		t.Fatalf("updated models = %#v, want weight 5", updated)
	}
}

func TestChangeProposalApprovalExpiresAcrossRuntimeManagerRestart(t *testing.T) {
	first, dsn := newFileServiceFixture(t)
	groupID := createGroupWithCredentials(t, first, "proposal-restart-secret")
	groupModels := loadCreatedGroupModels(t, first, groupID)
	principal := agent.Principal{CredentialID: 25, Scopes: []agent.Scope{agent.ScopeChangesPropose, agent.ScopeChangesApply}}
	proposal, err := first.service.CreateChangeProposal(t.Context(), principal, agent.CreateChangeProposalInput{Updates: []agent.ChangeProposalUpdateInput{{
		GroupID: groupID, EntryID: groupModels[0].EntryID,
		Weight: agent.ProposalInt{Set: true, Value: 6}, Priority: agent.ProposalInt{Set: true, Value: 1},
		ExpectedWeight: agent.ProposalInt{Set: true, Value: 1}, ExpectedPriority: agent.ProposalInt{Set: true, Value: 1},
	}}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := first.service.ApproveChangeProposal(t.Context(), proposal.ProposalID, ApproveChangeProposalInput{ApprovedBy: "admin"}); err != nil {
		t.Fatal(err)
	}

	second := newServiceFixtureWithDSN(t, dsn)
	if _, err := second.manager.Publish(mustBuildCompileInput(t, second.db)); err != nil {
		t.Fatalf("publish restarted runtime snapshot: %v", err)
	}
	_, err = second.service.ApplyChangeProposal(t.Context(), principal, proposal.ProposalID, "66666666-6666-4666-8666-666666666666")
	assertAPIErrorCode(t, err, errAgentChangeProposalRuntimeConflict.Code)
}
