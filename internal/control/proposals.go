package control

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"unicode"

	"github.com/gin-gonic/gin"
	"gorm.io/gorm"

	"gpt-load/internal/agent"
	"gpt-load/internal/platform/canonicaljson"
	app_errors "gpt-load/internal/platform/errors"
	"gpt-load/internal/platform/response"
	"gpt-load/internal/state"
	stateloader "gpt-load/internal/state/loader"
	"gpt-load/internal/storage/models"
)

// Keep the old local name available to package-level tests and callers while
// using the durable operation kind required by the control-plane contract.
const operationKindChangeProposalApply = operationKindModelRouteScheduleApply

const (
	ProposalStatePending  = agent.ProposalStatePending
	ProposalStateApproved = agent.ProposalStateApproved
	ProposalStateRevoked  = agent.ProposalStateRevoked
)

var (
	errAgentChangeProposalNotFound    = app_errors.ErrResourceNotFound
	errAgentChangeProposalNotApproved = &app_errors.APIError{
		HTTPStatus: http.StatusConflict,
		Code:       "CHANGE_PROPOSAL_NOT_APPROVED",
		Message:    "Change proposal is not approved",
	}
	errAgentChangeProposalRevoked = &app_errors.APIError{
		HTTPStatus: http.StatusConflict,
		Code:       "CHANGE_PROPOSAL_REVOKED",
		Message:    "Change proposal was revoked",
	}
	errAgentChangeProposalRuntimeConflict = &app_errors.APIError{
		HTTPStatus: http.StatusConflict,
		Code:       "CHANGE_PROPOSAL_RUNTIME_CONFLICT",
		Message:    "Change proposal approval is from another runtime epoch",
	}
	errAgentChangeProposalRevisionConflict = &app_errors.APIError{
		HTTPStatus: http.StatusConflict,
		Code:       "CHANGE_PROPOSAL_REVISION_CONFLICT",
		Message:    "Snapshot revision changed since the proposal was created",
	}
	errAgentChangeProposalValueConflict = &app_errors.APIError{
		HTTPStatus: http.StatusConflict,
		Code:       "CHANGE_PROPOSAL_VALUE_CONFLICT",
		Message:    "Route entry value changed since the proposal was created",
	}
	errAgentChangeProposalStateConflict = &app_errors.APIError{
		HTTPStatus: http.StatusConflict,
		Code:       "CHANGE_PROPOSAL_STATE_CONFLICT",
		Message:    "Change proposal is not in the required state",
	}
)

// ApproveChangeProposalInput is the administrator approval request body. The
// request is only reachable through the administrator-authenticated control
// module; Agent credentials cannot call this method or route.
type ApproveChangeProposalInput struct {
	ApprovedBy string `json:"approved_by"`
}

// ApplyChangeProposalResult is retained for control-side callers that need to
// distinguish a first execution from a proposal-bound replay.
type ApplyChangeProposalResult struct {
	View             agent.ChangeProposalView
	Replayed         bool
	OperationID      string
	ResourceIdentity string
}

var _ agent.ChangeProposalService = (*Service)(nil)

type proposalValueConflictData struct {
	GroupID  uint   `json:"group_id"`
	EntryID  string `json:"entry_id"`
	Field    string `json:"field"`
	Expected int    `json:"expected"`
	Actual   int    `json:"actual"`
}

type proposalOperationResult struct {
	ProposalID  string `json:"proposal_id"`
	OperationID string `json:"operation_id"`
}

type proposalApplyDigestBody struct {
	ProposalID string `json:"proposal_id"`
}

// CreateChangeProposal creates an immutable proposal and captures the
// published snapshot revision. It validates the target entries but performs
// no route or runtime mutation.
func (s *Service) CreateChangeProposal(
	ctx context.Context,
	principal agent.Principal,
	request agent.CreateChangeProposalInput,
) (agent.ChangeProposalView, error) {
	if s == nil || s.db == nil || s.manager == nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	if principal.CredentialID == 0 || !agentPrincipalHasScope(principal, agent.ScopeChangesPropose) {
		return agent.ChangeProposalView{}, app_errors.ErrForbidden
	}
	updates, err := normalizeProposalUpdates(request.Updates)
	if err != nil {
		return agent.ChangeProposalView{}, err
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	current := s.manager.Current()
	if current == nil || current.Revision == 0 {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	for _, update := range updates {
		if !snapshotHasProposalEntry(current, update.GroupID, update.EntryID) {
			return agent.ChangeProposalView{}, app_errors.NewAPIErrorWithData(
				app_errors.ErrValidation,
				proposalValueConflictData{
					GroupID: update.GroupID,
					EntryID: update.EntryID,
					Field:   "entry",
				},
			)
		}
	}
	nowMS, err := safeEpochMilliseconds(s.now())
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	proposalID, err := newProposalID()
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	encodedUpdates, err := json.Marshal(updates)
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	row := models.AgentChangeProposal{
		ID:                   proposalID,
		CreatedAtMS:          nowMS,
		UpdatedAtMS:          nowMS,
		BaseSnapshotRevision: current.Revision,
		CreatorCredentialID:  principal.CredentialID,
		State:                agent.ProposalStatePending,
		Updates:              models.JSON(encodedUpdates),
	}
	if err := s.withControlTransaction(ctx, func(tx *gorm.DB) error {
		return tx.Create(&row).Error
	}); err != nil {
		return agent.ChangeProposalView{}, app_errors.ParseDBError(err)
	}
	return mapChangeProposalView(row, nil)
}

// GetChangeProposal returns the proposal and the operation-derived execution
// projection. It is deliberately read-only: it never runs recovery.
func (s *Service) GetChangeProposal(
	ctx context.Context,
	proposalID string,
) (agent.ChangeProposalView, error) {
	if s == nil || s.db == nil || !validProposalID(proposalID) {
		return agent.ChangeProposalView{}, app_errors.ErrBadRequest
	}
	row, err := s.loadChangeProposal(ctx, proposalID)
	if err != nil {
		return agent.ChangeProposalView{}, err
	}
	operation, err := s.loadProposalOperation(ctx, proposalID)
	if err != nil {
		return agent.ChangeProposalView{}, err
	}
	return mapChangeProposalView(row, operation)
}

// ApproveChangeProposal stores an administrator approval, runtime epoch and
// approval timestamp. The proposal base revision must still be current.
func (s *Service) ApproveChangeProposal(
	ctx context.Context,
	proposalID string,
	input ApproveChangeProposalInput,
) (models.AgentChangeProposal, error) {
	if s == nil || s.db == nil || s.manager == nil || !validProposalID(proposalID) {
		return models.AgentChangeProposal{}, app_errors.ErrBadRequest
	}
	approvedBy := strings.TrimSpace(input.ApprovedBy)
	if approvedBy == "" || len(approvedBy) > 64 || strings.IndexFunc(approvedBy, unicode.IsSpace) >= 0 {
		return models.AgentChangeProposal{}, app_errors.ErrValidation
	}
	nowMS, err := safeEpochMilliseconds(s.now())
	if err != nil {
		return models.AgentChangeProposal{}, app_errors.ErrInternalServer
	}
	epoch := s.manager.RuntimeEpoch()
	if epoch <= 0 {
		return models.AgentChangeProposal{}, app_errors.ErrInternalServer
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	current := s.manager.Current()
	if current == nil || current.Revision == 0 {
		return models.AgentChangeProposal{}, app_errors.ErrInternalServer
	}
	var row models.AgentChangeProposal
	err = s.withControlTransaction(ctx, func(tx *gorm.DB) error {
		if err := tx.First(&row, "id = ?", proposalID).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return errAgentChangeProposalNotFound
			}
			return app_errors.ParseDBError(err)
		}
		if row.State != agent.ProposalStatePending {
			return errAgentChangeProposalStateConflict
		}
		if row.BaseSnapshotRevision != current.Revision {
			return errAgentChangeProposalRevisionConflict
		}
		row.State = agent.ProposalStateApproved
		row.ApprovedAtMS = &nowMS
		row.ApprovedBy = &approvedBy
		row.ApprovedRuntimeEpoch = &epoch
		row.UpdatedAtMS = nowMS
		return tx.Save(&row).Error
	})
	if err != nil {
		return models.AgentChangeProposal{}, err
	}
	return row, nil
}

// RevokeChangeProposal revokes a pending or approved proposal that has not yet
// been durably bound to an operation. A committed operation is never detached.
func (s *Service) RevokeChangeProposal(
	ctx context.Context,
	proposalID string,
) (models.AgentChangeProposal, error) {
	if s == nil || s.db == nil || !validProposalID(proposalID) {
		return models.AgentChangeProposal{}, app_errors.ErrBadRequest
	}
	nowMS, err := safeEpochMilliseconds(s.now())
	if err != nil {
		return models.AgentChangeProposal{}, app_errors.ErrInternalServer
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	var row models.AgentChangeProposal
	err = s.withControlTransaction(ctx, func(tx *gorm.DB) error {
		if err := tx.First(&row, "id = ?", proposalID).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return errAgentChangeProposalNotFound
			}
			return app_errors.ParseDBError(err)
		}
		if row.State != agent.ProposalStatePending && row.State != agent.ProposalStateApproved {
			return errAgentChangeProposalStateConflict
		}
		var operation models.ControlOperation
		bindingErr := tx.Where("proposal_id = ?", proposalID).Take(&operation).Error
		if bindingErr == nil {
			return errAgentChangeProposalStateConflict
		}
		if !errors.Is(bindingErr, gorm.ErrRecordNotFound) {
			return app_errors.ParseDBError(bindingErr)
		}
		row.State = agent.ProposalStateRevoked
		row.ApprovedAtMS = nil
		row.ApprovedBy = nil
		row.ApprovedRuntimeEpoch = nil
		row.UpdatedAtMS = nowMS
		return tx.Save(&row).Error
	})
	if err != nil {
		return models.AgentChangeProposal{}, err
	}
	return row, nil
}

// ApplyChangeProposal executes an approved proposal with the durable
// proposal->operation binding. Existing bindings are recovered/replayed before
// approval, revision or expected-current checks are considered.
func (s *Service) ApplyChangeProposal(
	ctx context.Context,
	principal agent.Principal,
	proposalID string,
	idempotencyKey string,
) (agent.ChangeProposalView, error) {
	if s == nil || s.db == nil || s.manager == nil || !validProposalID(proposalID) {
		return agent.ChangeProposalView{}, app_errors.ErrBadRequest
	}
	if principal.CredentialID == 0 || !agentPrincipalHasScope(principal, agent.ScopeChangesApply) {
		return agent.ChangeProposalView{}, app_errors.ErrForbidden
	}
	if err := validateIdempotencyKey(idempotencyKey); err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInvalidIdempotencyKey
	}
	digest, err := proposalApplyDigest(principal.CredentialID, proposalID)
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	var operation models.ControlOperation
	query := s.db.WithContext(ctx).Where("idempotency_key = ?", idempotencyKey).Take(&operation)
	if query.Error == nil {
		if err := validateProposalOperationComparator(&operation, digest.Digest); err != nil {
			return agent.ChangeProposalView{}, err
		}
		return s.replayBoundProposalLocked(ctx, &operation)
	}
	if !errors.Is(query.Error, gorm.ErrRecordNotFound) {
		return agent.ChangeProposalView{}, app_errors.ParseDBError(query.Error)
	}

	query = s.db.WithContext(ctx).Where("proposal_id = ?", proposalID).Take(&operation)
	if query.Error == nil {
		if operation.OperationKind != string(operationKindModelRouteScheduleApply) {
			return agent.ChangeProposalView{}, app_errors.ErrInternalServer
		}
		return s.replayBoundProposalLocked(ctx, &operation)
	}
	if !errors.Is(query.Error, gorm.ErrRecordNotFound) {
		return agent.ChangeProposalView{}, app_errors.ParseDBError(query.Error)
	}
	if err := s.enforceOperationRecoveryBarrierLocked(ctx, 0); err != nil {
		return agent.ChangeProposalView{}, err
	}

	proposal, err := s.loadChangeProposal(ctx, proposalID)
	if err != nil {
		return agent.ChangeProposalView{}, err
	}
	if proposal.State == agent.ProposalStateRevoked {
		return agent.ChangeProposalView{}, errAgentChangeProposalRevoked
	}
	if proposal.State != agent.ProposalStateApproved || proposal.ApprovedRuntimeEpoch == nil {
		return agent.ChangeProposalView{}, errAgentChangeProposalNotApproved
	}
	current := s.manager.Current()
	if current == nil || current.Revision == 0 {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	if proposal.BaseSnapshotRevision != current.Revision {
		return agent.ChangeProposalView{}, errAgentChangeProposalRevisionConflict
	}
	if *proposal.ApprovedRuntimeEpoch != s.manager.RuntimeEpoch() {
		return agent.ChangeProposalView{}, errAgentChangeProposalRuntimeConflict
	}

	operationID, err := newProposalOperationID(s.operationRandom)
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	requiredStages, err := operationRequiredStages(operationKindModelRouteScheduleApply)
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	encodedStages, err := json.Marshal(requiredStages)
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	canonicalResult, err := canonicaljson.Marshal(proposalOperationResult{
		ProposalID: proposalID, OperationID: operationID,
	})
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	nowMS, err := safeEpochMilliseconds(s.now())
	if err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	proposalIDCopy := proposalID
	operation = models.ControlOperation{
		OperationID:        operationID,
		IdempotencyKey:     idempotencyKey,
		DigestVersion:      1,
		RequestDigest:      append([]byte(nil), digest.Digest[:]...),
		OperationKind:      string(operationKindModelRouteScheduleApply),
		ResourceIdentity:   "proposal:" + proposalID,
		ProposalID:         &proposalIDCopy,
		CanonicalResult:    canonicalResult,
		RequiredStages:     models.JSON(encodedStages),
		LastCompletedStage: string(operationStageDBCommitted),
		CreatedAtMS:        nowMS,
		UpdatedAtMS:        nowMS,
	}

	err = s.withControlTransaction(ctx, func(tx *gorm.DB) error {
		var check models.AgentChangeProposal
		if err := tx.First(&check, "id = ?", proposalID).Error; err != nil {
			if errors.Is(err, gorm.ErrRecordNotFound) {
				return errAgentChangeProposalNotFound
			}
			return app_errors.ParseDBError(err)
		}
		if check.State == agent.ProposalStateRevoked {
			return errAgentChangeProposalRevoked
		}
		if check.State != agent.ProposalStateApproved || check.ApprovedRuntimeEpoch == nil {
			return errAgentChangeProposalNotApproved
		}
		if check.BaseSnapshotRevision != current.Revision {
			return errAgentChangeProposalRevisionConflict
		}
		if *check.ApprovedRuntimeEpoch != s.manager.RuntimeEpoch() {
			return errAgentChangeProposalRuntimeConflict
		}
		var updates []agent.ChangeProposalUpdate
		if err := json.Unmarshal(check.Updates, &updates); err != nil {
			return app_errors.ErrInternalServer
		}
		if err := s.applyStoredProposalUpdates(tx, updates); err != nil {
			return err
		}
		compileInput, err := stateloader.BuildCompileInputWithProxy(
			ctx,
			tx,
			s.encryption,
			s.environmentProxy,
			s.channelRegistry,
		)
		if err != nil {
			return err
		}
		if _, err := state.Compile(compileInput); err != nil {
			return err
		}
		return tx.Create(&operation).Error
	})
	if err != nil {
		var apiErr *app_errors.APIError
		if errors.As(err, &apiErr) {
			return agent.ChangeProposalView{}, err
		}
		return agent.ChangeProposalView{}, app_errors.ParseDBError(err)
	}
	if err := s.recoverOperationLocked(ctx, &operation); err != nil {
		s.wakeOperationRecovery()
		return agent.ChangeProposalView{}, s.operationIncompleteError(operation)
	}
	return mapChangeProposalView(proposal, &operation)
}

// GetControlOperation returns durable operation metadata without triggering
// recovery or changing any stage.
func (s *Service) GetControlOperation(
	ctx context.Context,
	operationID string,
) (agent.ControlOperationView, error) {
	if s == nil || s.db == nil || !validProposalID(operationID) {
		return agent.ControlOperationView{}, app_errors.ErrBadRequest
	}
	var operation models.ControlOperation
	if err := s.db.WithContext(ctx).Where("operation_id = ?", operationID).Take(&operation).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return agent.ControlOperationView{}, app_errors.ErrResourceNotFound
		}
		return agent.ControlOperationView{}, app_errors.ParseDBError(err)
	}
	if operation.ProposalID == nil || !validProposalID(*operation.ProposalID) {
		return agent.ControlOperationView{}, app_errors.ErrResourceNotFound
	}
	return mapControlOperationView(operation)
}

func normalizeProposalUpdates(
	inputs []agent.ChangeProposalUpdateInput,
) ([]agent.ChangeProposalUpdate, error) {
	if len(inputs) == 0 {
		return nil, app_errors.ErrValidation
	}
	updates := make([]agent.ChangeProposalUpdate, 0, len(inputs))
	seen := make(map[string]struct{}, len(inputs))
	for _, input := range inputs {
		entryID := strings.TrimSpace(input.EntryID)
		if input.GroupID == 0 || entryID == "" ||
			!input.Weight.Set || !input.Priority.Set ||
			!input.ExpectedWeight.Set || !input.ExpectedPriority.Set {
			return nil, app_errors.ErrValidation
		}
		if input.Weight.Value < 0 || input.Weight.Value > state.MaxWeight || input.Priority.Value < 1 ||
			input.ExpectedWeight.Value < 0 || input.ExpectedWeight.Value > state.MaxWeight ||
			input.ExpectedPriority.Value < 1 {
			return nil, app_errors.ErrValidation
		}
		key := fmt.Sprintf("%d\x00%s", input.GroupID, entryID)
		if _, exists := seen[key]; exists {
			return nil, app_errors.ErrValidation
		}
		seen[key] = struct{}{}
		updates = append(updates, agent.ChangeProposalUpdate{
			GroupID:          input.GroupID,
			EntryID:          entryID,
			Weight:           input.Weight.Value,
			Priority:         input.Priority.Value,
			ExpectedWeight:   input.ExpectedWeight.Value,
			ExpectedPriority: input.ExpectedPriority.Value,
		})
	}
	return updates, nil
}

func snapshotHasProposalEntry(snapshot *state.ConfigSnapshot, groupID uint, entryID string) bool {
	if snapshot == nil {
		return false
	}
	catalog, exists := snapshot.GroupCatalog[groupID]
	if exists {
		for _, model := range catalog.Models {
			if model.EntryID == entryID {
				return true
			}
		}
		return false
	}
	if group, exists := snapshot.Groups[groupID]; exists {
		for _, model := range group.Models {
			if model.EntryID == entryID {
				return true
			}
		}
	}
	if group, exists := snapshot.DisabledGroups[groupID]; exists {
		for _, model := range group.Models {
			if model.EntryID == entryID {
				return true
			}
		}
	}
	return false
}

func (s *Service) applyStoredProposalUpdates(tx *gorm.DB, updates []agent.ChangeProposalUpdate) error {
	if len(updates) == 0 {
		return app_errors.ErrValidation
	}
	byGroup := make(map[uint][]modelRouteSchedulePatchUpdate)
	seen := make(map[string]struct{}, len(updates))
	for _, update := range updates {
		if update.GroupID == 0 || strings.TrimSpace(update.EntryID) == "" ||
			update.Weight < 0 || update.Weight > state.MaxWeight || update.Priority < 1 {
			return app_errors.ErrValidation
		}
		key := fmt.Sprintf("%d\x00%s", update.GroupID, update.EntryID)
		if _, exists := seen[key]; exists {
			return app_errors.ErrInternalServer
		}
		seen[key] = struct{}{}
		group, err := loadGroupRow(tx, update.GroupID)
		if err != nil {
			return err
		}
		var entries []groupModelEntry
		if err := decodeGroupDiscoveryJSON(group.Models, &entries); err != nil {
			return app_errors.ErrInternalServer
		}
		found := false
		for _, entry := range entries {
			if entry.EntryID != update.EntryID {
				continue
			}
			found = true
			actualWeight, actualPriority := 1, 1
			if entry.Weight != nil {
				actualWeight = *entry.Weight
			}
			if entry.Priority != nil {
				actualPriority = *entry.Priority
			}
			if actualWeight != update.ExpectedWeight {
				return app_errors.NewAPIErrorWithData(
					errAgentChangeProposalValueConflict,
					proposalValueConflictData{
						GroupID: update.GroupID, EntryID: update.EntryID,
						Field: "weight", Expected: update.ExpectedWeight, Actual: actualWeight,
					},
				)
			}
			if actualPriority != update.ExpectedPriority {
				return app_errors.NewAPIErrorWithData(
					errAgentChangeProposalValueConflict,
					proposalValueConflictData{
						GroupID: update.GroupID, EntryID: update.EntryID,
						Field: "priority", Expected: update.ExpectedPriority, Actual: actualPriority,
					},
				)
			}
			break
		}
		if !found {
			return app_errors.NewAPIErrorWithData(
				app_errors.ErrValidation,
				proposalValueConflictData{GroupID: update.GroupID, EntryID: update.EntryID, Field: "entry"},
			)
		}
		byGroup[update.GroupID] = append(byGroup[update.GroupID], modelRouteSchedulePatchUpdate{
			GroupID:  update.GroupID,
			EntryID:  update.EntryID,
			Weight:   optionalField[int]{Set: true, Value: update.Weight},
			Priority: optionalField[int]{Set: true, Value: update.Priority},
		})
	}
	groupIDs := make([]uint, 0, len(byGroup))
	for groupID := range byGroup {
		groupIDs = append(groupIDs, groupID)
	}
	sort.Slice(groupIDs, func(left, right int) bool { return groupIDs[left] < groupIDs[right] })
	for _, groupID := range groupIDs {
		if err := s.applyModelRouteScheduleGroupPatch(tx, groupID, byGroup[groupID]); err != nil {
			return err
		}
	}
	return nil
}

func (s *Service) loadChangeProposal(ctx context.Context, proposalID string) (models.AgentChangeProposal, error) {
	var row models.AgentChangeProposal
	if err := s.db.WithContext(ctx).Where("id = ?", proposalID).Take(&row).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return models.AgentChangeProposal{}, errAgentChangeProposalNotFound
		}
		return models.AgentChangeProposal{}, app_errors.ParseDBError(err)
	}
	return row, nil
}

func (s *Service) loadProposalOperation(ctx context.Context, proposalID string) (*models.ControlOperation, error) {
	var operation models.ControlOperation
	err := s.db.WithContext(ctx).Where("proposal_id = ?", proposalID).Take(&operation).Error
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return nil, nil
	}
	if err != nil {
		return nil, app_errors.ParseDBError(err)
	}
	return &operation, nil
}

func (s *Service) replayBoundProposalLocked(
	ctx context.Context,
	operation *models.ControlOperation,
) (agent.ChangeProposalView, error) {
	if err := s.enforceOperationRecoveryBarrierLocked(ctx, operation.CommitSequence); err != nil {
		return agent.ChangeProposalView{}, err
	}
	if operation.CompactedAtMS != nil {
		if operation.CompletedAtMS == nil || validateSafeMilliseconds(*operation.CompletedAtMS) != nil {
			return agent.ChangeProposalView{}, app_errors.ErrInternalServer
		}
		return agent.ChangeProposalView{}, app_errors.NewAPIErrorWithData(
			app_errors.ErrIdempotencyResultExpired,
			operationExpiredData{
				OperationID:      operation.OperationID,
				OperationKind:    operationKind(operation.OperationKind),
				ResourceIdentity: operation.ResourceIdentity,
				CompletedAtMS:    *operation.CompletedAtMS,
			},
		)
	}
	if operation.LastCompletedStage != string(operationStageCompleted) {
		if err := s.recoverOperationLocked(ctx, operation); err != nil {
			s.wakeOperationRecovery()
			return agent.ChangeProposalView{}, s.operationIncompleteError(*operation)
		}
	}
	proposalID := ""
	if operation.ProposalID != nil {
		proposalID = *operation.ProposalID
	}
	if !validProposalID(proposalID) {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	proposal, err := s.loadChangeProposal(ctx, proposalID)
	if err != nil {
		return agent.ChangeProposalView{}, err
	}
	return mapChangeProposalView(proposal, operation)
}

func mapChangeProposalView(
	proposal models.AgentChangeProposal,
	operation *models.ControlOperation,
) (agent.ChangeProposalView, error) {
	var updates []agent.ChangeProposalUpdate
	if err := json.Unmarshal(proposal.Updates, &updates); err != nil {
		return agent.ChangeProposalView{}, app_errors.ErrInternalServer
	}
	view := agent.ChangeProposalView{
		SchemaVersion:        agent.SchemaVersion,
		ProposalID:           proposal.ID,
		State:                proposal.State,
		BaseSnapshotRevision: proposal.BaseSnapshotRevision,
		CreatorCredentialID:  proposal.CreatorCredentialID,
		CreatedAtMS:          proposal.CreatedAtMS,
		UpdatedAtMS:          proposal.UpdatedAtMS,
		Updates:              updates,
		ApprovedAtMS:         proposal.ApprovedAtMS,
		ApprovedBy:           proposal.ApprovedBy,
		ApprovedRuntimeEpoch: proposal.ApprovedRuntimeEpoch,
		Execution: agent.ChangeProposalExecution{
			State: agent.ProposalExecutionNotBound,
		},
	}
	if operation == nil {
		return view, nil
	}
	operationID := operation.OperationID
	operationKindValue := operation.OperationKind
	view.Execution = agent.ChangeProposalExecution{
		State:              deriveProposalExecutionState(*operation),
		OperationID:        &operationID,
		OperationKind:      &operationKindValue,
		LastCompletedStage: operation.LastCompletedStage,
		FailedStage:        operation.FailedStage,
		CompletedAtMS:      operation.CompletedAtMS,
		CanReconcile:       operation.CompletedAtMS == nil,
	}
	return view, nil
}

func deriveProposalExecutionState(operation models.ControlOperation) string {
	if operation.CompletedAtMS != nil {
		return agent.ProposalExecutionApplied
	}
	if operation.FailedStage != "" {
		return agent.ProposalExecutionFailed
	}
	switch operation.LastCompletedStage {
	case string(operationStageCompleted):
		return agent.ProposalExecutionApplied
	case string(operationStageDBCommitted), string(operationStageSnapshotPublished):
		return agent.ProposalExecutionExecuting
	default:
		return agent.ProposalExecutionFailed
	}
}

func mapControlOperationView(operation models.ControlOperation) (agent.ControlOperationView, error) {
	var required []operationStage
	if len(operation.RequiredStages) > 0 {
		if err := json.Unmarshal(operation.RequiredStages, &required); err != nil {
			return agent.ControlOperationView{}, app_errors.ErrInternalServer
		}
	}
	requiredStages := make([]string, 0, len(required))
	for _, stage := range required {
		requiredStages = append(requiredStages, string(stage))
	}
	return agent.ControlOperationView{
		SchemaVersion:      agent.SchemaVersion,
		OperationID:        operation.OperationID,
		OperationKind:      operation.OperationKind,
		ResourceIdentity:   operation.ResourceIdentity,
		RequiredStages:     requiredStages,
		LastCompletedStage: operation.LastCompletedStage,
		FailedStage:        operation.FailedStage,
		CompletedAtMS:      operation.CompletedAtMS,
		CanReconcile:       operation.CompletedAtMS == nil,
	}, nil
}

func proposalApplyDigest(credentialID uint, proposalID string) (idempotencyDigestResult, error) {
	body, err := canonicalIdempotencyBody(proposalApplyDigestBody{ProposalID: proposalID})
	if err != nil {
		return idempotencyDigestResult{}, err
	}
	return buildIdempotencyDigest(idempotencyDigestInput{
		Version:         1,
		Method:          "POST",
		OperationKind:   operationKindModelRouteScheduleApply,
		PathTemplate:    "/api/agent/v1/change-proposals/{proposal_id}/apply",
		ResourceLocator: proposalID,
		AuthScopeID:     fmt.Sprintf("agent-credential:%d", credentialID),
		CanonicalBody:   body,
	})
}

func validateProposalOperationComparator(operation *models.ControlOperation, digest [32]byte) error {
	if operation == nil || operation.DigestVersion != 1 || len(operation.RequestDigest) != len(digest) ||
		!operationKind(operation.OperationKind).valid() {
		return app_errors.ErrInternalServer
	}
	if operation.OperationKind != string(operationKindModelRouteScheduleApply) ||
		!strings.HasPrefix(operation.ResourceIdentity, "proposal:") ||
		subtle.ConstantTimeCompare(operation.RequestDigest, digest[:]) != 1 {
		return app_errors.NewAPIErrorWithData(
			app_errors.ErrIdempotencyKeyReused,
			operationErrorData{
				OperationID:   operation.OperationID,
				OperationKind: operationKind(operation.OperationKind),
			},
		)
	}
	return nil
}

func agentPrincipalHasScope(principal agent.Principal, scope agent.Scope) bool {
	if principal.HasScope(scope) {
		return true
	}
	for _, candidate := range principal.Scopes {
		if candidate == scope {
			return true
		}
	}
	return false
}

func validProposalID(value string) bool {
	return validateIdempotencyKey(value) == nil
}

func newProposalID() (string, error) {
	return newOperationID(rand.Reader)
}

func newProposalOperationID(random io.Reader) (string, error) {
	if random == nil {
		random = rand.Reader
	}
	return newOperationID(random)
}

func changeProposalMutationLocator(c *gin.Context) string {
	proposalID := c.Param("proposal_id")
	if !validProposalID(proposalID) {
		return "change-proposal:unknown"
	}
	return "change-proposal:" + proposalID
}

func (s *Server) handleApproveChangeProposal(c *gin.Context) {
	principal, ok := currentControlPrincipal(c)
	if !ok || principal.Type != controlPrincipalAdmin {
		writeServiceError(c, "approve_change_proposal", app_errors.ErrForbidden)
		return
	}
	proposalID := c.Param("proposal_id")
	if !validProposalID(proposalID) {
		writeServiceError(c, "approve_change_proposal", app_errors.ErrBadRequest)
		return
	}
	var request ApproveChangeProposalInput
	if err := bindStrictJSON(c, &request); err != nil {
		writeServiceError(c, "approve_change_proposal", mapControlJSONError(err))
		return
	}
	if _, err := s.service.ApproveChangeProposal(c.Request.Context(), proposalID, request); err != nil {
		writeServiceError(c, "approve_change_proposal", err)
		return
	}
	result, err := s.service.GetChangeProposal(c.Request.Context(), proposalID)
	if err != nil {
		writeServiceError(c, "approve_change_proposal", err)
		return
	}
	setMutationResourceLocator(c, "change-proposal:"+proposalID)
	response.SuccessI18n(c, "common.success", result)
}

func (s *Server) handleRevokeChangeProposal(c *gin.Context) {
	principal, ok := currentControlPrincipal(c)
	if !ok || principal.Type != controlPrincipalAdmin {
		writeServiceError(c, "revoke_change_proposal", app_errors.ErrForbidden)
		return
	}
	proposalID := c.Param("proposal_id")
	if !validProposalID(proposalID) {
		writeServiceError(c, "revoke_change_proposal", app_errors.ErrBadRequest)
		return
	}
	if _, err := s.service.RevokeChangeProposal(c.Request.Context(), proposalID); err != nil {
		writeServiceError(c, "revoke_change_proposal", err)
		return
	}
	result, err := s.service.GetChangeProposal(c.Request.Context(), proposalID)
	if err != nil {
		writeServiceError(c, "revoke_change_proposal", err)
		return
	}
	setMutationResourceLocator(c, "change-proposal:"+proposalID)
	response.SuccessI18n(c, "common.success", result)
}
