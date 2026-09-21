package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// Proposal approval states. Execution state is never stored on the proposal;
// it is derived from the single bound ControlOperation.
const (
	ProposalStatePending  = "pending"
	ProposalStateApproved = "approved"
	ProposalStateRevoked  = "revoked"
)

// Derived execution states for one change proposal.
const (
	ProposalExecutionNotBound  = "not_bound"
	ProposalExecutionExecuting = "executing"
	ProposalExecutionApplied   = "applied"
	ProposalExecutionFailed    = "failed"
)

// ProposalInt is an explicit integer JSON field. It distinguishes an absent
// field from an explicit zero and rejects JSON null outright: a null target is
// never equivalent to the persisted default value.
type ProposalInt struct {
	Set   bool
	Value int
}

// UnmarshalJSON accepts exactly one JSON integer and rejects null, floats and
// non-numeric values.
func (field *ProposalInt) UnmarshalJSON(data []byte) error {
	if field == nil {
		return fmt.Errorf("proposal integer receiver is nil")
	}
	field.Set = false
	field.Value = 0
	if bytes.Equal(bytes.TrimSpace(data), []byte("null")) {
		return fmt.Errorf("proposal integer must not be null")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var value int
	if err := decoder.Decode(&value); err != nil {
		return fmt.Errorf("proposal integer must be an integer: %w", err)
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return fmt.Errorf("proposal integer must contain one value")
	}
	field.Set = true
	field.Value = value
	return nil
}

// ChangeProposalUpdateInput is one requested weight/priority change. All four
// integer fields are required; expected_weight/expected_priority carry the
// persisted values observed before proposing.
type ChangeProposalUpdateInput struct {
	GroupID          uint        `json:"group_id"`
	EntryID          string      `json:"entry_id"`
	Weight           ProposalInt `json:"weight"`
	Priority         ProposalInt `json:"priority"`
	ExpectedWeight   ProposalInt `json:"expected_weight"`
	ExpectedPriority ProposalInt `json:"expected_priority"`
}

// CreateChangeProposalInput is the Agent proposal request body. The base
// snapshot revision is captured server-side at creation.
type CreateChangeProposalInput struct {
	Updates []ChangeProposalUpdateInput `json:"updates"`
}

// ChangeProposalUpdate is the immutable persisted target/expected tuple.
type ChangeProposalUpdate struct {
	GroupID          uint   `json:"group_id"`
	EntryID          string `json:"entry_id"`
	Weight           int    `json:"weight"`
	Priority         int    `json:"priority"`
	ExpectedWeight   int    `json:"expected_weight"`
	ExpectedPriority int    `json:"expected_priority"`
}

// ChangeProposalExecution is the operation-derived execution projection.
type ChangeProposalExecution struct {
	State              string  `json:"state"`
	OperationID        *string `json:"operation_id"`
	OperationKind      *string `json:"operation_kind"`
	LastCompletedStage string  `json:"last_completed_stage"`
	FailedStage        string  `json:"failed_stage"`
	CompletedAtMS      *int64  `json:"completed_at_ms"`
	CanReconcile       bool    `json:"can_reconcile"`
}

// ChangeProposalView is the stable Agent proposal projection.
type ChangeProposalView struct {
	SchemaVersion        int                     `json:"schema_version"`
	ProposalID           string                  `json:"proposal_id"`
	State                string                  `json:"state"`
	BaseSnapshotRevision uint64                  `json:"base_snapshot_revision"`
	CreatorCredentialID  uint                    `json:"creator_credential_id"`
	CreatedAtMS          int64                   `json:"created_at_ms"`
	UpdatedAtMS          int64                   `json:"updated_at_ms"`
	Updates              []ChangeProposalUpdate  `json:"updates"`
	ApprovedAtMS         *int64                  `json:"approved_at_ms"`
	ApprovedBy           *string                 `json:"approved_by"`
	ApprovedRuntimeEpoch *int64                  `json:"approved_runtime_epoch"`
	Execution            ChangeProposalExecution `json:"execution"`
}

// ControlOperationView is the read-only recovery projection of one control
// operation. It never exposes the canonical replay payload.
type ControlOperationView struct {
	SchemaVersion      int      `json:"schema_version"`
	OperationID        string   `json:"operation_id"`
	OperationKind      string   `json:"operation_kind"`
	ResourceIdentity   string   `json:"resource_identity"`
	RequiredStages     []string `json:"required_stages"`
	LastCompletedStage string   `json:"last_completed_stage"`
	FailedStage        string   `json:"failed_stage"`
	CompletedAtMS      *int64   `json:"completed_at_ms"`
	CanReconcile       bool     `json:"can_reconcile"`
}

// ChangeProposalService is the control-plane implementation injected into the
// control-owned Agent surface. The Agent package never imports control.
type ChangeProposalService interface {
	CreateChangeProposal(
		ctx context.Context,
		principal Principal,
		request CreateChangeProposalInput,
	) (ChangeProposalView, error)
	ApplyChangeProposal(
		ctx context.Context,
		principal Principal,
		proposalID string,
		idempotencyKey string,
	) (ChangeProposalView, error)
	GetChangeProposal(ctx context.Context, proposalID string) (ChangeProposalView, error)
	GetControlOperation(ctx context.Context, operationID string) (ControlOperationView, error)
}

// SetChangeProposalService wires the control-plane proposal lifecycle. Without
// it the proposal routes fail closed and capabilities report no support.
func (server *Server) SetChangeProposalService(service ChangeProposalService) {
	if server == nil {
		return
	}
	server.changeProposals = service
	server.features.ChangeProposals = service != nil
}

func (server *Server) changeProposalService() (ChangeProposalService, error) {
	if server == nil || server.changeProposals == nil {
		return nil, errAgentProposalsUnavailable
	}
	return server.changeProposals, nil
}

var errAgentProposalsUnavailable = errors.New("agent change proposals are unavailable")

// normalizeProposalID trims and validates one proposal path parameter.
func normalizeProposalID(raw string) (string, bool) {
	trimmed := strings.TrimSpace(raw)
	if !canonicalLowercaseUUIDv4.MatchString(trimmed) {
		return "", false
	}
	return trimmed, true
}
