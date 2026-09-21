package models

// AgentChangeProposal is an immutable, administrator-approved proposal that
// changes only explicit integer weight/priority values of model-route entries.
//
// The proposal stores its approval decision (state pending|approved|revoked,
// the approving administrator, the runtime epoch and time) but never stores an
// execution state: executing/applied/failed are derived from the single
// ControlOperation durably bound through ControlOperation.ProposalID, so the
// proposal can never fork into a second execution state.
type AgentChangeProposal struct {
	ID string `gorm:"primaryKey;type:char(36)"`
	// CreatedAtMS/UpdatedAtMS are epoch milliseconds owned by the application,
	// never by database auto-timestamps, so tests and recovery can drive time.
	CreatedAtMS int64 `gorm:"column:created_at_ms;not null;check:chk_agent_change_proposal_created_at,created_at_ms >= 0"`
	UpdatedAtMS int64 `gorm:"column:updated_at_ms;not null;check:chk_agent_change_proposal_updated_at,updated_at_ms >= 0"`
	// BaseSnapshotRevision is the published ConfigSnapshot revision captured at
	// creation. Approval and apply fail closed once the revision moves.
	BaseSnapshotRevision uint64 `gorm:"column:base_snapshot_revision;not null;check:chk_agent_change_proposal_base_revision,base_snapshot_revision > 0"`
	// CreatorCredentialID is the stable Agent credential identity that proposed
	// the change. It is part of the apply idempotency comparator.
	CreatorCredentialID uint `gorm:"column:creator_credential_id;not null;index:idx_agent_change_proposal_creator;check:chk_agent_change_proposal_creator,creator_credential_id > 0"`
	// State is the approval state only: pending, approved or revoked.
	State string `gorm:"column:state;type:varchar(16);not null;index:idx_agent_change_proposal_state;check:chk_agent_change_proposal_state,state IN ('pending','approved','revoked')"`
	// Updates is the immutable []ProposalUpdate payload with explicit integer
	// target and expected current values.
	Updates JSON `gorm:"column:updates;type:json;not null"`
	// ApprovedRuntimeEpoch is the process runtime epoch that approved the
	// proposal. A restarted process has a different epoch and cannot apply it.
	ApprovedRuntimeEpoch *int64 `gorm:"column:approved_runtime_epoch;check:chk_agent_change_proposal_approved_epoch,approved_runtime_epoch IS NULL OR approved_runtime_epoch > 0"`
	// ApprovedBy is the non-empty approval subject recorded by the control
	// plane. Agent tokens can never write it.
	ApprovedBy   *string `gorm:"column:approved_by;type:varchar(64);check:chk_agent_change_proposal_approved_by,approved_by IS NULL OR length(approved_by) > 0"`
	ApprovedAtMS *int64  `gorm:"column:approved_at_ms;check:chk_agent_change_proposal_approved_at,approved_at_ms IS NULL OR approved_at_ms >= 0"`
}

// ProposalUpdate is one immutable weight/priority target. Weight and Priority
// are the desired values; ExpectedWeight and ExpectedPriority are the
// persisted values observed at creation. All four are explicit integers: a
// JSON null is rejected instead of being treated as the default.
type ProposalUpdate struct {
	GroupID          uint   `json:"group_id"`
	EntryID          string `json:"entry_id"`
	Weight           int    `json:"weight"`
	Priority         int    `json:"priority"`
	ExpectedWeight   int    `json:"expected_weight"`
	ExpectedPriority int    `json:"expected_priority"`
}
