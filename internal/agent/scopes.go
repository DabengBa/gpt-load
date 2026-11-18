// Package agent implements the control-owned Agent machine surface: an
// independently authenticated surface for redacted diagnostics and scoped
// model-route maintenance. It never creates a second fact store.
package agent

// Scope is one explicitly grantable Agent capability. Unknown or empty scopes
// are rejected at credential creation and at request authorization.
type Scope string

const (
	// ScopeDiagnosticsRead grants the redacted read model and evidence metadata.
	ScopeDiagnosticsRead Scope = "diagnostics:read"
	// ScopeChangesPropose grants creating model-route change proposals.
	ScopeChangesPropose Scope = "changes:propose"
	// ScopeChangesApply grants applying an already approved change proposal.
	ScopeChangesApply Scope = "changes:apply"
	// ScopeRoutesWrite grants configuring model-route entries without a proposal.
	ScopeRoutesWrite Scope = "routes:write"
)

// GrantableScopes is the complete, ordered set of scopes an administrator may
// assign. Approval and raw-evidence scopes are intentionally unrepresentable.
var GrantableScopes = []Scope{
	ScopeDiagnosticsRead,
	ScopeChangesPropose,
	ScopeChangesApply,
	ScopeRoutesWrite,
}

// Valid reports whether the scope is grantable.
func (scope Scope) Valid() bool {
	for _, candidate := range GrantableScopes {
		if candidate == scope {
			return true
		}
	}
	return false
}
