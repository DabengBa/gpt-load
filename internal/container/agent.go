package container

import (
	"go.uber.org/dig"
	"gorm.io/gorm"

	"gpt-load/internal/agent"
	"gpt-load/internal/control"
	"gpt-load/internal/debugcapture"
	"gpt-load/internal/platform/config"
	"gpt-load/internal/platform/encryption"
	"gpt-load/internal/platform/redact"
	"gpt-load/internal/platform/version"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/state"
)

// agentOptionalDependencies carries the evidence store, which is absent on
// platforms without the database-backed debug capture runtime.
type agentOptionalDependencies struct {
	dig.In

	DebugCaptureStore   *debugcapture.Store   `optional:"true"`
	DebugCaptureRuntime *debugcapture.Runtime `optional:"true"`
}

func newAgentServer(
	cfg *config.Config,
	credentials *agent.CredentialStore,
	requestLogs *requestlog.Service,
	manager *state.Manager,
	redactor *redact.Redactor,
	optional agentOptionalDependencies,
) *agent.Server {
	dependencies := agent.Dependencies{
		Credentials:          credentials,
		RequestLogs:          requestLogs,
		RequestLogStats:      requestLogs,
		Usage:                requestLogs,
		Snapshots:            manager,
		Redactor:             redactor,
		Version:              version.Version,
		Features:             agent.Features{},
		MCPTrustedProxyCIDRs: cfg.MCPTrustedProxyCIDRs,
	}
	if optional.DebugCaptureStore != nil {
		dependencies.Evidence = optional.DebugCaptureStore
	}
	if optional.DebugCaptureRuntime != nil {
		dependencies.EvidenceHealth = optional.DebugCaptureRuntime
	}
	return agent.NewServer(dependencies)
}

func provideAgent(dependencyContainer *dig.Container) error {
	providers := []any{
		func(db *gorm.DB, encryptionService encryption.Service) *agent.CredentialStore {
			return agent.NewCredentialStore(db, encryptionService)
		},
		newAgentServer,
	}
	for _, provider := range providers {
		if err := dependencyContainer.Provide(provider); err != nil {
			return err
		}
	}
	return nil
}

func configureAgent(dependencyContainer *dig.Container) error {
	return dependencyContainer.Invoke(func(
		service *control.Service,
		credentials *agent.CredentialStore,
		server *agent.Server,
	) error {
		service.SetAgentCredentialStore(credentials)
		server.SetChangeProposalService(service)
		return nil
	})
}
