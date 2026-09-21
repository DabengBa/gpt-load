package agent

import (
	"context"
	"time"

	"gpt-load/internal/debugcapture"
	"gpt-load/internal/platform/redact"
	"gpt-load/internal/platform/version"
	"gpt-load/internal/requestlog"
	"gpt-load/internal/state"
)

// RequestLogReader is the read-only request-log source shared with the admin
// control plane. It exposes only List/Get and never a writer.
type RequestLogReader interface {
	List(context.Context, requestlog.ListQuery) (requestlog.Page, error)
	Get(context.Context, string) (requestlog.Record, error)
}

// RequestLogStatsReader exposes the bounded persistence health counters.
type RequestLogStatsReader interface {
	Stats() requestlog.Stats
}

// UsageReader is the read-only usage aggregate source.
type UsageReader interface {
	QueryUsage(context.Context, requestlog.UsageQuery) (requestlog.UsageReport, error)
}

// EvidenceReader is the read-only debug capture metadata source. It exposes
// queries only; raw export and per-session reads that can mark expiry are not
// part of the Agent surface.
type EvidenceReader interface {
	QuerySessions(debugcapture.SessionQuery) ([]debugcapture.SessionRecord, error)
}

// EvidenceHealthReader exposes the bounded capture store health.
type EvidenceHealthReader interface {
	Health() (debugcapture.Health, error)
}

// SnapshotProvider exposes the current immutable runtime configuration.
type SnapshotProvider interface {
	Current() *state.ConfigSnapshot
}

// Dependencies assembles the Agent surface. Read dependencies are query-only;
// ChangeProposals is the control-owned proposal lifecycle service.
type Dependencies struct {
	Credentials          *CredentialStore
	ChangeProposals      ChangeProposalService
	RequestLogs          RequestLogReader
	RequestLogStats      RequestLogStatsReader
	Usage                UsageReader
	Evidence             EvidenceReader
	EvidenceHealth       EvidenceHealthReader
	Snapshots            SnapshotProvider
	Redactor             *redact.Redactor
	Version              string
	Features             Features
	MCPTrustedProxyCIDRs []string
	Now                  func() time.Time
}

// Server implements the control-owned Agent HTTP module.
type Server struct {
	credentials          *CredentialStore
	requestLogs          RequestLogReader
	requestLogStats      RequestLogStatsReader
	usage                UsageReader
	evidence             EvidenceReader
	evidenceHealth       EvidenceHealthReader
	snapshots            SnapshotProvider
	redactor             *redact.Redactor
	version              string
	features             Features
	mcpTrustedProxyCIDRs []string
	mcpHTTP              mcpHTTPHandler
	now                  func() time.Time
	startedAt            time.Time
	changeProposals      ChangeProposalService
}

// NewServer builds the Agent surface; missing optional readers degrade the
// corresponding endpoint into an explicit unavailable result instead of
// silently returning an empty success.
func NewServer(deps Dependencies) *Server {
	now := deps.Now
	if now == nil {
		now = time.Now
	}
	redactor := deps.Redactor
	if redactor == nil {
		redactor = redact.New()
	}
	buildVersion := deps.Version
	if buildVersion == "" {
		buildVersion = version.Version
	}
	server := &Server{
		credentials:          deps.Credentials,
		changeProposals:      deps.ChangeProposals,
		requestLogs:          deps.RequestLogs,
		requestLogStats:      deps.RequestLogStats,
		usage:                deps.Usage,
		evidence:             deps.Evidence,
		evidenceHealth:       deps.EvidenceHealth,
		snapshots:            deps.Snapshots,
		redactor:             redactor,
		version:              buildVersion,
		features:             deps.Features,
		mcpTrustedProxyCIDRs: append([]string(nil), deps.MCPTrustedProxyCIDRs...),
		now:                  now,
		startedAt:            now().UTC(),
	}
	server.mcpHTTP = server.buildMCPServer()
	if deps.ChangeProposals != nil {
		server.features.ChangeProposals = true
	}
	return server
}

// Credentials exposes the credential store for the admin management routes.
func (server *Server) Credentials() *CredentialStore {
	if server == nil {
		return nil
	}
	return server.credentials
}

func (server *Server) snapshotRevision() uint64 {
	if server == nil || server.snapshots == nil {
		return 0
	}
	snapshot := server.snapshots.Current()
	if snapshot == nil {
		return 0
	}
	return snapshot.Revision
}

func (server *Server) uptimeSeconds() int64 {
	if server == nil {
		return 0
	}
	uptime := server.now().UTC().Sub(server.startedAt)
	if uptime < 0 {
		uptime = 0
	}
	return int64(uptime / time.Second)
}

func (server *Server) scrub(text string) string {
	if text == "" || server == nil || server.redactor == nil {
		return text
	}
	return server.redactor.String(text)
}
