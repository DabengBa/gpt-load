---
description: "The stateless MCP adapter that exposes scope-filtered Agent diagnostics and controlled proposal operations on the existing listener."
kind: technical
topic: agent-mcp
relations:
  related:
    - tech/agent-api.md
code:
  paths:
    - internal/agent/mcp.go
    - internal/agent/http_routes.go
    - internal/platform/config/config.go
    - internal/platform/httproute/registry.go
---
# Agent Operations MCP

## Scope

- **Owner:** `internal/agent/mcp.go` and the `/api/agent/v1/mcp` transport boundary.
- **Authoritative for:** MCP transport mode, request authentication, Origin trust rules, scope-filtered tools/resources, and error behavior.
- **Excludes:** Agent credential storage and proposal business rules owned by [Agent Control-Plane API](agent-api.md), and arbitrary MCP server functionality.

## Responsibility

The adapter exposes a stateless Streamable HTTP MCP endpoint on the existing HTTP listener. It translates MCP tool and resource calls into the Agent and Control service interfaces without direct database access, provider calls, shell execution, or a general-purpose mutation bridge.

## Architecture And Constraints

Each HTTP request passes the Agent Bearer authenticator again. MCP sessions do not carry identity or permission state. Only JSON `POST` requests are accepted by the adapter; authenticated `GET` and `DELETE` requests return `405 Method Not Allowed`.

The server uses `github.com/mark3labs/mcp-go` `v0.43.2` with stateless sessions and streaming disabled. Tool visibility is filtered from the authenticated principal: diagnostics tools require `diagnostics:read`, proposal creation requires `changes:propose`, and proposal apply requires `changes:apply`. Proposal approval remains an administrator-only control route.

When an `Origin` header is present, it must be a single origin whose scheme matches the request TLS state or a single trusted-proxy `X-Forwarded-Proto` value, and whose host matches `Host`. `X-Forwarded-Host` is never trusted. Trusted proxy CIDRs come from `MCP_TRUSTED_PROXY_CIDRS`; untrusted forwarded scheme headers cannot change the effective scheme.

## Core Implementation

- `internal/agent/mcp.go:buildMCPServer` configures the stateless, non-streaming MCP server.
- `filterMCPTools` and `mcpToolAllowed` apply scope-aware allowlisting before tools are listed.
- Tool handlers call the existing Agent/Control projection and proposal services; resource handlers expose redacted JSON projections for requests, evidence, proposals, and control operations.
- `validateMCPOrigin`, `mcpPeerTrusted`, and `mcpHostsEqual` enforce the origin trust boundary.
- `internal/platform/config/config.go` parses and normalizes the trusted proxy CIDR setting.

## Related Product Semantics And Binding Points

- Product-facing behavior and Agent credential semantics are owned by [Agent Control-Plane API](agent-api.md).
- Code binding: `internal/agent/http_routes.go:Module` route `mcp`
- Code binding: `internal/agent/mcp.go:handleMCP`, `validateMCPOrigin`

## Operations, Calculation, And Failure Boundaries

Missing or invalid Agent authentication is rejected before MCP dispatch. Missing `Origin` is allowed; malformed, multi-valued, mismatched, or untrusted-forwarded origins return `403`. Scope failures and service failures remain structured MCP tool results with GPT-Load error codes. Raw captures, credentials, full headers, request/response bodies, arbitrary provider URLs, SQL, shell commands, probes, and unrestricted mutations are never returned or executed through this adapter.
