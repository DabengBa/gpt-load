# Agent Operations MCP

GPT-Load exposes a stateless Streamable HTTP MCP endpoint at
`/api/agent/v1/mcp`. It is mounted on the existing HTTP listener and uses the
same Agent Bearer credential as the canonical Agent API.

## Transport and trust boundary

- Every HTTP request is authenticated again; MCP sessions never carry identity
  or permission state.
- Only JSON `POST` requests enter the MCP SDK. Authenticated `GET` and
  `DELETE` requests return `405 Method Not Allowed`.
- The adapter uses `github.com/mark3labs/mcp-go` `v0.43.2` under its MIT
  license, with `WithStateLess(true)` and `WithDisableStreaming(true)`.
- A missing `Origin` is allowed. When present, it must match the request
  scheme and `Host`; `X-Forwarded-Host` is never trusted.
- `X-Forwarded-Proto` is ignored unless the request peer is inside a CIDR from
  `MCP_TRUSTED_PROXY_CIDRS`. A trusted proxy must send exactly one value and it
  must be `http` or `https`.

## Exposed surface

The adapter has an explicit allowlist. It does not discover or proxy ordinary
HTTP routes.

Read tools require `diagnostics:read`:

- `agent_get_capabilities`
- `agent_get_request`
- `agent_get_evidence`
- `agent_list_routes`
- `agent_get_health`
- `agent_get_usage`
- `agent_get_proposal`
- `agent_get_operation`

Proposal tools require their matching scope:

- `agent_create_proposal` requires `changes:propose`.
- `agent_apply_proposal` requires `changes:apply` and accepts only a proposal
  ID plus a canonical UUID-v4 idempotency key. Approval remains an administrator
  control-plane operation.

Resources are redacted JSON projections for request, evidence, proposal, and
control-operation facts. Raw captures, credentials, SQL, shell commands,
provider URLs, probes, and arbitrary mutations are not exposed.

Tool errors are structured MCP results with the canonical GPT-Load error code;
side effects remain exclusively in the Agent/Control service layer.
