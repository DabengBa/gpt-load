---
id: page.groups
type: page
name: Groups
route: /groups
related: [feature.group-site-information, feature.realistic-probe]
---
# Groups

## ID Explanation

`page.groups` describes group collection and group maintenance entry points.
Site access and provider-account information are owned by
[Group Site Information](../features/group-site-information.md).

## Page Purpose

Administrators create, inspect and maintain the provider groups used by
GPT-Load. The collection supports finding a group and opening its settings;
group settings combine provider configuration, credentials, models and
runtime controls.

## Page Structure

The collection presents group status and available group actions. When a group
has configured site access, its latest known account balance and API Key group
ratio appear with the group. On narrow screens the table can be scrolled
horizontally to inspect all columns.

The group detail view uses full-width sections and a unified save action for
settings and model edits. Site access is maintained in its own section. On
mobile, the save controls stay in document flow so they do not cover the form.

In model maintenance, administrators can edit the model ID of an existing row
directly and save it without removing and re-adding the row. An empty model ID
cannot be saved. The retained row keeps its client alias and route settings;
its read-only test alias follows the [test alias lifecycle](feature.model-test-alias).

Saved model rows expose [Actual Request Testing](feature.realistic-probe) for
supported generation routes, alongside the existing diagnostic probe.

## Choosing a Channel for a New Group

An untouched new-group import defaults to OpenAI Compatible when available.
OpenAI, Anthropic, Gemini and Follow client protocol are also presented as
common choices. Other channels, including existing provider-specific presets,
remain available through searchable Other channels; this is not a removal of
existing channels. A recovered or prefilled draft is not replaced by the fresh
default.

Follow client protocol is for a gateway that accepts the declared client
operations: the next hop follows the client protocol, while the final provider
protocol depends on that gateway. Enter its root address or deployment prefix
and supply an API key. Protocol paths are appended automatically; an address
ending in `/v1` or `/v1beta` triggers an advisory warning because it may be an
API path, but it can also be a valid deployment prefix and is not blocked.
This differs from OpenAI Compatible, which expects the complete API prefix.
See the [multi-protocol channel contract](../../tech/multi-protocol-channel.md)
for its supported operations and probing boundary.

## One Upstream Credential per Group

A group owns at most one upstream credential. Deleting it preserves the group
but leaves the group unable to serve requests until another credential is
configured. An occupied group rejects a second credential even when its API key
is identical. Credential IDs remain stable references in request logs and
billing, while cross-group health counts and scheduling continue to aggregate
multiple groups.

Multiple text lines, including duplicate lines, are invalid; one structured JSON
credential may span lines. Cloud providers retain structured API credentials,
and external CLIProxyAPI uses its gateway API key. Plural collection routes,
batch deletion, download-all and multi-value import bodies have no compatibility
aliases.

## Reading Site Information

- A missing balance or ratio means no usable value is available; it is not a
  zero balance or a default ratio.
- Refresh failures show the last known successful data when available, with a
  failure reason and an explicit retry action.
- The provider account balance and group ratio are informational; they do not
  change local billing or routing.
