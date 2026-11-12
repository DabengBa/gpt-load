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

Saved model rows expose [Actual Request Testing](feature.realistic-probe) for
supported generation routes, alongside the existing diagnostic probe.

## Reading Site Information

- A missing balance or ratio means no usable value is available; it is not a
  zero balance or a default ratio.
- Refresh failures show the last known successful data when available, with a
  failure reason and an explicit retry action.
- The provider account balance and group ratio are informational; they do not
  change local billing or routing.
