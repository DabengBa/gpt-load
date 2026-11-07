---
description: Keep provider-site access and announcement snapshots outside the legacy groups table
kind: adr
topic: group-site-information-storage
status: accepted
date: 2026-11-07
---

# ADR-0004: Store provider site information in independent tables

## Context

Provider site access requires a user ID, encrypted token, refresh state and
account snapshot. Announcements are a separate, replaceable feed with
per-source status. Adding these fields directly to `groups` would make the
existing offline legacy database converter's exact source/target column
contracts reject a database shape that the converter does not own. The
converter intentionally validates that retired and current schemas match
known versions before converting data.

## Decision

Store site credentials and their latest account/source snapshot in
`group_site_configs`, and store aggregated announcement entries in
`site_announcements`. Link both to groups by group ID and manage their lifecycle
through explicit migrations, rather than extending the legacy `groups` table.

## Trade-offs

- A separate table requires additional joins and lifecycle cleanup when a group
  is deleted.
- The group read path must explicitly opt into site information to preserve
  older client response contracts.
- In return, the existing group schema and offline conversion contract remain
  unchanged, while credentials, refresh metadata and feed records have clear
  ownership and independent retention behavior.

## Consequences

Migrations 0029 and 0030 own table creation and source status. Group deletion
must remove related site configuration and announcement rows transactionally.
Schema changes to these tables still require updating migration, conversion
and exact-column tests, but do not redefine the legacy `groups` schema.
