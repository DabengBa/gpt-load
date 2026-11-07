---
id: page.announcements
type: page
name: Announcements
route: /announcements
related: [feature.group-site-information]
---
# Announcements

## ID Explanation

`page.announcements` owns the administrator-facing aggregate view of provider
site announcements. Connecting and refreshing a site is owned by
[Group Site Information](../features/group-site-information.md).

## Page Purpose

Administrators review announcements from the provider sites configured for
their groups in one place, without opening each provider site separately.

## Page Structure

The page shows source status and announcement entries ordered by event time.
Entries identify their site and associated group where available. Search and
site filters narrow the visible entries and are preserved in the page URL.

## Source and Empty States

- Announcement and notice sources display independent synchronization
  statuses. A failed source shows a sanitized reason and a retry action; retry
  queues a refresh and does not claim success before the result arrives.
- Existing announcements remain visible when a later source refresh fails.
- A genuinely empty feed is distinct from a search or filter with no matches;
  the latter provides an action to clear filters.
