---
target: /groups
total_score: 28
max_score: 40
na_heuristics: 
p0_count: 0
p1_count: 0
target_identity: "file:/mnt/projects/repos/gpt-load/web/src/frontends/astryx/features/groups/GroupsView.tsx"
target_fingerprint: "sha256:373d28c0fe107e102a98c92dbcb19b18b644e21c48e5569c9f04be63354a135e"
target_path: /mnt/projects/repos/gpt-load/web/src/frontends/astryx/features/groups/GroupsView.tsx
timestamp: 2026-10-10T00-32-37Z
slug: nds-astryx-features-groups-groupsview-tsx-0118d3c7
---
# /groups review

Method: dual-agent (A: 01a12333-7471-73e1-8fe5-edc3171c2411; B: 01a12333-7483-73e1-8fe5-edc40351fd60)

Target: web/src/frontends/astryx/features/groups/GroupsView.tsx
Mode: Operate
Score: 28/40

## Heuristics

| Heuristic | Score |
| --- | --- |
| Visibility of status | 3/4 |
| Real-world match | 3/4 |
| User control | 3/4 |
| Consistency | 2/4 |
| Error prevention | 3/4 |
| Recognition | 2/4 |
| Efficiency | 3/4 |
| Minimalism | 3/4 |
| Error recovery | 4/4 |
| Help | 2/4 |

## Verdict and strengths

Preserve the incumbent compact operational table. Its complete loading/error/empty/stale state handling, URL-backed filters, channel identity, and clickable status summaries serve the product well. No replacement visual world is needed.

## Priority issues

1. P2: Models render client_model_count / model_count without explaining the two quantities. Clarify the header or provide an accessible explanation without increasing row density.
2. P2: onSortChange returns to recent when a user attempts to reverse the active column sort. If retaining the directionless server enum, remove the misleading bidirectional header affordance and use the existing sort selector.
3. P2: Four icon-only row actions require interpretation. Consider visible text for credential management, a direct detail shortcut, and progressive disclosure for infrequent actions based on actual usage.
4. P2: screenshot/groups-overview.png differs from current source: progress bars and Add credential buttons versus switches, credential text, and icon actions. Use a current desktop/mobile rendered inspection before making visual conclusions.

## Cognitive load and personas

Moderate cognitive load: control counts are reasonable; ambiguity matters more than density. Power users encounter surprising sort behavior. First-time operators lack explanations for model ratios and action icons. Entry is reassuring through status summaries; uncertainty appears at ambiguous actions and quantities.

## Minor observations

Refresh has a loading state on the header button, in addition to a screen-reader announcement. Do not describe it as lacking all sighted feedback. Color dots have accompanying text. Do not introduce bulk controls, saved views, or new configuration without a demonstrated need.

## Evidence limits

Static detector: zero findings. TSX/StyleX/token coverage is limited; this is not proof of rendered contrast or responsive behavior. No browser tool was available. Historical screenshot is not current visual truth.
