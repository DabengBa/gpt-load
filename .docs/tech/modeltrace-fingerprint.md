---
description: "ModelTrace fingerprint quirk: closed-set attribution limits and the coupled SDK/data update rule."
kind: technical
topic: modeltrace-fingerprint
id: QUIRK-004
status: active
external_service: lm-detector
verified_date: 2026-11-25
paths:
  - internal/control/modeltrace_*.go
  - internal/channel/compiler.go
enforcement: fresh-audit
code:
  paths:
    - internal/control/modeltrace_detector.go
    - internal/channel/compiler.go
---
# ModelTrace Fingerprint Quirk

## Symptom

Fingerprint attribution is a closed-set comparison over the bundled reference
bank. It cannot prove upstream model identity. The lm-detector Go SDK and its
two embedded data files must stay in sync; a Dependabot PR updating only the
SDK can silently break attribution.

## Invariant Action

- Treat fingerprint results as candidate scores over the bundled reference
  bank only — never as proof of upstream model identity.
- The SDK and both embedded data files MUST be updated together in one change.
  A Go SDK update PR merged alone is not allowed.
- Fingerprint results MUST NOT trigger health changes, cooldowns, circuit
  breaking, retries, routing, or affinity.

## Boundary

Challenge dispatch records one request per challenge; result reporting is
candidate scores plus per-challenge outcomes. Method and threshold tables are
excluded — code is the source of truth.
