---
description: "The Hostinger release pipeline contract: how a pushed commit becomes a running image (local build, local image verification, upload, continuous switch) and exactly what the local verification does and does not prove."
kind: technical
topic: release-pipeline
code:
  paths:
    - scripts/deploy.sh
    - scripts/verify-built-image.py
    - scripts/local-rehearsal.py
---

# Release Pipeline

## Scope

- **Owner:** the build-verify-upload-switch pipeline in `scripts/deploy.sh`, the
  local image verification it runs (`scripts/verify-built-image.py`), and the
  opt-in snapshot rehearsal (`scripts/local-rehearsal.py`).
- **Authoritative for:** what each stage proves, the guard conditions around
  the production switch, and the capability boundary of local verification.
- **Excludes:** the operator runbook and rollback decisions
  (`docs/deployment.md`), offline legacy database conversion
  (`.docs/tech/legacy-db-conversion.md`), and ingress topology
  (`docs/godoxy-ingress.md`).

## Responsibility

Turn a pushed commit into the running Hostinger container: build it locally for
the server architecture, prove the built image boots and migrates on synthetic
fixtures, upload it, and switch the single production instance inside one
guarded remote script with an offline data archive taken at the stop.

## Stages

`scripts/deploy.sh <branch>` is one command. `--prepare-only` stops after the
upload; `--activate-only <40-hex sha>` skips build and upload and replays
verification plus switch against the already uploaded local image.

1. **Pin the commit and its scripts.** `git fetch origin refs/heads/<branch>`,
   resolve `FETCH_HEAD`, derive tag `gpt-load:<branch>-<short sha>`, then export
   that commit with `git archive` into a temporary directory and re-execute the
   exported `scripts/deploy.sh` from there. Build context, verification script,
   and switch script all come from the commit, so uncommitted local changes
   cannot enter the image or alter the pipeline. `--activate-only` aborts if the
   branch head no longer equals the given SHA, so a switch never activates an
   image that was not the verified one.
2. **Build locally for the server architecture.** `git archive <commit>` feeds
   `docker buildx build --builder default --load --platform linux/<arch>`, with
   `<arch>` from `ssh vps-kl uname -m`. The local Docker endpoint must be a
   `unix://` socket; the script rejects SSH/TCP contexts so a build cannot be
   dispatched to another daemon. Only the pushed commit enters the image.
3. **Verify the built image locally** before any upload. The previous image is
   the ID reported by `docker inspect gpt-load --format '{{.Image}}'`; if it is
   absent locally it is copied back with `docker save | docker load`. Any
   verification failure aborts with the old container untouched.
   `--activate-only` runs the same verification against the already uploaded
   local image, so replaying the switch is not a way to skip it, but it neither
   rebuilds nor re-uploads.
4. **Upload.** `docker save | gzip -1 | ssh vps-kl 'docker load'`. Production is
   still running during build, verification, and upload.
5. **Switch in one remote script.** Guarded, in order: the server-side
   `gpt-load:<tag>` image ID equals the image ID verified locally (a replaced or
   truncated upload is rejected); the old container is running (a pre-stopped
   host is rejected); the production container's `{{.Image}}` still equals the
   old image ID used for the upgrade verification (if the old container was
   replaced in between, the upgrade result no longer applies and the switch is
   rejected); the `/app/data` mount source and volume name are read from the
   running container rather than hardcoded; the backup directory is not tmpfs;
   free space exceeds twice `du -sb` of the data directory. Then: `docker compose
   stop -t 30`, confirm no running container still mounts that volume, `tar` the
   whole data directory into `pre-upgrade.XXXXXXXX` and list the archive, copy
   Compose aside, repoint the `image:` line, `docker compose up -d --no-build
   --pull never`, poll health 24×5s, then `curl` the loopback and public
   `/health`. Both health responses must report `version` exactly equal to
   `<branch>-<short sha>`; a healthy container running the previous image is a
   failed release.

**Failure semantics.** `trap recover_before_switch EXIT` restores the old
Compose file and starts the old container only when the stop happened and the
switch did not (`stopped && !switched`), i.e. while the database is still
unmodified. After `switched`, no automatic rollback happens because the new
version may already have migrated the database. `trap '' HUP` keeps an SSH
disconnect from aborting the started remote sequence, but host failure or a
killed process still needs manual recovery.

**Downtime.** Single instance: in-flight HTTP and streaming requests are
cancelled between stop and start. The pipeline does not promise zero downtime,
does not run `VACUUM`, and does not prune images or backups.

## Binding points

- Code: `scripts/deploy.sh` (switch guards), `scripts/verify-built-image.py`
  (local scenarios), `scripts/local-rehearsal.py` (opt-in snapshot rehearsal).
- Operator runbook: [`docs/deployment.md`](../../docs/deployment.md).
- No semantic product node owns release behavior; it is an operating boundary,
  not a user-visible capability.

## Local verification: what it proves

`scripts/verify-built-image.py` runs two scenarios against the same temporary
root, with no production data, no network, and no provider calls. Images are
resolved to IDs first so both scenarios exercise exactly the inspected images.

1. **Fresh install.** The target image starts on an empty bind-mounted
   directory and creates its schema through its own migrations. A group and one
   encrypted credential are then created through the management API with an
   `Idempotency-Key`; a debug capture session with one attempt and ordered,
   still-uncompressed chunk rows is inserted as data rows into the schema the
   image itself created.
2. **Real old-image upgrade.** The actual previously deployed image creates and
   populates the same fixture, including the capture rows; the target image then
   starts twice on that directory.

After each start the checks are: `/health` readiness, `/api/settings`,
`/api/access-keys`, group pagination total equal to 1, and `POST
/api/groups/<id>/credential/reveal` returning exactly the seeded plaintext —
which proves the credential was encrypted at rest and is still decryptable.
`/health` must also report `version` equal to the `--expected-version` the
pipeline derived from the tag, so a stale image cannot pass as verified.
SQLite `integrity_check` and `foreign_key_check` must pass, `groups` and
`schema_migrations` must exist, the stored credential row must not contain the
plaintext, the upgraded ledger must equal `fresh ∪ old`, and the group,
credential, and access-key counts must be unchanged.

Debug capture is covered end to end on the migrated database: the fixture
capture must still be listable through `/api/debug-captures`, its single attempt
must come back from `/api/debug-captures/<id>`, and the ZIP from
`/api/debug-captures/<id>/download` must contain `manifest.json` with the fixture
id plus one `attempts/<attempt>/parts/<direction>.<part>` entry per captured
chunk group, byte-identical to the data written before the upgrade. Because the
ZIP is the only surface that decodes every chunk through the chunk store, this
also proves chunks stay readable after the encoding migration. The database
check additionally requires the migrated rows to carry an `encoding` column
whose values are all `identity` or `zstd`, and the pre-upgrade id, capture,
attempt, part, direction, and data columns to be unchanged and in append order.

Containers run with `--network none`, `--user <operator uid>:<gid>`,
`DATA_DIR=/app/data`, empty `DATABASE_DSN`,
`MODELS_DEV_AUTO_SYNC_ENABLED=false`, and never mount the production volume or
bind a production port.

## Local verification: what it does not prove

- No provider traffic. There is no network, no upstream call, and no real
  request or response stream; the captured bytes in the fixture are constants in
  the script. What is real is the data: the group and credential rows come from
  the old image's own management API with real encryption, and the target image
  itself serves the version, management APIs, decryption, ZIP export, and
  migrations over those rows.
- Fixture size: one group and one credential. Production group shapes, volume,
  per-group credential counts, capture volume, and capture retention are not
  represented.
- Keys: fixed local placeholders (`local-verification-auth`,
  `local-verification-encryption`), not the production `auth.key` /
  `encryption.key`.
- Not exercised: retention startup cleanup, historical anomalous ledger rows
  (for example a manually removed `0009`), request-log paths, provider-observed
  capture writing (only pre-existing capture rows are replayed), and upgrade
  duration.
- The repository holds no migration fixture captured from production data, so
  no automated check covers a production-derived schema history.

The verification therefore proves "the target image boots and migrates from an
empty database and from the actual previous release's database, twice, to the
same ledger as a fresh install, keeps pre-existing debug capture data readable
after the chunk encoding migration, and still decrypts its credentials". It does
not prove compatibility with any specific production database, upgrade timing,
or data-level losslessness. Validating production data shapes is a separately
scoped, opt-in snapshot rehearsal; it is not a mandatory release step and it
does not generate or replace the final pre-switch backup. That rehearsal tool
currently resolves the volume by its default name, while the deployed volume
carries a version suffix, so it needs its own scope before it can run against
the current production volume.
