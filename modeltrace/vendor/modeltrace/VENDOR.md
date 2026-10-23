# Vendored ModelTrace core

This directory is a read-only copy of the upstream ModelTrace scoring core.
Only `fingerprint.py` and `unified_bank.json` are vendored; the upstream web app,
`enrollment.py`, `bank_builder.py`, `challenge_suite.py`, and the Codex plugin are
deliberately not included.

| Item | Value |
| --- | --- |
| Upstream | https://github.com/xqy2006/ModelTrace |
| Commit | `6cd4a8ff0f3f4e49d6d6969dd2fbadfa1569fde2` (2026-10-09) |
| License | MIT, (c) 2026 xqy2006 (`LICENSE` in this directory) |
| `fingerprint.py` sha256 | `387fe5c44cc5f859995432267b0d3b462fc26e82419fc22e3ee7f36cc2502d98` |
| `unified_bank.json` sha256 | `f267d0513aa06981d024ec5394746fceddf44e4e8a75fa6bf8304255f4b32d6d` |
| Bank built at | `2026-10-09T06:24:27.469312+00:00` |
| Bank models | 18 (8 GPT, 10 Claude) |

`fingerprint.py` is copied byte-for-byte. `unified_bank.json` is copied
byte-for-byte, so the digest still matches upstream
`codex-plugin/modeltrace-guard/assets/provenance.json`. The service compares the
digest of the loaded bank against `BANK_SHA256` in `__init__.py` at startup and
refuses to become ready on mismatch.

## Updating

1. Re-vendor both files from a reviewed upstream commit.
2. Update `UPSTREAM_COMMIT`, `UPSTREAM_COMMIT_DATE`, `BANK_SHA256`, `BANK_BUILT_AT`
   in `__init__.py`, and the table above.
3. Run `pnpm`-independent service tests: `python -m unittest discover -s modeltrace/tests`.
