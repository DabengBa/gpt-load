# ModelTrace analysis service

Standalone service that generates ModelTrace numeric challenges and attributes
model replies to a fingerprint library. GPT-Load executes the challenges against
a real route and posts the raw model text here; this service never contacts an
upstream provider, holds no provider credentials, enrolls no models, and writes
nothing to disk.

| Item | Value |
| --- | --- |
| Service version | `1.0.0` (`modeltrace/__init__.py`) |
| Upstream | https://github.com/xqy2006/ModelTrace |
| Upstream commit | `6cd4a8ff0f3f4e49d6d6969dd2fbadfa1569fde2` (2026-10-09), MIT |
| Fingerprint bank | `modeltrace/vendor/modeltrace/unified_bank.json`, built `2026-10-09T06:24:27Z`, 18 models (8 GPT / 10 Claude) |
| Bank sha256 | `f267d0513aa06981d024ec5394746fceddf44e4e8a75fa6bf8304255f4b32d6d` |
| Runtime | Python 3.14 + NumPy, `http.server` only |

## Endpoints

Errors are JSON `{"error": "<code>", "detail": "<text>"}`.

### `GET /health` (no auth)

Readiness plus the fingerprint bank summary. The service reports `ready` after
the read-only bank has been loaded and checked for model entries.

### `GET /api/challenges` (`Authorization: Bearer <token>`)

```json
{"challenges": [{"id": "probe-1-…", "expected_count": 301, "prompt": "…"}, "…", "…"]}
```

Three challenges per call, `expected_count` in `292..332`. Challenge IDs are
random per call and carry no server-side state.

### `POST /api/analyze` (`Authorization: Bearer <token>`, `Content-Type: application/json`)

```json
{"outputs": [{"prompt_id": "probe-1-…", "expected_count": 301, "text": "<model reply>"}]}
```

One to three outputs per request: GPT-Load executes the complete challenge set,
and each challenge is a real model call with its own usage and request-log
record. Bodies larger than 256 KiB are rejected with `413`,
`text` longer than 32768 chars with `400`, and non-JSON content types with `415`.

```json
{
  "prediction": "claude-sonnet-4-6",
  "prediction_name": "claude-sonnet-4-6",
  "probability": 0.93,
  "family_prediction": "claude",
  "family_prediction_name": "Claude",
  "family_probability": 0.99,
  "used_outputs": 1,
  "method": "统一全局稳健数字指纹",
  "calibration": {"queries": "1", "beta": 7.12, "cv_accuracy": 0.96},
  "candidates": [{"model": "claude-sonnet-4-6", "display_name": "…", "family": "claude", "family_name": "Claude", "probability": 0.93, "conditional_probability": 0.94, "profile_similarity": 0.81}],
  "families": [{"family": "claude", "display_name": "Claude", "probability": 0.99}],
  "diagnostics": [{"prompt_id": "probe-1-…", "expected_count": 301, "parsed_numbers": 301, "minimum_numbers": 166, "accepted": true}],
  "bank": {"models": 18, "built_at": "2026-10-09T06:24:27.469312+00:00", "bank_sha256": "f267d0…"},
  "upstream": {"repository": "https://github.com/xqy2006/ModelTrace", "commit": "6cd4a8f…"}
}
```

`candidates` holds all 18 library models sorted by probability and sums to 1
(closed-set, uniform prior). Input text is never echoed back and never logged.

Status codes: `401` missing/wrong token, `400` malformed JSON or invalid
`outputs`, `413` body over limit, `415` wrong content type, `422`
`no_usable_outputs` when the reply does not contain a long enough number run.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `MODELTRACE_TOKEN` | none | Required, minimum 16 characters. Bearer token shared with GPT-Load. |
| `MODELTRACE_HOST` | `127.0.0.1` | Image sets `0.0.0.0`. |
| `MODELTRACE_PORT` | `8080` | |

## Run

```bash
# local
python -m venv .venv && .venv/bin/pip install -r modeltrace/requirements.txt
MODELTRACE_TOKEN="$(openssl rand -hex 24)" python -m modeltrace.server

# container (no host port published; reach it over the compose network)
MODELTRACE_TOKEN="$(openssl rand -hex 24)" docker compose -f modeltrace/docker-compose.yml up -d --build
```

The image runs as UID 10001, drops all capabilities, and uses a read-only root
filesystem in compose. GPT-Load calls it at `http://modeltrace:8080` on the
shared compose network.

## Update and rollback

The image is self-contained, so updating never touches GPT-Load:

```bash
# update: re-vendor upstream, then rebuild under a new tag
docker build -f modeltrace/Dockerfile -t gpt-load-modeltrace:<version> .
MODELTRACE_IMAGE=gpt-load-modeltrace:<version> MODELTRACE_TOKEN=… \
  docker compose -f modeltrace/docker-compose.yml up -d

# rollback: point the tag back at the previous image
MODELTRACE_IMAGE=gpt-load-modeltrace:<previous> MODELTRACE_TOKEN=… \
  docker compose -f modeltrace/docker-compose.yml up -d
```

Pinning `MODELTRACE_IMAGE` (instead of `:latest`) is what makes that rollback a
one-line change. `docker compose ... config` fails fast when the token is unset.

## Tests

```bash
python -m unittest discover -s modeltrace/tests -t .
```

Covers health, authentication, content type, body/size/count boundaries,
unusable replies, the success analysis payload, and that input text is not
echoed. The suite starts a real HTTP server on a loopback port and uses the real
vendored scorer; it writes no files.

## Not included

No Flask UI, no `/api/test/auto` or other upstream-direct calls, no
`enrollment.py`, no bank builder or bank write endpoints, and no upstream API
key handling — those stay out of this image by design.
