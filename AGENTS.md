# Repository Identity

- Canonical repository: `DabengBa/gpt-load`
- Canonical URL: `https://github.com/DabengBa/gpt-load`
- Canonical remote: `origin` → `https://github.com/DabengBa/gpt-load.git`
- Default integration branch for this checkout: `dev`
- `upstream` may point to `tbphp/gpt-load` for reference only. Do not treat `tbphp/gpt-load` as the repository being maintained, and do not merge or push there unless the user explicitly requests it.

## Git and Documentation Rules

- PR discovery, merge, fetch, and push operations target `DabengBa/gpt-load` by default.
- README repository links, clone commands, and ownership statements must identify `DabengBa/gpt-load`.
- Keep links to `tbphp/gpt-load` only when they intentionally refer to upstream release/container artifacts, and label that relationship explicitly.
- Before merging, verify the target repository and base branch with `gh pr view -R DabengBa/gpt-load`.

## Hostinger Deployment

- Follow `docs/deployment.md` for requests to push and update the Hostinger GPT-Load service.
- Run `scripts/deploy.sh [branch]` locally after validation and pushing the intended commits; the default branch is `dev`.
- Build images locally and upload them over SSH. Hostinger only loads images and runs Compose; do not build images or invoke BuildKit on the server.
- Verify the deployed image/version, local and public health endpoints, and behavior relevant to the change before reporting success.
