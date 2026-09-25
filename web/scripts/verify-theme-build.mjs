// Rebuild-diff gate for the generated Astryx theme.
// `astryx theme build --check` compares committed artifacts against
// src/frontends/astryx/theme/gptload.theme.ts and exits non-zero on drift.
// Invoked through process.execPath because the .bin shell shim does not
// terminate reliably when spawned inside a pnpm script chain on Windows.
import { spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'

const webRoot = fileURLToPath(new URL('..', import.meta.url))
const astryxCli = fileURLToPath(
  new URL('../node_modules/@astryxdesign/cli/clients/cli/bin/astryx.mjs', import.meta.url),
)

const result = spawnSync(
  process.execPath,
  [
    astryxCli,
    'theme',
    'build',
    'src/frontends/astryx/theme/gptload.theme.ts',
    '-o',
    'src/frontends/astryx/theme/gptload.theme.css',
    '--check',
  ],
  { cwd: webRoot, stdio: ['ignore', 'inherit', 'inherit'] },
)

if (result.error) throw result.error
process.exit(result.status ?? 1)
