// B13 gate #4: first-screen JS+CSS gzip comparison between the classic
// (`index.html`) and Astryx (`astryx.html`) entries.
//
// "First screen" = what the browser fetches before the shell renders:
//   - the entry chunk and its recursively-static imports, plus
//   - the entry's own dynamic imports (the boot split — classic's
//     `import('./bootstrap')` runs unconditionally at startup) and their
//     static closures.
// Route-level lazy chunks are excluded for both frontends: they sit on
// `dynamicImports` of non-entry chunks, which this walk never follows.
// Note the astryx entry's dynamicImports are conditional DS internals
// (Tooltip, BottomSheet, MenuBottomSheet — loaded only when rendered), so
// the astryx total is a conservative upper bound.
//
// Requires a prior `vite build` (outputs ../internal/webui/dist with
// manifest:true). Run: node scripts/measure-first-screen.mjs

import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { gzipSync } from 'node:zlib'

const distDir = fileURLToPath(
  new URL('../../internal/webui/dist', import.meta.url),
)
const manifest = JSON.parse(
  readFileSync(join(distDir, '.vite', 'manifest.json'), 'utf8'),
)

function gzipSize(relativeFile) {
  return gzipSync(readFileSync(join(distDir, relativeFile))).byteLength
}

function collectFirstScreen(entryName) {
  const entry = manifest[entryName]
  if (entry === undefined || entry.isEntry !== true) {
    throw new Error(`entry ${entryName} not found in .vite/manifest.json`)
  }
  // Seed with the entry plus its boot-level dynamic imports, then walk
  // static `imports` only.
  const queue = [entryName, ...(entry.dynamicImports ?? [])]
  const seenChunks = new Set()
  const seenFiles = new Set()
  let js = 0
  let css = 0
  while (queue.length > 0) {
    const key = queue.shift()
    if (seenChunks.has(key)) continue
    const chunk = manifest[key]
    if (chunk === undefined) continue
    seenChunks.add(key)
    for (const file of [chunk.file, ...(chunk.css ?? [])]) {
      if (file === undefined || seenFiles.has(file)) continue
      seenFiles.add(file)
      if (file.endsWith('.css')) css += gzipSize(file)
      else js += gzipSize(file)
    }
    queue.push(...(chunk.imports ?? []))
  }
  return { js, css, files: seenFiles.size }
}

const classic = collectFirstScreen('index.html')
const astryx = collectFirstScreen('astryx.html')

const kb = (bytes) => (bytes / 1024).toFixed(1).padStart(7)
console.log('first-screen payload (gzip)')
console.log('  entry    |      JS |     CSS |   total | files')
for (const [name, m] of [
  ['index ', classic],
  ['astryx', astryx],
]) {
  console.log(
    `  ${name} | ${kb(m.js)} | ${kb(m.css)} | ${kb(m.js + m.css)} | ${String(m.files).padStart(5)}`,
  )
}
console.log(
  `  delta    | ${kb(astryx.js - classic.js)} | ${kb(astryx.css - classic.css)} | ${kb(astryx.js + astryx.css - classic.js - classic.css)} |`,
)
