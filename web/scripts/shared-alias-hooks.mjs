// Node customization hooks for --experimental-strip-types unit tests:
//  - `@shared/*` maps onto src/shared (the vite alias)
//  - extensionless relative specifiers inside src/ try .ts/.tsx so source
//    files authored for the bundler resolve under node.

import { existsSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'

const sharedRoot = fileURLToPath(new URL('../src/shared/', import.meta.url))
const srcRoot = fileURLToPath(new URL('../src/', import.meta.url))

function withExtension(pathname) {
  for (const candidate of [
    `${pathname}.ts`,
    `${pathname}.tsx`,
    `${pathname}/index.ts`,
    `${pathname}/index.tsx`,
    pathname,
  ]) {
    if (existsSync(candidate)) return pathToFileURL(candidate).href
  }
  return undefined
}

export function resolve(specifier, context, nextResolve) {
  if (specifier.startsWith('@shared/')) {
    const resolved = withExtension(sharedRoot + specifier.slice('@shared/'.length))
    if (resolved !== undefined) return nextResolve(resolved, context)
    return nextResolve(specifier, context)
  }
  const parent = context.parentURL ?? ''
  if (
    (specifier.startsWith('./') || specifier.startsWith('../')) &&
    parent.startsWith('file://') &&
    fileURLToPath(parent).replaceAll('\\', '/').startsWith(
      srcRoot.replaceAll('\\', '/'),
    ) &&
    !/\.[a-z0-9]+$/u.test(specifier)
  ) {
    const base = fileURLToPath(new URL(specifier, parent))
    const resolved = withExtension(base)
    if (resolved !== undefined) return nextResolve(resolved, context)
  }
  return nextResolve(specifier, context)
}
