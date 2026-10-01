import { createServer } from 'vite'
import { fileURLToPath } from 'node:url'
import { runGroupCollectionContractTests } from './verify-group-collection.test.mjs'

const webRoot = fileURLToPath(new URL('..', import.meta.url))

const server = await createServer({
  root: webRoot,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})

try {
  const [route, queryKeys, groups, recovery] = await Promise.all([
    server.ssrLoadModule('/src/shared/routing/group-collection-route.ts'),
    server.ssrLoadModule('/src/shared/control/query-keys.ts'),
    server.ssrLoadModule('/src/shared/control/resources/groups.ts'),
    server.ssrLoadModule('/src/shared/controllers/import-recovery.ts'),
  ])
  runGroupCollectionContractTests({
    parseGroupCollectionRouteQuery: route.parseGroupCollectionRouteQuery,
    normalizeGroupCollectionFilters: queryKeys.normalizeGroupCollectionFilters,
    projectGroupCollection: groups.projectGroupCollection,
    createImportRecoveryService: recovery.createImportRecoveryService,
    importRecoveryStorageKey: recovery.importRecoveryStorageKey,
    importRecoveryTtlMs: recovery.importRecoveryTtlMs,
  })
  console.log('group-collection contract: PASS')
} finally {
  await server.close()
}

// The stylex unplugin keeps worker handles alive after server.close();
// exit explicitly so the contract verdict does not hang the gate.
process.exit(process.exitCode ?? 0)
