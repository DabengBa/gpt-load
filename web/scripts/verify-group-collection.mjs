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
    server.ssrLoadModule('/src/frontends/classic/features/groups/group-collection-route.ts'),
    server.ssrLoadModule('/src/frontends/classic/app/query-keys.ts'),
    server.ssrLoadModule('/src/frontends/classic/app/resources/groups.ts'),
    server.ssrLoadModule('/src/frontends/classic/features/import/import-recovery.ts'),
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
