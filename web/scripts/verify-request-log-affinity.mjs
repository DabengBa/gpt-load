import { createServer } from 'vite'
import { fileURLToPath } from 'node:url'
import { runRequestLogAffinityContractTests } from './verify-request-log-affinity.test.mjs'

const webRoot = fileURLToPath(new URL('..', import.meta.url))

const server = await createServer({
  root: webRoot,
  server: { middlewareMode: true },
  appType: 'custom',
  logLevel: 'error',
})

try {
  const affinity = await server.ssrLoadModule('/src/frontends/classic/features/monitor/request-log-affinity.ts')
  runRequestLogAffinityContractTests(affinity)
  console.log('request-log-affinity contract: PASS')
} finally {
  await server.close()
}
