import { fileURLToPath, URL } from 'node:url'

import babel from '@rolldown/plugin-babel'
import tailwindcss from '@tailwindcss/vite'
import stylex from '@stylexjs/unplugin/vite'
import react from '@vitejs/plugin-react'
import vue from '@vitejs/plugin-vue'
import { defineConfig, type Connect, type Plugin } from 'vite'

import { pagePathMatches, pageRouteEntries } from './src/shared/routing/page-routes'

export const webRootPath = fileURLToPath(new URL('.', import.meta.url))
export const pageRouteManifestPath = fileURLToPath(
  new URL('../internal/webui/page_routes.json', import.meta.url),
)
export const devServerFileSystemAllow = [webRootPath, pageRouteManifestPath]

const astryxRoot = fileURLToPath(new URL('./src/frontends/astryx', import.meta.url))
const astryxScriptInclude = /frontends[\\/]astryx.*\.[cm]?[jt]sx?$/
const proxyTarget = process.env.VITE_DEV_PROXY_TARGET || 'http://127.0.0.1:3001'

export default defineConfig({
  root: webRootPath,
  plugins: [
    stylex({
      unstable_moduleResolution: { type: 'commonJS', rootDir: webRootPath },
      // Layer order must mirror src/frontends/astryx/entry.css.
      useCSSLayers: { before: ['reset', 'astryx-base', 'astryx-theme'], prefix: 'app' },
      // Two entries ship CSS: keep collected StyleX atoms out of classic chunks.
      cssInjectionTarget: (fileName) => /(^|\/)astryx(-[\w-]+)?\.css$/i.test(fileName),
    }),
    vue(),
    react({ include: astryxScriptInclude }),
    babel({
      include: astryxScriptInclude,
      plugins: ['babel-plugin-react-compiler'],
    }),
    tailwindcss(),
    frontendSelectorDevPlugin(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src/frontends/classic', import.meta.url)),
      '@app': astryxRoot,
      '@shared': fileURLToPath(new URL('./src/shared', import.meta.url)),
    },
  },
  server: {
    fs: {
      allow: devServerFileSystemAllow,
    },
    proxy: Object.fromEntries(
      ['/api', '/health', '/v1', '/v1beta'].map((path) => [
        path,
        { target: proxyTarget, changeOrigin: true },
      ]),
    ),
  },
  build: {
    outDir: '../internal/webui/dist',
    emptyOutDir: true,
    manifest: true,
    target: 'chrome125',
    rollupOptions: {
      input: {
        index: fileURLToPath(new URL('./index.html', import.meta.url)),
        astryx: fileURLToPath(new URL('./astryx.html', import.meta.url)),
      },
    },
  },
})

// Mirrors internal/webui/server.go frontendCookieName. The dev selector must
// keep the same name so e2e selection exercises the production contract.
const frontendCookieName = 'gpt-load.frontend'
const astryxEntryUrl = '/astryx.html'

function cookieValue(header: string | undefined, name: string): string | undefined {
  if (header === undefined) return undefined
  for (const pair of header.split(';')) {
    const separator = pair.indexOf('=')
    if (separator === -1) continue
    if (pair.slice(0, separator).trim() === name) {
      return pair.slice(separator + 1).trim()
    }
  }
  return undefined
}

// Dev-side mirror of the Go server's document selection (B4):
//   - flagged manifest route + cookie "astryx"  -> astryx.html
//   - unknown path + cookie "astryx"            -> astryx.html (404 fallback parity)
//   - anything else                             -> classic index.html
// Only GET requests that accept HTML are rewritten; API/assets pass through.
function frontendSelectorDevPlugin(): Plugin {
  return {
    name: 'gpt-load:frontend-selector',
    configureServer(server) {
      const middleware: Connect.NextHandleFunction = (req, _res, next) => {
        if (req.method !== 'GET') return next()
        const accept = req.headers.accept
        if (typeof accept !== 'string' || !accept.includes('text/html')) return next()
        if (cookieValue(req.headers.cookie, frontendCookieName) !== 'astryx') {
          return next()
        }

        const pathname = new URL(req.url ?? '/', 'http://localhost').pathname
        const entry = pageRouteEntries.find((route) =>
          pagePathMatches(route.name, pathname),
        )
        if (entry === undefined ? pathname === astryxEntryUrl : entry.astryx !== true) {
          return next()
        }
        req.url = astryxEntryUrl
        next()
      }
      server.middlewares.use(middleware)
    },
  }
}
