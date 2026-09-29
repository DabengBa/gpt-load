import { fileURLToPath, URL } from 'node:url'

import babel from '@rolldown/plugin-babel'
import tailwindcss from '@tailwindcss/vite'
import stylex from '@stylexjs/unplugin/vite'
import react from '@vitejs/plugin-react'
import vue from '@vitejs/plugin-vue'
import { defineConfig, type Connect, type Plugin } from 'vite'

import { readFrontendPreference } from './src/shared/controllers/frontend-preference'
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
      useCSSLayers: {
        before: ['reset', 'astryx-base', 'astryx-theme', 'tokens'],
        prefix: 'app',
      },
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
  optimizeDeps: {
    // Both html entries must be crawled up front; otherwise astryx-only deps
    // are discovered on first page load and a mid-run re-optimization reloads
    // in-flight pages (observed flake: .vite/deps pre-transform errors).
    entries: ['./index.html', './astryx.html'],
    // Astryx is consumed via deep subpath imports (@astryxdesign/core/Button,
    // ...) which the crawl can miss behind lazy chunks; pin every one used so
    // they are bundled in the initial optimization pass.
    include: [
      'react',
      'react-dom',
      'react-dom/client',
      // Injected by babel-plugin-react-compiler / plugin-react transforms, so
      // the dependency crawl can never see them — pin or they trigger a
      // mid-run re-optimization on first transformed module.
      'react/compiler-runtime',
      'react/jsx-runtime',
      'react/jsx-dev-runtime',
      'react-intl',
      'lucide-react',
      '@tanstack/react-router',
      '@tanstack/react-query',
      '@tanstack/query-core',
      '@internationalized/date',
      '@stylexjs/stylex',
      '@astryxdesign/core',
      '@astryxdesign/core/Banner',
      '@astryxdesign/core/Button',
      '@astryxdesign/core/Card',
      '@astryxdesign/core/Collapsible',
      '@astryxdesign/core/DateTimeInput',
      '@astryxdesign/core/Dialog',
      '@astryxdesign/core/IconButton',
      '@astryxdesign/core/Layout',
      '@astryxdesign/core/Link',
      '@astryxdesign/core/Popover',
      '@astryxdesign/core/TextInput',
      '@astryxdesign/core/i18n',
      '@astryxdesign/core/theme',
      '@astryxdesign/theme-neutral',
    ],
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

const astryxEntryUrl = '/astryx.html'

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
        if (readFrontendPreference(req.headers.cookie) !== 'astryx') {
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
