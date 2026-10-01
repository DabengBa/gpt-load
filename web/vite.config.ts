import { fileURLToPath, URL } from 'node:url'

import babel from '@rolldown/plugin-babel'
import stylex from '@stylexjs/unplugin/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export const webRootPath = fileURLToPath(new URL('.', import.meta.url))
// The shared route manifest lives outside webRoot; the dev server needs
// explicit fs.allow access for src/shared/routing/page-routes.ts.
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
    }),
    react({ include: astryxScriptInclude }),
    babel({
      include: astryxScriptInclude,
      plugins: ['babel-plugin-react-compiler'],
    }),
  ],
  resolve: {
    alias: {
      '@app': astryxRoot,
      '@shared': fileURLToPath(new URL('./src/shared', import.meta.url)),
    },
  },
  optimizeDeps: {
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
      input: fileURLToPath(new URL('./index.html', import.meta.url)),
    },
  },
})
