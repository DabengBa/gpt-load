import { fileURLToPath, URL } from 'node:url'

import babel from '@rolldown/plugin-babel'
import tailwindcss from '@tailwindcss/vite'
import stylex from '@stylexjs/unplugin/vite'
import react from '@vitejs/plugin-react'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

export const webRootPath = fileURLToPath(new URL('.', import.meta.url))
export const pageRouteManifestPath = fileURLToPath(
  new URL('../internal/webui/page_routes.json', import.meta.url),
)
export const devServerFileSystemAllow = [webRootPath, pageRouteManifestPath]

const astryxRoot = fileURLToPath(new URL('./src/frontends/astryx', import.meta.url))
const astryxInclude = /frontends[\\/]astryx/
const proxyTarget = process.env.VITE_DEV_PROXY_TARGET || 'http://127.0.0.1:3001'

export default defineConfig({
  root: webRootPath,
  plugins: [
    stylex({
      unstable_moduleResolution: { type: 'commonJS', rootDir: webRootPath },
      // Two entries ship CSS: keep collected StyleX atoms out of classic chunks.
      cssInjectionTarget: (fileName) => /(^|\/)astryx(-[\w-]+)?\.css$/i.test(fileName),
    }),
    vue(),
    react({ include: astryxInclude }),
    babel({
      include: astryxInclude,
      plugins: ['babel-plugin-react-compiler'],
    }),
    tailwindcss(),
    // B5 fills this in: cookie + manifest-flag frontend selection in dev.
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

function frontendSelectorDevPlugin() {
  return {
    name: 'gpt-load:frontend-selector',
    // Implemented in B5.
  }
}
