import eslintConfigPrettier from 'eslint-config-prettier'
import { globalIgnores } from 'eslint/config'
import reactHooks from 'eslint-plugin-react-hooks'
import pluginVue from 'eslint-plugin-vue'
import stylex from '@stylexjs/eslint-plugin'
import { defineConfigWithVueTs, vueTsConfigs } from '@vue/eslint-config-typescript'

export default defineConfigWithVueTs(
  globalIgnores(['node_modules/**']),
  pluginVue.configs['flat/recommended'],
  vueTsConfigs.recommended,
  // The shared layer must stay framework-free so both frontends can consume it.
  {
    files: ['src/shared/**/*.ts'],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              regex: '^(vue|vue-router|vue-i18n|reka-ui|@tanstack/vue-query|@lucide/vue)(/|$)',
            },
            { regex: '^@/' },
            { regex: '(^|/)frontends/' },
            { regex: '\\.vue(\\?|$)' },
          ],
        },
      ],
    },
  },
  // React 19 frontend: hooks + compiler diagnostics are errors, StyleX validity
  // is enforced by the design system's own plugin.
  {
    files: ['src/frontends/astryx/**/*.{ts,tsx}'],
    plugins: {
      'react-hooks': reactHooks,
      '@stylexjs': stylex,
    },
    rules: {
      ...reactHooks.configs['recommended-latest'].rules,
      '@stylexjs/valid-styles': 'error',
    },
  },
  eslintConfigPrettier,
)
