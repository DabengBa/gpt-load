import { defineTheme } from '@astryxdesign/core/theme'
import { neutralTheme } from '@astryxdesign/theme-neutral'

/**
 * GPT-Load theme — neutral base with classic density and colors.
 *
 * Token values mirror `src/frontends/astryx/theme/tokens.css`; light/dark
 * tuples follow its `:root` / `:root[data-theme='dark']` pairs.
 * Rebuild artifacts with `pnpm run theme:build` (verified by `theme:check`).
 */
export default defineTheme({
  name: 'gptload',
  extends: neutralTheme,
  color: {
    // classic --color-action / --color-focus
    accent: ['#1c4f6e', '#6fb2d6'],
  },
  typography: {
    // classic --text-body 13.5px anchors the scale
    scale: { base: 13.5, ratio: 1.2 },
    // classic --font-sans: no webfont, satisfies font-src 'self'
    body: {
      family: 'system-ui',
      fallbacks: "-apple-system, 'Segoe UI', sans-serif",
    },
    code: {
      family: 'ui-monospace',
      fallbacks: "SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace",
    },
  },
  tokens: {
    // Surfaces — classic canvas/surface/surface-raised/surface-sunken
    '--color-background-body': ['#eeede9', '#0b0d10'],
    '--color-background-surface': ['#ffffff', '#171b20'],
    '--color-background-card': ['#ffffff', '#171b20'],
    '--color-background-popover': ['#ffffff', '#171b20'],
    '--color-background-muted': ['#f5f4f1', '#12151a'],

    // Text — classic text / text-muted / text-faint
    '--color-text-primary': ['#15181b', '#e8eaec'],
    '--color-text-secondary': ['#4e545b', '#969ca3'],
    '--color-text-disabled': ['#687078', '#858c94'],

    // Borders — classic border-subtle / border-control
    '--color-border': ['#e6e5e0', '#232830'],
    '--color-border-emphasized': ['#cfcfc9', '#333a43'],

    // Accent-adjacent — classic action-soft / action-ink / info
    '--color-accent-muted': ['#e8eff4', '#142633'],
    '--color-on-accent': ['#ffffff', '#0c1a22'],
    '--color-background-blue': ['#e8eff4', '#142633'],
    '--color-text-blue': ['#1c4f6e', '#6fb2d6'],
    '--color-icon-blue': ['#1c4f6e', '#6fb2d6'],
    '--color-border-blue': ['#b9ccd7', '#27455a'],

    // Status — classic success/warning/danger/neutral + muted backgrounds
    '--color-success': ['#1a6b3f', '#4fb178'],
    '--color-success-muted': ['#e6f3ec', '#112a1d'],
    '--color-on-success': ['#ffffff', '#0d1f14'],
    '--color-warning': ['#8f6212', '#d5a341'],
    '--color-warning-muted': ['#f8f0dd', '#241d10'],
    '--color-on-warning': ['#ffffff', '#241d10'],
    '--color-error': ['#d03b3b', '#e66767'],
    '--color-error-muted': ['#fbebe9', '#2a1613'],
    '--color-on-error': ['#ffffff', '#2a1613'],
    '--color-neutral': ['#5a5f66', '#9aa0a8'],

    // Feedback — classic overlay / interactive hover / skeleton mix / tag track
    '--color-overlay': ['rgba(20, 22, 24, 0.34)', 'rgba(0, 0, 0, 0.68)'],
    '--color-tint-hover': [
      'color-mix(in srgb, #f5f4f1 58%, transparent)',
      'color-mix(in srgb, #12151a 58%, transparent)',
    ],
    '--color-skeleton': ['#f2f1ee', '#15181e'],
    '--color-track': ['#eceded', '#1c1e21'],

    // Focus ring color — classic --color-focus
    '--focus-outline-color': ['#1c4f6e', '#6fb2d6'],

    // Shadows — classic card / sheet / overlay pairs
    '--shadow-low': ['0 1px 2px rgba(28, 26, 20, 0.05)', '0 1px 2px rgba(0, 0, 0, 0.45)'],
    '--shadow-med': [
      '0 1px 2px rgba(28, 26, 20, 0.05), 0 12px 32px rgba(28, 26, 20, 0.06)',
      '0 1px 2px rgba(0, 0, 0, 0.45), 0 12px 32px rgba(0, 0, 0, 0.3)',
    ],
    '--shadow-high': [
      '0 2px 8px rgba(0, 0, 0, 0.08), 0 14px 30px rgba(0, 0, 0, 0.09)',
      '0 2px 8px rgba(0, 0, 0, 0.5), 0 14px 30px rgba(0, 0, 0, 0.4)',
    ],

    // Radius — classic tag 6 / control 7 / sheet 10 (fixed scale can't express them)
    '--radius-inner': '6px',
    '--radius-element': '7px',
    '--radius-container': '10px',
    '--radius-page': '10px',
    '--radius-chat': '10px',

    // Control heights — classic compact/sm/md stops (26/32/42 gaps handled
    // via component overrides once the shell lands)
    '--size-element-sm': '30px',
    '--size-element-md': '34px',
    '--size-element-lg': '38px',

    // Semantic text sizes — classic body/meta/label/title tokens
    '--text-body-size': '13.5px',
    '--text-large-size': '16px',
    '--text-supporting-size': '12px',
    '--text-label-size': '11.5px',
    '--text-code-size': '13.5px',
    '--text-heading-1-size': '26px',
    '--text-heading-2-size': '22px',
    '--text-heading-3-size': '16px',
    '--text-heading-4-size': '15px',
    '--text-heading-5-size': '13.5px',
    '--text-heading-6-size': '13.5px',
    '--text-display-1-size': '34px',
    '--text-display-2-size': '30px',
    '--text-display-3-size': '26px',
  },
  adaptations: {
    // Classic applies --touch-target (44px) to shell actions at
    // max-width:860px; `below md` with md=861 matches that range.
    widthBreakpoints: { md: 861 },
    rules: [
      {
        when: { width: { below: 'md' } },
        value: {
          components: {
            // IconButton renders Button — one key covers the preferences
            // trigger and the import action. The bump is component-wide;
            // classic scopes it to shell controls only.
            button: {
              base: { minHeight: '44px', minWidth: '44px' },
            },
          },
        },
      },
    ],
  },
})
