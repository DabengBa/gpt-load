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
    accent: ['#3155ef', '#6fb2d6'],
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
    '--color-background-body': ['#f7f8fa', '#0b0d10'],
    '--color-background-surface': ['#ffffff', '#171b20'],
    '--color-background-card': ['#ffffff', '#171b20'],
    '--color-background-popover': ['#ffffff', '#171b20'],
    '--color-background-muted': ['#f5f7fa', '#12151a'],

    // Text — classic text / text-muted / text-faint
    '--color-text-primary': ['#202634', '#e8eaec'],
    '--color-text-secondary': ['#515c70', '#969ca3'],
    '--color-text-disabled': ['#626c7e', '#858c94'],

    // Borders — classic border-subtle / border-control
    '--color-border': ['#e4e7ee', '#232830'],
    '--color-border-emphasized': ['#c6cedb', '#333a43'],

    // Accent-adjacent — classic action-soft / action-ink / info
    '--color-accent-muted': ['#edf1ff', '#142633'],
    '--color-on-accent': ['#ffffff', '#0c1a22'],
    '--color-background-blue': ['#edf1ff', '#142633'],
    '--color-text-blue': ['#3155ef', '#6fb2d6'],
    '--color-icon-blue': ['#3155ef', '#6fb2d6'],
    '--color-border-blue': ['#b9ccd7', '#27455a'],

    // Status — classic success/warning/danger/neutral + muted backgrounds
    '--color-success': ['#1a6b3f', '#4fb178'],
    '--color-success-muted': ['#e6f3ec', '#112a1d'],
    '--color-on-success': ['#ffffff', '#0d1f14'],
    '--color-warning': ['#8a5b00', '#d5a341'],
    '--color-warning-muted': ['#f8f0dd', '#241d10'],
    '--color-on-warning': '#241d10',
    '--color-error': ['#d03b3b', '#e66767'],
    '--color-error-muted': ['#fbebe9', '#2a1613'],
    '--color-on-error': ['#ffffff', '#2a1613'],
    // Astryx interaction wash (segmented track, secondary button, nav hover,
    // scrollbar) — NOT classic's opaque neutral status color; that lives in
    // tokens.css as --color-neutral-fg/--color-neutral-bg.
    '--color-neutral': ['#161c2212', '#ffffff1a'],

    // Feedback — classic overlay / interactive hover / skeleton mix / tag track
    '--color-overlay': ['rgba(20, 22, 24, 0.34)', 'rgba(0, 0, 0, 0.68)'],
    // Solid tint mixed via color-mix() into hover borders/fills (checkbox,
    // radio, switch) — must be opaque; a translucent value fades the result.
    '--color-tint-hover': ['#161c22', '#ffffff'],
    '--color-skeleton': ['#f2f1ee', '#15181e'],
    '--color-track': ['#eceded', '#1c1e21'],

    // Focus ring color — classic --color-focus
    '--focus-outline-color': ['#3155ef', '#6fb2d6'],

    // Shadows — classic card / sheet / overlay pairs
    '--shadow-low': ['0 1px 3px rgba(32, 38, 52, 0.04)', '0 1px 2px rgba(0, 0, 0, 0.45)'],
    '--shadow-med': [
      '0 1px 3px rgba(32, 38, 52, 0.04)',
      '0 1px 2px rgba(0, 0, 0, 0.45), 0 12px 32px rgba(0, 0, 0, 0.3)',
    ],
    '--shadow-high': [
      '0 2px 8px rgba(0, 0, 0, 0.08), 0 14px 30px rgba(0, 0, 0, 0.09)',
      '0 2px 8px rgba(0, 0, 0, 0.5), 0 14px 30px rgba(0, 0, 0, 0.4)',
    ],

    // Radius — classic tag 6 / control 7 / sheet 10 (fixed scale can't express them)
    '--radius-inner': '6px',
    '--radius-element': '8px',
    '--radius-container': '14px',
    '--radius-page': '14px',
    '--radius-chat': '14px',

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
