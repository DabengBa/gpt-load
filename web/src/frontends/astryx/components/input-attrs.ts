/**
 * `inputMode`/`spellCheck`/`autoCorrect` are intentionally omitted from the
 * Astryx BaseProps prop surface but still forward through `...rest` onto the
 * underlying `<input>` — pass them as spread objects so the numeric-keyboard
 * and no-spellcheck contracts survive the prop types.
 */
export const numericInputAttrs = {
  inputMode: 'numeric',
  spellCheck: false,
  autoCorrect: 'off',
} as const

export const plainTextInputAttrs = {
  spellCheck: false,
  autoCorrect: 'off',
} as const
