import * as stylex from '@stylexjs/stylex'
import type { JSX } from 'react'

const narrow = '@media (max-width: 767px)'
const touch = '@media (pointer: coarse)'

const styles = stylex.create({
  numberField: { display: 'grid', minWidth: 0, gap: 4 },
  inputShell: {
    display: 'flex',
    width: '100%',
    minWidth: 0,
    boxSizing: 'border-box',
    alignItems: 'center',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: { default: 'var(--color-border-control)', ':focus-within': 'var(--color-focus)' },
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    paddingInline: 6,
    minHeight: { default: 32, [narrow]: 44, [touch]: 44 },
    fontSize: 'var(--text-meta)',
  },
  inputShellInvalid: { borderColor: 'var(--color-danger)' },
  inputError: {
    color: 'var(--color-danger)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
  },
  inputShellDisabled: { cursor: 'not-allowed', opacity: 0.55 },
  inputInner: {
    width: '100%',
    minWidth: 0,
    minHeight: 'inherit',
    borderWidth: 0,
    outlineStyle: 'none',
    backgroundColor: 'transparent',
    color: { default: 'var(--color-text)', '::placeholder': 'var(--color-text-faint)' },
    padding: 0,
    fontFamily: 'inherit',
    fontSize: { default: 'inherit', [narrow]: '14px' },
    fontWeight: 'inherit',
    lineHeight: 'inherit',
    appearance: 'textfield',
    fontVariantNumeric: 'tabular-nums',
    cursor: { ':disabled': 'not-allowed' },
  },
  srOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clipPath: 'inset(50%)',
    whiteSpace: 'nowrap',
  },
})

export function ScheduleNumberInput({
  id,
  label,
  value,
  placeholder,
  invalid,
  invalidMessage,
  min,
  max,
  disabled,
  onChange,
}: {
  id: string
  label: string
  value: string
  placeholder?: string
  invalid?: boolean
  invalidMessage?: string
  min?: number
  max?: number
  disabled?: boolean
  onChange(value: string): void
}): JSX.Element {
  return (
    <div {...stylex.props(styles.numberField)}>
      <div
        data-input-shell
        {...stylex.props(
          styles.inputShell,
          invalid === true && styles.inputShellInvalid,
          disabled === true && styles.inputShellDisabled,
        )}
      >
        <label {...stylex.props(styles.srOnly)} htmlFor={id}>
          {label}
        </label>
        <input
          id={id}
          data-input-inner
          {...stylex.props(styles.inputInner)}
          type="number"
          value={value}
          placeholder={placeholder}
          min={min}
          max={max}
          disabled={disabled}
          aria-invalid={invalid === true || undefined}
          aria-describedby={invalid === true ? `${id}-error` : undefined}
          autoComplete="off"
          spellCheck={false}
          inputMode="numeric"
          onChange={(event) => onChange(event.currentTarget.value)}
        />
      </div>
      {invalid === true && (
        <span id={`${id}-error`} {...stylex.props(styles.inputError)} role="alert">
          {invalidMessage} (
          {min !== undefined ? `${min}${max !== undefined ? `–${max}` : '+'}` : ''})
        </span>
      )}
    </div>
  )
}
