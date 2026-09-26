import { TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'

import { useT } from '../../app/i18n'
import { numericInputAttrs } from '../../components/input-attrs'
import {
  cloneSettingsDraft,
  setSettingsOverride,
  type SettingsDraft,
} from '@shared/domain/settings/settings-patch'
import type {
  RuntimeSettingKey,
  SettingsResource,
} from '@shared/control/resources/settings'

const sectionStyles = stylex.create({
  section: {
    display: 'grid',
    gap: 'var(--space-4)',
    scrollMarginTop: '76px',
    minWidth: 0,
  },
  title: {
    margin: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
  },
  description: {
    margin: 0,
    marginTop: 'var(--space-1)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  rows: {
    display: 'grid',
    gap: 'var(--space-1)',
  },
  numberInput: {
    display: 'inline-grid',
    gridTemplateColumns: 'minmax(0, 112px) auto',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
})

/** Section chrome matching the classic `.settings-section` heading+rows shape. */
export function SettingsSectionFrame({
  id,
  title,
  description,
  children,
}: {
  id: string
  title: string
  description: string
  children: React.ReactNode
}) {
  return (
    <section id={id} tabIndex={-1} {...stylex.props(sectionStyles.section)}>
      <header>
        <h2 {...stylex.props(sectionStyles.title)}>{title}</h2>
        <p {...stylex.props(sectionStyles.description)}>{description}</p>
      </header>
      <div {...stylex.props(sectionStyles.rows)}>{children}</div>
    </section>
  )
}

export interface SettingsSectionProps {
  base: SettingsResource
  draft: SettingsDraft
  /** Classic `disabled` — true while a save/restore operation owns the draft. */
  disabled: boolean
  publish: (key: RuntimeSettingKey, draft: SettingsDraft) => void
}

/**
 * Per-section row helpers. Classic duplicated this exact block in all six
 * sections; React sections share it via one hook.
 */
export function useSettingTools({ base, draft, publish }: SettingsSectionProps) {
  const t = useT()
  return {
    hasOverride: (key: RuntimeSettingKey) => draft.overrides.has(key),
    isPendingRestore: (key: RuntimeSettingKey) =>
      !draft.overrides.has(key) && base.settings.overrides.includes(key),
    toggleOverride: (key: RuntimeSettingKey) => {
      publish(key, setSettingsOverride(base.settings, draft, key, !draft.overrides.has(key)))
    },
    /** Clone the draft, mutate it, publish. */
    update: (key: RuntimeSettingKey, mutate: (next: SettingsDraft) => void) => {
      const next = cloneSettingsDraft(draft)
      mutate(next)
      publish(key, next)
    },
    /** `'12'` → `12`, `''`/garbage → `NaN` (classic's invalid-intermediate state). */
    parseCount: (value: string) => (value.trim() === '' ? Number.NaN : Number(value)),
    sourceLabel: (key: RuntimeSettingKey) => {
      if (draft.overrides.has(key)) return t('settings.runtime.overrideSource')
      if (base.settings.overrides.includes(key)) return t('settings.runtime.pendingRestoreSource')
      return t('settings.runtime.defaultSource')
    },
    actionLabel: (key: RuntimeSettingKey) =>
      draft.overrides.has(key)
        ? t('settings.runtime.restoreDefault')
        : t('settings.runtime.override'),
  }
}

export type SettingTools = ReturnType<typeof useSettingTools>

/**
 * Draft number → input text. `NaN` renders as the empty string, matching what
 * a number input shows for `String(NaN)` in classic.
 */
export function numberText(value: number): string {
  return Number.isNaN(value) ? '' : String(value)
}

/**
 * Numeric draft editor — `TextInput` (text, not number) so `NaN` intermediate
 * states surface through `error` exactly like classic's `type="number"` +
 * CompactFieldError flow. `statusVariant="tooltip"` matches the compact
 * error affordance.
 */
export function SettingNumberInput({
  id,
  value,
  label,
  error,
  unit,
  min = '1',
  max = '9223372036',
  disabled,
  onChange,
}: {
  id: string
  /** Input text — use `numberText(draft.values[key])` or a local buffer. */
  value: string
  label: string
  error?: string
  unit: string
  min?: string
  max?: string
  disabled: boolean
  onChange: (value: string) => void
}) {
  const bounds = { min, max, step: '1' }
  return (
    <div {...stylex.props(sectionStyles.numberInput)}>
      <TextInput
        id={id}
        type="text"
        {...numericInputAttrs}
        {...bounds}
        data-gptload-mono
        value={value}
        label={label}
        isLabelHidden
        size="sm"
        isDisabled={disabled}
        status={error === undefined ? undefined : { type: 'error', message: error }}
        statusVariant="tooltip"
        onChange={onChange}
      />
      <span aria-hidden="true">{unit}</span>
    </div>
  )
}
