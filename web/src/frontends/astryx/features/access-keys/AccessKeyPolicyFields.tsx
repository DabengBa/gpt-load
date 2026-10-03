import * as stylex from '@stylexjs/stylex'
import { DateTimeInput, Selector, TextArea, type ISODateTimeString } from '@astryxdesign/core'

import { useState } from 'react'

import { currentTimeZone } from '@shared/lib/time'

import { useT } from '../../app/i18n'

const TIGHT = '@media (max-width: 560px)'

const styles = stylex.create({
  fields: {
    display: 'grid',
    gap: 12,
  },
  row: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) auto',
      [TIGHT]: 'minmax(0, 1fr)',
    },
    alignItems: 'center',
    gap: 'var(--space-3)',
  },
  rowLabelTitle: {
    display: 'block',
    fontSize: 'var(--text-meta)',
  },
  rowLabelText: {
    marginTop: 3,
    marginBottom: 0,
    marginInline: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  // Classic styles the CIDR textarea in monospace; xstyle lands on the field
  // wrapper and cascades into the control.
  cidrField: {
    fontFamily: 'var(--font-mono)',
  },
  cidrWarning: {
    margin: 0,
    color: 'var(--color-warning)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.55,
  },
})

function pad(value: number): string {
  return String(value).padStart(2, '0')
}

function localDateTimeValue(epochMS: number | null): string {
  if (epochMS === null || !Number.isSafeInteger(epochMS) || epochMS <= 0) return ''
  const date = new Date(epochMS)
  if (Number.isNaN(date.getTime())) return ''
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

// Port of classic AccessKeyPolicyFields.vue: expiration mode + datetime,
// connection-source mode, and the allowed-CIDRs textarea. v-model pairs become
// value + onXChange props.
export function AccessKeyPolicyFields({
  expirationMode,
  expiresAt,
  baseExpiresAt,
  sourceMode,
  allowedCidrs,
  disabled,
  onExpirationModeChange,
  onExpiresAtChange,
  onSourceModeChange,
  onAllowedCidrsChange,
}: {
  expirationMode: 'never' | 'specified'
  expiresAt: number | null
  baseExpiresAt?: number | null
  sourceMode: 'all' | 'restricted'
  allowedCidrs: string[]
  disabled: boolean
  onExpirationModeChange(value: 'never' | 'specified'): void
  onExpiresAtChange(value: number | null): void
  onSourceModeChange(value: 'all' | 'restricted'): void
  onAllowedCidrsChange(value: string[]): void
}) {
  const t = useT()

  const expirationOptions = [
    { value: 'never', label: t('accessKeys.drawer.expirationNever') },
    { value: 'specified', label: t('accessKeys.drawer.expirationSpecified') },
  ]
  const sourceOptions = [
    { value: 'all', label: t('accessKeys.drawer.sourceAll') },
    { value: 'restricted', label: t('accessKeys.drawer.sourceRestricted') },
  ]

  // `now` is seeded at mount and refreshed in the expiration change handlers —
  // the classic recomputes Date.now() on each re-render, so the clock is
  // current whenever the user edits; render itself stays pure (Compiler rule).
  const [now, setNow] = useState(() => Date.now())
  const expirationInput = localDateTimeValue(expiresAt)
  const minimumExpiration = (() => {
    const nextSecond = now + 1_000
    if (expiresAt !== null && expiresAt === baseExpiresAt) {
      return localDateTimeValue(Math.min(expiresAt, nextSecond))
    }
    return localDateTimeValue(nextSecond)
  })()
  const expirationError = (() => {
    if (expirationMode !== 'specified') return undefined
    if (expiresAt === null || expiresAt <= 0) {
      return t('accessKeys.drawer.expirationRequired')
    }
    if (expiresAt !== baseExpiresAt && expiresAt <= now) {
      return t('accessKeys.drawer.expirationFuture')
    }
    return undefined
  })()
  const cidrInput = allowedCidrs.join('\n')
  const normalizedCIDRCount = new Set(allowedCidrs.map((value) => value.trim()).filter(Boolean))
    .size
  const cidrError = (() => {
    if (sourceMode !== 'restricted') return undefined
    if (normalizedCIDRCount === 0) return t('accessKeys.drawer.sourceRequired')
    if (normalizedCIDRCount > 64) return t('accessKeys.drawer.sourceLimit')
    return undefined
  })()

  function setExpirationMode(value: string): void {
    if (value !== 'never' && value !== 'specified') return
    setNow(Date.now())
    onExpirationModeChange(value)
  }

  function setExpiration(value: string | undefined): void {
    setNow(Date.now())
    if (value === undefined || value === '') {
      onExpiresAtChange(0)
      return
    }
    const epochMS = new Date(value).getTime()
    onExpiresAtChange(Number.isSafeInteger(epochMS) ? epochMS : 0)
  }

  function setSourceMode(value: string): void {
    if (value === 'all' || value === 'restricted') onSourceModeChange(value)
  }

  function setCIDRs(value: string): void {
    onAllowedCidrsChange(value.split(/\r?\n/u))
  }

  return (
    <div {...stylex.props(styles.fields)}>
      <div {...stylex.props(styles.row)}>
        <div>
          <strong {...stylex.props(styles.rowLabelTitle)}>
            {t('accessKeys.drawer.expiration')}
          </strong>
          <p {...stylex.props(styles.rowLabelText)}>
            {t('accessKeys.drawer.expirationDescription')}
          </p>
        </div>
        <Selector
          label={t('accessKeys.drawer.expiration')}
          isLabelHidden
          options={expirationOptions}
          value={expirationMode}
          size="sm"
          isDisabled={disabled}
          onChange={setExpirationMode}
        />
      </div>

      {expirationMode === 'specified' && (
        <DateTimeInput
          label={t('accessKeys.drawer.expirationTime')}
          description={t('accessKeys.drawer.expirationTimezone', {
            timezone: currentTimeZone(),
          })}
          status={
            expirationError !== undefined ? { type: 'error', message: expirationError } : undefined
          }
          size="sm"
          hasSeconds
          isDisabled={disabled}
          value={expirationInput === '' ? undefined : (expirationInput as ISODateTimeString)}
          min={minimumExpiration === '' ? undefined : (minimumExpiration as ISODateTimeString)}
          onChange={setExpiration}
        />
      )}

      <div {...stylex.props(styles.row)}>
        <div>
          <strong {...stylex.props(styles.rowLabelTitle)}>{t('accessKeys.drawer.sourceIP')}</strong>
          <p {...stylex.props(styles.rowLabelText)}>{t('accessKeys.drawer.sourceIPDescription')}</p>
        </div>
        <Selector
          label={t('accessKeys.drawer.sourceIP')}
          isLabelHidden
          options={sourceOptions}
          value={sourceMode}
          size="sm"
          isDisabled={disabled}
          onChange={setSourceMode}
        />
      </div>

      {sourceMode === 'restricted' && (
        <div {...stylex.props(styles.fields)}>
          <TextArea
            label={t('accessKeys.drawer.allowedCIDRs')}
            description={t('accessKeys.drawer.allowedCIDRsDescription')}
            status={cidrError !== undefined ? { type: 'error', message: cidrError } : undefined}
            value={cidrInput}
            rows={4}
            placeholder={t('accessKeys.drawer.allowedCIDRsPlaceholder')}
            isDisabled={disabled}
            hasSpellCheck={false}
            size="sm"
            xstyle={styles.cidrField}
            onChange={setCIDRs}
          />
          <p {...stylex.props(styles.cidrWarning)}>{t('accessKeys.drawer.proxyIPWarning')}</p>
        </div>
      )}
    </div>
  )
}
