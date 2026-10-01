import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { useState } from 'react'
import { useIntl } from 'react-intl'

import type { AccessKeyCostLimitRuleStatusDto } from '@shared/control/types'
import { formatLocalInstant } from '@shared/lib/format'

import { useT } from '../../app/i18n'
import { RelativeInstant } from '../../components/RelativeInstant'

const styles = stylex.create({
  preview: {
    cursor: 'help',
    textDecorationLine: 'underline',
    textDecorationStyle: 'dotted',
    textDecorationColor: 'var(--color-border-control)',
    textUnderlineOffset: 3,
    whiteSpace: 'nowrap',
    outline: { ':focus-visible': '2px solid var(--color-focus)' },
    outlineOffset: { ':focus-visible': 2 },
    borderRadius: { ':focus-visible': 3 },
  },
})

type PeriodUnit = 'day' | 'hour' | 'minute' | 'second'

function relativePeriod(seconds: number): { value: number; unit: PeriodUnit } | null {
  if (!Number.isSafeInteger(seconds) || seconds <= 0) return null
  for (const candidate of [
    { seconds: 86_400, unit: 'day' as const },
    { seconds: 3_600, unit: 'hour' as const },
    { seconds: 60, unit: 'minute' as const },
  ]) {
    if (seconds % candidate.seconds === 0) {
      return { value: seconds / candidate.seconds, unit: candidate.unit }
    }
  }
  return { value: seconds, unit: 'second' }
}

// Port of classic AccessKeyCostLimitWindowTime: a live window shows the real
// reset instant; an inactive one previews "when the window would end" from the
// hover/focus moment.
export function CostLimitWindowTime({ rule }: { rule: AccessKeyCostLimitRuleStatusDto }) {
  const intl = useIntl()
  const t = useT()
  const [previewStartedAtMS, setPreviewStartedAtMS] = useState(() => Date.now())

  if (rule.window_ends_at_ms !== null) {
    const actualTooltip =
      rule.window_started_at_ms === null
        ? undefined
        : t('accessKeys.costLimits.windowPeriod', {
            start: formatLocalInstant(rule.window_started_at_ms, intl.locale),
            end: formatLocalInstant(rule.window_ends_at_ms, intl.locale),
          })
    return (
      <RelativeInstant
        instant={rule.window_ends_at_ms}
        emptyLabel={t('accessKeys.costLimits.status.inactive')}
        hint
        tooltipContent={actualTooltip}
      />
    )
  }

  const period = relativePeriod(rule.period_seconds)
  const previewLabel =
    period === null
      ? t('accessKeys.costLimits.status.inactive')
      : new Intl.RelativeTimeFormat(intl.locale, { numeric: 'always' }).format(
          period.value,
          period.unit,
        )
  const durationMS = rule.period_seconds * 1_000
  const previewEndsAtMS = previewStartedAtMS + durationMS
  const previewTooltip =
    !Number.isSafeInteger(durationMS) || !Number.isSafeInteger(previewEndsAtMS)
      ? previewLabel
      : t('accessKeys.costLimits.inactiveWindowPeriod', {
          start: formatLocalInstant(previewStartedAtMS, intl.locale),
          end: formatLocalInstant(previewEndsAtMS, intl.locale),
        })

  return (
    <Tooltip content={previewTooltip}>
      <span
        {...stylex.props(styles.preview)}
        tabIndex={0}
        onPointerEnter={() => setPreviewStartedAtMS(Date.now())}
        onFocus={() => setPreviewStartedAtMS(Date.now())}
      >
        {previewLabel}
      </span>
    </Tooltip>
  )
}
