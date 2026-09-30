import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { useEffect, useState } from 'react'
import { useIntl } from 'react-intl'

import { formatISOInstant, formatLocalInstant, formatRelativeInstant } from '@shared/lib/format'
import { currentTimeZone } from '@shared/lib/time'

const styles = stylex.create({
  time: {
    whiteSpace: 'nowrap',
  },
  hint: {
    cursor: 'help',
    textDecorationLine: 'underline',
    textDecorationStyle: 'dotted',
    textDecorationColor: 'var(--color-border-control)',
    textUnderlineOffset: 3,
  },
})

// Port of the classic AppRelativeTime: 30s tick, optional help-hint underline
// with an absolute-time (or custom) tooltip. `instant` is epoch ms.
export function RelativeInstant({
  instant,
  emptyLabel,
  hint = false,
  tooltipContent,
}: {
  instant: number | null
  emptyLabel: string
  hint?: boolean
  tooltipContent?: string
}) {
  const intl = useIntl()
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 30_000)
    return () => window.clearInterval(timer)
  }, [])

  if (instant === null) return <>{emptyLabel}</>
  const dateTime = formatISOInstant(instant)
  const label = formatRelativeInstant(instant, now, intl.locale)
  const content =
    tooltipContent ??
    formatLocalInstant(instant, intl.locale, {
      timeZone: currentTimeZone(),
    })

  if (!hint) {
    return (
      <time {...stylex.props(styles.time)} dateTime={dateTime} title={content}>
        {label}
      </time>
    )
  }
  return (
    <Tooltip content={content}>
      <time {...stylex.props(styles.time, styles.hint)} dateTime={dateTime} tabIndex={0}>
        {label}
      </time>
    </Tooltip>
  )
}
