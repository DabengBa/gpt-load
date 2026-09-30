import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { ChartNoAxesColumnIncreasing, Zap } from 'lucide-react'
import { useIntl } from 'react-intl'

import { formatLogTokenCount } from '@shared/domain/monitor/log-format'

import { useT } from '../../app/i18n'

// stylex.props consumes CompiledStyles (the create() output), not the
// authored StyleXStyles — conditional values (`:focus-visible` etc.) only
// typecheck as the compiled shape.
type StyleProp = stylex.StyleXArray<null | undefined | false | stylex.CompiledStyles>

/**
 * Classic PricingModeIndicator.vue — tier-threshold or fast-mode hint icon
 * shown next to a cost value. `xstyle` carries the caller's chrome (the list
 * uses the compact hint slot; the drawer its own indicator style).
 */
export function PricingModeIndicator({
  mode,
  contextThresholdTokens,
  xstyle,
}: {
  mode: string | null
  contextThresholdTokens: string | null
  xstyle?: StyleProp
}) {
  const t = useT()
  const { locale } = useIntl()
  if (contextThresholdTokens !== null) {
    const tierLabel = t('monitor.logs.pricingMode.tierLabel', {
      threshold: formatLogTokenCount(contextThresholdTokens, locale),
    })
    return (
      <Tooltip content={tierLabel}>
        <span tabIndex={0} aria-label={tierLabel} {...stylex.props(xstyle)}>
          <ChartNoAxesColumnIncreasing size={13} aria-hidden="true" />
        </span>
      </Tooltip>
    )
  }
  if (mode === 'fast') {
    const fastLabel = t('monitor.logs.pricingMode.fastLabel')
    return (
      <Tooltip content={fastLabel}>
        <span tabIndex={0} aria-label={fastLabel} {...stylex.props(xstyle)}>
          <Zap size={13} aria-hidden="true" />
        </span>
      </Tooltip>
    )
  }
  return null
}
