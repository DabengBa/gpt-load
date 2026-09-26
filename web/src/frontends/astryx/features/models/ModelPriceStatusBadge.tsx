import * as stylex from '@stylexjs/stylex'
import { Badge, Tooltip } from '@astryxdesign/core'
import { CircleAlert, CircleCheck, CircleOff, Info, Pencil } from 'lucide-react'
import type { ReactNode } from 'react'

import type { ModelPriceDto } from '@shared/control/resources/model-prices'

import { useT } from '../../app/i18n'

const styles = stylex.create({
  wrap: {
    display: 'inline-flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    justifyContent: 'flex-end',
    gap: 'var(--space-1)',
    outline: 'none',
  },
})

type Presentation = { icon: ReactNode; label: string; variant: 'success' | 'info' | 'warning' | 'neutral' }

export function ModelPriceStatusBadge({
  price,
  providerName,
}: {
  price: Pick<ModelPriceDto, 'method' | 'pricing_status' | 'matched_provider_id' | 'match_source'>
  providerName?: string
}) {
  const t = useT()

  const presentation = ((): Presentation => {
    const icon = { size: 12, 'aria-hidden': true } as const
    switch (price.method) {
      case 'auto_sync':
        return price.match_source === 'provider_priority_fallback'
          ? {
              icon: <Info {...icon} />,
              label: t('modelPrices.method.reference_price'),
              variant: 'info',
            }
          : {
              icon: <CircleCheck {...icon} />,
              label: t('modelPrices.method.auto_sync'),
              variant: 'success',
            }
      case 'user_set':
        return {
          icon: <Pencil {...icon} />,
          label: t('modelPrices.method.user_set'),
          variant: 'neutral',
        }
      case 'user_marked_unpriced':
        return {
          icon: <CircleOff {...icon} />,
          label: t('modelPrices.status.unpriced'),
          variant: 'neutral',
        }
      default:
        return price.pricing_status === 'pending'
          ? {
              icon: <CircleAlert {...icon} />,
              label: t('modelPrices.status.pending'),
              variant: 'warning',
            }
          : {
              icon: <CircleCheck {...icon} />,
              label: t('modelPrices.status.configured'),
              variant: 'success',
            }
    }
  })()

  const sourceDetail = (() => {
    if (price.method !== 'auto_sync' || price.match_source === null) return null
    const provider = providerName?.trim() || price.matched_provider_id
    if (!provider) return null
    return t(`modelPrices.source.${price.match_source}`, { provider })
  })()

  return (
    <Tooltip content={sourceDetail ?? ''} isEnabled={sourceDetail !== null}>
      <span
        {...stylex.props(styles.wrap)}
        tabIndex={sourceDetail === null ? undefined : 0}
      >
        <Badge variant={presentation.variant} label={presentation.label} icon={presentation.icon} />
      </span>
    </Tooltip>
  )
}
