import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { ArrowRightLeft } from 'lucide-react'

import type {
  RequestLogRouteMode,
  RequestLogUpstreamProtocol,
} from '@shared/control/resources/request-logs'
import type { AccessProtocol } from '@shared/control/types'

import { useT } from '../../app/i18n'

const styles = stylex.create({
  marker: {
    display: 'inline-flex',
    width: { default: 20, '@media (max-width: 860px)': 44 },
    height: { default: 20, '@media (max-width: 860px)': 44 },
    flex: '0 0 auto',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 'var(--radius-tag)',
    color: 'var(--color-warning)',
    cursor: 'help',
    backgroundColor: {
      ':hover': 'color-mix(in srgb, var(--color-warning) 10%, transparent)',
    },
    outlineWidth: { ':focus-visible': 2 },
    outlineStyle: { ':focus-visible': 'solid' },
    outlineColor: { ':focus-visible': 'var(--color-focus)' },
    outlineOffset: { ':focus-visible': 1 },
  },
})

/**
 * Protocol-conversion marker for a request log — classic
 * LogProtocolConversion.vue. Only renders when the request was routed through
 * a protocol conversion (`mode === 'converted'`).
 */
export function LogProtocolConversion({
  mode,
  clientProtocol,
  upstreamProtocol,
}: {
  mode: RequestLogRouteMode | null
  clientProtocol: AccessProtocol
  upstreamProtocol: RequestLogUpstreamProtocol | null
}) {
  const t = useT()
  if (mode !== 'converted') return null

  const clientProtocolLabel: string = clientProtocol
  const upstreamProtocolLabel =
    upstreamProtocol === null ? t('monitor.logs.protocolConversion.notRecorded') : upstreamProtocol
  const tooltip =
    upstreamProtocol === null
      ? t('monitor.logs.protocolConversion.tooltipNotRecorded', {
          client: clientProtocolLabel,
        })
      : t('monitor.logs.protocolConversion.tooltip', {
          client: clientProtocolLabel,
          upstream: upstreamProtocolLabel,
        })
  const label = t('monitor.logs.protocolConversion.label', {
    client: clientProtocolLabel,
    upstream: upstreamProtocolLabel,
  })

  return (
    <Tooltip content={tooltip}>
      <span {...stylex.props(styles.marker)} tabIndex={0} aria-label={label}>
        <ArrowRightLeft size={14} strokeWidth={1.8} aria-hidden="true" />
      </span>
    </Tooltip>
  )
}
