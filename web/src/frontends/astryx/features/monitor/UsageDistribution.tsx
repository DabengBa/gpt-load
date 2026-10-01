import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { Boxes, KeyRound, Layers3 } from 'lucide-react'
import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { useIntl } from 'react-intl'

import type { AccessKeyOptionDto, GroupOptionDto } from '@shared/control/types'
import type { ChannelDto } from '@shared/control/resources/channels'
import type {
  UsageAggregateDto,
  UsageDistributionAggregateDto,
  UsageDistributionDto,
  UsageDistributionMetric,
} from '@shared/control/resources/usage'
import { formatEstimatedCost, formatInteger, formatPercent, formatTokens } from '@shared/lib/format'

import { useT } from '../../app/i18n'
import { ChannelIcon } from '../../components/ChannelIcon'

const MEDIUM = '@media (max-width: 820px)'
const NARROW = '@media (max-width: 560px)'
const REDUCED_MOTION = '@media (prefers-reduced-motion: reduce)'

type DistributionItem = UsageDistributionDto['items'][number]

interface DistributionRow {
  key: string
  item: UsageDistributionAggregateDto
  identity?: DistributionItem
  rank?: number
}

const styles = stylex.create({
  list: {
    display: 'grid',
    minWidth: 0,
    overflow: 'hidden',
    margin: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
    padding: 0,
  },
  item: {
    minWidth: 0,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    listStyle: 'none',
  },
  // Classic clears the divider via :last-child; StyleX disallows it, so the
  // last <li> applies this style conditionally by index.
  itemLast: {
    borderBottomWidth: 0,
  },
  row: {
    display: 'grid',
    minWidth: 0,
    minHeight: {
      default: 62,
      [NARROW]: 68,
    },
    gridTemplateColumns: {
      default: '34px minmax(180px, 0.85fr) minmax(160px, 1.25fr) 176px 64px',
      [MEDIUM]: '30px minmax(0, 1fr) 154px 58px',
      [NARROW]: '26px minmax(0, 1fr) 132px',
    },
    alignItems: 'center',
    gap: {
      default: 14,
      [MEDIUM]: 10,
    },
    paddingBlock: 10,
    paddingInline: {
      default: 15,
      [MEDIUM]: 12,
    },
    textAlign: 'left',
  },
  rowOther: {
    backgroundColor: 'var(--color-surface-sunken)',
  },
  rank: {
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    letterSpacing: '0.04em',
  },
  identity: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 10,
  },
  icon: {
    display: 'grid',
    width: {
      default: 30,
      [NARROW]: 28,
    },
    height: {
      default: 30,
      [NARROW]: 28,
    },
    flex: 'none',
    placeItems: 'center',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text-muted)',
    fontSize: 17,
  },
  copy: {
    display: 'flex',
    minWidth: 0,
    flexDirection: 'column',
    gap: 3,
  },
  name: {
    maxWidth: '100%',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: 'var(--text-sm)',
    fontWeight: 620,
  },
  meta: {
    overflow: 'hidden',
    color: 'var(--color-text-faint)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    fontSize: 'var(--text-label-xs)',
  },
  visual: {
    display: {
      default: 'block',
      [MEDIUM]: 'none',
    },
    minWidth: 0,
  },
  track: {
    display: 'block',
    overflow: 'hidden',
    height: 8,
    borderRadius: 999,
    backgroundColor: 'var(--color-surface-sunken)',
  },
  fill: {
    display: 'block',
    // --distribution-share is set inline per row.
    width: 'var(--distribution-share)',
    minWidth: 'min(2px, var(--distribution-share))',
    height: '100%',
    borderRadius: 'inherit',
    backgroundColor: 'var(--color-action)',
    transitionProperty: {
      default: 'width',
      [REDUCED_MOTION]: 'none',
    },
    transitionDuration: 'var(--duration-normal)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  fillOther: {
    backgroundColor: 'var(--color-text-faint)',
  },
  values: {
    display: 'flex',
    minWidth: 0,
    flexDirection: 'column',
    gap: 3,
    alignItems: 'flex-end',
    textAlign: 'right',
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
  },
  valuePrimary: {
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    fontWeight: 620,
  },
  share: {
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    color: 'var(--color-text-muted)',
    textAlign: 'right',
    fontSize: 'var(--text-sm)',
    display: {
      default: 'block',
      [NARROW]: 'none',
    },
  },
})

/**
 * OverflowTooltip port: the rich tooltip arms only while the label is
 * truncated, measured by ResizeObserver like the classic component. The row
 * and identity spans keep their native `title` tooltips as in classic.
 */
function DistributionName({ label }: { label: string }) {
  const ref = useRef<HTMLElement | null>(null)
  const [overflowing, setOverflowing] = useState(false)

  useEffect(() => {
    const element = ref.current
    if (element === null) return
    const update = (): void => {
      setOverflowing(element.scrollWidth > element.clientWidth + 1)
    }
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(update) : undefined
    observer?.observe(element)
    // Without ResizeObserver there is no initial delivery — defer the mount
    // measure a frame so it stays out of the synchronous effect body.
    const fallbackFrame = observer === undefined ? requestAnimationFrame(update) : undefined
    return () => {
      observer?.disconnect()
      if (fallbackFrame !== undefined) cancelAnimationFrame(fallbackFrame)
    }
  }, [label])

  return (
    <Tooltip content={label} isEnabled={overflowing}>
      <strong ref={ref} {...stylex.props(styles.name)}>
        {label}
      </strong>
    </Tooltip>
  )
}

const distributionMetrics: UsageDistributionMetric[] = ['requests', 'tokens', 'cost']

/**
 * Distribution ranking list — classic UsageDistribution.vue. Ordered rank +
 * channel icon + proportional share bar per group / access key / model.
 */
export function UsageDistribution({
  distribution,
  summary,
  groups,
  channels,
  accessKeys,
}: {
  distribution: UsageDistributionDto
  summary: UsageAggregateDto
  groups: GroupOptionDto[]
  channels: ChannelDto[]
  accessKeys: AccessKeyOptionDto[]
}) {
  const intl = useIntl()
  const t = useT()
  const locale = intl.locale

  const rows: DistributionRow[] = distribution.items.map((item, index) => ({
    key:
      distribution.dimension === 'group'
        ? `group:${item.group_id ?? 0}`
        : distribution.dimension === 'access_key'
          ? `access-key:${item.access_key_id ?? 0}`
          : `model:${item.model ?? ''}`,
    item,
    identity: item,
    rank: index + 1,
  }))
  if (distribution.other !== null) {
    rows.push({ key: 'other', item: distribution.other })
  }

  function group(item: DistributionItem): GroupOptionDto | undefined {
    return groups.find(({ id }) => id === item.group_id)
  }

  function channel(item: DistributionItem): ChannelDto | undefined {
    const channelID = group(item)?.channel_id
    return channels.find(({ channel_id }) => channel_id === channelID)
  }

  function accessKey(item: DistributionItem): AccessKeyOptionDto | undefined {
    return accessKeys.find(({ id }) => id === item.access_key_id)
  }

  function identityLabel(row: DistributionRow): string {
    if (row.identity === undefined) return t('monitor.usage.distribution.other')
    if (distribution.dimension === 'model') {
      return row.identity.model || t('monitor.usage.distribution.unknownModel')
    }
    if (distribution.dimension === 'access_key') {
      const accessKeyID = row.identity.access_key_id ?? 0
      return (
        accessKey(row.identity)?.name ??
        t('monitor.usage.distribution.deletedOrUnknownAccessKey', { id: accessKeyID })
      )
    }
    const groupID = row.identity.group_id ?? 0
    return group(row.identity)?.name ?? t('monitor.usage.filters.deletedOrUnknown', { id: groupID })
  }

  function identityMeta(row: DistributionRow): string {
    if (row.identity === undefined) {
      if (distribution.dimension === 'group') {
        return t('monitor.usage.distribution.otherGroupHint')
      }
      if (distribution.dimension === 'access_key') {
        return t('monitor.usage.distribution.otherAccessKeyHint')
      }
      return t('monitor.usage.distribution.otherHint')
    }
    if (distribution.dimension === 'model') {
      return t('monitor.usage.distribution.modelHint')
    }
    if (distribution.dimension === 'access_key') {
      return t('monitor.usage.distribution.accessKeyValue', {
        id: row.identity.access_key_id ?? 0,
      })
    }
    const groupID = row.identity.group_id ?? 0
    const groupChannel = channel(row.identity)
    const groupLabel = t('monitor.usage.distribution.groupValue', { id: groupID })
    return groupChannel ? `${groupChannel.name} · ${groupLabel}` : groupLabel
  }

  function metricValue(item: UsageDistributionAggregateDto): number | bigint {
    if (distribution.metric === 'cost') return BigInt(item.estimated_cost_nano_usd)
    if (distribution.metric === 'tokens') return item.total_tokens
    return item.request_count
  }

  function totalValue(): number | bigint {
    if (distribution.metric === 'cost') {
      return BigInt(summary.estimated_cost_nano_usd)
    }
    if (distribution.metric === 'tokens') return summary.total_tokens
    return summary.request_count
  }

  function shareBasisPoints(item: UsageDistributionAggregateDto): number {
    const value = metricValue(item)
    const total = totalValue()
    if (typeof value === 'bigint' && typeof total === 'bigint') {
      if (total === 0n) return 0
      return Number((value * 10_000n + total / 2n) / total)
    }
    if (typeof value === 'number' && typeof total === 'number') {
      if (total === 0) return 0
      return Math.round((value / total) * 10_000)
    }
    return 0
  }

  function shareLabel(item: UsageDistributionAggregateDto): string {
    return formatPercent(shareBasisPoints(item), 10_000, locale)
  }

  function primaryValue(item: UsageDistributionAggregateDto): string {
    if (distribution.metric === 'cost') {
      return formatEstimatedCost(item.estimated_cost_nano_usd, locale)
    }
    if (distribution.metric === 'tokens') {
      return formatTokens(item.total_tokens, locale)
    }
    return formatInteger(item.request_count, locale)
  }

  function secondaryMetricValue(
    metric: UsageDistributionMetric,
    item: UsageDistributionAggregateDto,
  ): string {
    if (metric === 'requests') {
      return t('monitor.usage.distribution.requestsValue', {
        value: formatInteger(item.request_count, locale),
      })
    }
    if (metric === 'tokens') {
      return t('monitor.usage.distribution.tokensValue', {
        value: formatTokens(item.total_tokens, locale),
      })
    }
    return formatEstimatedCost(item.estimated_cost_nano_usd, locale)
  }

  function secondaryValue(item: UsageDistributionAggregateDto): string {
    return distributionMetrics
      .filter((metric) => metric !== distribution.metric)
      .map((metric) => secondaryMetricValue(metric, item))
      .join(' · ')
  }

  return (
    <ol {...stylex.props(styles.list)} aria-label={t('monitor.usage.distribution.caption')}>
      {rows.map((row, index) => {
        const label = identityLabel(row)
        const rowChannel = row.identity !== undefined ? channel(row.identity) : undefined
        const isOther = row.identity === undefined
        return (
          <li
            key={row.key}
            {...stylex.props(styles.item, index === rows.length - 1 && styles.itemLast)}
          >
            <div {...stylex.props(styles.row, isOther && styles.rowOther)} title={label}>
              <span {...stylex.props(styles.rank)} aria-hidden="true">
                {row.rank === undefined ? '∑' : String(row.rank).padStart(2, '0')}
              </span>

              <span {...stylex.props(styles.identity)} title={label}>
                <span {...stylex.props(styles.icon)} aria-hidden="true">
                  {distribution.dimension === 'group' ? (
                    rowChannel !== undefined ? (
                      <ChannelIcon icon={rowChannel.icon} mark={rowChannel.mark} />
                    ) : (
                      <Boxes size={17} aria-hidden="true" />
                    )
                  ) : distribution.dimension === 'access_key' ? (
                    <KeyRound size={17} aria-hidden="true" />
                  ) : (
                    <Layers3 size={17} aria-hidden="true" />
                  )}
                </span>
                <span {...stylex.props(styles.copy)}>
                  <DistributionName label={label} />
                  <small {...stylex.props(styles.meta)}>{identityMeta(row)}</small>
                </span>
              </span>

              <span {...stylex.props(styles.visual)}>
                <span {...stylex.props(styles.track)} aria-hidden="true">
                  <span
                    {...stylex.props(styles.fill, isOther && styles.fillOther)}
                    style={
                      {
                        '--distribution-share': `${shareBasisPoints(row.item) / 100}%`,
                      } as CSSProperties
                    }
                  />
                </span>
              </span>

              <span {...stylex.props(styles.values)}>
                <strong {...stylex.props(styles.valuePrimary)}>{primaryValue(row.item)}</strong>
                <small {...stylex.props(styles.meta)} title={secondaryValue(row.item)}>
                  {secondaryValue(row.item)}
                </small>
              </span>

              <span {...stylex.props(styles.share)}>{shareLabel(row.item)}</span>
            </div>
          </li>
        )
      })}
    </ol>
  )
}
