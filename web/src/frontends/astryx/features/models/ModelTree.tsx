import * as stylex from '@stylexjs/stylex'
import { IconButton, Tooltip } from '@astryxdesign/core'
import { ChevronRight, CircleHelp, Zap } from 'lucide-react'
import { useIntl } from 'react-intl'
import { useState } from 'react'

import { enabledDataProtocols } from '@shared/control/protocols'
import type { GroupProtocol } from '@shared/control/types'
import type {
  ClientModelDto,
  ModelRouteGroupDto,
  ModelUpstreamDto,
} from '@shared/control/resources/models'
import { modelPriceFields } from '@shared/domain/model-prices/model-price-form'
import {
  presentClientModel,
  type ClientModelRow,
  type ModelUpstreamRow,
} from '@shared/domain/models/model-presenter'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { ChannelIcon } from '../../components/ChannelIcon'
import { CopyButton } from '../../components/CopyButton'
import { ModelPriceStatusBadge } from './ModelPriceStatusBadge'

const NARROW = '@media (max-width: 860px)'
// 树线锚点:客户端模型行的主干与上游行的转角共用同一条竖线位置。
const RAIL = '20px'
const CONTROL_XS = 'var(--control-xs, 32px)'
const WIDE_COLUMNS = `minmax(220px, 1fr) repeat(4, minmax(78px, 104px)) auto ${CONTROL_XS}`
const WIDE_COLUMNS_READONLY = 'minmax(220px, 1fr) repeat(4, minmax(78px, 104px))'
const CARD_COLUMNS = 'minmax(0, 1fr)'

const styles = stylex.create({
  tree: {
    minWidth: 0,
    borderWidth: { default: 1, [NARROW]: 0 },
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: { default: 'var(--color-surface)', [NARROW]: 'transparent' },
  },
  scroll: {
    overflowX: { default: 'auto', [NARROW]: 'visible' },
    borderRadius: 'inherit',
  },
  grid: {
    display: 'grid',
    minWidth: { default: '760px', [NARROW]: 0 },
    gridTemplateColumns: { default: WIDE_COLUMNS, [NARROW]: CARD_COLUMNS },
    gap: { default: 0, [NARROW]: 'var(--space-2)' },
  },
  gridReadOnly: {
    gridTemplateColumns: { default: WIDE_COLUMNS_READONLY, [NARROW]: CARD_COLUMNS },
  },
  row: {
    display: 'grid',
    gridColumn: { default: '1 / -1', [NARROW]: '1' },
    gridTemplateColumns: {
      default: WIDE_COLUMNS,
      [NARROW]: CARD_COLUMNS,
    },
    alignItems: 'center',
    columnGap: { default: 'var(--space-3)', [NARROW]: 0 },
  },
  rowReadOnly: {
    gridTemplateColumns: { default: WIDE_COLUMNS_READONLY, [NARROW]: CARD_COLUMNS },
  },
  rowHead: {
    display: { default: 'grid', [NARROW]: 'none' },
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    letterSpacing: '0.03em',
  },
  rowClient: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    // 窄屏卡片化:分组行自身成为卡片容器。
    borderRadius: { default: 0, [NARROW]: 'var(--radius-control)' },
    padding: { default: 0, [NARROW]: 'var(--space-1)' },
  },
  rowClientFirst: {
    borderTopWidth: 0,
  },
  rowUpstream: {
    transitionProperty: 'background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
    // 卡片布局:价格铺成 2×2,箭头脱离网格钉在右上角。
    position: { default: 'static', [NARROW]: 'relative' },
    gridTemplateColumns: {
      default: WIDE_COLUMNS,
      [NARROW]: 'repeat(2, minmax(0, 1fr))',
    },
    alignItems: { default: 'center', [NARROW]: 'start' },
    gap: { default: 0, [NARROW]: 'var(--space-2) var(--space-3)' },
    borderWidth: { default: 0, [NARROW]: 1 },
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    padding: { default: 0, [NARROW]: 'var(--space-3)' },
    backgroundColor: { default: 'transparent', [NARROW]: 'var(--color-surface)' },
  },
  rowUpstreamReadOnly: {
    gridTemplateColumns: {
      default: WIDE_COLUMNS_READONLY,
      [NARROW]: 'repeat(2, minmax(0, 1fr))',
    },
  },
  rowUpstreamHovered: {
    backgroundColor: { default: 'var(--color-interactive-hover)', [NARROW]: 'var(--color-surface)' },
  },
  rowUpstreamSibling: {
    borderTopWidth: 1,
    borderTopColor: 'var(--color-border-subtle)',
  },
  cell: {
    minWidth: 0,
    paddingBlock: { default: 'var(--space-1-75)', [NARROW]: 0 },
  },
  cellFirst: {
    paddingInlineStart: { default: 'var(--space-3-5)', [NARROW]: 'var(--space-2-5)' },
  },
  cellLast: {
    paddingInlineEnd: { default: 'var(--space-2-5)', [NARROW]: 'var(--space-2-5)' },
  },
  headCell: {
    paddingBlock: 'var(--space-2)',
  },
  cellPrice: {
    justifySelf: { default: 'end', [NARROW]: 'stretch' },
    textAlign: { default: 'right', [NARROW]: 'left' },
    display: { default: 'block', [NARROW]: 'grid' },
    gap: { default: 0, [NARROW]: '1px' },
  },
  cellStatus: {
    justifySelf: { default: 'end', [NARROW]: 'start' },
    gridColumn: { default: 'auto', [NARROW]: '1 / -1' },
  },
  cellAction: {
    justifySelf: 'end',
    position: { default: 'static', [NARROW]: 'absolute' },
    top: { default: 'auto', [NARROW]: 'var(--space-2)' },
    insetInlineEnd: { default: 'auto', [NARROW]: 'var(--space-2)' },
    padding: { default: 0, [NARROW]: 0 },
  },
  cellClient: {
    position: 'relative',
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gridColumn: '1 / -1',
    gap: 'var(--space-1) var(--space-2-5)',
    paddingInlineEnd: { default: 'var(--space-3-5)', [NARROW]: 'var(--space-2-5)' },
  },
  cellUpstream: {
    position: 'relative',
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-1) var(--space-2)',
    paddingInlineStart: { default: `calc(${RAIL} + var(--space-4))`, [NARROW]: 0 },
    paddingInlineEnd: { default: 0, [NARROW]: CONTROL_XS },
    gridColumn: { default: 'auto', [NARROW]: '1 / -1' },
  },
  railClient: {
    position: 'absolute',
    display: { default: 'block', [NARROW]: 'none' },
    top: '62%',
    bottom: 0,
    insetInlineStart: RAIL,
    width: '1px',
    backgroundColor: 'var(--color-border-control)',
  },
  railMidVertical: {
    position: 'absolute',
    display: { default: 'block', [NARROW]: 'none' },
    top: '-1px',
    bottom: '-1px',
    insetInlineStart: RAIL,
    width: '1px',
    backgroundColor: 'var(--color-border-control)',
  },
  railMidStub: {
    position: 'absolute',
    display: { default: 'block', [NARROW]: 'none' },
    top: '50%',
    insetInlineStart: RAIL,
    width: '9px',
    height: '1px',
    backgroundColor: 'var(--color-border-control)',
  },
  railLastCorner: {
    position: 'absolute',
    display: { default: 'block', [NARROW]: 'none' },
    top: '-1px',
    bottom: '50%',
    insetInlineStart: RAIL,
    width: '9px',
    borderBottomLeftRadius: '5px',
    borderLeftWidth: 1,
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-control)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
  },
  clientName: {
    overflowWrap: 'anywhere',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-body)',
    fontWeight: 600,
  },
  ident: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-0-5)',
  },
  tag: {
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-tag)',
    paddingBlock: '1px',
    paddingInline: '6px',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
  },
  protocolRestriction: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-1)',
    borderWidth: 0,
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-warning-bg)',
    cursor: 'help',
    paddingBlock: '1px',
    paddingInline: '6px',
    color: 'var(--color-text-muted)',
    fontFamily: 'inherit',
    fontSize: 'var(--text-label-xs)',
  },
  protocolRestrictionIcon: {
    color: 'var(--color-warning)',
  },
  channelIcon: {
    display: 'inline-flex',
    width: '20px',
    height: '20px',
    flex: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 'var(--radius-tag)',
    color: 'var(--color-text-muted)',
    cursor: 'help',
    fontSize: '16px',
    outline: 'none',
  },
  channelIconDecorative: {
    cursor: 'default',
  },
  open: {
    borderWidth: 0,
    backgroundColor: 'transparent',
    cursor: 'pointer',
    padding: 0,
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
    textAlign: 'left',
  },
  openHovered: {
    color: 'var(--color-action)',
    textDecoration: 'underline',
  },
  upstreamName: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
  },
  groups: {
    display: 'inline-flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '4px',
  },
  group: {
    maxWidth: '168px',
    overflow: 'hidden',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    paddingBlock: '1px',
    paddingInline: '7px',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 620,
    textDecoration: 'none',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  groupDisabled: {
    backgroundColor: 'var(--color-neutral-bg)',
    color: 'var(--color-text-faint)',
  },
  groupMore: {
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    cursor: 'help',
  },
  actionIcon: {
    color: 'var(--color-border-control)',
  },
  actionIconHovered: {
    color: 'var(--color-action)',
  },
  price: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    fontVariantNumeric: 'tabular-nums',
  },
  priceEmpty: {
    color: 'var(--color-text-faint)',
  },
  priceValues: {
    display: 'grid',
    justifyItems: { default: 'end', [NARROW]: 'start' },
    gap: '1px',
  },
  fastPrice: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '3px',
    borderRadius: 'var(--radius-tag)',
    color: 'var(--color-text-faint)',
    cursor: 'help',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    outline: 'none',
  },
  priceLabel: {
    display: { default: 'none', [NARROW]: 'block' },
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    padding: 0,
    margin: '-1px',
    overflow: 'hidden',
    clip: 'rect(0, 0, 0, 0)',
    whiteSpace: 'nowrap',
    borderWidth: 0,
  },
})

// 一行里的分组必定同渠道,两枚足够点出归属,其余在抽屉里看。
const visibleRouteGroupCount = 2

function visibleRouteGroups(upstream: ModelUpstreamDto): ModelRouteGroupDto[] {
  return upstream.route_groups.slice(0, visibleRouteGroupCount)
}

function hiddenRouteGroups(upstream: ModelUpstreamDto): ModelRouteGroupDto[] {
  return upstream.route_groups.slice(visibleRouteGroupCount)
}

function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}?tab=models`
}

function RouteGroupsChips({ upstream }: { upstream: ModelUpstreamDto }) {
  const t = useT()
  const intl = useIntl()
  const hidden = hiddenRouteGroups(upstream)
  // narrow 去掉"和 / and"只留分隔符;unit 类型在中日文下不插分隔符,不能用。
  const hiddenNames = (() => {
    try {
      return new Intl.ListFormat(intl.locale, { style: 'narrow', type: 'conjunction' }).format(
        hidden.map(({ name }) => name),
      )
    } catch {
      return hidden.map(({ name }) => name).join(', ')
    }
  })()

  return (
    <span {...stylex.props(styles.groups)} aria-label={t('models.tree.routeGroups')}>
      {visibleRouteGroups(upstream).map((group) => (
        <RouteLink
          key={group.id}
          to={groupDetailHref(group.id)}
          title={
            group.enabled
              ? t('models.tree.routeGroupLink', { name: group.name })
              : t('models.tree.routeGroupDisabled', { name: group.name })
          }
          {...stylex.props(styles.group, !group.enabled && styles.groupDisabled)}
        >
          {group.name}
        </RouteLink>
      ))}
      {hidden.length > 0 && (
        <Tooltip
          content={t('models.tree.routeGroupMore', {
            count: hidden.length,
            names: hiddenNames,
          })}
        >
          <span {...stylex.props(styles.group, styles.groupMore)} tabIndex={0}>
            +{hidden.length}
          </span>
        </Tooltip>
      )}
    </span>
  )
}

function UpstreamRow({
  row,
  index,
  isLast,
  readOnly,
  onOpen,
}: {
  row: ModelUpstreamRow
  index: number
  isLast: boolean
  readOnly: boolean
  onOpen: (upstream: ModelUpstreamDto) => void
}) {
  const t = useT()
  const [hovered, setHovered] = useState(false)
  const upstream = row.upstream
  const pricingIdentity = t('models.tree.pricingIdentityHelp', {
    channel: upstream.price.channel_name.trim() || upstream.price.channel_id,
    model: upstream.model_id,
  })

  return (
    <div
      role="row"
      {...stylex.props(
        styles.row,
        styles.rowUpstream,
        readOnly && styles.rowUpstreamReadOnly,
        index > 0 && styles.rowUpstreamSibling,
        hovered && styles.rowUpstreamHovered,
      )}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div role="cell" {...stylex.props(styles.cell, styles.cellUpstream)}>
        {/* 树轨:非末行 ├ 竖线+横短杆;末行 └ 圆角转角。 */}
        {isLast ? (
          <span {...stylex.props(styles.railLastCorner)} aria-hidden />
        ) : (
          <>
            <span {...stylex.props(styles.railMidVertical)} aria-hidden />
            <span {...stylex.props(styles.railMidStub)} aria-hidden />
          </>
        )}
        <span {...stylex.props(styles.ident)}>
          {!readOnly ? (
            <Tooltip content={pricingIdentity} alignment="start">
              <span
                {...stylex.props(styles.channelIcon)}
                tabIndex={0}
                aria-label={pricingIdentity}
              >
                <ChannelIcon icon={upstream.price.channel_icon} mark={upstream.price.channel_mark} />
              </span>
            </Tooltip>
          ) : (
            <span
              {...stylex.props(styles.channelIcon, styles.channelIconDecorative)}
              aria-hidden
            >
              <ChannelIcon icon={upstream.price.channel_icon} mark={upstream.price.channel_mark} />
            </span>
          )}
          {!readOnly ? (
            <button
              type="button"
              {...stylex.props(styles.open, hovered && styles.openHovered)}
              aria-label={t('models.tree.open', { model: upstream.model_id })}
              onClick={() => onOpen(upstream)}
            >
              {upstream.model_id}
            </button>
          ) : (
            <span {...stylex.props(styles.upstreamName)}>{upstream.model_id}</span>
          )}
          <CopyButton
            value={upstream.model_id}
            label={t('models.tree.copyUpstream', { model: upstream.model_id })}
            successLabel={t('models.tree.copySucceeded')}
            failureLabel={t('models.tree.copyFailed')}
          />
        </span>
        {row.tierCount > 0 && (
          <span {...stylex.props(styles.tag)}>
            {t('models.tree.tierCount', { count: row.tierCount })}
          </span>
        )}
        {!readOnly && upstream.route_groups.length > 0 && (
          <RouteGroupsChips upstream={upstream} />
        )}
      </div>

      {modelPriceFields.map((field, fieldIndex) => (
        <div
          key={field}
          role="cell"
          {...stylex.props(
            styles.cell,
            styles.cellPrice,
            readOnly && fieldIndex === modelPriceFields.length - 1 && styles.cellLast,
          )}
        >
          <span {...stylex.props(styles.priceLabel)} aria-hidden>
            {t(`modelPrices.fields.${field}`)}
          </span>
          <span {...stylex.props(styles.priceValues)}>
            <span
              {...stylex.props(styles.price, row.prices[field] === null && styles.priceEmpty)}
            >
              {row.prices[field] ?? t('models.tree.noPrice')}
            </span>
            {row.fastPrices && (
              <Tooltip content={t('models.tree.fastPrice')}>
                <span
                  {...stylex.props(styles.fastPrice)}
                  tabIndex={0}
                  aria-label={t('models.tree.fastPriceValue', {
                    field: t(`modelPrices.fields.${field}`),
                    price: row.fastPrices[field] ?? t('models.tree.noPrice'),
                  })}
                >
                  <Zap size={11} aria-hidden />
                  <span
                    {...stylex.props(
                      row.fastPrices[field] === null ? styles.priceEmpty : {},
                    )}
                  >
                    {row.fastPrices[field] ?? t('models.tree.noPrice')}
                  </span>
                </span>
              </Tooltip>
            )}
          </span>
        </div>
      ))}

      {!readOnly && (
        <div role="cell" {...stylex.props(styles.cell, styles.cellStatus)}>
          <ModelPriceStatusBadge
            price={upstream.price}
            providerName={upstream.catalog_reference?.provider_name}
          />
        </div>
      )}

      {!readOnly && (
        <div role="cell" {...stylex.props(styles.cell, styles.cellLast, styles.cellAction)}>
          <span {...stylex.props(styles.actionIcon, hovered && styles.actionIconHovered)}>
            <IconButton
              variant="ghost"
              size="sm"
              label={t('models.tree.open', { model: upstream.model_id })}
              icon={<ChevronRight size={17} aria-hidden />}
              onClick={() => onOpen(upstream)}
            />
          </span>
        </div>
      )}
    </div>
  )
}

export function ModelTree({
  items,
  readOnly = false,
  onOpen,
}: {
  items: ClientModelDto[]
  readOnly?: boolean
  onOpen: (upstream: ModelUpstreamDto) => void
}) {
  const t = useT()
  const rows: ClientModelRow[] = items.map(presentClientModel)

  const hasProtocolRestriction = (protocols: GroupProtocol[]): boolean =>
    readOnly && protocols.length < enabledDataProtocols.length

  return (
    <div {...stylex.props(styles.tree)}>
      <div {...stylex.props(styles.scroll)}>
        <div
          {...stylex.props(styles.grid, readOnly && styles.gridReadOnly)}
          role="table"
          aria-label={t('models.tree.label')}
        >
          <div role="row" {...stylex.props(styles.row, styles.rowHead)}>
            <span role="columnheader" {...stylex.props(styles.cell, styles.cellFirst, styles.headCell)}>
              {t('models.tree.modelColumn')}
            </span>
            {modelPriceFields.map((field) => (
              <span
                key={field}
                role="columnheader"
                {...stylex.props(styles.cell, styles.cellPrice, styles.headCell)}
              >
                {t(`modelPrices.fields.${field}`)}
              </span>
            ))}
            {!readOnly && (
              <span
                role="columnheader"
                {...stylex.props(styles.cell, styles.cellStatus, styles.headCell)}
              >
                {t('models.tree.statusColumn')}
              </span>
            )}
            {!readOnly && (
              <span role="columnheader" {...stylex.props(styles.cell, styles.cellLast, styles.headCell)}>
                <span {...stylex.props(styles.srOnly)}>{t('models.tree.actionColumn')}</span>
              </span>
            )}
          </div>

          {rows.map((row, rowIndex) => (
            <ModelClientGroup
              key={row.model.client_model}
              row={row}
              isFirst={rowIndex === 0}
              readOnly={readOnly}
              onOpen={onOpen}
              hasProtocolRestriction={hasProtocolRestriction}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

function ModelClientGroup({
  row,
  isFirst,
  readOnly,
  onOpen,
  hasProtocolRestriction,
}: {
  row: ClientModelRow
  isFirst: boolean
  readOnly: boolean
  onOpen: (upstream: ModelUpstreamDto) => void
  hasProtocolRestriction: (protocols: GroupProtocol[]) => boolean
}) {
  const t = useT()
  const restricted = hasProtocolRestriction(row.model.protocols)
  const restrictionTooltip = t('models.tree.protocolRestrictedHelp', {
    protocols: row.model.protocols.join('\n'),
  })

  return (
    <>
      <div
        role="row"
        {...stylex.props(styles.row, styles.rowClient, isFirst && styles.rowClientFirst)}
      >
        <div role="cell" {...stylex.props(styles.cell, styles.cellFirst, styles.cellClient)}>
          {/* 树干从组标题行内长出,延伸到行底,交给下方第一个上游行接续转角。 */}
          <span {...stylex.props(styles.railClient)} aria-hidden />
          <span {...stylex.props(styles.ident)}>
            <span {...stylex.props(styles.clientName)}>{row.model.client_model}</span>
            <CopyButton
              value={row.model.client_model}
              label={t('models.tree.copy', { model: row.model.client_model })}
              successLabel={t('models.tree.copySucceeded')}
              failureLabel={t('models.tree.copyFailed')}
            />
          </span>
          <span {...stylex.props(styles.tag)}>
            {t('models.tree.upstreamCount', { count: row.upstreams.length })}
          </span>
          {restricted && (
            <Tooltip content={restrictionTooltip} alignment="start">
              <button
                type="button"
                {...stylex.props(styles.protocolRestriction)}
                aria-label={restrictionTooltip}
              >
                <span {...stylex.props(styles.protocolRestrictionIcon)}>
                  <CircleHelp size={13} aria-hidden />
                </span>
                {t('models.tree.protocolRestricted')}
              </button>
            </Tooltip>
          )}
        </div>
      </div>

      {row.upstreams.map((entry, index) => (
        <UpstreamRow
          key={entry.upstream.price.id}
          row={entry}
          index={index}
          isLast={index === row.upstreams.length - 1}
          readOnly={readOnly}
          onOpen={onOpen}
        />
      ))}
    </>
  )
}
