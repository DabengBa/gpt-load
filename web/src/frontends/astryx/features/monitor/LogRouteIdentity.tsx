import * as stylex from '@stylexjs/stylex'
import { Tooltip } from '@astryxdesign/core'
import { ArrowRight, ExternalLink } from 'lucide-react'

import type { ChannelDto } from '@shared/control/resources/channels'
import { formatRouteEntity } from '@shared/domain/monitor/log-format'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { ChannelIcon } from '../../components/ChannelIcon'

const styles = stylex.create({
  identity: {
    display: 'grid',
    minWidth: 0,
    gap: 2,
  },
  // compact：图标/渠道名在左列,分组行与凭据行在右列,长名互不挤压。
  identityCompact: {
    gridTemplateColumns: 'auto minmax(0, 1fr)',
    columnGap: 5,
    alignItems: 'center',
  },
  identityCompactSolo: {
    gridTemplateColumns: 'minmax(0, 1fr)',
  },
  asideCompact: {
    gridColumn: 1,
    gridRow: 1,
    maxWidth: 64,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  lineCompact: {
    gridColumn: 2,
    gridRow: 1,
  },
  credentialRow: {
    gridColumn: 2,
  },
  // 抽屉空间充裕，排成一行并跟随所在字段字号。
  identityPlain: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    rowGap: 4,
    columnGap: 8,
  },
  line: {
    display: 'flex',
    flexWrap: 'nowrap',
    minWidth: 0,
    alignItems: 'center',
    gap: 4,
  },
  linePlain: {
    display: 'contents',
  },
  icon: {
    display: 'inline-flex',
    flex: 'none',
    fontSize: 'var(--text-label-xs)',
  },
  channel: {
    flex: 'none',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  // 名称可任意长，一律省略；凭据独占一行，两者不挤压彼此。
  // （astryx reset 已给 button 补 font: inherit / color: inherit。）
  value: {
    display: 'block',
    minWidth: 0,
    overflow: 'hidden',
    borderWidth: 0,
    backgroundColor: 'transparent',
    padding: 0,
    // 行高须高于字号，否则悬停下划线被 overflow 裁掉。
    lineHeight: 1.5,
    textAlign: 'left',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  group: {
    flex: '1 1 auto',
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    fontWeight: 560,
  },
  credential: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  credentialPlain: {
    color: 'var(--color-text-muted)',
  },
  code: {
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
  },
  groupCode: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  // Classic .filterable-value：平时跟随所在列的字色字号，只在悬停时点出可交互。
  filterable: {
    color: { ':hover': 'var(--color-action)' },
    textDecorationLine: { ':hover': 'underline' },
    textUnderlineOffset: { ':hover': 2 },
    cursor: 'pointer',
    borderRadius: { ':focus-visible': 3 },
    outlineWidth: { ':focus-visible': 2 },
    outlineStyle: { ':focus-visible': 'solid' },
    outlineColor: { ':focus-visible': 'var(--color-focus)' },
    outlineOffset: { ':focus-visible': 2 },
  },
  actionIcon: {
    display: 'inline-flex',
    flex: 'none',
    alignItems: 'center',
    justifyContent: 'center',
    width: 'var(--control-compact)',
    height: 'var(--control-compact)',
    minWidth: 'var(--control-compact)',
    minHeight: 'var(--control-compact)',
    borderRadius: 'var(--radius-control)',
    color: { default: 'var(--color-text-faint)', ':hover': 'var(--color-action)' },
  },
  groupLink: {
    color: { ':focus-visible': 'var(--color-action)' },
  },
  // plain：图标、渠道名、分组、凭据跟随所在字段字号。
  inheritSize: {
    fontSize: 'inherit',
  },
})

function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}`
}

export interface LogRouteIdentityProps {
  groupId: number | null
  groupName?: string | null
  providerUrl?: string | null
  channelId: string | null
  channel?: Pick<ChannelDto, 'name' | 'icon' | 'mark'> | null
  credentialId: number | null
  credentialName?: string | null
  /** 名称来源已就绪却查不到，即该实体已被删除。 */
  groupDeleted?: boolean
  /** 分组 options 未完成加载时保持维护入口关闭，但保留日志中的历史名称。 */
  groupResolved?: boolean
  credentialDeleted?: boolean
  appearance?: 'compact' | 'plain'
  /** 列表里点分组、凭据就地筛选；详情抽屉没有筛选上下文，保持纯展示。 */
  filterable?: boolean
  onFilterGroup?: (groupId: number) => void
  onFilterCredential?: (credentialId: number) => void
}

/**
 * Route identity chain for a request log — classic LogRouteIdentity.vue.
 * Compact mode is the two-line list cell; `plain` is the drawer detail row.
 */
export function LogRouteIdentity({
  groupId,
  groupName,
  providerUrl,
  channelId,
  channel,
  credentialId,
  credentialName,
  groupDeleted = false,
  groupResolved = true,
  credentialDeleted = false,
  appearance = 'compact',
  filterable = false,
  onFilterGroup,
  onFilterCredential,
}: LogRouteIdentityProps) {
  const t = useT()

  // 图标优先表达渠道；图标资源缺失时退回渠道名，ID 是最后的诊断兜底。
  const showsIcon = Boolean(channel?.mark)
  const channelLabel = channel?.name.trim() || channelId || '—'

  // 取不到名称（已删除）才退回编号，凭据与分组同一套规则。
  const deletedText = (id: number): string => t('monitor.logs.deletedRef', { id })
  const groupLabel =
    groupDeleted && groupId !== null
      ? deletedText(groupId)
      : formatRouteEntity({
          id: groupId,
          name: groupName,
          deleted: false,
          prefix: 'G',
          deletedText,
        })
  const credentialLabel =
    credentialId === null
      ? ''
      : formatRouteEntity({
          id: credentialId,
          name: credentialName,
          deleted: credentialDeleted,
          prefix: 'K',
          deletedText,
        })
  const hasGroupName = Boolean(groupName?.trim()) || groupDeleted
  const hasCredentialName = Boolean(credentialName?.trim()) || credentialDeleted
  const canFilterGroup = filterable && groupId !== null
  const canFilterCredential = filterable && credentialId !== null
  const canOpenGroup =
    groupResolved && groupId !== null && !groupDeleted && Boolean(groupName?.trim())

  // 四个字段共用一条提示：列里放不下的名称和已折叠的供应商外链在这里给全。
  const tooltipLines: string[] = []
  if (channelId !== null) {
    tooltipLines.push(t('monitor.logs.routeIdentity.channel', { name: channelLabel }))
  }
  if (groupId !== null) {
    tooltipLines.push(t('monitor.logs.routeIdentity.group', { name: groupLabel }))
  }
  if (credentialLabel !== '') {
    tooltipLines.push(t('monitor.logs.routeIdentity.credential', { name: credentialLabel }))
  }
  if (providerUrl) {
    tooltipLines.push(t('monitor.logs.routeIdentity.provider', { url: providerUrl }))
  }
  const routeTooltip = tooltipLines.join('\n')

  const groupAction = t('monitor.logs.routeIdentity.filterGroup', { name: groupLabel })
  const credentialAction = t('monitor.logs.routeIdentity.filterCredential', {
    name: credentialLabel,
  })
  const groupLinkAction = t('monitor.logs.routeIdentity.openGroup', { name: groupLabel })

  const isCompact = appearance === 'compact'
  const hasAside = showsIcon || channelId !== null
  const groupChipStyle = [
    styles.value,
    styles.group,
    !hasGroupName && styles.groupCode,
    !isCompact && styles.inheritSize,
  ] as const
  // 分组、凭据各占一行——compact 下凭据落在分组行下方的同一网格列。
  const credentialChipStyle = [
    styles.value,
    styles.credential,
    !hasCredentialName && styles.code,
    isCompact
      ? hasAside
        ? styles.credentialRow
        : undefined
      : styles.credentialPlain,
    !isCompact && styles.inheritSize,
  ] as const

  const credentialChip = () =>
    canFilterCredential ? (
      <button
        type="button"
        data-testid="log-route-identity__credential"
        {...stylex.props(...credentialChipStyle, styles.filterable)}
        aria-label={credentialAction}
        onClick={() => {
          if (credentialId !== null) onFilterCredential?.(credentialId)
        }}
      >
        {credentialLabel}
      </button>
    ) : (
      <span
        data-testid="log-route-identity__credential"
        {...stylex.props(...credentialChipStyle)}
      >
        {credentialLabel}
      </span>
    )

  return (
    // 抽屉里字段已完整展开，无需再提示。
    <Tooltip
      content={routeTooltip}
      isEnabled={appearance !== 'plain' && routeTooltip !== ''}
      placement="above"
      alignment="start"
    >
      <span
        data-testid="log-route-identity"
        tabIndex={isCompact ? 0 : undefined}
        aria-label={isCompact ? routeTooltip : undefined}
        {...stylex.props(
          styles.identity,
          isCompact
            ? hasAside
              ? styles.identityCompact
              : styles.identityCompactSolo
            : styles.identityPlain,
        )}
      >
        {showsIcon ? (
          <span
            {...stylex.props(
              styles.icon,
              isCompact && hasAside && styles.asideCompact,
              !isCompact && styles.inheritSize,
            )}
          >
            <ChannelIcon icon={channel?.icon ?? ''} mark={channel?.mark ?? ''} />
          </span>
        ) : channelId !== null ? (
          <span
            {...stylex.props(
              styles.channel,
              isCompact && hasAside && styles.asideCompact,
              !isCompact && styles.inheritSize,
            )}
          >
            {channelLabel}
          </span>
        ) : null}
        <span
          {...stylex.props(
            styles.line,
            isCompact && hasAside && styles.lineCompact,
            !isCompact && styles.linePlain,
          )}
        >
          {canFilterGroup ? (
            <button
              type="button"
              data-testid="log-route-identity__group"
              {...stylex.props(...groupChipStyle, styles.filterable)}
              aria-label={groupAction}
              onClick={() => {
                if (groupId !== null) onFilterGroup?.(groupId)
              }}
            >
              {groupLabel}
            </button>
          ) : (
            <span data-testid="log-route-identity__group" {...stylex.props(...groupChipStyle)}>
              {groupLabel}
            </span>
          )}
          {/* 分组维护页与供应商外链只在抽屉（plain）保留。 */}
          {!isCompact && (
            <>
              {canOpenGroup && (
                <RouteLink
                  to={groupDetailHref(groupId)}
                  data-testid="log-route-identity__group-link"
                  {...stylex.props(styles.actionIcon, styles.groupLink)}
                  aria-label={groupLinkAction}
                  onClick={(event) => event.stopPropagation()}
                >
                  <ArrowRight size={14} aria-hidden="true" />
                </RouteLink>
              )}
              {providerUrl && (
                <a
                  data-testid="log-route-identity__provider"
                  {...stylex.props(styles.actionIcon)}
                  href={providerUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label={t('monitor.logs.routeIdentity.openProviderUrl', {
                    url: providerUrl,
                  })}
                  onClick={(event) => event.stopPropagation()}
                >
                  <ExternalLink size={12} aria-hidden="true" />
                </a>
              )}
            </>
          )}
        </span>
        {credentialLabel !== '' && credentialChip()}
      </span>
    </Tooltip>
  )
}
