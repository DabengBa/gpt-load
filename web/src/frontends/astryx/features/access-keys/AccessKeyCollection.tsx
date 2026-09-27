import * as stylex from '@stylexjs/stylex'
import { Badge, Button, IconButton, Table, pixel, proportional, type TableColumn } from '@astryxdesign/core'
import { ArrowRight, RotateCcw, Trash2 } from 'lucide-react'
import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type Ref,
} from 'react'
import { useIntl } from 'react-intl'

import { revealAccessKey } from '@shared/control/resources/access-keys'
import type {
  AccessKeyCollectionItemDto,
  AccessKeyDto,
  GroupOptionDto,
} from '@shared/control/types'
import {
  presentAccessKeyCollection,
  type AccessKeyPresentation,
} from '@shared/domain/access-keys/access-key-presenter'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { RelativeInstant } from '../../components/RelativeInstant'
import { CopyChip } from './AccessKeyCopyChip'
import { AccessKeyCostLimitResetDialog } from './AccessKeyCostLimitResetDialog'
import { AccessKeyDeleteDialog } from './AccessKeyDeleteDialog'

const styles = stylex.create({
  nameCell: {
    minWidth: 0,
    color: 'var(--color-text)',
    fontWeight: 600,
    overflowWrap: 'anywhere',
  },
  statusCell: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 5,
  },
  scopeCell: {
    display: 'grid',
    gap: 3,
    margin: 0,
  },
  scopeRow: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: '46px minmax(0, 1fr)',
    alignItems: 'baseline',
    gap: 6,
  },
  scopeTerm: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    textAlign: 'right',
  },
  scopeValue: {
    minWidth: 0,
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  limitsCell: {
    display: 'grid',
    gap: 2,
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
  },
  timeCell: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
  },
  actionsCell: {
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
})

// Table requires rows to satisfy Record<string, unknown>.
type AccessKeyRow = AccessKeyPresentation & Record<string, unknown>

/** Imperative handle mirroring the classic defineExpose({ conceal }). */
export interface AccessKeyCollectionHandle {
  /**
   * Abort every in-flight key reveal and remount the copy chips so transient
   * copy feedback/fallback state is cleared (classic conceal()).
   */
  conceal(): void
}

export function AccessKeyCollection({
  accessKeys,
  groups,
  total,
  filteredTotal,
  page,
  pageSize,
  busyIds,
  lockedIds,
  onOpen,
  onToggle,
  onDeleted,
  onReset,
  ref,
}: {
  accessKeys: readonly AccessKeyCollectionItemDto[]
  groups: readonly GroupOptionDto[]
  total: number
  filteredTotal: number
  page: number
  pageSize: number
  busyIds: ReadonlySet<number>
  lockedIds: ReadonlySet<number>
  onOpen(accessKey: AccessKeyCollectionItemDto, trigger: HTMLElement): void
  onToggle(accessKey: AccessKeyCollectionItemDto): void
  onDeleted(name: string): void
  onReset(name: string): void
  ref?: Ref<AccessKeyCollectionHandle>
}) {
  const t = useT()
  const intl = useIntl()
  const { apiClient } = useAppServices()
  const sources = useMemo(
    () => new Map(accessKeys.map((accessKey) => [accessKey.id, accessKey])),
    [accessKeys],
  )
  // Classic conceals whenever page or the visible id set changes — a bump in
  // copyGeneration remounts the chips (clearing transient copy state), and the
  // effect below aborts the in-flight reveal requests themselves.
  const ids = accessKeys.map(({ id }) => id).join(',')
  const [copyGeneration, setCopyGeneration] = useState(0)
  const [concealed, setConcealed] = useState({ page, ids })
  if (concealed.page !== page || concealed.ids !== ids) {
    setConcealed({ page, ids })
    setCopyGeneration((generation) => generation + 1)
  }
  const revealControllersRef = useRef(new Map<number, AbortController>())

  const abortReveals = useCallback(() => {
    for (const controller of revealControllersRef.current.values()) controller.abort()
    revealControllersRef.current.clear()
  }, [])

  const conceal = useCallback(() => {
    abortReveals()
    setCopyGeneration((generation) => generation + 1)
  }, [abortReveals])
  useImperativeHandle(ref, () => ({ conceal }), [conceal])

  useEffect(() => {
    abortReveals()
  }, [page, ids, abortReveals])
  useEffect(() => abortReveals, [abortReveals])

  const source = (id: number): AccessKeyCollectionItemDto => {
    const accessKey = sources.get(id)
    if (!accessKey) throw new Error(`ACCESS_KEY_SOURCE_MISSING:${id}`)
    return accessKey
  }

  const presentations = useMemo(
    () =>
      presentAccessKeyCollection(accessKeys, groups, {
        locale: intl.locale,
        labels: {
          groups: t('accessKeys.filterGroups'),
          protocols: t('accessKeys.filterProtocols'),
          models: t('accessKeys.filterModels'),
          allGroups: t('accessKeys.allGroups'),
          allProtocols: t('accessKeys.allProtocols'),
          allModels: t('accessKeys.allModels'),
          unlimited: t('accessKeys.unlimited'),
          costRules: (count) => t('accessKeys.costLimits.ruleCount', { count }),
          priceMultiplier: (value) => t('common.priceMultiplier.value', { value }),
        },
        protocolLabel: (protocol) => protocol,
      }),
    [accessKeys, groups, intl.locale, t],
  )

  const resolveCopyValue = async (id: number): Promise<string> => {
    const controller = new AbortController()
    revealControllersRef.current.set(id, controller)
    try {
      const result = await revealAccessKey(apiClient, id, controller.signal)
      return result.key
    } finally {
      if (revealControllersRef.current.get(id) === controller) {
        revealControllersRef.current.delete(id)
      }
    }
  }

  // Rebuilt per render so cell callbacks never close over stale props.
  const columns: TableColumn<AccessKeyRow>[] = [
      {
        key: 'name',
        header: t('accessKeys.columns.name'),
        width: proportional(1.05),
        renderCell: (record): ReactNode => (
          <span {...stylex.props(styles.nameCell)}>{record.name}</span>
        ),
      },
      {
        key: 'key',
        header: t('accessKeys.columns.key'),
        width: proportional(1.3),
        renderCell: (record): ReactNode => (
          <CopyChip
            key={`${copyGeneration}:${source(record.id).updated_at_ms}`}
            value={record.maskedKey}
            label={t('accessKeys.copy')}
            successLabel={t('common.copied')}
            failureLabel={t('common.copyFailed')}
            resolveValue={() => resolveCopyValue(record.id)}
          />
        ),
      },
      {
        key: 'status',
        header: t('accessKeys.columns.status'),
        width: pixel(120),
        renderCell: (record): ReactNode => (
          <span {...stylex.props(styles.statusCell)}>
            <Badge
              variant={record.status === 'active' ? 'success' : 'neutral'}
              label={t(`accessKeys.status.${record.status}` as MessageId)}
            />
            {record.expired && (
              <Badge variant="error" label={t('accessKeys.status.expired')} />
            )}
            {record.ipRestricted && (
              <Badge variant="neutral" label={t('accessKeys.status.ipRestricted')} />
            )}
            {record.quotaExhausted && (
              <Badge variant="error" label={t('accessKeys.costLimits.exhausted')} />
            )}
          </span>
        ),
      },
      {
        key: 'scope',
        header: t('accessKeys.columns.scope'),
        width: proportional(1.35),
        renderCell: (record): ReactNode => (
          <dl {...stylex.props(styles.scopeCell)}>
            {record.scopeRows.map((scope) => (
              <div key={scope.label} {...stylex.props(styles.scopeRow)}>
                <dt {...stylex.props(styles.scopeTerm)}>{scope.label}</dt>
                <dd {...stylex.props(styles.scopeValue)} title={scope.value}>
                  {scope.value}
                </dd>
              </div>
            ))}
          </dl>
        ),
      },
      {
        key: 'limits',
        header: t('accessKeys.columns.limits'),
        width: pixel(110),
        renderCell: (record): ReactNode => (
          <span {...stylex.props(styles.limitsCell)}>
            {record.limits.map((limit) => (
              <span key={limit}>{limit}</span>
            ))}
          </span>
        ),
      },
      {
        key: 'lastRequest',
        header: t('accessKeys.columns.lastRequest'),
        width: pixel(120),
        renderCell: (record): ReactNode => (
          <span {...stylex.props(styles.timeCell)}>
            <RelativeInstant
              instant={record.lastRequestAt}
              emptyLabel={t('accessKeys.collection.neverRequested')}
            />
          </span>
        ),
      },
      {
        key: 'actions',
        header: t('accessKeys.columns.actions'),
        width: pixel(170),
        renderCell: (record): ReactNode => (
          <AccessKeyRowActions
            item={source(record.id)}
            status={record.status}
            name={record.name}
            costLimitRuleCount={record.costLimitRuleCount}
            total={total}
            busy={busyIds.has(record.id)}
            locked={lockedIds.has(record.id)}
            onToggle={onToggle}
            onReset={onReset}
            onDeleted={onDeleted}
            onOpen={onOpen}
          />
        ),
      },
  ]

  return (
    <Table
      data={presentations as AccessKeyRow[]}
      columns={columns}
      density="compact"
      dividers="rows"
      hasHover
      aria-label={t('accessKeys.collection.tableLabel')}
      rowIndexStart={(page - 1) * pageSize + 1}
      rowCount={filteredTotal}
    />
  )
}

/** Per-row action cluster — owns the open state for its dialogs (renderCell
 * itself is a plain function, so state lives in this leaf component). */
function AccessKeyRowActions({
  item,
  status,
  name,
  costLimitRuleCount,
  total,
  busy,
  locked,
  onToggle,
  onReset,
  onDeleted,
  onOpen,
}: {
  item: AccessKeyCollectionItemDto
  status: AccessKeyDto['status']
  name: string
  costLimitRuleCount: number
  total: number
  busy: boolean
  locked: boolean
  onToggle(accessKey: AccessKeyCollectionItemDto): void
  onReset(name: string): void
  onDeleted(name: string): void
  onOpen(accessKey: AccessKeyCollectionItemDto, trigger: HTMLElement): void
}) {
  const t = useT()
  const [resetOpen, setResetOpen] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)

  return (
    <span {...stylex.props(styles.actionsCell)}>
      <Button
        variant="secondary"
        size="sm"
        isLoading={busy}
        isDisabled={locked}
        label={status === 'active' ? t('accessKeys.actions.disable') : t('accessKeys.actions.enable')}
        onClick={() => onToggle(item)}
      />
      {costLimitRuleCount > 0 && (
        <>
          <IconButton
            variant="ghost"
            size="sm"
            label={t('accessKeys.reset.open')}
            icon={<RotateCcw size={15} />}
            isDisabled={busy || locked}
            onClick={() => setResetOpen(true)}
          />
          <AccessKeyCostLimitResetDialog
            accessKey={item}
            open={resetOpen}
            onOpenChange={setResetOpen}
            onReset={() => onReset(name)}
          />
        </>
      )}
      <IconButton
        variant="ghost"
        size="sm"
        label={t('accessKeys.delete.open')}
        icon={<Trash2 size={15} />}
        isDisabled={busy || locked}
        onClick={() => setDeleteOpen(true)}
      />
      <AccessKeyDeleteDialog
        accessKey={item}
        total={total}
        open={deleteOpen}
        onOpenChange={setDeleteOpen}
        onDeleted={onDeleted}
      />
      <IconButton
        variant="ghost"
        size="sm"
        label={t('accessKeys.collection.openDetailsFor', { name })}
        icon={<ArrowRight size={15} />}
        isDisabled={busy || locked}
        onClick={(event) => onOpen(item, event.currentTarget)}
      />
    </span>
  )
}
