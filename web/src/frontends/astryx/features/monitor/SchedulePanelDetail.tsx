import * as stylex from '@stylexjs/stylex'
import { AlertDialog, Button, Popover, Selector } from '@astryxdesign/core'
import { LoaderCircle, RefreshCw, TriangleAlert } from 'lucide-react'
import {
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type JSX,
  type ReactNode,
  type Ref,
} from 'react'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import type { ModelProbeTargetDto } from '@shared/control/resources/model-probe'
import {
  isModelRouteScheduleRevisionConflict,
  isReasoningEffort,
  reasoningEffortValues,
  recoverModelRouteScheduleEntry,
  updateModelRouteSchedule,
  type ModelRouteScheduleDetailDto,
  type ModelRouteScheduleEntryDto,
  type ModelRouteScheduleGroupDto,
  type ModelRouteSchedulePatchRequest,
  type ModelRouteSchedulePatchUpdate,
} from '@shared/control/resources/model-route-schedule'
import type { ReasoningEffortDto } from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import { formatLocalInstant } from '@shared/lib/format'
import type { ScheduleDraft, ScheduleDrafts } from '@shared/routing/monitor-route'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useAppServices } from '../../app/services'
import { StickySaveBar } from '../../components/StickySaveBar'
import { ModelProbeScopeDialog } from '../models/ModelProbeScopeDialog'

export interface SchedulePanelDetailLabels {
  title: string
  loading: string
  refresh: string
  stale: string
  observedAt?: string
  routeUnavailable: string
  group: string
  upstreamModel: string
  weight: string
  priority: string
  share: string
  status: string
  available: string
  cooldown: string
  blacklisted: string
  failures: string
  recover: string
  breakerRecovery: string
  breakerThreshold?: string
  breakerCooldown?: string
  cooldownUntil?: string
  scheduledReleaseAt?: string
  invalidValue: string
  derivedReadOnly: string
  save: string
  discard: string
  unsaved: string
  saved: string
  saveFailed: string
  conflict: string
  refreshToResolve: string
  recoverFailed: string
  noEntries: string
  unknownReason?: string
  reasonLabels?: Partial<Record<string, string>>
  draftPreview?: string
  toggleFailed?: string
  calls24h?: string
  successRate24h?: string
  reasoning: string
  inherit: string
}

export interface SchedulePanelDetailProps {
  ref?: Ref<SchedulePanelDetailHandle>
  contextControl?: ReactNode
  detail?: ModelRouteScheduleDetailDto
  loading?: boolean
  error?: string
  stale?: boolean
  labels?: Partial<SchedulePanelDetailLabels>
  locale?: string
  selectedRow?: string
  sourceGroupId?: number
  drafts?: ScheduleDrafts
  onRefresh(): void
  onSaved(snapshotRevision: number, clearDrafts: boolean): void
  onRecovered(groupId: number, entryId: string): void
  onProbe(groupId: number, modelId: string, disabled: boolean): void
  onProbeAll(targets: ModelProbeTargetDto[], disabledGroupIds: number[]): void
  onOpenPrice(groupId: number, modelId: string): void
  onDraftChange(drafts: ScheduleDrafts): void
  onRowChange(row: string | undefined): void
}

export interface SchedulePanelDetailHandle {
  save(): Promise<boolean>
  discard(): void
  readonly invalid: boolean
  readonly pending: boolean
}

type EditableField = 'weight' | 'priority' | 'concurrency_limit'
type Draft = ScheduleDraft

interface ScheduleRow {
  readonly group: ModelRouteScheduleGroupDto
  readonly entry: ModelRouteScheduleEntryDto
}

type DetailKey = readonly [revision: number, externalModel: string | null, protocol: string]

const EMPTY_DETAIL_KEY: DetailKey = [0, '', '']

function scheduleRows(detail: ModelRouteScheduleDetailDto | undefined): ScheduleRow[] {
  return (detail?.groups ?? [])
    .flatMap((group) => group.entries.map((entry) => ({ group, entry })))
    .sort(
      (a, b) =>
        a.entry.priority - b.entry.priority ||
        a.group.group_id - b.group.group_id ||
        (a.entry.entry_id < b.entry.entry_id ? -1 : a.entry.entry_id > b.entry.entry_id ? 1 : 0),
    )
}

function rowKey(groupID: number, entryID: string): string {
  return `${groupID}:${entryID}`
}

function draftKey(groupID: number, entryID: string): string {
  return rowKey(groupID, entryID)
}

function fieldKey(groupID: number, entryID: string, field: EditableField): string {
  return `${draftKey(groupID, entryID)}\u0000${field}`
}

function hasOwn(source: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(source, key)
}

function configuredValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number | null {
  if (field === 'concurrency_limit') return entry.concurrency_limit
  if (field === 'weight') return null
  return entry.priority === 1 ? null : entry.priority
}

function effectiveValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number {
  return entry[field]
}

function isValidValue(field: EditableField, value: number): boolean {
  if (!Number.isSafeInteger(value)) return false
  if (field === 'concurrency_limit') return value >= 0 && value <= 2147483647
  return field === 'weight' ? value >= 0 && value <= 100 : value >= 1
}

function draftFingerprint(source: ScheduleDrafts): string {
  return JSON.stringify(
    Object.keys(source)
      .sort()
      .map((key) => [
        key,
        Object.entries(source[key] ?? {}).sort(([a], [b]) => a.localeCompare(b)),
      ]),
  )
}

function cloneDrafts(source: ScheduleDrafts): ScheduleDrafts {
  return Object.fromEntries(
    Object.entries(source).map(([key, draft]) => [key, draft === undefined ? {} : { ...draft }]),
  )
}

// hydrateDraftState: resets the local edit surface, then re-populates drafts
// for keys that still resolve to rendered rows; raw inputs mirror the draft
// values so a URL-decode round trip keeps the same visible text.
function buildHydration(
  rows: ScheduleRow[],
  source: ScheduleDrafts,
): { draftMap: Record<string, Draft>; rawInputs: Record<string, string> } {
  const rowsByKey = new Map(
    rows.map(({ group, entry }) => [draftKey(group.group_id, entry.entry_id), { group, entry }]),
  )
  const nextDraftMap: Record<string, Draft> = {}
  const nextRawInputs: Record<string, string> = {}
  for (const [key, draft] of Object.entries(source)) {
    const row = rowsByKey.get(key)
    if (row === undefined) continue
    const next = { ...draft }
    nextDraftMap[key] = next
    for (const field of ['weight', 'priority', 'concurrency_limit'] as const) {
      if (!hasOwn(next, field)) continue
      const value = next[field]
      nextRawInputs[fieldKey(row.group.group_id, row.entry.entry_id, field)] =
        value === null || value === undefined ? '' : String(value)
    }
  }
  return { draftMap: nextDraftMap, rawInputs: nextRawInputs }
}

function buildPatchUpdates(
  detail: ModelRouteScheduleDetailDto | undefined,
  draftMap: Record<string, Draft>,
): ModelRouteSchedulePatchUpdate[] {
  const result: ModelRouteSchedulePatchUpdate[] = []
  for (const group of detail?.groups ?? []) {
    for (const entry of group.entries) {
      const key = draftKey(group.group_id, entry.entry_id)
      const draft = draftMap[key]
      if (draft === undefined || entry.entry_id.startsWith('derived:')) {
        continue
      }
      const update: ModelRouteSchedulePatchUpdate = {
        group_id: group.group_id,
        entry_id: entry.entry_id,
      }
      if (draft !== undefined && hasOwn(draft, 'weight')) update.weight = draft.weight
      if (draft !== undefined && hasOwn(draft, 'priority')) update.priority = draft.priority
      if (draft?.concurrency_limit !== undefined && draft.concurrency_limit !== null) {
        update.concurrency_limit = draft.concurrency_limit
      }
      if (hasOwn(draft, 'reasoning_effort')) {
        update.reasoning_effort = draft.reasoning_effort ?? null
      }
      if (Object.keys(update).length > 2) result.push(update)
    }
  }
  return result
}

function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}`
}

function revealScheduleRow(container: HTMLElement | null, key: string): void {
  if (container === null) return
  container
    .querySelector(`[data-row-key="${CSS.escape(key)}"]`)
    ?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
}

function withSetValue<T>(set: ReadonlySet<T>, value: T): ReadonlySet<T> {
  return new Set(set).add(value)
}

function withoutSetValue<T>(set: ReadonlySet<T>, value: T): ReadonlySet<T> {
  if (!set.has(value)) return set
  const next = new Set(set)
  next.delete(value)
  return next
}

function withMapValue<K, V>(map: ReadonlyMap<K, V>, key: K, value: V): ReadonlyMap<K, V> {
  return new Map(map).set(key, value)
}

function withoutMapValue<K, V>(map: ReadonlyMap<K, V>, key: K): ReadonlyMap<K, V> {
  if (!map.has(key)) return map
  const next = new Map(map)
  next.delete(key)
  return next
}

// Classic `await nextTick()` after the mutation: give the invalidation-driven
// parent refresh a macrotask to land before the preserve markers clear.
function waitNextTick(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0))
}

const narrow = '@media (max-width: 767px)'
const touch = '@media (pointer: coarse)'
const reduceMotion = '@media (prefers-reduced-motion: reduce)'

const spin = stylex.keyframes({
  to: { transform: 'rotate(360deg)' },
})

const styles = stylex.create({
  detail: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-3)',
  },
  header: {
    display: 'grid',
    gridTemplateColumns: { default: 'minmax(0, 1fr) auto', [narrow]: '1fr' },
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 8,
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    flexWrap: 'wrap',
    gap: 'var(--space-2) var(--space-3)',
  },
  observed: {
    display: { default: 'flex', '@media (max-width: 1199px)': 'none' },
    flexWrap: 'wrap',
    justifyContent: { default: 'flex-end', [narrow]: 'flex-start' },
    gap: '6px 12px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
    textAlign: { default: 'right', [narrow]: 'left' },
  },
  // QueryFeedback port (loading/error/stale strips).
  feedback: {
    display: 'flex',
    minHeight: 48,
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    padding: 'var(--space-3)',
  },
  feedbackError: {
    borderColor: 'var(--color-danger)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
  },
  feedbackStale: {
    borderColor: 'color-mix(in srgb, var(--color-warning) 32%, var(--color-border-subtle))',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
    paddingTop: 9,
    paddingBottom: 9,
    paddingInline: 11,
    fontSize: 'var(--text-sm)',
  },
  feedbackText: {
    minWidth: 0,
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 'auto',
  },
  feedbackRetry: {
    display: 'inline-flex',
    flex: 'none',
    minHeight: { default: 44, [touch]: 'var(--touch-target)' },
    alignItems: 'center',
    gap: 'var(--space-1)',
    marginLeft: 'auto',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'inherit',
    fontWeight: 650,
    whiteSpace: 'nowrap',
    cursor: 'pointer',
  },
  feedbackRetryStale: {
    minHeight: { default: 'var(--control-compact)', [touch]: 'var(--touch-target)' },
  },
  spin: {
    animationName: { default: spin, [reduceMotion]: 'none' },
    animationDuration: '1s',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  // InlineFeedback tone="danger" port.
  inlineDanger: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-danger)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
    paddingTop: 9,
    paddingBottom: 9,
    paddingInline: 10,
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  inlineGlyph: {
    display: 'grid',
    width: 18,
    height: 18,
    flex: 'none',
    placeItems: 'center',
    fontWeight: 600,
    lineHeight: 1,
  },
  inlineMessage: {
    minWidth: 0,
    flex: '1',
  },
  inlineAction: {
    display: 'inline-flex',
    flex: 'none',
    alignItems: 'center',
    alignSelf: 'flex-start',
  },
  // AppButton variant="link" size="inline" port.
  linkButton: {
    display: 'inline',
    minHeight: { default: 18, [touch]: 'var(--touch-target)' },
    alignItems: 'center',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'inherit',
    padding: 0,
    paddingInline: 4,
    fontWeight: 650,
    whiteSpace: 'nowrap',
    cursor: 'pointer',
    verticalAlign: 'baseline',
    textDecorationLine: { default: 'none', ':hover': 'underline' },
  },
  empty: {
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: 'var(--color-border-control)',
    color: 'var(--color-text-muted)',
    padding: 28,
    textAlign: 'center',
  },
  tableWrap: {
    position: 'relative',
    minWidth: 0,
    maxWidth: '100%',
    maxHeight: { default: '72vh', [narrow]: 'none' },
    overflow: { default: 'auto', [narrow]: 'visible' },
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
  },
  table: { minWidth: 0 },
  row: {
    display: 'grid',
    gridTemplateColumns: {
      // The actions track must fit switch + recovery + More (115px of controls
      // plus two 5px gaps) because it is right-aligned and would otherwise spill
      // leftwards over the share value.
      default: 'minmax(120px, 1.6fr) 116px 100px minmax(206px, 1.2fr) 56px 128px',
      '@media (min-width: 768px) and (max-width: 1199px)':
        'minmax(100px, 1.3fr) 108px 90px minmax(194px, 1fr) 52px 128px',
      [narrow]: 'minmax(0, 1fr) minmax(0, 1fr) 148px',
    },
    gridTemplateAreas: {
      default: '"identity status metrics settings share actions"',
      [narrow]:
        '"identity identity status" "metrics metrics share" "settings settings settings" "actions actions actions"',
    },
    minHeight: 58,
    alignItems: { default: 'center', [narrow]: 'start' },
    columnGap: { default: 10, '@media (max-width: 1199px)': 8 },
    rowGap: { default: 0, [narrow]: 4 },
    paddingBlock: 8,
    paddingInline: { default: 12, '@media (max-width: 1199px)': 10 },
    borderBottomWidth: { default: 1, ':last-child': 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    backgroundColor: { default: 'var(--color-surface)', ':hover': 'var(--color-surface-sunken)' },
  },
  rowHeader: {
    display: { default: 'grid', [narrow]: 'none' },
    position: 'sticky',
    zIndex: 1,
    top: 0,
    minHeight: 42,
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  headerCell: { minWidth: 0, fontWeight: 600 },
  rowSelected: { backgroundColor: 'var(--color-action-soft)' },
  rowModified: {
    backgroundColor: 'color-mix(in srgb, var(--color-action-soft) 35%, var(--color-surface))',
  },
  cell: { minWidth: 0, color: 'var(--color-text)', fontSize: 'var(--text-body)' },
  identity: { gridArea: 'identity' },
  statusCell: { gridArea: 'status', textAlign: { default: 'left', [narrow]: 'right' } },
  eligibility: {
    display: 'block',
    fontWeight: 600,
    fontSize: 'var(--text-meta)',
    lineHeight: 1.3,
    overflowWrap: 'anywhere',
  },
  cellStrong: {
    display: 'block',
    overflowWrap: 'anywhere',
    color: 'var(--color-text)',
    fontWeight: 650,
    lineHeight: 1.45,
  },
  cellSmall: {
    display: 'block',
    marginTop: 2,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
  },
  groupLink: {
    display: 'block',
    overflowWrap: 'anywhere',
    marginTop: 2,
    color: { default: 'var(--color-text-muted)', ':hover': 'var(--color-action)' },
    fontSize: 'var(--text-meta)',
    textDecoration: { default: 'none', ':hover': 'underline' },
  },
  changed: { color: 'var(--color-action)', fontSize: 'var(--text-meta)' },
  metrics: {
    gridArea: 'metrics',
    display: { default: 'grid', [narrow]: 'flex' },
    gap: { default: 1, [narrow]: 10 },
    fontVariantNumeric: 'tabular-nums',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  metricValue: { fontWeight: 500, whiteSpace: 'nowrap' },
  metricLabel: {
    display: { default: 'none', [narrow]: 'inline' },
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    marginInlineEnd: 4,
  },
  metricRate: { color: 'var(--color-text-muted)' },
  settings: {
    gridArea: 'settings',
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: '44px 48px minmax(82px, 1fr)',
      [narrow]: 'minmax(56px, .8fr) minmax(56px, .8fr) minmax(100px, 1.5fr)',
    },
    gap: 6,
  },
  settingsHead: {
    display: 'grid',
    gridTemplateColumns: '44px 48px minmax(82px, 1fr)',
    gap: 6,
    minWidth: 0,
  },
  controlGroup: {
    display: 'grid',
    minWidth: 0,
    // A bare `grid` gives the control an `auto` track, which sizes to the
    // Selector's max-content width and refuses to shrink — the reasoning
    // trigger then spills over the share column at 768–1199px. `minmax(0, 1fr)`
    // lets the control use the column width it is given.
    gridTemplateColumns: 'minmax(0, 1fr)',
    alignContent: 'start',
    gap: 4,
  },
  controlLabel: {
    display: { default: 'none', [narrow]: 'block' },
    fontSize: 'var(--text-meta)',
    color: 'var(--color-text-muted)',
  },
  cellShare: {
    gridArea: 'share',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 600,
    textAlign: { default: 'left', [narrow]: 'right' },
  },
  shareLabel: {
    display: { default: 'none', [narrow]: 'inline' },
    color: 'var(--color-text-muted)',
    fontWeight: 400,
    fontSize: 'var(--text-meta)',
  },
  actions: {
    gridArea: 'actions',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'flex-end',
    gap: { default: 5, [narrow]: 8 },
  },
  rowAction: {
    minWidth: 0,
    paddingInline: { default: 5, [narrow]: 10 },
    minHeight: { default: 30, [narrow]: 44, [touch]: 44 },
    fontSize: 'var(--text-meta)',
    flexShrink: 0,
  },
  selector: { minWidth: 0, width: '100%', minHeight: { default: 32, [narrow]: 44, [touch]: 44 } },
  menu: { display: 'grid', gap: 12, minWidth: 0, padding: 4 },
  menuActions: { display: 'flex', gap: 8, flexWrap: 'wrap' },
  menuField: { display: 'grid', gap: 5, fontSize: 'var(--text-meta)' },
  menuHelp: { color: 'var(--color-text-muted)', fontSize: 'var(--text-meta)' },
  breaker: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 8,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  toggle: {
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: { default: 32, [narrow]: 44, [touch]: 44 },
    minHeight: { default: 32, [narrow]: 44, [touch]: 44 },
    padding: 0,
    borderWidth: 0,
    backgroundColor: 'transparent',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
    flexShrink: 0,
  },
  toggleTrack: {
    display: 'flex',
    width: 30,
    height: 18,
    padding: 2,
    borderRadius: 12,
    backgroundColor: 'var(--color-border-control)',
  },
  toggleTrackOn: { backgroundColor: 'var(--color-action)' },
  toggleThumb: {
    width: 14,
    height: 14,
    borderRadius: '50%',
    backgroundColor: 'var(--color-surface)',
    transitionProperty: 'transform',
    transitionDuration: { default: '140ms', [reduceMotion]: '0s' },
  },
  toggleThumbOn: { transform: 'translateX(12px)' },
  numberField: { display: 'grid', minWidth: 0, gap: 4 },
  inputShell: {
    display: 'flex',
    width: '100%',
    minWidth: 0,
    boxSizing: 'border-box',
    alignItems: 'center',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: { default: 'var(--color-border-control)', ':focus-within': 'var(--color-focus)' },
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    paddingInline: 6,
    minHeight: { default: 32, [narrow]: 44, [touch]: 44 },
    fontSize: 'var(--text-meta)',
  },
  inputShellInvalid: { borderColor: 'var(--color-danger)' },
  inputError: {
    color: 'var(--color-danger)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
  },
  inputShellDisabled: { cursor: 'not-allowed', opacity: 0.55 },
  inputInner: {
    width: '100%',
    minWidth: 0,
    minHeight: 'inherit',
    borderWidth: 0,
    outlineStyle: 'none',
    backgroundColor: 'transparent',
    color: { default: 'var(--color-text)', '::placeholder': 'var(--color-text-faint)' },
    padding: 0,
    fontFamily: 'inherit',
    fontSize: { default: 'inherit', [narrow]: '14px' },
    fontWeight: 'inherit',
    lineHeight: 'inherit',
    appearance: 'textfield',
    fontVariantNumeric: 'tabular-nums',
    cursor: { ':disabled': 'not-allowed' },
  },
  srOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clipPath: 'inset(50%)',
    whiteSpace: 'nowrap',
  },
})

function ScheduleFeedback({
  state,
  message,
  retryLabel,
  onRetry,
}: {
  state: 'loading' | 'error' | 'stale'
  message: string
  retryLabel?: string
  onRetry?(): void
}): JSX.Element {
  return (
    <div
      {...stylex.props(
        styles.feedback,
        state === 'error' && styles.feedbackError,
        state === 'stale' && styles.feedbackStale,
      )}
      role={state === 'error' ? 'alert' : 'status'}
    >
      {state === 'loading' ? (
        <LoaderCircle size={18} aria-hidden {...stylex.props(styles.spin)} />
      ) : (
        <TriangleAlert size={state === 'stale' ? 14 : 18} aria-hidden />
      )}
      <span {...stylex.props(styles.feedbackText)}>{message}</span>
      {state !== 'loading' && retryLabel !== undefined && (
        <button
          type="button"
          {...stylex.props(styles.feedbackRetry, state === 'stale' && styles.feedbackRetryStale)}
          onClick={onRetry}
        >
          <RefreshCw size={15} aria-hidden />
          {retryLabel}
        </button>
      )}
    </div>
  )
}

function ScheduleNumberInput({
  id,
  label,
  value,
  placeholder,
  invalid,
  invalidMessage,
  min,
  max,
  disabled,
  onChange,
}: {
  id: string
  label: string
  value: string
  placeholder?: string
  invalid?: boolean
  invalidMessage?: string
  min?: number
  max?: number
  disabled?: boolean
  onChange(value: string): void
}): JSX.Element {
  return (
    <div {...stylex.props(styles.numberField)}>
      <div
        data-input-shell
        {...stylex.props(
          styles.inputShell,
          invalid === true && styles.inputShellInvalid,
          disabled === true && styles.inputShellDisabled,
        )}
      >
        <label {...stylex.props(styles.srOnly)} htmlFor={id}>
          {label}
        </label>
        <input
          id={id}
          data-input-inner
          {...stylex.props(styles.inputInner)}
          type="number"
          value={value}
          placeholder={placeholder}
          min={min}
          max={max}
          disabled={disabled}
          aria-invalid={invalid === true || undefined}
          aria-describedby={invalid === true ? `${id}-error` : undefined}
          autoComplete="off"
          spellCheck={false}
          inputMode="numeric"
          onChange={(event) => onChange(event.currentTarget.value)}
        />
      </div>
      {invalid === true && (
        <span id={`${id}-error`} {...stylex.props(styles.inputError)} role="alert">
          {invalidMessage} (
          {min !== undefined ? `${min}${max !== undefined ? `–${max}` : '+'}` : ''})
        </span>
      )}
    </div>
  )
}

export function SchedulePanelDetail({
  ref,
  contextControl,
  detail,
  loading = false,
  error = '',
  stale = false,
  labels = {},
  locale = 'en-US',
  selectedRow,
  sourceGroupId,
  drafts = {},
  onRefresh,
  onSaved,
  onRecovered,
  onProbe,
  onProbeAll,
  onOpenPrice,
  onDraftChange,
  onRowChange,
}: SchedulePanelDetailProps): JSX.Element {
  const t = useT()
  const { apiClient, queryClient, toast } = useAppServices()

  function text(key: keyof SchedulePanelDetailLabels): string {
    const value = labels[key]
    return typeof value === 'string' && value !== ''
      ? value
      : t(`monitor.schedule.detail.${key}` as MessageId)
  }

  const rows = useMemo(() => scheduleRows(detail), [detail])
  const countFormatter = useMemo(() => new Intl.NumberFormat(locale), [locale])
  const rateFormatter = useMemo(
    () =>
      new Intl.NumberFormat(locale, {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1,
      }),
    [locale],
  )

  const [draftMap, setDraftMap] = useState<Record<string, Draft>>(
    () => buildHydration(scheduleRows(detail), drafts).draftMap,
  )

  const [rawInputs, setRawInputs] = useState<Record<string, string>>(
    () => buildHydration(scheduleRows(detail), drafts).rawInputs,
  )
  const [invalidInputs, setInvalidInputs] = useState<Record<string, boolean>>({})
  const [pending, setPending] = useState(false)
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saved' | 'error'>('idle')
  const [saveError, setSaveError] = useState('')
  const [recovering, setRecovering] = useState('')
  const [togglingEntries, setTogglingEntries] = useState<ReadonlySet<string>>(new Set())
  const [optimisticEnabled, setOptimisticEnabled] = useState<ReadonlyMap<string, boolean>>(
    new Map(),
  )
  const [preserveDraftRevisions, setPreserveDraftRevisions] = useState<ReadonlySet<number>>(
    new Set(),
  )
  const [preserveDraftSnapshots, setPreserveDraftSnapshots] = useState<
    ReadonlyMap<number, ScheduleDrafts>
  >(new Map())
  // Ignore the one URL echo caused by a local edit; later history changes
  // hydrate normally.
  const [pendingLocalDraftFingerprint, setPendingLocalDraftFingerprint] = useState<
    string | undefined
  >(undefined)
  const [probeScopeOpen, setProbeScopeOpen] = useState(false)
  const [menuRow, setMenuRow] = useState<string | null>(null)
  const [pendingProbe, setPendingProbe] = useState<{
    groupId: number
    modelId: string
  } | null>(null)

  const scheduleTableRef = useRef<HTMLDivElement | null>(null)
  const callbacksRef = useRef({
    onDraftChange,
    onRowChange,
  })
  useEffect(() => {
    callbacksRef.current = { onDraftChange, onRowChange }
  })

  const detailKey: DetailKey =
    detail === undefined
      ? EMPTY_DETAIL_KEY
      : [detail.snapshot_revision, detail.external_model, detail.protocol]
  const draftsFingerprint = draftFingerprint(drafts)
  const [synced, setSynced] = useState<{ key: DetailKey; drafts: string }>(() => ({
    key: detailKey,
    drafts: draftsFingerprint,
  }))

  // Classic watch(immediate) on [snapshot_revision, external_model, protocol]
  // plus the deep watch on `drafts`, replayed as a render-time adjustment so
  // controlled inputs never paint stale values. Watch A runs first, then B.
  const keyChanged =
    synced.key[0] !== detailKey[0] ||
    synced.key[1] !== detailKey[1] ||
    synced.key[2] !== detailKey[2]
  const draftsChanged = synced.drafts !== draftsFingerprint
  if (keyChanged || draftsChanged) {
    let nextRevisions = preserveDraftRevisions
    let nextSnapshots = preserveDraftSnapshots
    let nextFingerprint: string | undefined = pendingLocalDraftFingerprint
    let hydrationSource: ScheduleDrafts | null = null
    if (keyChanged) {
      const previousRevision = synced.key[0]
      const sameContext = synced.key[1] === detailKey[1] && synced.key[2] === detailKey[2]
      if (sameContext && previousRevision !== 0 && previousRevision !== detailKey[0]) {
        if (nextRevisions.has(previousRevision)) {
          nextRevisions = withoutSetValue(nextRevisions, previousRevision)
          const snapshot = nextSnapshots.get(previousRevision) ?? drafts
          nextSnapshots = withoutMapValue(nextSnapshots, previousRevision)
          hydrationSource = snapshot
        } else {
          // A newer observation is not an instruction to discard local edits.
          hydrationSource = draftMap
        }
      } else {
        hydrationSource = drafts
      }
      nextFingerprint = undefined
    }
    // Watch B runs after A: the consume branch clears the pending fingerprint
    // and the hydrate branch clears it too, so either way it ends cleared.
    if (draftsChanged) {
      if (nextFingerprint !== draftsFingerprint) hydrationSource = drafts
      nextFingerprint = undefined
    }
    if (hydrationSource !== null) {
      const hydrated = buildHydration(rows, hydrationSource)
      setDraftMap(hydrated.draftMap)
      const sameContext = synced.key[1] === detailKey[1] && synced.key[2] === detailKey[2]
      if (keyChanged && sameContext) {
        const existingFields = new Set<string>()
        const editedFields = new Set<string>()
        for (const { group, entry } of rows) {
          for (const field of ['weight', 'priority', 'concurrency_limit'] as const) {
            const key = fieldKey(group.group_id, entry.entry_id, field)
            existingFields.add(key)
            if (
              hasOwn(hydrated.draftMap[draftKey(group.group_id, entry.entry_id)] ?? {}, field) ||
              invalidInputs[key]
            )
              editedFields.add(key)
          }
        }
        setRawInputs({
          ...hydrated.rawInputs,
          ...Object.fromEntries(Object.entries(rawInputs).filter(([key]) => editedFields.has(key))),
        })
        setInvalidInputs(
          Object.fromEntries(
            Object.entries(invalidInputs).filter(([key]) => existingFields.has(key)),
          ),
        )
      } else {
        setRawInputs(hydrated.rawInputs)
        setInvalidInputs({})
      }
      setSaveStatus('idle')
      setSaveError('')
    }
    if (nextRevisions !== preserveDraftRevisions) {
      setPreserveDraftRevisions(nextRevisions)
    }
    if (nextSnapshots !== preserveDraftSnapshots) {
      setPreserveDraftSnapshots(nextSnapshots)
    }
    if (nextFingerprint !== pendingLocalDraftFingerprint) {
      setPendingLocalDraftFingerprint(nextFingerprint)
    }
    setSynced({ key: detailKey, drafts: draftsFingerprint })
  }

  // Row-resolution watch on [detail, sourceGroupId, selectedRow] — the classic
  // watch has no immediate flag, so mounting with a loaded detail does not
  // resolve a row.
  const rowWatchMountedRef = useRef(false)
  useEffect(() => {
    if (!rowWatchMountedRef.current) {
      rowWatchMountedRef.current = true
      return
    }
    if (detail === undefined || sourceGroupId === undefined) return
    const rowsByKey = new Map(
      rows.map((row) => [rowKey(row.group.group_id, row.entry.entry_id), row] as const),
    )
    if (selectedRow !== undefined) {
      if (rowsByKey.has(selectedRow)) {
        revealScheduleRow(scheduleTableRef.current, selectedRow)
        return
      }
      callbacksRef.current.onRowChange(undefined)
      return
    }
    const externalModel = detail.external_model ?? ''
    const target = rows.find(
      ({ group, entry }) =>
        group.group_id === sourceGroupId &&
        (entry.alias === externalModel || entry.model_id === externalModel),
    )
    if (target !== undefined) {
      callbacksRef.current.onRowChange(rowKey(target.group.group_id, target.entry.entry_id))
    }
  }, [detail, sourceGroupId, selectedRow, rows])

  const invalid =
    Object.values(invalidInputs).some(Boolean) ||
    Object.values(draftMap).some(
      (draft) => hasOwn(draft, 'concurrency_limit') && draft.concurrency_limit === null,
    )
  const dirty = Object.keys(draftMap).length > 0 || invalid
  const hasScheduleDraft = Object.values(draftMap).some(
    (draft) => hasOwn(draft, 'weight') || hasOwn(draft, 'priority'),
  )

  // Batch scope is exactly what is on screen: the same rows the table renders,
  // deduplicated to (group, model) targets. Disabled groups are split out so
  // the operator decides whether to spend an upstream call on a group that is
  // not serving traffic.
  const probeScopes = useMemo(() => {
    const all: ModelProbeTargetDto[] = []
    const enabled: ModelProbeTargetDto[] = []
    const disabledGroupIds = new Set<number>()
    const seen = new Set<string>()
    for (const { group, entry } of rows) {
      const key = `${group.group_id}:${entry.model_id}`
      if (seen.has(key)) continue
      seen.add(key)
      const target = { group_id: group.group_id, model: entry.model_id }
      all.push(target)
      if (group.enabled) enabled.push(target)
      else disabledGroupIds.add(group.group_id)
    }
    return { all, enabled, disabledGroupIds: [...disabledGroupIds] }
  }, [rows])

  const previewShares = useMemo(() => {
    const result = new Map<string, number>()
    if (detail === undefined || !hasScheduleDraft) return result
    const candidates = detail.groups.flatMap((group) =>
      group.entries.map((entry) => {
        const draft = draftMap[draftKey(group.group_id, entry.entry_id)]
        return {
          group,
          entry,
          weight: draft?.weight ?? entry.weight,
          priority: draft?.priority ?? entry.priority,
        }
      }),
    )
    const totals = new Map<number, number>()
    for (const candidate of candidates) {
      const key = rowKey(candidate.group.group_id, candidate.entry.entry_id)
      const enabled = optimisticEnabled.get(key) ?? candidate.entry.enabled
      if (enabled && candidate.group.enabled) {
        totals.set(candidate.priority, (totals.get(candidate.priority) ?? 0) + candidate.weight)
      }
    }
    for (const candidate of candidates) {
      const key = rowKey(candidate.group.group_id, candidate.entry.entry_id)
      const enabled = optimisticEnabled.get(key) ?? candidate.entry.enabled
      const total = totals.get(candidate.priority) ?? 0
      result.set(
        key,
        enabled && candidate.group.enabled && total > 0 ? Math.max(0, candidate.weight) / total : 0,
      )
    }
    return result
  }, [detail, draftMap, optimisticEnabled, hasScheduleDraft])

  const observed = detail?.observed_at_ms
  const observedLabel =
    observed === undefined ? '' : `${text('observedAt')} ${formatLocalInstant(observed, locale)}`

  function entryEnabled(groupID: number, entry: ModelRouteScheduleEntryDto): boolean {
    return optimisticEnabled.get(rowKey(groupID, entry.entry_id)) ?? entry.enabled
  }

  async function toggleEntryEnabled(
    groupID: number,
    entry: ModelRouteScheduleEntryDto,
    next: boolean,
  ): Promise<void> {
    const current = detail
    const key = rowKey(groupID, entry.entry_id)
    if (
      current === undefined ||
      togglingEntries.has(key) ||
      entry.entry_id.startsWith('derived:')
    ) {
      return
    }
    const revision = current.snapshot_revision
    setTogglingEntries((previous) => withSetValue(previous, key))
    setOptimisticEnabled((previous) => withMapValue(previous, key, next))
    setPreserveDraftRevisions((previous) => withSetValue(previous, revision))
    setPreserveDraftSnapshots((previous) => withMapValue(previous, revision, cloneDrafts(draftMap)))
    try {
      const response = await updateModelRouteSchedule(apiClient, {
        snapshot_revision: revision,
        protocol: current.protocol,
        external_model: current.external_model ?? '',
        operation: current.operation,
        updates: [{ group_id: groupID, entry_id: entry.entry_id, enabled: next }],
      })
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.update)
      onSaved(response.snapshot_revision_new, false)
    } catch (toggleFailure: unknown) {
      toast.show({
        message: isModelRouteScheduleRevisionConflict(toggleFailure)
          ? text('conflict')
          : text('toggleFailed'),
        tone: 'danger',
      })
    } finally {
      await waitNextTick()
      setPreserveDraftRevisions((previous) => withoutSetValue(previous, revision))
      setPreserveDraftSnapshots((previous) => withoutMapValue(previous, revision))
      setOptimisticEnabled((previous) => withoutMapValue(previous, key))
      setTogglingEntries((previous) => withoutSetValue(previous, key))
    }
  }

  function formatCount(value: number): string {
    return countFormatter.format(value)
  }

  function formatRate(value: number): string {
    return rateFormatter.format(value)
  }

  function entryReasoningValue(
    groupID: number,
    entry: ModelRouteScheduleEntryDto,
  ): ReasoningEffortDto | '' {
    const key = draftKey(groupID, entry.entry_id)
    return hasOwn(draftMap[key] ?? {}, 'reasoning_effort')
      ? (draftMap[key]?.reasoning_effort ?? '')
      : (entry.reasoning.configured ?? '')
  }

  // Unconfigured entries fold the effective value into the option label: the
  // cell has a single line, so the operator can see where the effective
  // reasoning effort comes from without opening the menu.
  function reasoningOptionsFor(entry: ModelRouteScheduleEntryDto): {
    value: string
    label: string
  }[] {
    const effective = entry.reasoning.effective
    const inheritLabel = effective === null ? text('inherit') : `${text('inherit')} · ${effective}`
    return [
      { value: '', label: inheritLabel },
      ...reasoningEffortValues.map((value) => ({ value, label: value })),
    ]
  }

  function setEntryReasoning(
    groupID: number,
    entry: ModelRouteScheduleEntryDto,
    value: string,
  ): void {
    if (value !== '' && !isReasoningEffort(value)) return
    const next: ReasoningEffortDto | null = value === '' ? null : value
    const key = draftKey(groupID, entry.entry_id)
    const nextDraftMap = { ...draftMap }
    const draft = { ...nextDraftMap[key] }
    if (next === entry.reasoning.configured) delete draft.reasoning_effort
    else draft.reasoning_effort = next
    if (Object.keys(draft).length === 0) delete nextDraftMap[key]
    else nextDraftMap[key] = draft
    setDraftMap(nextDraftMap)
    setSaveStatus('idle')
    emitDraftChange(nextDraftMap)
  }

  function inputValue(
    groupID: number,
    entry: ModelRouteScheduleEntryDto,
    field: EditableField,
  ): string {
    const key = fieldKey(groupID, entry.entry_id, field)
    const raw = rawInputs[key]
    if (raw !== undefined) return raw
    const value = draftMap[draftKey(groupID, entry.entry_id)]?.[field]
    return value === null || value === undefined
      ? String(effectiveValue(entry, field))
      : String(value)
  }

  function emitDraftChange(source: ScheduleDrafts): void {
    if (preserveDraftRevisions.size > 0) {
      setPreserveDraftSnapshots((previous) => {
        let next = previous
        for (const revision of preserveDraftRevisions) {
          next = new Map(next).set(revision, cloneDrafts(source))
        }
        return next
      })
    }
    setPendingLocalDraftFingerprint(draftFingerprint(source))
    onDraftChange(source)
  }

  function setDraftValue(
    groupID: number,
    entry: ModelRouteScheduleEntryDto,
    field: EditableField,
    value: number | null,
  ): void {
    const key = draftKey(groupID, entry.entry_id)
    const baseline = configuredValue(entry, field)
    const effective = effectiveValue(entry, field)
    const sameAsServer = value !== null && value === (baseline ?? effective)
    const nextDraftMap = { ...draftMap }
    const draft: Draft = { ...nextDraftMap[key] }
    if (value === null) draft[field] = null
    else if (sameAsServer) delete draft[field]
    else draft[field] = value
    if (Object.keys(draft).length === 0) delete nextDraftMap[key]
    else nextDraftMap[key] = draft
    setDraftMap(nextDraftMap)
    emitDraftChange({ ...nextDraftMap })
    onRowChange(key)
  }

  function setInput(
    groupID: number,
    entry: ModelRouteScheduleEntryDto,
    field: EditableField,
    value: string,
  ): void {
    setSaveStatus('idle')
    const key = fieldKey(groupID, entry.entry_id, field)
    setRawInputs((previous) => ({ ...previous, [key]: value }))
    setInvalidInputs((previous) => {
      if (!hasOwn(previous, key)) return previous
      const next = { ...previous }
      delete next[key]
      return next
    })
    if (value.trim() === '') {
      setDraftValue(groupID, entry, field, null)
      return
    }
    const parsed = Number(value)
    if (!/^\d+$/.test(value.trim()) || !isValidValue(field, parsed)) {
      setInvalidInputs((previous) => ({ ...previous, [key]: true }))
      if (field === 'concurrency_limit') setDraftValue(groupID, entry, field, null)
      return
    }
    setDraftValue(groupID, entry, field, parsed)
  }

  function resetDrafts(): void {
    setDraftMap({})

    setRawInputs({})
    setInvalidInputs({})
    setSaveStatus('idle')
    setSaveError('')
  }

  function discard(): void {
    resetDrafts()
    emitDraftChange({})
  }

  async function save(): Promise<boolean> {
    if (detail === undefined || pending || invalid) return false
    if (!dirty) return true
    const body: ModelRouteSchedulePatchRequest = {
      snapshot_revision: detail.snapshot_revision,
      protocol: detail.protocol,
      external_model: detail.external_model ?? '',
      operation: detail.operation,
      updates: buildPatchUpdates(detail, draftMap),
    }
    if (body.updates.length === 0) return false
    setPending(true)
    setSaveStatus('idle')
    setSaveError('')
    try {
      const response = await updateModelRouteSchedule(apiClient, body)
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.update)
      resetDrafts()
      emitDraftChange({})
      setSaveStatus('saved')
      onSaved(response.snapshot_revision_new, true)
      return true
    } catch (saveFailure: unknown) {
      setSaveStatus('error')
      setSaveError(
        isModelRouteScheduleRevisionConflict(saveFailure) ? text('conflict') : text('saveFailed'),
      )
      return false
    } finally {
      setPending(false)
    }
  }

  async function recover(groupID: number, entry: ModelRouteScheduleEntryDto): Promise<void> {
    const key = rowKey(groupID, entry.entry_id)
    setRecovering(key)
    setSaveError('')
    try {
      await recoverModelRouteScheduleEntry(apiClient, {
        group_id: groupID,
        entry_id: entry.entry_id,
        failure_version: entry.runtime.failure_version,
      })
      await applyInvalidationPlan(queryClient, mutationInvalidationPlans.modelRouteSchedule.recover)
      onRecovered(groupID, entry.entry_id)
    } catch {
      setSaveError(text('recoverFailed'))
    } finally {
      setRecovering('')
    }
  }

  function shareValue(groupID: number, entry: ModelRouteScheduleEntryDto): number {
    if (!entryEnabled(groupID, entry)) return 0
    return previewShares.get(rowKey(groupID, entry.entry_id)) ?? entry.configured_share
  }

  function breakerRecoveryLabel(entry: ModelRouteScheduleEntryDto): string {
    const breaker = entry.circuit_breaker.effective
    const parts: string[] = []
    if (breaker.blacklist_threshold !== null) {
      parts.push(`${text('breakerThreshold')} ${breaker.blacklist_threshold}`)
    }
    if (breaker.cooldown_seconds !== null) {
      parts.push(`${text('breakerCooldown')} ${breaker.cooldown_seconds}s`)
    }
    if (entry.runtime.cooldown_until_ms !== null) {
      parts.push(
        `${text('cooldownUntil')}: ${formatLocalInstant(entry.runtime.cooldown_until_ms, locale)}`,
      )
    }
    if (entry.runtime.blacklist_release_at_ms !== null) {
      parts.push(
        `${text('scheduledReleaseAt')}: ${formatLocalInstant(entry.runtime.blacklist_release_at_ms, locale)}`,
      )
    }
    return parts.join(' · ')
  }

  // A disabled row still probes on explicit request, so it asks first.
  function requestProbe(groupId: number, modelId: string, disabled: boolean): void {
    if (!disabled) {
      onProbe(groupId, modelId, false)
      return
    }
    setPendingProbe({ groupId, modelId })
  }

  function confirmSingleProbe(): void {
    const target = pendingProbe
    setPendingProbe(null)
    if (target !== null) onProbe(target.groupId, target.modelId, true)
  }

  function handleSingleProbeOpen(value: boolean): void {
    if (!value) setPendingProbe(null)
  }

  function probeVisibleRows(): void {
    const { all, enabled } = probeScopes
    if (all.length === 0) return
    if (enabled.length === all.length) {
      onProbeAll(all, [])
      return
    }
    setProbeScopeOpen(true)
  }

  function confirmProbeAll(): void {
    const { all, disabledGroupIds } = probeScopes
    setProbeScopeOpen(false)
    onProbeAll(all, disabledGroupIds)
  }

  function confirmProbeEnabled(): void {
    const { enabled } = probeScopes
    setProbeScopeOpen(false)
    onProbeAll(enabled, [])
  }

  useImperativeHandle(ref, () => ({ save, discard, invalid, pending }))

  return (
    <section {...stylex.props(styles.detail)} aria-labelledby="schedule-detail-title">
      <h2 id="schedule-detail-title" {...stylex.props(styles.srOnly)}>
        {text('title')}
      </h2>
      <header {...stylex.props(styles.header)}>
        {contextControl}
        <div {...stylex.props(styles.headerActions)}>
          {observedLabel !== '' && <span {...stylex.props(styles.observed)}>{observedLabel}</span>}
          {error === '' && (
            <Button
              label={text('refresh')}
              variant="secondary"
              size="sm"
              xstyle={styles.rowAction}
              isDisabled={pending || loading}
              onClick={onRefresh}
            />
          )}
          {rows.length > 0 && (
            <Button
              variant="secondary"
              size="sm"
              xstyle={styles.rowAction}
              isDisabled={pending}
              tooltip={t('monitor.modelProbe.description')}
              label={t('monitor.modelProbe.batch', { count: probeScopes.all.length })}
              onClick={probeVisibleRows}
            />
          )}
        </div>
      </header>

      {loading ? (
        <ScheduleFeedback state="loading" message={text('loading')} />
      ) : error !== '' ? (
        <ScheduleFeedback
          state="error"
          message={error}
          retryLabel={text('refresh')}
          onRetry={onRefresh}
        />
      ) : detail !== undefined ? (
        <>
          {stale && (
            <ScheduleFeedback
              state="stale"
              message={text('stale')}
              retryLabel={text('refresh')}
              onRetry={onRefresh}
            />
          )}
          {(detail.external_model === null || !detail.routable) && (
            <ScheduleFeedback
              state="stale"
              message={text('routeUnavailable')}
              retryLabel={text('refresh')}
              onRetry={onRefresh}
            />
          )}
          {saveError !== '' && (
            <div
              {...stylex.props(styles.inlineDanger)}
              role="alert"
              aria-live="assertive"
              aria-atomic="true"
            >
              <span {...stylex.props(styles.inlineGlyph)} aria-hidden="true">
                <TriangleAlert size={16} aria-hidden />
              </span>
              <span {...stylex.props(styles.inlineMessage)}>{saveError}</span>
              <span {...stylex.props(styles.inlineAction)}>
                <button type="button" {...stylex.props(styles.linkButton)} onClick={onRefresh}>
                  {text('refreshToResolve')}
                </button>
              </span>
            </div>
          )}

          {rows.length === 0 ? (
            <div {...stylex.props(styles.empty)} role="status">
              {text('noEntries')}
            </div>
          ) : (
            <div ref={scheduleTableRef} {...stylex.props(styles.tableWrap)}>
              {hasScheduleDraft && (
                <span {...stylex.props(styles.cellSmall)}>{text('draftPreview')}</span>
              )}
              <div role="table" aria-label={text('title')} {...stylex.props(styles.table)}>
                <div role="row" {...stylex.props(styles.row, styles.rowHeader)}>
                  <div role="columnheader" {...stylex.props(styles.headerCell, styles.identity)}>
                    {text('upstreamModel')} / {text('group')}
                  </div>
                  <div role="columnheader" {...stylex.props(styles.headerCell, styles.statusCell)}>
                    {text('status')}
                  </div>
                  <div role="columnheader" {...stylex.props(styles.headerCell, styles.metrics)}>
                    <span>{text('calls24h')}</span>
                    <span>{text('successRate24h')}</span>
                  </div>
                  <div
                    role="columnheader"
                    {...stylex.props(styles.headerCell, styles.settingsHead)}
                  >
                    <span>{text('priority')}</span>
                    <span>{text('weight')}</span>
                    <span>{text('reasoning')}</span>
                  </div>
                  <div role="columnheader" {...stylex.props(styles.headerCell, styles.cellShare)}>
                    {text('share')}
                  </div>
                  <div role="columnheader" {...stylex.props(styles.headerCell, styles.actions)}>
                    {t('monitor.schedule.detail.actions')}
                  </div>
                </div>
                {rows.map(({ group, entry }, index) => {
                  const key = rowKey(group.group_id, entry.entry_id)
                  const derived = entry.entry_id.startsWith('derived:')
                  const enabled = entryEnabled(group.group_id, entry)
                  const needsRecovery = entry.runtime.state !== 'available'
                  const rowChanged = Object.keys(draftMap[key] ?? {}).length > 0
                  const rowLabel = `${entry.model_id} · ${group.group_name}`
                  const reason = !group.enabled
                    ? 'group_disabled'
                    : !enabled
                      ? 'entry_disabled'
                      : entry.reason_code
                  const eligibility =
                    reason !== null
                      ? (labels.reasonLabels?.[reason] ?? text('unknownReason'))
                      : entry.routable
                        ? t('monitor.schedule.detail.schedulable')
                        : text('unknownReason')
                  const toggleLabel = `${t(enabled ? 'monitor.schedule.detail.disableEntry' : 'monitor.schedule.detail.enableEntry')} ${rowLabel}`
                  return (
                    <div
                      key={key}
                      data-row-key={key}
                      role="row"
                      tabIndex={0}
                      aria-selected={selectedRow === key}
                      onClick={() => onRowChange(key)}
                      onKeyDown={(event) => {
                        if (
                          event.target === event.currentTarget &&
                          (event.key === 'Enter' || event.key === ' ')
                        ) {
                          event.preventDefault()
                          onRowChange(key)
                        }
                      }}
                      {...stylex.props(
                        styles.row,
                        rowChanged && styles.rowModified,
                        selectedRow === key && styles.rowSelected,
                      )}
                    >
                      <div role="cell" {...stylex.props(styles.cell, styles.identity)}>
                        <strong {...stylex.props(styles.cellStrong)}>{entry.model_id}</strong>
                        <RouteLink
                          to={groupDetailHref(group.group_id)}
                          {...stylex.props(styles.groupLink)}
                          onClick={(event) => event.stopPropagation()}
                        >
                          {group.group_name}
                        </RouteLink>
                        {entry.alias !== '' &&
                          entry.alias !== detail.external_model &&
                          entry.alias !== entry.model_id && (
                            <span {...stylex.props(styles.cellSmall)}>{entry.alias}</span>
                          )}
                        {derived && (
                          <span {...stylex.props(styles.cellSmall)}>{text('derivedReadOnly')}</span>
                        )}
                      </div>
                      <div role="cell" {...stylex.props(styles.cell, styles.statusCell)}>
                        <span {...stylex.props(styles.eligibility)}>{eligibility}</span>
                        <span {...stylex.props(styles.cellSmall)}>
                          {t('monitor.schedule.detail.entryHealth')}:{' '}
                          {entry.runtime.state === 'available'
                            ? text('available')
                            : entry.runtime.state === 'cooldown'
                              ? text('cooldown')
                              : text('blacklisted')}
                        </span>
                        {entry.runtime.failure_count > 0 && (
                          <span {...stylex.props(styles.cellSmall)}>
                            {entry.runtime.failure_count} {text('failures')}
                          </span>
                        )}
                      </div>
                      <div role="cell" {...stylex.props(styles.cell, styles.metrics)}>
                        <span {...stylex.props(styles.metricValue)}>
                          <span {...stylex.props(styles.metricLabel)}>{text('calls24h')}</span>
                          {formatCount(group.request_count)}
                        </span>
                        <span {...stylex.props(styles.metricValue, styles.metricRate)}>
                          <span {...stylex.props(styles.metricLabel)}>
                            {text('successRate24h')}
                          </span>
                          {group.request_count === 0 ? '—' : formatRate(group.success_rate)}
                        </span>
                      </div>
                      <div
                        role="cell"
                        data-schedule-settings
                        {...stylex.props(styles.cell, styles.settings)}
                        onClick={(event) => event.stopPropagation()}
                      >
                        <div {...stylex.props(styles.controlGroup)}>
                          <span aria-hidden {...stylex.props(styles.controlLabel)}>
                            {text('priority')}
                          </span>
                          <ScheduleNumberInput
                            id={`priority-${index}`}
                            label={`${text('priority')} ${rowLabel}`}
                            value={inputValue(group.group_id, entry, 'priority')}
                            min={1}
                            invalid={
                              invalidInputs[
                                fieldKey(group.group_id, entry.entry_id, 'priority')
                              ] === true
                            }
                            invalidMessage={text('invalidValue')}
                            disabled={pending || derived}
                            onChange={(value) => setInput(group.group_id, entry, 'priority', value)}
                          />
                        </div>
                        <div {...stylex.props(styles.controlGroup)}>
                          <span aria-hidden {...stylex.props(styles.controlLabel)}>
                            {text('weight')}
                          </span>
                          <ScheduleNumberInput
                            id={`weight-${index}`}
                            max={100}
                            label={`${text('weight')} ${rowLabel}`}
                            value={inputValue(group.group_id, entry, 'weight')}
                            min={0}
                            invalid={
                              invalidInputs[fieldKey(group.group_id, entry.entry_id, 'weight')] ===
                              true
                            }
                            invalidMessage={text('invalidValue')}
                            disabled={pending || derived}
                            onChange={(value) => setInput(group.group_id, entry, 'weight', value)}
                          />
                        </div>
                        <div {...stylex.props(styles.controlGroup)}>
                          <span aria-hidden {...stylex.props(styles.controlLabel)}>
                            {text('reasoning')}
                          </span>
                          <Selector
                            label={`${text('reasoning')} ${rowLabel}`}
                            isLabelHidden
                            size="sm"
                            width="100%"
                            xstyle={styles.selector}
                            value={entryReasoningValue(group.group_id, entry)}
                            options={reasoningOptionsFor(entry)}
                            isDisabled={pending || derived}
                            onChange={(value) => setEntryReasoning(group.group_id, entry, value)}
                          />
                        </div>
                      </div>
                      <div role="cell" {...stylex.props(styles.cell, styles.cellShare)}>
                        <span {...stylex.props(styles.shareLabel)}>{text('share')} </span>
                        {`${(shareValue(group.group_id, entry) * 100).toFixed(1)}%`}
                      </div>
                      <div
                        role="cell"
                        {...stylex.props(styles.cell, styles.actions)}
                        onClick={(event) => event.stopPropagation()}
                      >
                        <button
                          type="button"
                          role="switch"
                          aria-checked={enabled}
                          aria-label={toggleLabel}
                          title={toggleLabel}
                          disabled={pending || togglingEntries.has(key) || derived}
                          {...stylex.props(styles.toggle)}
                          onClick={() => {
                            void toggleEntryEnabled(group.group_id, entry, !enabled)
                          }}
                        >
                          <span
                            {...stylex.props(styles.toggleTrack, enabled && styles.toggleTrackOn)}
                          >
                            <span
                              {...stylex.props(styles.toggleThumb, enabled && styles.toggleThumbOn)}
                            />
                          </span>
                        </button>
                        {needsRecovery ? (
                          <Button
                            label={text('recover')}
                            isLoading={recovering === key}
                            variant="secondary"
                            size="sm"
                            xstyle={styles.rowAction}
                            isDisabled={pending || derived}
                            onClick={() => {
                              void recover(group.group_id, entry)
                            }}
                          />
                        ) : (
                          <Button
                            label={t('monitor.modelProbe.button')}
                            variant="secondary"
                            size="sm"
                            xstyle={styles.rowAction}
                            isDisabled={pending || derived}
                            onClick={() =>
                              requestProbe(group.group_id, entry.model_id, !group.enabled)
                            }
                          />
                        )}
                        <Popover
                          isOpen={menuRow === key}
                          onOpenChange={(open) => setMenuRow(open ? key : null)}
                          placement="below"
                          alignment="end"
                          width={300}
                          label={t('monitor.schedule.detail.moreActions', {
                            model: entry.model_id,
                            group: group.group_name,
                          })}
                          closeButtonLabel={t('common.close')}
                          content={
                            <div
                              {...stylex.props(styles.menu)}
                              onClick={(event) => event.stopPropagation()}
                            >
                              <div {...stylex.props(styles.menuActions)}>
                                <Button
                                  label={t('modelPrices.matrix.heading')}
                                  variant="secondary"
                                  size="sm"
                                  isDisabled={pending || onOpenPrice === undefined}
                                  onClick={() => {
                                    setMenuRow(null)
                                    onOpenPrice(group.group_id, entry.model_id)
                                  }}
                                />
                                {needsRecovery && (
                                  <Button
                                    label={t('monitor.modelProbe.button')}
                                    variant="secondary"
                                    size="sm"
                                    isDisabled={pending || derived}
                                    onClick={() => {
                                      setMenuRow(null)
                                      requestProbe(group.group_id, entry.model_id, !group.enabled)
                                    }}
                                  />
                                )}
                              </div>
                              <div {...stylex.props(styles.menuField)}>
                                <span>{t('monitor.schedule.detail.concurrencyLimit')}</span>
                                <ScheduleNumberInput
                                  id={`concurrency-${index}`}
                                  label={`${t('monitor.schedule.detail.concurrencyLimit')} ${rowLabel}`}
                                  value={inputValue(group.group_id, entry, 'concurrency_limit')}
                                  placeholder={String(entry.concurrency_limit ?? 0)}
                                  min={0}
                                  max={2147483647}
                                  invalid={
                                    invalidInputs[
                                      fieldKey(group.group_id, entry.entry_id, 'concurrency_limit')
                                    ] === true || draftMap[key]?.concurrency_limit === null
                                  }
                                  invalidMessage={t(
                                    'monitor.schedule.detail.invalidConcurrencyLimit',
                                  )}
                                  disabled={pending || derived}
                                  onChange={(value) =>
                                    setInput(group.group_id, entry, 'concurrency_limit', value)
                                  }
                                />
                                <span {...stylex.props(styles.menuHelp)}>
                                  {t('monitor.schedule.detail.concurrencyHelp')}
                                </span>
                              </div>
                              <div {...stylex.props(styles.breaker)}>
                                <strong>{text('breakerRecovery')}</strong>
                                <span {...stylex.props(styles.cellSmall)}>
                                  {breakerRecoveryLabel(entry)}
                                </span>
                              </div>
                            </div>
                          }
                        >
                          <Button
                            label={t('monitor.schedule.detail.more')}
                            variant="ghost"
                            size="sm"
                            xstyle={styles.rowAction}
                            isDisabled={pending}
                            aria-label={`${t('monitor.schedule.detail.more')} ${rowLabel}`}
                          />
                        </Popover>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          <StickySaveBar
            appearance="ledger"
            dirty={dirty}
            pending={pending}
            status={saveStatus}
            error={saveStatus === 'error' ? saveError : ''}
            errorActionLabel={saveStatus === 'error' ? text('refreshToResolve') : ''}
            onErrorAction={onRefresh}
            statusContent={
              saveStatus === 'saved' ? (
                <span>{text('saved')}</span>
              ) : dirty ? (
                <span>
                  {Object.keys(draftMap).length > 0 &&
                    t('monitor.schedule.detail.dirtySummary', {
                      entries: Object.keys(draftMap).length,
                      fields: Object.values(draftMap).reduce(
                        (total, draft) => total + Object.keys(draft).length,
                        0,
                      ),
                    })}
                  {invalid &&
                    `${Object.keys(draftMap).length > 0 ? ' · ' : ''}${t('monitor.schedule.detail.invalidInputs', { count: Math.max(1, Object.values(invalidInputs).filter(Boolean).length) })}`}
                </span>
              ) : null
            }
            actions={
              <>
                <Button
                  variant="ghost"
                  size="sm"
                  isDisabled={pending}
                  label={text('discard')}
                  onClick={discard}
                />
                <Button
                  variant="primary"
                  size="sm"
                  isDisabled={pending || invalid}
                  label={text('save')}
                  onClick={() => void save()}
                />
              </>
            }
          />
          <details>
            <summary {...stylex.props(styles.menuHelp)}>
              {t('monitor.schedule.detail.rules')}
            </summary>
            <p {...stylex.props(styles.menuHelp)}>{t('monitor.schedule.detail.rulesHelp')}</p>
          </details>
        </>
      ) : null}

      <ModelProbeScopeDialog
        open={probeScopeOpen}
        total={probeScopes.all.length}
        disabledCount={probeScopes.all.length - probeScopes.enabled.length}
        onOpenChange={setProbeScopeOpen}
        onProbeAll={confirmProbeAll}
        onProbeEnabled={confirmProbeEnabled}
      />
      <AlertDialog
        isOpen={pendingProbe !== null}
        title={t('monitor.modelProbe.disabledConfirm.title')}
        description={t('monitor.modelProbe.disabledConfirm.description')}
        cancelLabel={t('monitor.modelProbe.disabledConfirm.cancel')}
        actionLabel={t('monitor.modelProbe.disabledConfirm.confirm')}
        actionVariant="primary"
        onOpenChange={handleSingleProbeOpen}
        onAction={confirmSingleProbe}
      />
    </section>
  )
}
