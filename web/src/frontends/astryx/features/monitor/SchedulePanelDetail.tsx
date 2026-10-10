import * as stylex from '@stylexjs/stylex'
import { AlertDialog, Button, Selector, Switch } from '@astryxdesign/core'
import { LoaderCircle, RefreshCw, TriangleAlert } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type JSX } from 'react'

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
import type { ScheduleDrafts } from '@shared/routing/monitor-route'
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
  toggleEnabled?: string
  toggleFailed?: string
  disabled?: string
  calls24h?: string
  successRate24h?: string
  reasoning: string
  inherit: string
  groupDisabled: string
}

export interface SchedulePanelDetailProps {
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

type EditableField = 'weight' | 'priority'
type Draft = Partial<Record<EditableField, number | null>>

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
  if (field === 'weight') return null
  return entry.priority === 1 ? null : entry.priority
}

function effectiveValue(entry: ModelRouteScheduleEntryDto, field: EditableField): number {
  return field === 'weight' ? entry.weight : entry.priority
}

function isValidValue(field: EditableField, value: number): boolean {
  if (!Number.isSafeInteger(value)) return false
  return field === 'weight' ? value >= 0 && value <= 100 : value >= 1
}

function draftFingerprint(source: ScheduleDrafts): string {
  return JSON.stringify(
    Object.keys(source)
      .sort()
      .map((key) => [key, source[key]]),
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
    for (const field of ['weight', 'priority'] as const) {
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
  reasoningDrafts: Record<string, ReasoningEffortDto | null>,
): ModelRouteSchedulePatchUpdate[] {
  const result: ModelRouteSchedulePatchUpdate[] = []
  for (const group of detail?.groups ?? []) {
    for (const entry of group.entries) {
      const key = draftKey(group.group_id, entry.entry_id)
      const draft = draftMap[key]
      if (
        (draft === undefined && !hasOwn(reasoningDrafts, key)) ||
        entry.entry_id.startsWith('derived:')
      ) {
        continue
      }
      const update: ModelRouteSchedulePatchUpdate = {
        group_id: group.group_id,
        entry_id: entry.entry_id,
      }
      if (draft !== undefined && hasOwn(draft, 'weight')) update.weight = draft.weight
      if (draft !== undefined && hasOwn(draft, 'priority')) update.priority = draft.priority
      if (hasOwn(reasoningDrafts, key)) {
        update.reasoning_effort = reasoningDrafts[key] ?? null
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

const narrow = '@media (max-width: 620px)'
const wide = '@media (min-width: 621px)'
const touch = '@media (max-width: 860px)'
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
    gap: 'var(--space-3)',
    paddingBottom: 'var(--space-2)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
  },
  headerActions: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: { default: 'flex-end', [narrow]: 'space-between' },
    flexWrap: 'wrap',
    gap: 'var(--space-2) var(--space-3)',
  },
  title: {
    margin: 0,
    color: 'var(--color-text)',
    fontSize: 'var(--text-lg)',
  },
  observed: {
    display: 'flex',
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
    minHeight: 0,
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
    // position lets the scroll container be the containing block for the 1px
    // hidden cell labels so they cannot widen the page on narrow viewports.
    position: 'relative',
    minWidth: 0,
    maxWidth: '100%',
    maxHeight: { default: '72vh', [narrow]: 'none' },
    overflow: { default: 'auto', [narrow]: 'visible' },
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
  },
  table: {
    minWidth: { default: 1120, [narrow]: 0 },
  },
  row: {
    display: 'grid',
    gridTemplateColumns: {
      default:
        '96px minmax(154px, 1.2fr) minmax(144px, 1fr) minmax(128px, 0.8fr) 96px 100px 120px minmax(164px, 1.1fr)',
      [narrow]: 'minmax(0, 1fr) minmax(0, 1fr)',
    },
    minHeight: 44,
    alignItems: 'center',
    gap: { default: 0, [narrow]: '10px 12px' },
    borderBottomWidth: { default: 1, ':last-child': 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border)',
    paddingTop: { default: 0, [narrow]: 12 },
    paddingBottom: { default: 0, [narrow]: 12 },
    paddingInline: { default: 0, [narrow]: 12 },
    contentVisibility: 'auto',
    containIntrinsicSize: '0 64px',
  },

  rowHeader: {
    display: { default: 'grid', [narrow]: 'none' },
    position: 'sticky',
    zIndex: 1,
    top: 0,
    minHeight: 34,
  },
  headerCell: {
    paddingBlock: 'var(--spacing-1)',
    paddingInline: 'var(--spacing-2)',
    minWidth: 0,
    boxSizing: 'border-box',
  },
  rowPriorityStart: {
    borderTopWidth: 2,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-strong)',
  },
  rowSelected: {
    backgroundColor: 'var(--color-action-soft)',
  },
  cell: {
    minWidth: 0,
    color: 'var(--color-text-primary)',
    fontSize: 'var(--text-body-size)',
    paddingBlock: { default: 'var(--spacing-1)', [narrow]: 0 },
    paddingInline: { default: 'var(--spacing-2)', [narrow]: 0 },
    boxSizing: 'border-box',
  },
  cellPriority: {
    alignSelf: 'stretch',
    display: 'grid',
    alignContent: 'center',
    justifyItems: 'start',
    fontVariantNumeric: 'tabular-nums',
  },
  priorityEdit: {
    display: 'flex',
    alignItems: 'center',
    gap: 3,
    marginTop: 4,
  },
  cellLabel: {
    display: 'block',
    marginBottom: 2,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-size)',
    fontWeight: 650,
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
    position: { default: 'static', [wide]: 'absolute' },
    width: { default: 'auto', [wide]: 1 },
    height: { default: 'auto', [wide]: 1 },
    overflow: { default: 'visible', [wide]: 'hidden' },
    clip: { default: 'none', [wide]: 'rect(0 0 0 0)' },
    whiteSpace: { default: 'normal', [wide]: 'nowrap' },
  },
  cellStrong: {
    display: 'block',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    color: 'var(--color-text)',
    fontWeight: 650,
  },
  cellSmall: {
    display: 'block',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    marginTop: 2,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  cellStats: {
    fontFamily: 'var(--font-mono)',
  },
  cellReasoning: {
    display: 'grid',
    alignContent: 'center',
    gap: 2,
    gridColumn: 'auto',
  },
  cellInput: {
    display: 'grid',
    alignContent: 'center',
    gap: 2,
  },
  cellShare: {
    display: 'grid',
    alignContent: 'center',
    color: 'var(--color-action)',
    fontFamily: 'var(--font-mono)',
    fontWeight: 700,
  },
  cellBreaker: {
    display: 'flex',
    alignItems: { default: 'center', [narrow]: 'flex-start' },
    flexDirection: { default: 'row', [narrow]: 'column' },
    flexWrap: 'wrap',
    gap: 6,
    fontFamily: 'var(--font-mono)',
  },
  shareTrack: {
    display: 'block',
    height: 4,
    marginTop: 5,
    overflow: 'hidden',
    borderRadius: '999px',
    backgroundColor: 'var(--color-border-subtle)',
  },
  shareFill: {
    display: 'block',
    height: '100%',
    borderRadius: 'inherit',
    backgroundColor: 'var(--color-action)',
  },
  groupLink: {
    display: 'block',
    // Classic `:hover strong` recolors the name through the link — the strong
    // inherits the link color so the hover cascade survives.
    color: { default: 'var(--color-text)', ':hover': 'var(--color-action)' },
    textDecoration: 'none',
  },
  groupLinkStrong: {
    color: 'inherit',
  },
  runtimeAvailable: {
    color: 'var(--color-success)',
  },
  runtimeCooldown: {
    color: 'var(--color-warning)',
  },
  runtimeBlacklisted: {
    color: 'var(--color-danger)',
  },
  // AppTextInput (size="compact", appearance="surface") port — a native input
  // is required because the DS TextInput always assigns its own internal id,
  // while the e2e contract depends on `priority-${index}`/`weight-${index}`.
  inputShell: {
    display: 'flex',
    width: '100%',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    color: 'var(--color-text-muted)',
    paddingInlineStart: 'var(--space-3)',
    backgroundColor: 'var(--color-surface)',
    minHeight: { default: 'var(--control-xs)', [touch]: 'var(--touch-target)' },
    fontSize: 'var(--text-meta)',
  },
  inputShellInvalid: {
    borderColor: 'var(--color-danger)',
  },
  inputError: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  inputShellDisabled: {
    cursor: 'not-allowed',
    opacity: 0.55,
  },
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
    fontSize: 'inherit',
    fontStyle: 'inherit',
    fontWeight: 'inherit',
    lineHeight: 'inherit',
    appearance: 'textfield',
    opacity: { '::placeholder': 1 },
    cursor: { ':disabled': 'not-allowed' },
  },
  priorityInputShell: {
    width: 58,
    flex: 'none',
  },
  // Classic caps the weight field's inner input at 62px (100% at ≤620px) while
  // the bordered shell keeps stretching to the cell width.
  weightInputInner: {
    width: { default: 62, [narrow]: '100%' },
  },
  srOnly: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  // display: contents keeps the click-stop boundary without adding a layout
  // box, so the classic `@click.stop` wrappers don't change the cell flow.
  stopPropagation: {
    display: 'contents',
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
  mono,
  shellStyle,
  innerStyle,
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
  mono?: boolean
  shellStyle?: stylex.StyleXStyles
  innerStyle?: stylex.StyleXStyles
  onChange(value: string): void
}): JSX.Element {
  return (
    <div
      data-input-shell
      {...(mono === true ? { 'data-gptload-mono': true } : {})}
      {...stylex.props(
        styles.inputShell,
        invalid === true && styles.inputShellInvalid,
        disabled === true && styles.inputShellDisabled,
        shellStyle,
      )}
    >
      <label {...stylex.props(styles.srOnly)} htmlFor={id}>
        {label}
      </label>
      <input
        id={id}
        data-input-inner
        {...stylex.props(styles.inputInner, innerStyle)}
        type="number"
        value={value}
        placeholder={placeholder}
        disabled={disabled}
        aria-invalid={invalid === true || undefined}
        aria-describedby={invalid === true ? `${id}-error` : undefined}
        autoComplete="off"
        spellCheck={false}
        inputMode="numeric"
        onChange={(event) => onChange(event.currentTarget.value)}
      />
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
  const [reasoningDrafts, setReasoningDrafts] = useState<Record<string, ReasoningEffortDto | null>>(
    {},
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
  const [pendingProbe, setPendingProbe] = useState<{
    groupId: number
    modelId: string
  } | null>(null)
  // Watch-A hydration can emit a draft reset; renders are pure, so the emit is
  // queued as state and dispatched from an effect after commit.
  const [queuedDraftEmit, setQueuedDraftEmit] = useState<ScheduleDrafts | null>(null)

  const scheduleTableRef = useRef<HTMLDivElement | null>(null)
  const callbacksRef = useRef({
    onDraftChange,
    onRowChange,
  })
  useEffect(() => {
    callbacksRef.current = { onDraftChange, onRowChange }
  })
  const emittedDraftsRef = useRef<ScheduleDrafts | null>(null)

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
    let emitAfter = false
    if (keyChanged) {
      const previousRevision = synced.key[0]
      if (previousRevision !== 0 && previousRevision !== detailKey[0]) {
        if (nextRevisions.has(previousRevision)) {
          nextRevisions = withoutSetValue(nextRevisions, previousRevision)
          const snapshot = nextSnapshots.get(previousRevision) ?? drafts
          nextSnapshots = withoutMapValue(nextSnapshots, previousRevision)
          hydrationSource = snapshot
        } else {
          hydrationSource = {}
          for (const revision of nextRevisions) {
            nextSnapshots = withMapValue(nextSnapshots, revision, {})
          }
          emitAfter = true
        }
      } else {
        hydrationSource = drafts
      }
      // hydrateDraftState clears pendingLocalDraftFingerprint; in the emit
      // branch emitDraftChange then re-sets it to the '{}' fingerprint.
      nextFingerprint = emitAfter ? draftFingerprint({}) : undefined
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
      setReasoningDrafts({})
      setRawInputs(hydrated.rawInputs)
      setInvalidInputs({})
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
    if (emitAfter) setQueuedDraftEmit({})
  }

  useEffect(() => {
    const queued = queuedDraftEmit
    if (queued === null || emittedDraftsRef.current === queued) return
    emittedDraftsRef.current = queued
    callbacksRef.current.onDraftChange(queued)
  }, [queuedDraftEmit])

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

  const dirty = Object.keys(draftMap).length > 0 || Object.keys(reasoningDrafts).length > 0
  const invalid = Object.values(invalidInputs).some(Boolean)
  const hasScheduleDraft = Object.keys(draftMap).length > 0

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
      if (enabled) {
        totals.set(candidate.priority, (totals.get(candidate.priority) ?? 0) + candidate.weight)
      }
    }
    for (const candidate of candidates) {
      const key = rowKey(candidate.group.group_id, candidate.entry.entry_id)
      const enabled = optimisticEnabled.get(key) ?? candidate.entry.enabled
      const total = totals.get(candidate.priority) ?? 0
      result.set(key, enabled && total > 0 ? Math.max(0, candidate.weight) / total : 0)
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

  function isPriorityStart(index: number): boolean {
    return index === 0 || rows[index - 1]?.entry.priority !== rows[index]?.entry.priority
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
    return hasOwn(reasoningDrafts, key)
      ? (reasoningDrafts[key] ?? '')
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
    setReasoningDrafts((previous) => {
      const draft = { ...previous }
      if (next === entry.reasoning.configured) delete draft[key]
      else draft[key] = next
      return draft
    })
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

  function placeholder(entry: ModelRouteScheduleEntryDto, field: EditableField): string {
    const configured = configuredValue(entry, field)
    return String(configured === null ? effectiveValue(entry, field) : configured)
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
      return
    }
    setDraftValue(groupID, entry, field, parsed)
  }

  function resetDrafts(): void {
    setDraftMap({})
    setReasoningDrafts({})
    setRawInputs({})
    setInvalidInputs({})
    setSaveStatus('idle')
    setSaveError('')
  }

  function discard(): void {
    resetDrafts()
    emitDraftChange({})
  }

  async function save(): Promise<void> {
    if (detail === undefined || !dirty || invalid) return
    const body: ModelRouteSchedulePatchRequest = {
      snapshot_revision: detail.snapshot_revision,
      protocol: detail.protocol,
      external_model: detail.external_model ?? '',
      operation: detail.operation,
      updates: buildPatchUpdates(detail, draftMap, reasoningDrafts),
    }
    if (body.updates.length === 0) return
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
    } catch (saveFailure: unknown) {
      setSaveStatus('error')
      setSaveError(
        isModelRouteScheduleRevisionConflict(saveFailure) ? text('conflict') : text('saveFailed'),
      )
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

  function runtimeLabel(entry: ModelRouteScheduleEntryDto): string {
    if (entry.runtime.state === 'blacklisted') return text('blacklisted')
    if (entry.runtime.state === 'cooldown') return text('cooldown')
    return text('available')
  }

  function shareValue(groupID: number, entry: ModelRouteScheduleEntryDto): number {
    if (!entryEnabled(groupID, entry)) return 0
    return previewShares.get(rowKey(groupID, entry.entry_id)) ?? entry.configured_share
  }

  function isDraftShare(groupID: number, entry: ModelRouteScheduleEntryDto): boolean {
    return previewShares.has(rowKey(groupID, entry.entry_id))
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

  const runtimeToneStyle = {
    available: styles.runtimeAvailable,
    cooldown: styles.runtimeCooldown,
    blacklisted: styles.runtimeBlacklisted,
  } as const

  return (
    <section {...stylex.props(styles.detail)} aria-labelledby="schedule-detail-title">
      <header {...stylex.props(styles.header)}>
        <h2 id="schedule-detail-title" {...stylex.props(styles.title)}>
          {text('title')}
        </h2>
        <div {...stylex.props(styles.headerActions)}>
          {observedLabel !== '' && <span {...stylex.props(styles.observed)}>{observedLabel}</span>}
          {rows.length > 0 && (
            <Button
              variant="secondary"
              size="sm"
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
                ▲
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
              <div {...stylex.props(styles.table)} role="table" aria-label={text('title')}>
                <div {...stylex.props(styles.row, styles.rowHeader)} role="row">
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('priority')}
                  </span>
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('upstreamModel')}
                  </span>
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('group')}
                  </span>
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('reasoning')}
                  </span>
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('weight')}
                  </span>
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('share')}
                  </span>
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('status')}
                  </span>
                  <span {...stylex.props(styles.headerCell)} role="columnheader">
                    {text('breakerRecovery')}
                  </span>
                </div>
                {rows.map(({ group, entry }, index) => {
                  const key = rowKey(group.group_id, entry.entry_id)
                  const derived = entry.entry_id.startsWith('derived:')
                  const enabled = entryEnabled(group.group_id, entry)
                  const share = shareValue(group.group_id, entry)
                  const shareLabel = `${(share * 100).toFixed(1)}%`
                  const priorityFieldKey = fieldKey(group.group_id, entry.entry_id, 'priority')
                  const weightFieldKey = fieldKey(group.group_id, entry.entry_id, 'weight')
                  return (
                    <div
                      key={key}
                      data-row-key={key}
                      aria-selected={selectedRow === key}
                      {...stylex.props(
                        styles.row,

                        selectedRow === key && styles.rowSelected,
                        isPriorityStart(index) && styles.rowPriorityStart,
                      )}
                      role="row"
                      tabIndex={0}
                      onClick={() => onRowChange(key)}
                      onKeyDown={(event) => {
                        if (event.target !== event.currentTarget) return
                        if (event.key === 'Enter' || event.key === ' ') {
                          event.preventDefault()
                          onRowChange(key)
                        }
                      }}
                    >
                      <div {...stylex.props(styles.cell, styles.cellPriority)} role="cell">
                        <span {...stylex.props(styles.cellLabel)}>{text('priority')}</span>
                        <div {...stylex.props(styles.priorityEdit)}>
                          <ScheduleNumberInput
                            id={`priority-${index}`}
                            label={`${text('priority')} ${entry.model_id}`}
                            value={inputValue(group.group_id, entry, 'priority')}
                            placeholder={placeholder(entry, 'priority')}
                            invalid={invalidInputs[priorityFieldKey] === true}
                            invalidMessage={text('invalidValue')}
                            min={1}
                            disabled={derived}
                            shellStyle={styles.priorityInputShell}
                            onChange={(value) => setInput(group.group_id, entry, 'priority', value)}
                          />
                        </div>
                      </div>
                      <div {...stylex.props(styles.cell)} role="cell">
                        <span {...stylex.props(styles.cellLabel)}>{text('upstreamModel')}</span>
                        <strong {...stylex.props(styles.cellStrong)}>
                          {entry.alias || entry.model_id}
                        </strong>
                        {entry.alias !== '' && (
                          <small {...stylex.props(styles.cellSmall)}>{entry.model_id}</small>
                        )}
                        <span
                          {...stylex.props(styles.stopPropagation)}
                          onClick={(event) => event.stopPropagation()}
                        >
                          <Switch
                            label={`${text('toggleEnabled')} ${entry.model_id}`}
                            isLabelHidden
                            size="sm"
                            value={enabled}
                            isDisabled={pending || togglingEntries.has(key) || derived}
                            onChange={(next) =>
                              void toggleEntryEnabled(group.group_id, entry, next)
                            }
                          />
                        </span>
                      </div>
                      <div {...stylex.props(styles.cell)} role="cell">
                        <span {...stylex.props(styles.cellLabel)}>{text('group')}</span>
                        <RouteLink
                          to={groupDetailHref(group.group_id)}
                          {...stylex.props(styles.groupLink)}
                          onClick={(event) => event.stopPropagation()}
                        >
                          <strong {...stylex.props(styles.cellStrong, styles.groupLinkStrong)}>
                            {group.group_name}
                          </strong>
                        </RouteLink>
                        <small {...stylex.props(styles.cellSmall, styles.cellStats)}>
                          {text('calls24h')}: {formatCount(group.request_count)} ·{' '}
                          {text('successRate24h')}: {formatRate(group.success_rate)}
                        </small>
                      </div>
                      <div {...stylex.props(styles.cell, styles.cellReasoning)} role="cell">
                        <span {...stylex.props(styles.cellLabel)}>{text('reasoning')}</span>
                        <span
                          {...stylex.props(styles.stopPropagation)}
                          onClick={(event) => event.stopPropagation()}
                        >
                          <Selector
                            label={`${text('reasoning')} ${entry.model_id}`}
                            isLabelHidden
                            size="sm"
                            options={reasoningOptionsFor(entry)}
                            value={entryReasoningValue(group.group_id, entry)}
                            isDisabled={pending || derived}
                            width="auto"
                            onChange={(value) => setEntryReasoning(group.group_id, entry, value)}
                          />
                        </span>
                      </div>
                      <div {...stylex.props(styles.cell, styles.cellInput)} role="cell">
                        <label {...stylex.props(styles.cellLabel)} htmlFor={`weight-${index}`}>
                          {text('weight')}
                        </label>
                        <ScheduleNumberInput
                          id={`weight-${index}`}
                          label={`${text('weight')} ${entry.model_id}`}
                          value={inputValue(group.group_id, entry, 'weight')}
                          placeholder={placeholder(entry, 'weight')}
                          invalid={invalidInputs[weightFieldKey] === true}
                          invalidMessage={text('invalidValue')}
                          min={0}
                          max={100}
                          disabled={derived}
                          mono
                          innerStyle={styles.weightInputInner}
                          onChange={(value) => setInput(group.group_id, entry, 'weight', value)}
                        />
                      </div>
                      <div {...stylex.props(styles.cell, styles.cellShare)} role="cell">
                        <span {...stylex.props(styles.cellLabel)}>{text('share')}</span>
                        <strong {...stylex.props(styles.cellStrong)}>{shareLabel}</strong>
                        <span
                          {...stylex.props(styles.shareTrack)}
                          aria-label={`${text('share')} ${shareLabel}`}
                        >
                          <span
                            {...stylex.props(styles.shareFill)}
                            style={{ width: `${share * 100}%` }}
                          />
                        </span>
                        {isDraftShare(group.group_id, entry) && (
                          <small {...stylex.props(styles.cellSmall)}>{text('draftPreview')}</small>
                        )}
                      </div>
                      <div {...stylex.props(styles.cell)} role="cell">
                        <span {...stylex.props(styles.cellLabel)}>{text('status')}</span>
                        {!enabled ? (
                          <strong {...stylex.props(styles.cellStrong, styles.runtimeBlacklisted)}>
                            {text('disabled')}
                          </strong>
                        ) : !group.enabled ? (
                          <strong {...stylex.props(styles.cellStrong, styles.runtimeCooldown)}>
                            {text('groupDisabled')}
                          </strong>
                        ) : (
                          <strong
                            {...stylex.props(
                              styles.cellStrong,
                              runtimeToneStyle[entry.runtime.state],
                            )}
                          >
                            {runtimeLabel(entry)}
                          </strong>
                        )}
                        {entry.runtime.failure_count !== 0 && (
                          <small {...stylex.props(styles.cellSmall)}>
                            {entry.runtime.failure_count} {text('failures')}
                          </small>
                        )}
                      </div>
                      <div {...stylex.props(styles.cell, styles.cellBreaker)} role="cell">
                        <span {...stylex.props(styles.cellLabel)}>{text('breakerRecovery')}</span>
                        <span>{breakerRecoveryLabel(entry)}</span>
                        <Button
                          variant="secondary"
                          size="sm"
                          label={t('modelPrices.matrix.heading')}
                          onClick={(event) => {
                            event.stopPropagation()
                            onOpenPrice(group.group_id, entry.model_id)
                          }}
                        />
                        {entry.runtime.state !== 'available' && (
                          <Button
                            variant="secondary"
                            size="sm"
                            isLoading={recovering === key}
                            isDisabled={derived}
                            label={text('recover')}
                            onClick={(event) => {
                              event.stopPropagation()
                              void recover(group.group_id, entry)
                            }}
                          />
                        )}
                        <Button
                          variant="secondary"
                          size="sm"
                          isDisabled={pending || derived}
                          label={t('monitor.modelProbe.button')}
                          onClick={(event) => {
                            event.stopPropagation()
                            requestProbe(group.group_id, entry.model_id, !group.enabled)
                          }}
                        />
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
                <span>{text('unsaved')}</span>
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
