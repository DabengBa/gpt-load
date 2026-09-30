import * as stylex from '@stylexjs/stylex'
import { Button, IconButton, TextInput, Tooltip } from '@astryxdesign/core'
import { ChevronDown, ChevronRight, CircleAlert, Plus, Search, X } from 'lucide-react'
import {
  useEffect,
  useId,
  useImperativeHandle,
  useRef,
  useState,
  type ReactNode,
  type Ref,
} from 'react'

import {
  clientModel,
  modelDraftValidity,
  routeEntryShares,
  type ModelAliasEditorLabels,
  type ModelDraftKey,
  type ModelDraftValue,
  type ModelNameConflict,
} from '@shared/domain/models/model-draft'

import { LedgerRecordList, ledgerRecordStyles } from '../../components/LedgerRecordList'

const narrow = '@media (max-width: 860px)'
const small = '@media (max-width: 640px)'

const styles = stylex.create({
  root: {
    minWidth: 0,
  },
  toolbar: {
    display: 'flex',
    alignItems: { default: 'flex-end', [small]: 'stretch' },
    flexDirection: { default: 'row', [small]: 'column' },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
    paddingBlock: '15px 13px',
  },
  search: {
    display: 'grid',
    width: { default: 'min(100%, 420px)', [small]: '100%' },
    minWidth: 0,
    gap: '5px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  count: {
    flexShrink: 0,
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
  },
  groupCell: {
    gridColumn: '1 / -1',
    display: 'grid',
    gap: '6px',
  },
  groupToggle: {
    display: 'inline-flex',
    width: 'fit-content',
    alignItems: 'center',
    gap: '7px',
    paddingBlock: '2px',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-text)',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.7 },
    font: 'inherit',
    fontWeight: 600,
    padding: 0,
  },
  groupCount: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    fontWeight: 500,
  },
  distribution: {
    display: 'flex',
    width: 'min(100%, 360px)',
    height: '6px',
    overflow: 'hidden',
    borderRadius: '999px',
    backgroundColor: 'var(--color-border-subtle)',
  },
  distributionSegment: {
    height: '100%',
    backgroundColor: 'var(--color-action)',
  },
  distributionSegmentZero: {
    backgroundColor: 'var(--color-border-subtle)',
  },
  recordInvalid: {
    backgroundColor: 'var(--color-danger-bg)',
  },
  recordNested: {
    borderLeftWidth: '2px',
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-subtle)',
    paddingLeft: '10px',
  },
  idCell: {
    display: 'grid',
    alignContent: 'center',
    gap: '5px',
    gridColumn: { [narrow]: '1 / -1' },
  },
  idCode: {
    display: 'block',
    width: '100%',
    paddingInlineEnd: '38px',
    overflowWrap: 'anywhere',
    fontSize: 'var(--text-sm)',
  },
  fieldWrap: {
    position: 'relative',
    display: 'flex',
    width: '100%',
    minHeight: { default: 'var(--control-sm)', [narrow]: 'var(--touch-target)' },
    maxWidth: { default: '300px', [narrow]: 'none' },
    alignItems: 'center',
  },
  input: {
    width: '100%',
    minHeight: 'var(--control-xs)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingBlock: '4px',
    paddingInline: '9px',
    paddingInlineEnd: '30px',
    fontFamily: 'var(--font-mono)',
    fontSize: { default: 'var(--text-meta)', [narrow]: '16px' },
  },
  inputInvalid: {
    borderColor: 'var(--color-danger)',
  },
  errorIndicator: {
    position: 'absolute',
    insetInlineEnd: '3px',
    display: 'grid',
    width: '28px',
    height: '28px',
    placeItems: 'center',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-danger)',
    cursor: 'help',
    padding: 0,
  },
  aliasCell: {
    display: 'grid',
    gap: 'var(--space-1)',
    gridColumn: { [narrow]: '1 / -1' },
    borderTopWidth: { [narrow]: '1px' },
    borderTopStyle: { [narrow]: 'solid' },
    borderTopColor: { [narrow]: 'var(--color-border-subtle)' },
    paddingTop: { [narrow]: '11px' },
  },
  aliasControl: {
    display: 'flex',
    minHeight: 'var(--control-sm)',
    alignItems: 'center',
    gap: '9px',
  },
  aliasToggle: {
    display: 'grid',
    width: { default: '32px', [narrow]: 'var(--touch-target)' },
    height: { default: '32px', [narrow]: 'var(--touch-target)' },
    flexBasis: { default: '32px', [narrow]: 'var(--touch-target)' },
    flexShrink: 0,
    placeItems: 'center',
    cursor: { default: 'pointer', ':has(:disabled)': 'not-allowed' },
  },
  aliasToggleInput: {
    width: '16px',
    height: '16px',
    margin: 0,
    accentColor: 'var(--color-action)',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
  },
  aliasField: {
    width: 'min(100%, 300px)',
    minWidth: 0,
    flexGrow: 1,
    maxWidth: { [narrow]: 'none' },
  },
  routeCell: {
    display: 'grid',
    minWidth: 0,
    alignContent: 'center',
    gap: '4px',
    gridColumn: { [narrow]: '1 / -1' },
  },
  routeInputs: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: '8px',
  },
  routeField: {
    width: 'min(96px, 100%)',
    position: 'relative',
  },
  routeSummary: {
    display: 'inline-flex',
    minWidth: '42px',
    minHeight: 'var(--control-compact)',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
  },
  routeMeta: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    minHeight: '16px',
  },
  routeFlag: {
    paddingBlock: '1px',
    paddingInline: '7px',
    borderRadius: '999px',
    backgroundColor: 'var(--color-border-subtle)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
    fontWeight: 560,
  },
  routeShare: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
  },
  thirdColumn: {
    minWidth: 0,
    gridColumn: { [narrow]: '1 / -1' },
  },
  actions: {
    display: 'flex',
    justifyContent: 'flex-end',
    position: { [narrow]: 'absolute' },
    top: { [narrow]: '4px' },
    insetInlineEnd: { [narrow]: '4px' },
  },
  mobileLabel: {
    display: { default: 'none', [narrow]: 'inline' },
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 560,
  },
  disabledCell: {
    opacity: 0.55,
  },
  empty: {
    gridTemplateColumns: 'minmax(0, 1fr)',
    minHeight: '58px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    textAlign: 'center',
  },
  emptyCell: {
    gridColumn: '1 / -1',
  },
  add: {
    marginTop: '10px',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
})

export interface ModelAliasEditorHandle {
  addManual(): Promise<void>
  focusFirstInvalid(): Promise<void>
}

interface VisibleRow<T extends ModelDraftValue> {
  item: T
  index: number
}

interface VisibleGroup<T extends ModelDraftValue> {
  clientModel: string
  rows: VisibleRow<T>[]
}

type RenderItem<T extends ModelDraftValue> =
  | { kind: 'header'; key: string; group: VisibleGroup<T> }
  | { kind: 'row'; key: ModelDraftKey; item: T; index: number; nested: boolean }

/** CompactFieldError contract: positioned danger indicator (tooltip) + a
 *  visually-hidden error node the control references via aria-describedby. */
function FieldError({ id, error }: { id: string; error: string }) {
  return (
    <>
      <Tooltip content={error}>
        <button
          type="button"
          tabIndex={-1}
          aria-label={error}
          {...stylex.props(styles.errorIndicator)}
        >
          <CircleAlert size={15} aria-hidden />
        </button>
      </Tooltip>
      <span id={`${id}-error`} {...stylex.props(styles.srOnly)}>
        {error}
      </span>
    </>
  )
}

function sameIndexSet(a: Set<number>, b: Set<number>): boolean {
  if (a.size !== b.size) return false
  for (const value of a) if (!b.has(value)) return false
  return true
}

export interface ModelAliasEditorProps<T extends ModelDraftValue> {
  value: T[]
  conflicts: readonly ModelNameConflict[]
  labels: ModelAliasEditorLabels
  createRow?: () => T
  disabled?: boolean
  searchable?: boolean
  addable?: boolean
  search?: string
  validationMode?: 'immediate' | 'blur'
  showAllErrors?: boolean
  readonlyRouteFields?: boolean
  renderThirdColumn?: (item: T, index: number) => ReactNode
  onChange(value: T[]): void
  onSearchChange?(value: string): void
  onVisibleValidationChange?(indexes: Set<number>): void
  ref?: Ref<ModelAliasEditorHandle>
}

export function ModelAliasEditor<T extends ModelDraftValue>({
  value,
  conflicts,
  labels,
  createRow,
  disabled = false,
  searchable = true,
  addable = true,
  search,
  validationMode = 'immediate',
  showAllErrors = false,
  readonlyRouteFields = false,
  renderThirdColumn,
  onChange,
  onSearchChange,
  onVisibleValidationChange,
  ref,
}: ModelAliasEditorProps<T>) {
  const instanceId = useId()
  const rootRef = useRef<HTMLDivElement>(null)
  const [internalSearch, setInternalSearch] = useState(search ?? '')
  const [touchedModelIDs, setTouchedModelIDs] = useState<ReadonlySet<ModelDraftKey>>(new Set())
  const [touchedAliases, setTouchedAliases] = useState<ReadonlySet<ModelDraftKey>>(new Set())

  // Classic watch(props.search): external search value mirrors into the
  // internal draft (render adjustment — the change must not paint stale).
  const [lastSearch, setLastSearch] = useState(search)
  if (lastSearch !== search) {
    setLastSearch(search)
    setInternalSearch(search ?? '')
  }

  const searchValue = internalSearch
  function setSearchValue(next: string): void {
    setInternalSearch(next)
    onSearchChange?.(next)
  }

  const validity = modelDraftValidity(value, conflicts)
  const shares = routeEntryShares(value)

  const query = searchValue.trim().toLocaleLowerCase()
  const rows = value.flatMap<VisibleRow<T>>((item, index) =>
    !query || `${item.id} ${item.name} ${item.alias}`.toLocaleLowerCase().includes(query)
      ? [{ item, index }]
      : [],
  )
  const visibleGroups: VisibleGroup<T>[] = []
  const byName = new Map<string, VisibleGroup<T>>()
  for (const row of rows) {
    const name = clientModel(row.item) || '\u0000'
    let group = byName.get(name)
    if (!group) {
      group = { clientModel: name, rows: [] }
      byName.set(name, group)
      visibleGroups.push(group)
    }
    group.rows.push(row)
  }
  const visibleRowCount = visibleGroups.reduce((total, group) => total + group.rows.length, 0)
  const groupHeaderCount = visibleGroups.filter((group) => group.rows.length > 1).length

  const [collapsedGroups, setCollapsedGroups] = useState<ReadonlySet<string>>(new Set())
  function toggleGroup(name: string): void {
    const next = new Set(collapsedGroups)
    if (next.has(name)) next.delete(name)
    else next.add(name)
    setCollapsedGroups(next)
  }

  const renderList: RenderItem<T>[] = []
  for (const group of visibleGroups) {
    const collapsed = collapsedGroups.has(group.clientModel)
    if (group.rows.length > 1) {
      renderList.push({ kind: 'header', key: `group:${group.clientModel}`, group })
      if (collapsed) continue
    }
    const nested = group.rows.length > 1
    for (const row of group.rows) {
      renderList.push({ kind: 'row', key: row.item.key, item: row.item, index: row.index, nested })
    }
  }

  function sharePercent(index: number): string {
    const share = shares[index] ?? 0
    return `${Math.round(share * 1_000) / 10}%`
  }

  function routeCountText(routeValue: number | null): string {
    return routeValue === null ? '' : String(routeValue)
  }

  type RouteCountField = 'weight' | 'priority'

  function updateRow(
    index: number,
    patch: Partial<Pick<ModelDraftValue, 'id' | 'alias' | 'alias_enabled' | RouteCountField>>,
  ): void {
    onChange(
      value.map((item, current) =>
        current === index
          ? ({
              ...item,
              ...patch,
              alias: patch.alias_enabled === false ? '' : (patch.alias ?? item.alias),
            } as T)
          : ({ ...item, sources: [...item.sources] } as T),
      ),
    )
  }

  function updateRouteCount(index: number, field: RouteCountField, raw: string): void {
    if (readonlyRouteFields) return
    const trimmed = raw.trim()
    if (trimmed === '') {
      updateRow(index, { [field]: null })
      return
    }
    updateRow(index, { [field]: Number(trimmed) })
  }

  function removeRow(index: number): void {
    onChange(
      value
        .filter((_, current) => current !== index)
        .map((item) => ({ ...item, sources: [...item.sources] }) as T),
    )
  }

  function conflictMessage(index: number): string {
    const conflict = conflicts.find((item) => item.indexes.includes(index))
    return conflict ? labels.nameConflict(conflict.client_model) : ''
  }

  function modelIDError(item: ModelDraftValue, index: number): string {
    if (validity.emptyIDIndexes.has(index)) return labels.manualIdRequired
    if (!item.alias_enabled && validity.conflictIndexes.has(index)) {
      return conflictMessage(index)
    }
    return ''
  }

  function modelAliasError(item: ModelDraftValue, index: number): string {
    if (!item.alias_enabled) return ''
    if (validity.emptyAliasIndexes.has(index)) return labels.aliasRequired
    return validity.conflictIndexes.has(index) ? conflictMessage(index) : ''
  }

  function visibleWeightError(index: number): string {
    if (validity.invalidWeightIndexes.has(index)) return labels.invalidWeight
    if (validity.zeroShareIndexes.has(index)) return labels.zeroShare
    return ''
  }

  function visiblePriorityError(index: number): string {
    return validity.invalidPriorityIndexes.has(index) ? labels.invalidPriority : ''
  }

  function visibleModelIDError(item: ModelDraftValue, index: number): string {
    const error = modelIDError(item, index)
    if (!error) return ''
    if (
      validationMode === 'immediate' ||
      showAllErrors ||
      !item.editable_id ||
      touchedModelIDs.has(item.key)
    ) {
      return error
    }
    return ''
  }

  function visibleModelAliasError(item: ModelDraftValue, index: number): string {
    const error = modelAliasError(item, index)
    if (!error) return ''
    if (validationMode === 'immediate' || showAllErrors || touchedAliases.has(item.key)) {
      return error
    }
    return ''
  }

  const visibleInvalidIndexes = new Set(
    value.flatMap((item, index) =>
      visibleModelIDError(item, index) ||
      visibleModelAliasError(item, index) ||
      visibleWeightError(index) ||
      visiblePriorityError(index)
        ? [index]
        : [],
    ),
  )

  // Classic watch(visibleInvalidIndexes, {immediate}) → emit only when the
  // set actually changes (content equality, not identity).
  const emittedInvalidRef = useRef<Set<number>>(new Set())
  useEffect(() => {
    if (sameIndexSet(emittedInvalidRef.current, visibleInvalidIndexes)) return
    emittedInvalidRef.current = new Set(visibleInvalidIndexes)
    onVisibleValidationChange?.(new Set(visibleInvalidIndexes))
  })

  function touchModelID(key: ModelDraftKey): void {
    if (touchedModelIDs.has(key)) return
    setTouchedModelIDs(new Set(touchedModelIDs).add(key))
  }

  function touchAlias(key: ModelDraftKey): void {
    if (touchedAliases.has(key)) return
    setTouchedAliases(new Set(touchedAliases).add(key))
  }

  async function setAliasEnabled(index: number, enabled: boolean): Promise<void> {
    const item = value[index]
    if (enabled && item && validationMode === 'blur') {
      const nextTouched = new Set(touchedAliases)
      nextTouched.delete(item.key)
      setTouchedAliases(nextTouched)
    }
    updateRow(index, { alias_enabled: enabled })
    if (!enabled) return
    // Focus the freshly-enabled alias input after the parent re-renders.
    requestAnimationFrame(() => {
      rootRef.current
        ?.querySelector<HTMLInputElement>(`[data-alias-input-index="${index}"]`)
        ?.focus()
    })
  }

  async function addManual(): Promise<void> {
    if (disabled || !createRow) return
    setSearchValue('')
    const index = value.length
    onChange([...value.map((item) => ({ ...item, sources: [...item.sources] }) as T), createRow()])
    requestAnimationFrame(() => {
      rootRef.current?.querySelector<HTMLInputElement>(`[data-model-id-index="${index}"]`)?.focus()
    })
  }

  async function focusFirstInvalid(): Promise<void> {
    const index = Math.min(...validity.invalidIndexes)
    if (!Number.isFinite(index)) return
    setSearchValue('')
    const item = value[index]
    const targetsModelID =
      validity.emptyIDIndexes.has(index) ||
      (validity.conflictIndexes.has(index) && item?.editable_id && !item.alias_enabled)
    const targetsWeight =
      validity.invalidWeightIndexes.has(index) || validity.zeroShareIndexes.has(index)
    const targetsPriority = validity.invalidPriorityIndexes.has(index)
    if (item) {
      if (targetsModelID) touchModelID(item.key)
      else if (item.alias_enabled && !targetsWeight && !targetsPriority) touchAlias(item.key)
    }
    await Promise.resolve()
    requestAnimationFrame(() => {
      const selector = targetsModelID
        ? `[data-model-id-index="${index}"]`
        : targetsWeight
          ? `[data-model-weight-index="${index}"]`
          : targetsPriority
            ? `[data-model-priority-index="${index}"]`
            : item?.alias_enabled
              ? `[data-alias-input-index="${index}"]`
              : `[data-alias-toggle-index="${index}"]`
      rootRef.current?.querySelector<HTMLElement>(selector)?.focus()
    })
  }

  useImperativeHandle(ref, () => ({ addManual, focusFirstInvalid }))

  return (
    <div ref={rootRef} {...stylex.props(styles.root)}>
      {searchable && (
        <div {...stylex.props(styles.toolbar)}>
          <label {...stylex.props(styles.search)}>
            <span>{labels.searchLabel}</span>
            <TextInput
              label={labels.search}
              isLabelHidden
              placeholder={labels.search}
              value={searchValue}
              isDisabled={disabled}
              hasClear
              size="sm"
              startIcon={<Search size={14} aria-hidden="true" />}
              onChange={setSearchValue}
            />
          </label>
          <span {...stylex.props(styles.count)} aria-live="polite">
            {labels.count(value.length)}
          </span>
        </div>
      )}

      <LedgerRecordList
        label={labels.tableLabel}
        rowCount={visibleRowCount + groupHeaderCount + 1}
        grid="minmax(170px, 22fr) minmax(230px, 36fr) minmax(200px, 22fr) minmax(110px, 15fr) 40px"
        cardGrid="minmax(0, 0.7fr) minmax(0, 1.3fr)"
        recordMinHeight="58px"
        recordPadding="9px 0"
        columnGap="16px"
        header={
          <>
            <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
              {labels.id}
            </span>
            <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
              {labels.alias}
            </span>
            <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
              {labels.weight} / {labels.priority}
            </span>
            <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
              {labels.thirdColumn}
            </span>
            <span role="columnheader" {...stylex.props(ledgerRecordStyles.cell)}>
              <span {...stylex.props(styles.srOnly)}>{labels.actions}</span>
            </span>
          </>
        }
      >
        {renderList.map((render, renderIndex) =>
          render.kind === 'header' ? (
            <article
              key={render.key}
              {...stylex.props(ledgerRecordStyles.record)}
              role="row"
              aria-rowindex={renderIndex + 2}
            >
              <div {...stylex.props(ledgerRecordStyles.cell, styles.groupCell)} role="cell">
                <button
                  type="button"
                  {...stylex.props(styles.groupToggle)}
                  aria-expanded={!collapsedGroups.has(render.group.clientModel)}
                  disabled={disabled}
                  onClick={() => toggleGroup(render.group.clientModel)}
                >
                  {collapsedGroups.has(render.group.clientModel) ? (
                    <ChevronRight size={14} aria-hidden="true" />
                  ) : (
                    <ChevronDown size={14} aria-hidden="true" />
                  )}
                  <strong>{render.group.clientModel}</strong>
                  <span {...stylex.props(styles.groupCount)}>{render.group.rows.length}</span>
                </button>
                <div {...stylex.props(styles.distribution)} aria-hidden="true">
                  {render.group.rows.map((row) => (
                    <i
                      key={row.item.key}
                      {...stylex.props(
                        styles.distributionSegment,
                        (row.item.weight ?? 1) === 0 && styles.distributionSegmentZero,
                      )}
                      style={{ width: sharePercent(row.index) }}
                    />
                  ))}
                </div>
              </div>
            </article>
          ) : (
            <article
              key={render.key}
              data-testid="model-alias-editor__record"
              {...stylex.props(
                ledgerRecordStyles.record,
                visibleInvalidIndexes.has(render.index) && styles.recordInvalid,
                render.nested && styles.recordNested,
              )}
              role="row"
              aria-rowindex={renderIndex + 2}
            >
              <div
                {...stylex.props(
                  ledgerRecordStyles.cell,
                  styles.idCell,
                  (render.item.weight ?? 1) === 0 && styles.disabledCell,
                )}
                role="cell"
              >
                <span {...stylex.props(styles.mobileLabel)}>{labels.id}</span>
                {(() => {
                  const idError = visibleModelIDError(render.item, render.index)
                  const idFieldId = `${instanceId}-model-id-${render.index}`
                  return (
                    <span {...stylex.props(styles.fieldWrap)}>
                      {render.item.editable_id ? (
                        <input
                          id={idFieldId}
                          {...stylex.props(styles.input, idError !== '' && styles.inputInvalid)}
                          value={render.item.id}
                          aria-label={labels.id}
                          placeholder={labels.manualId}
                          aria-invalid={idError !== '' || undefined}
                          aria-describedby={idError !== '' ? `${idFieldId}-error` : undefined}
                          data-model-id-index={render.index}
                          spellCheck={false}
                          disabled={disabled}
                          onChange={(event) => updateRow(render.index, { id: event.target.value })}
                          onBlur={() => touchModelID(render.item.key)}
                        />
                      ) : (
                        <code
                          {...stylex.props(styles.idCode)}
                          aria-describedby={idError !== '' ? `${idFieldId}-error` : undefined}
                        >
                          {render.item.id}
                        </code>
                      )}
                      {idError !== '' && <FieldError id={idFieldId} error={idError} />}
                    </span>
                  )
                })()}
              </div>

              <div
                {...stylex.props(
                  ledgerRecordStyles.cell,
                  styles.aliasCell,
                  (render.item.weight ?? 1) === 0 && styles.disabledCell,
                )}
                role="cell"
              >
                <span {...stylex.props(styles.mobileLabel)}>{labels.alias}</span>
                <div {...stylex.props(styles.aliasControl)}>
                  <label {...stylex.props(styles.aliasToggle)}>
                    <span {...stylex.props(styles.srOnly)}>
                      {labels.aliasEnabledFor(render.item.id)}
                    </span>
                    <input
                      {...stylex.props(styles.aliasToggleInput)}
                      data-alias-toggle-index={render.index}
                      type="checkbox"
                      checked={render.item.alias_enabled}
                      disabled={disabled}
                      onChange={(event) => void setAliasEnabled(render.index, event.target.checked)}
                    />
                  </label>
                  {render.item.alias_enabled &&
                    (() => {
                      const aliasError = visibleModelAliasError(render.item, render.index)
                      const aliasFieldId = `${instanceId}-model-alias-${render.index}`
                      return (
                        <span {...stylex.props(styles.fieldWrap, styles.aliasField)}>
                          <input
                            id={aliasFieldId}
                            {...stylex.props(
                              styles.input,
                              aliasError !== '' && styles.inputInvalid,
                            )}
                            value={render.item.alias}
                            aria-label={labels.aliasFor(render.item.id)}
                            disabled={disabled}
                            placeholder={labels.aliasPlaceholder}
                            aria-invalid={aliasError !== '' || undefined}
                            aria-describedby={
                              aliasError !== '' ? `${aliasFieldId}-error` : undefined
                            }
                            data-alias-input-index={render.index}
                            spellCheck={false}
                            onChange={(event) =>
                              updateRow(render.index, { alias: event.target.value })
                            }
                            onBlur={() => touchAlias(render.item.key)}
                          />
                          {aliasError !== '' && <FieldError id={aliasFieldId} error={aliasError} />}
                        </span>
                      )
                    })()}
                </div>
              </div>

              <div {...stylex.props(ledgerRecordStyles.cell, styles.routeCell)} role="cell">
                <span {...stylex.props(styles.mobileLabel)}>
                  {labels.weight} / {labels.priority}
                </span>
                <div {...stylex.props(styles.routeInputs)}>
                  {readonlyRouteFields ? (
                    <>
                      <span {...stylex.props(styles.routeSummary)} aria-label={labels.weight}>
                        {routeCountText(render.item.weight ?? null) || '—'}
                      </span>
                      <span {...stylex.props(styles.routeSummary)} aria-label={labels.priority}>
                        {routeCountText(render.item.priority ?? null) || '—'}
                      </span>
                    </>
                  ) : (
                    <>
                      {(() => {
                        const weightError = visibleWeightError(render.index)
                        const weightId = `${instanceId}-model-weight-${render.index}`
                        return (
                          <span {...stylex.props(styles.routeField)}>
                            <input
                              id={weightId}
                              {...stylex.props(
                                styles.input,
                                weightError !== '' && styles.inputInvalid,
                              )}
                              value={routeCountText(render.item.weight ?? null)}
                              aria-label={labels.weight}
                              placeholder="1"
                              aria-invalid={weightError !== '' || undefined}
                              aria-describedby={
                                weightError !== '' ? `${weightId}-error` : undefined
                              }
                              data-model-weight-index={render.index}
                              spellCheck={false}
                              disabled={disabled}
                              onChange={(event) =>
                                updateRouteCount(render.index, 'weight', event.target.value)
                              }
                            />
                            {weightError !== '' && <FieldError id={weightId} error={weightError} />}
                          </span>
                        )
                      })()}
                      {(() => {
                        const priorityError = visiblePriorityError(render.index)
                        const priorityId = `${instanceId}-model-priority-${render.index}`
                        return (
                          <span {...stylex.props(styles.routeField)}>
                            <input
                              id={priorityId}
                              {...stylex.props(
                                styles.input,
                                priorityError !== '' && styles.inputInvalid,
                              )}
                              value={routeCountText(render.item.priority ?? null)}
                              aria-label={labels.priority}
                              placeholder="1"
                              aria-invalid={priorityError !== '' || undefined}
                              aria-describedby={
                                priorityError !== '' ? `${priorityId}-error` : undefined
                              }
                              data-model-priority-index={render.index}
                              spellCheck={false}
                              disabled={disabled}
                              onChange={(event) =>
                                updateRouteCount(render.index, 'priority', event.target.value)
                              }
                            />
                            {priorityError !== '' && (
                              <FieldError id={priorityId} error={priorityError} />
                            )}
                          </span>
                        )
                      })()}
                    </>
                  )}
                </div>
                <div {...stylex.props(styles.routeMeta)}>
                  {(render.item.weight ?? 1) === 0 ? (
                    <span {...stylex.props(styles.routeFlag)}>{labels.weightDisabled}</span>
                  ) : (
                    <span {...stylex.props(styles.routeShare)}>{sharePercent(render.index)}</span>
                  )}
                </div>
              </div>

              <div {...stylex.props(ledgerRecordStyles.cell, styles.thirdColumn)} role="cell">
                <span {...stylex.props(styles.mobileLabel)}>{labels.thirdColumn}</span>
                {renderThirdColumn?.(render.item, render.index)}
              </div>

              <div {...stylex.props(ledgerRecordStyles.cell, styles.actions)} role="cell">
                <IconButton
                  variant="ghost"
                  size="sm"
                  isDisabled={disabled}
                  label={labels.removeFor(render.item.id || labels.manualId)}
                  icon={<X size={16} aria-hidden="true" />}
                  onClick={() => removeRow(render.index)}
                />
              </div>
            </article>
          ),
        )}

        {visibleRowCount === 0 && (
          <article
            {...stylex.props(ledgerRecordStyles.record, styles.empty)}
            role="row"
            aria-rowindex={2}
          >
            <span {...stylex.props(ledgerRecordStyles.cell, styles.emptyCell)} role="cell">
              {value.length ? labels.noMatches : labels.empty}
            </span>
          </article>
        )}
      </LedgerRecordList>

      {addable && (
        <Button
          variant="ghost"
          size="sm"
          xstyle={styles.add}
          isDisabled={disabled || !createRow}
          icon={<Plus size={16} aria-hidden="true" />}
          label={labels.addInline}
          onClick={() => void addManual()}
        />
      )}
    </div>
  )
}
