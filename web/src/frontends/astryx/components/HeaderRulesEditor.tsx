import { Tooltip } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { CircleAlert, Plus, X } from 'lucide-react'
import { useEffect, useId, useMemo, useState } from 'react'

import { useT } from '../app/i18n'
import type { HeaderRulesDto } from '@shared/control/types'
import {
  validateHeaderRuleRows,
  type HeaderRuleAction,
  type HeaderRuleValidationError,
  type HeaderRuleValidationPolicy,
} from '@shared/domain/settings/header-rules-validation'

interface RuleRow {
  key: number
  action: HeaderRuleAction
  name: string
  value: string
}

// Module-scoped so render-phase row rebuilds (StrictMode double-invocation,
// render-time adjustments) can burn keys freely — rows only need per-list
// uniqueness, not per-instance sequences.
let nextRowKey = 1

const narrow = '@media (max-width: 800px)'
const compact = '@media (max-width: 520px)'

const styles = stylex.create({
  // Ledger appearance — the only variant either classic consumer uses
  // (settings + group settings). The 'default' appearance from the classic
  // editor is dead code there and intentionally not ported.
  root: {
    display: 'grid',
    gap: '11px',
  },
  rows: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  rule: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(120px, 0.8fr) auto minmax(0, 1.2fr) 32px',
      [compact]: 'minmax(0, 1fr) 32px',
    },
    gap: 'var(--space-2)',
    alignItems: 'center',
  },
  field: {
    position: 'relative',
    minWidth: 0,
    gridColumnStart: { [compact]: 1 },
    gridRow: { [compact]: 'auto' },
  },
  input: {
    width: '100%',
    minHeight: { default: 'var(--control-xs)', [narrow]: 'var(--touch-target)' },
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingBlock: '6px',
    paddingInline: '9px',
    fontFamily: 'var(--font-mono)',
    fontSize: { default: 'var(--text-meta)', [narrow]: '16px' },
  },
  inputInvalid: {
    borderColor: 'var(--color-danger)',
    paddingInlineEnd: '26px',
  },
  errorIndicator: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    insetInlineEnd: 0,
    display: 'inline-flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '24px',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-danger)',
    padding: 0,
    cursor: 'help',
  },
  mode: {
    display: 'inline-flex',
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    gridColumnStart: { [compact]: 1 },
    gridRow: { [compact]: 'auto' },
  },
  modeButton: {
    minWidth: '52px',
    minHeight: { default: 'var(--control-xs)', [narrow]: 'var(--touch-target)' },
    borderWidth: 0,
    borderLeftWidth: { default: '1px', ':first-child': 0 },
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text-faint)',
    paddingBlock: '5px',
    paddingInline: '8px',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
  },
  modeButtonSet: {
    backgroundColor: 'var(--color-text)',
    color: 'var(--color-surface)',
    fontWeight: 560,
  },
  modeButtonRemove: {
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
  },
  removeHint: {
    display: 'flex',
    alignItems: 'center',
    borderWidth: '1px',
    borderStyle: 'dashed',
    borderColor: 'color-mix(in srgb, var(--color-warning) 46%, var(--color-border-subtle))',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
    paddingBlock: '6px',
    paddingInline: '9px',
    fontFamily: 'var(--font-mono)',
    fontSize: { default: 'var(--text-meta)', [narrow]: '16px' },
    lineHeight: 1.4,
    minHeight: { default: 'var(--control-xs)', [narrow]: 'var(--touch-target)' },
    gridColumnStart: { [compact]: 1 },
    gridRow: { [compact]: 'auto' },
  },
  iconButton: {
    display: 'inline-flex',
    minWidth: { default: '32px', [narrow]: 'var(--touch-target)' },
    minHeight: { default: '32px', [narrow]: 'var(--touch-target)' },
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'transparent',
    color: 'var(--color-action)',
    padding: 0,
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
    gridColumnStart: { [compact]: 2 },
    gridRowStart: { [compact]: 1 },
  },
  add: {
    display: 'inline-flex',
    width: 'fit-content',
    minWidth: 0,
    minHeight: { default: 'var(--control-md)', [narrow]: 'var(--touch-target)' },
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: 'var(--space-2)',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-action)',
    marginTop: '2px',
    paddingBlock: '5px',
    paddingInline: '1px',
    fontWeight: 650,
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
  },
})

function compareHeaderNames(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0
}

function normalizeRules(value: HeaderRulesDto): HeaderRulesDto {
  const set = Object.fromEntries(
    Object.entries(value.set).sort(([left], [right]) => compareHeaderNames(left, right)),
  )
  const remove = [...value.remove].sort(compareHeaderNames)
  return { set, remove }
}

function rulesFromRows(rows: readonly RuleRow[]): HeaderRulesDto {
  const rules: HeaderRulesDto = { set: {}, remove: [] }
  for (const row of rows) {
    if (row.action === 'set') rules.set[row.name] = row.value
    else rules.remove.push(row.name)
  }
  return rules
}

function rowsMatchRules(rows: readonly RuleRow[], rules: HeaderRulesDto): boolean {
  const expected = [
    ...Object.entries(rules.set).map(([name, value]) => ({
      action: 'set' as const,
      name,
      value,
    })),
    ...rules.remove.map((name) => ({ action: 'remove' as const, name, value: '' })),
  ]
  return (
    rows.length === expected.length &&
    rows.every(
      (row, index) =>
        row.action === expected[index].action &&
        row.name === expected[index].name &&
        row.value === expected[index].value,
    )
  )
}

function FieldErrorMarker({ id, error }: { id: string; error: string | undefined }) {
  if (error === undefined) return null
  return (
    <>
      <Tooltip content={error}>
        <button type="button" tabIndex={-1} {...stylex.props(styles.errorIndicator)}>
          <CircleAlert size={13} aria-hidden />
        </button>
      </Tooltip>
      <span id={`${id}-error`} style={{ display: 'none' }}>
        {error}
      </span>
    </>
  )
}

/**
 * Ledger-density header rules editor — the React port of classic's
 * HeaderRulesEditor `appearance="ledger"` (the only appearance in use).
 * Local row state + `publishedRef` mirror Vue's rows/publishedRules split:
 * invalid rows stay editable without touching the outer draft, and the outer
 * `onChange` only ever receives normalized, valid rule sets.
 */
export function HeaderRulesEditor({
  value,
  disabled = false,
  removeLabel,
  removeHint,
  resetKey = 0,
  showAdd = true,
  validationPolicy = 'request',
  onChange,
  onValidChange,
  onInvalidEditsChange,
}: {
  value: HeaderRulesDto
  disabled?: boolean
  removeLabel?: string
  removeHint?: string
  resetKey?: number
  showAdd?: boolean
  validationPolicy?: HeaderRuleValidationPolicy
  onChange: (value: HeaderRulesDto) => void
  onValidChange?: (valid: boolean) => void
  onInvalidEditsChange?: (invalidEdits: boolean) => void
}) {
  const t = useT()
  const scope = useId()

  const [published, setPublished] = useState<HeaderRulesDto>(() => normalizeRules(value))
  const [rows, setRows] = useState<RuleRow[]>(() => createRows(published))

  function createRows(source: HeaderRulesDto): RuleRow[] {
    return [
      ...Object.entries(source.set).map(([name, headerValue]) => ({
        key: nextRowKey++,
        action: 'set' as const,
        name,
        value: headerValue,
      })),
      ...source.remove.map((name) => ({
        key: nextRowKey++,
        action: 'remove' as const,
        name,
        value: '',
      })),
    ]
  }

  const validationErrors = useMemo(
    () =>
      validateHeaderRuleRows(
        rows.map(({ key, action, name, value: rowValue }) => ({
          rowKey: key,
          action,
          name,
          value: rowValue,
        })),
        validationPolicy,
      ),
    [rows, validationPolicy],
  )
  const validationErrorsByRow = useMemo(() => {
    const map = new Map<number, HeaderRuleValidationError>()
    for (const error of validationErrors) {
      if (!map.has(error.rowKey)) map.set(error.rowKey, error)
    }
    return map
  }, [validationErrors])
  const valid = validationErrors.length === 0
  const invalidEdits = validationErrors.length > 0 && !rowsMatchRules(rows, published)

  useEffect(() => onValidChange?.(valid), [onValidChange, valid])
  useEffect(() => onInvalidEditsChange?.(invalidEdits), [onInvalidEditsChange, invalidEdits])

  // External value changes (save responses, discard) rebuild rows unless the
  // payload merely echoes what this editor already published. Render-time
  // adjustment, same idiom as GroupsView's searchStr sync.
  const valueJson = JSON.stringify(normalizeRules(value))
  const [lastValueJson, setLastValueJson] = useState(valueJson)
  if (lastValueJson !== valueJson) {
    setLastValueJson(valueJson)
    if (JSON.stringify(normalizeRules(published)) !== valueJson) {
      const external = normalizeRules(value)
      setPublished(external)
      setRows(createRows(external))
    }
  }

  const [lastResetKey, setLastResetKey] = useState(resetKey)
  if (lastResetKey !== resetKey) {
    setLastResetKey(resetKey)
    setRows(createRows(published))
  }

  const publish = (nextRows: RuleRow[]): void => {
    const errors = validateHeaderRuleRows(
      nextRows.map(({ key, action, name, value: rowValue }) => ({
        rowKey: key,
        action,
        name,
        value: rowValue,
      })),
      validationPolicy,
    )
    if (errors.length > 0) return
    const next = normalizeRules(rulesFromRows(nextRows))
    setPublished(next)
    onChange(next)
  }

  // Side effects stay out of state updaters: compute the next rows, then set
  // state and publish in one step (StrictMode would otherwise double-publish).
  const applyRows = (nextRows: RuleRow[]): void => {
    setRows(nextRows)
    publish(nextRows)
  }

  const mutateRow = (key: number, mutate: (row: RuleRow) => void): void => {
    applyRows(
      rows.map((row) => {
        if (row.key !== key) return row
        const next = { ...row }
        mutate(next)
        return next
      }),
    )
  }

  const addRow = (): void => {
    applyRows([...rows, { key: nextRowKey++, action: 'set' as const, name: '', value: '' }])
  }

  const removeRow = (key: number): void => {
    applyRows(rows.filter((row) => row.key !== key))
  }

  const rowError = (row: RuleRow, field: 'name' | 'value'): string | undefined => {
    const error = validationErrorsByRow.get(row.key)
    if (!error || (field === 'value' && error.code !== 'invalid_value')) return undefined
    if (field === 'value') return t(`common.headerRules.errors.${error.code}`)
    if (error.code === 'invalid_value') return undefined
    return t(`common.headerRules.errors.${error.code}`)
  }

  return (
    <section {...stylex.props(styles.root)} aria-label={t('common.headerRules.title')}>
      {rows.length > 0 && (
        <div {...stylex.props(styles.rows)}>
          {rows.map((row) => {
            const nameId = `${scope}-header-name-${row.key}`
            const valueId = `${scope}-header-value-${row.key}`
            const nameError = rowError(row, 'name')
            const valueError = row.action === 'set' ? rowError(row, 'value') : undefined
            return (
              <div key={row.key} {...stylex.props(styles.rule)}>
                <div {...stylex.props(styles.field)}>
                  <label htmlFor={nameId} style={{ display: 'none' }}>
                    {t('common.headerRules.name')}
                  </label>
                  <input
                    id={nameId}
                    {...stylex.props(styles.input, nameError !== undefined && styles.inputInvalid)}
                    value={row.name}
                    placeholder={t('common.headerRules.name')}
                    autoComplete="off"
                    spellCheck={false}
                    disabled={disabled}
                    aria-invalid={nameError !== undefined || undefined}
                    aria-describedby={nameError !== undefined ? `${nameId}-error` : undefined}
                    onChange={(event) =>
                      mutateRow(row.key, (next) => {
                        next.name = event.target.value
                      })
                    }
                  />
                  <FieldErrorMarker id={nameId} error={nameError} />
                </div>

                <div
                  {...stylex.props(styles.mode)}
                  role="group"
                  aria-label={t('common.headerRules.action')}
                >
                  <button
                    type="button"
                    {...stylex.props(
                      styles.modeButton,
                      row.action === 'set' && styles.modeButtonSet,
                    )}
                    aria-pressed={row.action === 'set'}
                    disabled={disabled}
                    onClick={() =>
                      mutateRow(row.key, (next) => {
                        next.action = 'set'
                      })
                    }
                  >
                    {t('common.headerRules.set')}
                  </button>
                  <button
                    type="button"
                    data-mode="remove"
                    {...stylex.props(
                      styles.modeButton,
                      row.action === 'remove' && styles.modeButtonRemove,
                    )}
                    aria-pressed={row.action === 'remove'}
                    disabled={disabled}
                    onClick={() =>
                      mutateRow(row.key, (next) => {
                        next.action = 'remove'
                        next.value = ''
                      })
                    }
                  >
                    {removeLabel ?? t('common.headerRules.remove')}
                  </button>
                </div>

                {row.action === 'set' ? (
                  <div {...stylex.props(styles.field)}>
                    <label htmlFor={valueId} style={{ display: 'none' }}>
                      {t('common.headerRules.value')}
                    </label>
                    <input
                      id={valueId}
                      type="text"
                      {...stylex.props(
                        styles.input,
                        valueError !== undefined && styles.inputInvalid,
                      )}
                      value={row.value}
                      placeholder={t('common.headerRules.value')}
                      autoComplete="off"
                      spellCheck={false}
                      disabled={disabled}
                      aria-invalid={valueError !== undefined || undefined}
                      aria-describedby={valueError !== undefined ? `${valueId}-error` : undefined}
                      onChange={(event) =>
                        mutateRow(row.key, (next) => {
                          next.value = event.target.value
                        })
                      }
                    />
                    <FieldErrorMarker id={valueId} error={valueError} />
                  </div>
                ) : (
                  <span {...stylex.props(styles.removeHint)}>
                    {removeHint ?? t('common.headerRules.removeHint')}
                  </span>
                )}

                <button
                  type="button"
                  {...stylex.props(styles.iconButton)}
                  aria-label={t('common.headerRules.delete')}
                  disabled={disabled}
                  onClick={() => removeRow(row.key)}
                >
                  <X size={16} aria-hidden />
                </button>
              </div>
            )
          })}
        </div>
      )}
      {showAdd && (
        <button type="button" {...stylex.props(styles.add)} disabled={disabled} onClick={addRow}>
          <Plus size={16} aria-hidden />
          {t('common.headerRules.add')}
        </button>
      )}
    </section>
  )
}
