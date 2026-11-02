import { Button } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { ArrowDown, Plus } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'

import type {
  AccessProtocol,
  GroupModelItemDto,
  ParameterOverrideRuleDto,
} from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import { expandParameterSet, flattenParameterSet } from '@shared/lib/parameter-paths'
import {
  inferValueKind,
  parameterValueKinds,
  type ParameterValueKind,
} from '@shared/lib/parameter-values'

import { useT } from '../../../app/i18n'
import { ParameterOverrideRuleCard } from './ParameterOverrideRuleCard'
import {
  collectErrors,
  createRows,
  createSetParam,
  matchCount,
  rulesValid,
  serializeRows,
  setEntries,
  summarizeRules,
  switchBlockedByRule,
  tryParseSetText,
  type ParamRow,
  type RuleRow,
} from './parameter-override-rules'

export function ParameterOverrideRulesEditor({
  value,
  protocols,
  models,
  disabled = false,
  onChange,
  onValidChange,
  onInvalidEditsChange,
  resetKey = 0,
}: {
  value: ParameterOverrideRuleDto[]
  protocols: AccessProtocol[]
  models: GroupModelItemDto[]
  disabled?: boolean
  onChange(value: ParameterOverrideRuleDto[]): void
  onValidChange(value: boolean): void
  onInvalidEditsChange(value: boolean): void
  /** 与 classic `:key="parameterOverridesEditorRevision"` 等价：变化即重建行状态。 */
  resetKey?: number
}) {
  const t = useT()
  const instanceId = useId()
  // Handler-allocated keys come from this ref (event handlers may mutate refs).
  // Seeded rows get deterministic negative keys derived from their index —
  // createRows also runs during render (useState init + resetKey adjustment),
  // where ref access is forbidden.
  const nextKeyRef = useRef(1)

  function newKey(): number {
    return nextKeyRef.current++
  }

  const [rows, setRows] = useState<RuleRow[]>(() => createRows(value))
  const [moveAnnouncement, setMoveAnnouncement] = useState('')
  const [lastResetKey, setLastResetKey] = useState(resetKey)
  if (lastResetKey !== resetKey) {
    setLastResetKey(resetKey)
    setRows(createRows(value))
  }

  const modelSuggestions = [...new Set(models.map(({ client_model }) => client_model))].sort(
    (left, right) => left.localeCompare(right),
  )
  const availableProtocols = (() => {
    const values = new Set<AccessProtocol>(protocols)
    for (const row of rows) {
      if (row.protocol) values.add(row.protocol as AccessProtocol)
    }
    return [...values]
  })()
  const protocolOptions = [
    { value: '', label: t('group.settings.parameterOverrides.allProtocols') },
    ...availableProtocols.map((value) => ({ value, label: value })),
  ]
  const opOptions = [
    { value: 'set', label: t('group.settings.parameterOverrides.opSet') },
    { value: 'remove', label: t('group.settings.parameterOverrides.opRemove') },
  ]
  const kindOptions = parameterValueKinds.map((kind) => ({
    value: kind,
    label: t(`group.settings.parameterOverrides.kind.${kind}` as MessageId),
  }))
  const booleanOptions = [
    { value: 'true', label: 'true' },
    { value: 'false', label: 'false' },
  ]
  const ruleOptions = {
    protocols: protocolOptions,
    operations: opOptions,
    kinds: kindOptions,
    booleans: booleanOptions,
  }

  /** Card copy never needs interpolation, so it gets the message id alone. */
  function translate(id: MessageId): string {
    return t(id)
  }

  const errorsByRule = collectErrors(rows, t)
  const valid = rulesValid(errorsByRule)
  const summaryByRule = summarizeRules(rows)

  const publishRef = useRef({ onChange, onValidChange, onInvalidEditsChange, rows })
  useEffect(() => {
    publishRef.current = { onChange, onValidChange, onInvalidEditsChange, rows }
  })

  useEffect(() => {
    const { onChange, onValidChange, onInvalidEditsChange, rows: current } = publishRef.current
    onValidChange(valid)
    onInvalidEditsChange(!valid)
    if (valid) onChange(serializeRows(current))
    // valid/serializeRows derive from rows; publish mirrors the classic
    // watch(rows, { deep, immediate }) emission contract.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows])

  function mutateRows(mutate: (draft: RuleRow[]) => void): void {
    setRows((current) => {
      const next = current.map((row) => ({
        ...row,
        params: row.params.map((param) => ({ ...param })),
      }))
      mutate(next)
      return next
    })
  }

  function updateRule(key: number, patch: Partial<RuleRow>): void {
    mutateRows((draft) => {
      const row = draft.find((item) => item.key === key)
      if (row) Object.assign(row, patch)
    })
  }

  function updateParam(rowKey: number, paramKey: number, patch: Partial<ParamRow>): void {
    mutateRows((draft) => {
      const param = draft
        .find((item) => item.key === rowKey)
        ?.params.find((item) => item.key === paramKey)
      if (param) Object.assign(param, patch)
    })
  }

  function addRule(): void {
    setRows((current) => [
      ...current,
      {
        key: newKey(),
        open: true,
        protocol: '',
        model: '',
        json: false,
        params: [
          { key: newKey(), op: 'set', path: '', valueText: '', kind: 'text', kindPinned: false },
        ],
        setText: '',
      },
    ])
  }

  function copyRule(index: number): void {
    setRows((current) => {
      const source = current[index]
      if (!source) return current
      const next = [...current]
      next.splice(index + 1, 0, {
        key: newKey(),
        open: true,
        protocol: source.protocol,
        model: source.model,
        json: source.json,
        params: source.params.map((param) => ({ ...param, key: newKey() })),
        setText: source.setText,
      })
      return next
    })
  }

  function moveRule(index: number, offset: -1 | 1): void {
    setRows((current) => {
      const target = index + offset
      if (target < 0 || target >= current.length) return current
      const next = [...current]
      const [row] = next.splice(index, 1)
      if (!row) return current
      next.splice(target, 0, row)
      return next
    })
    setMoveAnnouncement(
      t('group.settings.parameterOverrides.moved', { position: index + offset + 1 }),
    )
  }

  function removeRule(index: number): void {
    setRows((current) => current.filter((_, rowIndex) => rowIndex !== index))
  }

  /** 新增行追加末尾，切换动作原地不动：行序不参与语义，不该跳走。 */
  function addParam(row: RuleRow): void {
    mutateRows((draft) => {
      const target = draft.find((item) => item.key === row.key)
      target?.params.push({
        key: newKey(),
        op: row.json ? 'remove' : 'set',
        path: '',
        valueText: '',
        kind: 'text',
        kindPinned: false,
      })
    })
  }

  function setParamOp(row: RuleRow, param: ParamRow, value: string): void {
    if (value !== 'set' && value !== 'remove') return
    updateParam(row.key, param.key, {
      op: value,
      ...(value === 'remove' ? { valueText: '', kind: 'text' as const, kindPinned: false } : {}),
    })
  }

  function setParamKind(row: RuleRow, param: ParamRow, value: string): void {
    if (!(parameterValueKinds as readonly string[]).includes(value)) return
    const kind = value as ParameterValueKind
    updateParam(row.key, param.key, {
      kind,
      kindPinned: true,
      ...(kind === 'boolean' && param.valueText.trim() !== 'false' ? { valueText: 'true' } : {}),
    })
  }

  function setParamValue(row: RuleRow, param: ParamRow, text: string): void {
    updateParam(row.key, param.key, {
      valueText: text,
      ...(param.kindPinned ? {} : { kind: inferValueKind(text) }),
    })
  }

  function removeParam(row: RuleRow, key: number): void {
    mutateRows((draft) => {
      const target = draft.find((item) => item.key === row.key)
      if (target) target.params = target.params.filter((param) => param.key !== key)
    })
  }

  const switchBlocked = switchBlockedByRule(rows, errorsByRule)

  function toggleJSON(row: RuleRow): void {
    if (switchBlocked.get(row.key)) return
    if (row.json) {
      const parsed = tryParseSetText(row.setText)
      if (!parsed) return
      updateRule(row.key, {
        params: [
          ...flattenParameterSet(parsed).map((entry) => createSetParam(entry, newKey())),
          ...row.params.filter(({ op }) => op === 'remove'),
        ],
        json: false,
      })
      return
    }
    const entries = setEntries(row)
    if (entries === undefined) return
    const set = expandParameterSet(entries)
    updateRule(row.key, {
      setText: Object.keys(set).length > 0 ? JSON.stringify(set, null, 2) : '',
      json: true,
    })
  }

  function formatSetText(row: RuleRow): void {
    const parsed = tryParseSetText(row.setText)
    if (parsed)
      updateRule(row.key, {
        setText: Object.keys(parsed).length > 0 ? JSON.stringify(parsed, null, 2) : '',
      })
  }

  const modelMatchCounts = new Map(
    rows.map((row) => [row.key, matchCount(row.model, modelSuggestions)] as const),
  )

  return (
    <div {...stylex.props(styles.root)}>
      <div {...stylex.props(styles.bar)}>
        <span {...stylex.props(styles.order)}>
          <ArrowDown size={13} aria-hidden="true" />
          {t('group.settings.parameterOverrides.order')}
        </span>
        <Button
          size="sm"
          isDisabled={disabled}
          onClick={addRule}
          icon={<Plus size={15} aria-hidden="true" />}
          label={t('group.settings.parameterOverrides.addRule')}
        />
      </div>

      <p {...stylex.props(styles.srOnly)} aria-live="polite">
        {moveAnnouncement}
      </p>

      {rows.length === 0 ? (
        <p {...stylex.props(styles.empty)}>{t('group.settings.parameterOverrides.empty')}</p>
      ) : (
        <div {...stylex.props(styles.list)}>
          {rows.map((row, index) => (
            <ParameterOverrideRuleCard
              key={row.key}
              row={row}
              index={index}
              count={rows.length}
              instanceId={instanceId}
              disabled={disabled}
              errors={errorsByRule.get(row.key)}
              summary={summaryByRule.get(row.key)}
              matches={modelMatchCounts.get(row.key)}
              switchBlocked={switchBlocked.get(row.key) === true}
              options={ruleOptions}
              tr={translate}
              onToggleOpen={(target) => updateRule(target.key, { open: !target.open })}
              onUpdateRule={updateRule}
              onUpdateParam={updateParam}
              onSetParamOp={setParamOp}
              onSetParamKind={setParamKind}
              onSetParamValue={setParamValue}
              onRemoveParam={removeParam}
              onAddParam={addParam}
              onToggleJSON={toggleJSON}
              onFormatSetText={formatSetText}
              onMove={moveRule}
              onCopy={copyRule}
              onRemove={removeRule}
            />
          ))}
        </div>
      )}

      <datalist id={`${instanceId}-models`}>
        {modelSuggestions.map((model) => (
          <option key={model} value={model} />
        ))}
      </datalist>
    </div>
  )
}

const styles = stylex.create({
  bar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-4)',
  },
  empty: {
    margin: 0,
    borderWidth: '1px',
    borderStyle: 'dashed',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingTop: '12px',
    paddingBottom: '12px',
    paddingLeft: '14px',
    paddingRight: '14px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
  },
  list: {
    display: 'grid',
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
  },
  order: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '5px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  root: {
    display: 'grid',
    gap: '9px',
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
