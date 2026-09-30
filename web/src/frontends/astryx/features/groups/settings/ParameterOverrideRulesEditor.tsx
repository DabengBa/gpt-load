import { Button, IconButton, Selector, Tooltip } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { ArrowDown, ArrowUp, ChevronDown, Copy, Plus, Trash2, X } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'

import type {
  AccessProtocol,
  GroupModelItemDto,
  ParameterJSONValue,
  ParameterOverrideRuleDto,
} from '@shared/control/types'
import { InlineNotice } from '../../../components/InlineNotice'
import { assertJSONNumbersRoundTrip, JSONNumberPrecisionError } from '@shared/lib/json-number'
import {
  decodeParameterPath,
  expandParameterSet,
  flattenParameterSet,
  fromParameterPointer,
  parameterPathsCross,
  toParameterPointer,
  type ParameterPathEntry,
} from '@shared/lib/parameter-paths'
import {
  formatValueText,
  hasEmptyParameterKey,
  inferValueKind,
  parameterValueKinds,
  ParameterValueError,
  tryValueFromText,
  valueFromText,
  valueKind,
  type ParameterValueKind,
} from '@shared/lib/parameter-values'
import { useT } from '../../../app/i18n'
import type { MessageId } from '@shared/i18n/message-ids'

type ParamOp = 'set' | 'remove'

/** 一行参数动作。行序只是录入次序，不参与语义。 */
interface ParamRow {
  key: number
  op: ParamOp
  path: string
  valueText: string
  kind: ParameterValueKind
  /** 用户动过类型列之后就不再跟随输入推断，否则一打字又被改回去。 */
  kindPinned: boolean
}

interface RuleRow {
  key: number
  open: boolean
  protocol: string
  model: string
  /** true 时 set 用整段 JSON 编辑，remove 仍是行。 */
  json: boolean
  params: ParamRow[]
  setText: string
}

interface RuleErrors {
  model?: string
  set?: string
  action?: string
  params: Map<number, string>
}

interface SummaryChip {
  key: string
  op: ParamOp
  path: string
  value: string
}

interface RuleSummary {
  chips: SummaryChip[]
  overflow: number
}

/** 折叠行放不下太多摘要；超出的用 +N 兜住，避免静默裁切成看不见。 */
const summaryLimit = 3

const forbiddenRootFields = ['model', 'stream', 'store']

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

  function createSetParam({ path, value }: ParameterPathEntry, key: number): ParamRow {
    return {
      key,
      op: 'set',
      path,
      valueText: formatValueText(value),
      kind: valueKind(value),
      kindPinned: false,
    }
  }

  function createRows(source: ParameterOverrideRuleDto[]): RuleRow[] {
    return source.map((rule, ruleIndex) => {
      const setEntries = (rule.set ? flattenParameterSet(rule.set) : []).map((entry, paramIndex) =>
        createSetParam(entry, -(ruleIndex * 1_000_000 + paramIndex + 1)),
      )
      const removeEntries = (rule.remove ?? []).map((pointer, paramIndex) => ({
        key: -(ruleIndex * 1_000_000 + setEntries.length + paramIndex + 1),
        op: 'remove' as const,
        path: fromParameterPointer(pointer),
        valueText: '',
        kind: 'text' as const,
        kindPinned: false,
      }))
      return {
        key: -(ruleIndex + 1),
        open: false,
        protocol: rule.match.protocol ?? '',
        model: rule.match.model ?? '',
        json: false,
        params: [...setEntries, ...removeEntries],
        setText: rule.set ? JSON.stringify(rule.set, null, 2) : '',
      }
    })
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

  function parseSetText(text: string): Record<string, ParameterJSONValue> | undefined {
    const trimmed = text.trim()
    if (!trimmed) return {}
    const parsed: unknown = JSON.parse(trimmed)
    assertJSONNumbersRoundTrip(trimmed)
    if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined
    if (hasEmptyParameterKey(parsed)) throw new ParameterValueError('empty-key')
    return parsed as Record<string, ParameterJSONValue>
  }

  function tryParseSetText(text: string): Record<string, ParameterJSONValue> | undefined {
    try {
      return parseSetText(text)
    } catch {
      return undefined
    }
  }

  /** 刚加出来还没填的行：不标红、不阻止保存，序列化时直接忽略。 */
  function isBlankParam(param: ParamRow): boolean {
    if (param.path.trim()) return false
    return param.op === 'remove' || param.kind === 'null' || !param.valueText.trim()
  }

  function setEntries(row: RuleRow): ParameterPathEntry[] | undefined {
    if (row.json) {
      const parsed = tryParseSetText(row.setText)
      return parsed ? flattenParameterSet(parsed) : undefined
    }
    const entries: ParameterPathEntry[] = []
    for (const param of row.params) {
      if (param.op !== 'set' || isBlankParam(param)) continue
      const paramValue = tryValueFromText(param.valueText, param.kind)
      if (paramValue === undefined) return undefined
      entries.push({ path: param.path, value: paramValue })
    }
    return entries
  }

  function removePointers(row: RuleRow): string[] {
    return row.params
      .filter((param) => param.op === 'remove' && !isBlankParam(param))
      .map(({ path }) => toParameterPointer(path))
  }

  function pathError(row: ParamRow, siblings: ParamRow[]): string {
    if (!row.path.trim()) return t('group.settings.parameterOverrides.errors.pathRequired')
    const segments = decodeParameterPath(row.path, row.op)
    if (!segments) return t('group.settings.parameterOverrides.errors.pathInvalid')
    if (forbiddenRootFields.includes((segments[0] ?? '').toLowerCase()))
      return t('group.settings.parameterOverrides.errors.forbiddenField')
    for (const other of siblings) {
      if (other.key === row.key) continue
      if (other.op === row.op && other.path === row.path)
        return t('group.settings.parameterOverrides.errors.pathDuplicate')
      // 同为 set 时祖先与后代互斥：还原成嵌套对象会互相顶掉。
      if (row.op === 'set' && other.op === 'set' && parameterPathsCross(row.path, other.path))
        return t('group.settings.parameterOverrides.errors.pathAncestor')
    }
    return ''
  }

  function valueError(row: ParamRow): string {
    // 空值类型没有值可填；文本类型允许空字符串。
    if (row.kind === 'null') return ''
    if (!row.valueText.trim() && row.kind !== 'text')
      return t('group.settings.parameterOverrides.errors.valueRequired')
    try {
      valueFromText(row.valueText, row.kind)
      return ''
    } catch (cause: unknown) {
      if (cause instanceof JSONNumberPrecisionError)
        return t('group.settings.parameterOverrides.errors.unsafeNumber')
      if (cause instanceof ParameterValueError && cause.message === 'number')
        return t('group.settings.parameterOverrides.errors.valueNumber')
      if (cause instanceof ParameterValueError && cause.message === 'boolean')
        return t('group.settings.parameterOverrides.errors.valueBoolean')
      if (cause instanceof ParameterValueError && cause.message === 'empty-key')
        return t('group.settings.parameterOverrides.errors.emptyKey')
      return t('group.settings.parameterOverrides.errors.valueJSON', {
        brace: '{',
        bracket: '[',
      })
    }
  }

  function ruleErrors(row: RuleRow): RuleErrors {
    const errors: RuleErrors = { params: new Map() }
    const model = row.model.trim()
    if (model.includes('*') && !/^[^*]+\*$/u.test(model)) {
      errors.model = t('group.settings.parameterOverrides.errors.modelPattern')
    }

    const visible = row.json ? row.params.filter(({ op }) => op === 'remove') : row.params
    const filled = visible.filter((param) => !isBlankParam(param))
    for (const param of filled) {
      const error = pathError(param, filled) || (param.op === 'set' ? valueError(param) : '')
      if (error) errors.params.set(param.key, error)
    }

    let parsedSet: Record<string, ParameterJSONValue> | undefined
    if (row.json) {
      try {
        parsedSet = parseSetText(row.setText)
        if (!parsedSet) {
          errors.set = t('group.settings.parameterOverrides.errors.setObject')
        } else {
          const roots = new Set(
            flattenParameterSet(parsedSet).map(({ path }) =>
              (decodeParameterPath(path, 'set')?.[0] ?? '').toLowerCase(),
            ),
          )
          if (forbiddenRootFields.some((field) => roots.has(field)))
            errors.set = t('group.settings.parameterOverrides.errors.forbiddenField')
        }
      } catch (cause: unknown) {
        errors.set = t(
          cause instanceof JSONNumberPrecisionError
            ? 'group.settings.parameterOverrides.errors.unsafeNumber'
            : cause instanceof ParameterValueError && cause.message === 'empty-key'
              ? 'group.settings.parameterOverrides.errors.emptyKey'
              : 'group.settings.parameterOverrides.errors.invalidJSON',
        )
      }
    }

    const hasSet = row.json
      ? Object.keys(parsedSet ?? {}).length > 0
      : filled.some(({ op }) => op === 'set')
    const hasRemove = row.params.some((param) => param.op === 'remove' && !isBlankParam(param))
    if (!hasSet && !hasRemove) {
      errors.action = t('group.settings.parameterOverrides.errors.actionRequired')
    }
    return errors
  }

  const errorsByRule = new Map(rows.map((row) => [row.key, ruleErrors(row)] as const))
  const valid = [...errorsByRule.values()].every(
    (errors) => !errors.model && !errors.set && !errors.action && errors.params.size === 0,
  )

  function ruleInvalid(row: RuleRow): boolean {
    const errors = errorsByRule.get(row.key)
    return Boolean(
      errors && (errors.model || errors.set || errors.action || errors.params.size > 0),
    )
  }

  /** 设置与删除只在路径相交时才互相影响，这时后端固定先删后设。 */
  function pathsCrossing(row: RuleRow): boolean {
    const removes = row.params.filter(({ op }) => op === 'remove').map(({ path }) => path)
    if (removes.length === 0) return false
    const sets = (setEntries(row) ?? []).map(({ path }) => path)
    return removes.some((left) => sets.some((right) => parameterPathsCross(left, right)))
  }

  function previewValue(value: ParameterJSONValue): string {
    return typeof value === 'string' ? value : formatValueText(value)
  }

  function buildSummary(row: RuleRow): SummaryChip[] {
    const chips: SummaryChip[] = []
    if (row.json) {
      for (const { path, value } of setEntries(row) ?? []) {
        chips.push({ key: `s${path}`, op: 'set', path, value: previewValue(value) })
      }
      for (const param of row.params) {
        if (param.op === 'remove')
          chips.push({ key: `r${param.key}`, op: 'remove', path: param.path, value: '' })
      }
      return chips
    }
    for (const param of row.params) {
      const paramValue =
        param.op === 'set' ? tryValueFromText(param.valueText, param.kind) : undefined
      chips.push({
        key: `p${param.key}`,
        op: param.op,
        path: param.path,
        value: paramValue === undefined ? '' : previewValue(paramValue),
      })
    }
    return chips
  }

  const summaryByRule = new Map<number, RuleSummary>(
    rows.map((row) => {
      const summary = buildSummary(row)
      return [
        row.key,
        {
          chips: summary.slice(0, summaryLimit),
          overflow: Math.max(0, summary.length - summaryLimit),
        },
      ] as const
    }),
  )

  function serializeRows(): ParameterOverrideRuleDto[] {
    return rows.map((row) => {
      const protocol = row.protocol as AccessProtocol
      const model = row.model.trim()
      const set = expandParameterSet(setEntries(row) ?? [])
      const remove = removePointers(row)
      return {
        match: {
          ...(protocol ? { protocol } : {}),
          ...(model ? { model } : {}),
        },
        ...(Object.keys(set).length > 0 ? { set } : {}),
        ...(remove.length > 0 ? { remove } : {}),
      }
    })
  }

  const publishRef = useRef({ onChange, onValidChange, onInvalidEditsChange, serializeRows })
  useEffect(() => {
    publishRef.current = { onChange, onValidChange, onInvalidEditsChange, serializeRows }
  })

  useEffect(() => {
    const { onChange, onValidChange, onInvalidEditsChange, serializeRows } = publishRef.current
    onValidChange(valid)
    onInvalidEditsChange(!valid)
    if (valid) onChange(serializeRows())
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

  const switchBlocked = new Map(
    rows.map((row) => {
      const errors = errorsByRule.get(row.key)
      const blocked = row.json
        ? Boolean(errors?.set) || tryParseSetText(row.setText) === undefined
        : setEntries(row) === undefined ||
          [...(errors?.params ?? [])].some(([key]) =>
            row.params.some((param) => param.key === key && param.op === 'set'),
          )
      return [row.key, blocked] as const
    }),
  )

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

  function protocolLabel(value: string): string {
    return value || t('group.settings.parameterOverrides.allProtocols')
  }

  function modelLabel(row: RuleRow): string {
    return row.model.trim() || t('group.settings.parameterOverrides.allModels')
  }

  function matchCount(model: string): number | undefined {
    const text = model.trim()
    if (!text || (text.includes('*') && !/^[^*]+\*$/u.test(text))) return undefined
    const prefix = text.endsWith('*') ? text.slice(0, -1) : undefined
    return modelSuggestions.filter((candidate) =>
      prefix === undefined ? candidate === text : candidate.startsWith(prefix),
    ).length
  }

  const modelMatchCounts = new Map(rows.map((row) => [row.key, matchCount(row.model)] as const))

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
          {rows.map((row, index) => {
            const errors = errorsByRule.get(row.key)
            const summary = summaryByRule.get(row.key)
            const matches = modelMatchCounts.get(row.key)
            return (
              <article
                key={row.key}
                {...stylex.props(styles.rule, ruleInvalid(row) && styles.ruleInvalid)}
              >
                <div {...stylex.props(styles.ruleHead)}>
                  <button
                    type="button"
                    {...stylex.props(styles.summaryButton)}
                    aria-expanded={row.open}
                    aria-controls={`${instanceId}-rule-${row.key}`}
                    onClick={() => updateRule(row.key, { open: !row.open })}
                  >
                    <ChevronDown
                      size={15}
                      aria-hidden="true"
                      {...stylex.props(styles.chevron, row.open && styles.chevronOpen)}
                    />
                    <span {...stylex.props(styles.index)}>{index + 1}</span>
                    <span {...stylex.props(styles.match)}>
                      <span {...stylex.props(styles.tag, !row.protocol && styles.tagAny)}>
                        {protocolLabel(row.protocol)}
                      </span>
                      <span {...stylex.props(styles.tag, !row.model.trim() && styles.tagAny)}>
                        {modelLabel(row)}
                      </span>
                    </span>
                    <span {...stylex.props(styles.summaryTail)}>
                      <span {...stylex.props(styles.chips)}>
                        {summary?.chips.map((chip) => (
                          <span
                            key={chip.key}
                            {...stylex.props(styles.chip, chip.op === 'remove' && styles.chipDrop)}
                          >
                            {chip.op === 'remove' && (
                              <span {...stylex.props(styles.chipMark)} aria-hidden="true">
                                −
                              </span>
                            )}
                            <b>{chip.path}</b>
                            {chip.op === 'set' && (
                              <span {...stylex.props(styles.chipValue)}>{chip.value}</span>
                            )}
                          </span>
                        ))}
                      </span>
                      {summary !== undefined && summary.overflow > 0 && (
                        <span {...stylex.props(styles.chipMore)}>+{summary.overflow}</span>
                      )}
                    </span>
                  </button>
                  <div {...stylex.props(styles.tools)}>
                    <IconButton
                      variant="ghost"
                      size="sm"
                      label={t('group.settings.parameterOverrides.moveUp')}
                      isDisabled={disabled || index === 0}
                      onClick={() => moveRule(index, -1)}
                      icon={<ArrowUp size={15} aria-hidden="true" />}
                    />
                    <IconButton
                      variant="ghost"
                      size="sm"
                      label={t('group.settings.parameterOverrides.moveDown')}
                      isDisabled={disabled || index === rows.length - 1}
                      onClick={() => moveRule(index, 1)}
                      icon={<ArrowDown size={15} aria-hidden="true" />}
                    />
                    <span {...stylex.props(styles.toolsGap)} aria-hidden="true" />
                    <IconButton
                      variant="ghost"
                      size="sm"
                      label={t('group.settings.parameterOverrides.copy')}
                      isDisabled={disabled}
                      onClick={() => copyRule(index)}
                      icon={<Copy size={15} aria-hidden="true" />}
                    />
                    <IconButton
                      variant="ghost"
                      size="sm"
                      label={t('group.settings.parameterOverrides.deleteRule')}
                      isDisabled={disabled}
                      onClick={() => removeRule(index)}
                      icon={<Trash2 size={15} aria-hidden="true" />}
                      xstyle={styles.dangerIcon}
                    />
                  </div>
                </div>

                {row.open && (
                  <div id={`${instanceId}-rule-${row.key}`} {...stylex.props(styles.ruleBody)}>
                    <div {...stylex.props(styles.field)}>
                      <span {...stylex.props(styles.fieldLabel)}>
                        {t('group.settings.parameterOverrides.match')}
                      </span>
                      <div {...stylex.props(styles.matchInputs)}>
                        <Selector
                          label={t('group.settings.parameterOverrides.protocol')}
                          isLabelHidden
                          options={protocolOptions}
                          value={row.protocol}
                          size="sm"
                          isDisabled={disabled}
                          onChange={(value) => updateRule(row.key, { protocol: value })}
                          xstyle={styles.protocolSelect}
                        />
                        <span {...stylex.props(styles.modelCell)}>
                          <input
                            {...stylex.props(
                              styles.input,
                              errors?.model !== undefined && styles.inputInvalid,
                            )}
                            id={`${instanceId}-model-${row.key}`}
                            value={row.model}
                            list={`${instanceId}-models`}
                            placeholder={t('group.settings.parameterOverrides.allModels')}
                            aria-label={t('group.settings.parameterOverrides.model')}
                            aria-invalid={errors?.model !== undefined ? true : undefined}
                            disabled={disabled}
                            autoComplete="off"
                            onChange={(event) => updateRule(row.key, { model: event.target.value })}
                          />
                          {errors?.model !== undefined && (
                            <small {...stylex.props(styles.errorText)} role="alert">
                              {errors.model}
                            </small>
                          )}
                        </span>
                        {matches !== undefined && (
                          <span {...stylex.props(styles.count, matches === 0 && styles.countZero)}>
                            {matches === 0
                              ? t('group.settings.parameterOverrides.modelNoMatch')
                              : t('group.settings.parameterOverrides.modelMatch', {
                                  count: matches,
                                })}
                          </span>
                        )}
                      </div>
                    </div>

                    <div {...stylex.props(styles.field, styles.fieldParams)}>
                      <span {...stylex.props(styles.fieldLabel)}>
                        {t('group.settings.parameterOverrides.params')}
                      </span>
                      <div {...stylex.props(styles.paramsStack)}>
                        {row.json && (
                          <div>
                            <textarea
                              {...stylex.props(
                                styles.input,
                                styles.textarea,
                                errors?.set !== undefined && styles.inputInvalid,
                              )}
                              id={`${instanceId}-set-${row.key}`}
                              value={row.setText}
                              aria-label={t('group.settings.parameterOverrides.setLabel')}
                              aria-invalid={errors?.set !== undefined ? true : undefined}
                              disabled={disabled}
                              spellCheck={false}
                              onChange={(event) =>
                                updateRule(row.key, { setText: event.target.value })
                              }
                            />
                            {errors?.set !== undefined && (
                              <small {...stylex.props(styles.errorText)} role="alert">
                                {errors.set}
                              </small>
                            )}
                          </div>
                        )}

                        {(!row.json || row.params.some(({ op }) => op === 'remove')) && (
                          <div {...stylex.props(styles.params)}>
                            {row.params
                              .filter(({ op }) => !row.json || op === 'remove')
                              .map((param) => {
                                const paramError = errors?.params.get(param.key)
                                return (
                                  <div
                                    key={param.key}
                                    {...stylex.props(
                                      styles.param,
                                      param.op === 'remove' && styles.paramDrop,
                                    )}
                                  >
                                    {/* JSON 视图下 set 归 textarea，行只可能是删除，下拉就成了死控件。 */}
                                    {row.json ? (
                                      <span {...stylex.props(styles.opStatic)}>
                                        {t('group.settings.parameterOverrides.opRemove')}
                                      </span>
                                    ) : (
                                      <span {...stylex.props(styles.op)}>
                                        <Selector
                                          label={t('group.settings.parameterOverrides.paramAction')}
                                          isLabelHidden
                                          options={opOptions}
                                          value={param.op}
                                          size="sm"
                                          isDisabled={disabled}
                                          onChange={(value) => setParamOp(row, param, value)}
                                        />
                                      </span>
                                    )}
                                    <span {...stylex.props(styles.pathCell)}>
                                      <input
                                        {...stylex.props(
                                          styles.input,
                                          styles.path,
                                          paramError !== undefined && styles.inputInvalid,
                                        )}
                                        id={`${instanceId}-path-${param.key}`}
                                        value={param.path}
                                        placeholder={
                                          param.op === 'remove'
                                            ? 'generationConfig/topP'
                                            : 'thinking/type'
                                        }
                                        aria-label={t(
                                          'group.settings.parameterOverrides.paramPath',
                                        )}
                                        aria-invalid={paramError !== undefined ? true : undefined}
                                        disabled={disabled}
                                        spellCheck={false}
                                        onChange={(event) =>
                                          updateParam(row.key, param.key, {
                                            path: event.target.value,
                                          })
                                        }
                                      />
                                      {paramError !== undefined && (
                                        <small {...stylex.props(styles.errorText)} role="alert">
                                          {paramError}
                                        </small>
                                      )}
                                    </span>
                                    {param.op === 'set' && (
                                      <>
                                        <span {...stylex.props(styles.kind)}>
                                          <Selector
                                            label={t('group.settings.parameterOverrides.paramKind')}
                                            isLabelHidden
                                            options={kindOptions}
                                            value={param.kind}
                                            size="sm"
                                            isDisabled={disabled}
                                            onChange={(value) => setParamKind(row, param, value)}
                                          />
                                        </span>
                                        {/* 值控件跟着类型走：布尔只有两个取值，空值没有值可填。 */}
                                        {param.kind === 'null' ? (
                                          <span {...stylex.props(styles.valueNone)} aria-hidden>
                                            —
                                          </span>
                                        ) : param.kind === 'boolean' ? (
                                          <span {...stylex.props(styles.bool)}>
                                            <Selector
                                              label={t(
                                                'group.settings.parameterOverrides.paramValue',
                                              )}
                                              isLabelHidden
                                              options={booleanOptions}
                                              value={
                                                param.valueText.trim() === 'false'
                                                  ? 'false'
                                                  : 'true'
                                              }
                                              size="sm"
                                              isDisabled={disabled}
                                              onChange={(value) =>
                                                updateParam(row.key, param.key, {
                                                  valueText: value,
                                                })
                                              }
                                            />
                                          </span>
                                        ) : (
                                          <input
                                            {...stylex.props(styles.input, styles.valueInput)}
                                            value={param.valueText}
                                            placeholder={
                                              param.kind === 'json' ? '[&quot;\n\n&quot;]' : '0.7'
                                            }
                                            aria-label={t(
                                              'group.settings.parameterOverrides.paramValue',
                                            )}
                                            disabled={disabled}
                                            spellCheck={false}
                                            onChange={(event) =>
                                              updateParam(row.key, param.key, {
                                                valueText: event.target.value,
                                              })
                                            }
                                            onBlur={(event) =>
                                              setParamValue(row, param, event.target.value)
                                            }
                                          />
                                        )}
                                      </>
                                    )}
                                    <IconButton
                                      variant="ghost"
                                      size="sm"
                                      label={t('group.settings.parameterOverrides.deleteParam')}
                                      isDisabled={disabled}
                                      onClick={() => removeParam(row, param.key)}
                                      icon={<X size={15} aria-hidden="true" />}
                                    />
                                  </div>
                                )
                              })}
                          </div>
                        )}

                        {pathsCrossing(row) && (
                          <InlineNotice tone="warning" appearance="ledger-hint">
                            {t('group.settings.parameterOverrides.pathsCross')}
                          </InlineNotice>
                        )}

                        <div {...stylex.props(styles.paramsFoot)}>
                          <Button
                            variant="ghost"
                            size="sm"
                            isDisabled={disabled}
                            onClick={() => addParam(row)}
                            icon={<Plus size={14} aria-hidden="true" />}
                            label={
                              row.json
                                ? t('group.settings.parameterOverrides.addRemovePath')
                                : t('group.settings.parameterOverrides.addParam')
                            }
                            xstyle={styles.linkButton}
                          />
                          <span {...stylex.props(styles.paramsActions)}>
                            {row.json && (
                              <Button
                                variant="ghost"
                                size="sm"
                                isDisabled={disabled || !row.setText.trim()}
                                onClick={() => formatSetText(row)}
                                label={t('group.settings.parameterOverrides.formatJSON')}
                                xstyle={styles.linkButton}
                              />
                            )}
                            <Button
                              variant="ghost"
                              size="sm"
                              isDisabled={disabled || switchBlocked.get(row.key)}
                              onClick={() => toggleJSON(row)}
                              label={
                                row.json
                                  ? t('group.settings.parameterOverrides.toRows')
                                  : t('group.settings.parameterOverrides.toJSON')
                              }
                              xstyle={styles.linkButton}
                            />
                            <Tooltip
                              content={`${t('group.settings.parameterOverrides.pathHint')}　${t('group.settings.parameterOverrides.mergeHint')}`}
                            >
                              <button
                                {...stylex.props(styles.hint)}
                                type="button"
                                aria-label={t('group.settings.parameterOverrides.pathHint')}
                              >
                                ?
                              </button>
                            </Tooltip>
                          </span>
                        </div>
                      </div>
                    </div>

                    {errors?.action !== undefined && (
                      <InlineNotice tone="danger">{errors.action}</InlineNotice>
                    )}
                  </div>
                )}
              </article>
            )
          })}
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
  root: {
    display: 'grid',
    gap: '9px',
  },
  bar: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-4)',
  },
  order: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '5px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
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
  rule: {
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  ruleInvalid: {
    boxShadow: 'inset 2px 0 var(--color-danger)',
  },
  ruleHead: {
    display: 'flex',
    alignItems: 'center',
    minHeight: '42px',
  },
  summaryButton: {
    display: 'grid',
    minWidth: 0,
    flexGrow: 1,
    gridTemplateColumns: 'auto auto minmax(0, auto) minmax(0, 1fr)',
    alignItems: 'center',
    gap: '8px',
    borderWidth: 0,
    backgroundColor: 'transparent',
    paddingTop: '8px',
    paddingBottom: '8px',
    paddingLeft: '12px',
    paddingRight: '8px',
    fontFamily: 'inherit',
    fontSize: 'var(--text-sm)',
    textAlign: 'left',
    cursor: 'pointer',
    color: 'var(--color-text)',
  },
  chevron: {
    color: 'var(--color-text-faint)',
    transitionProperty: 'transform',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  chevronOpen: {
    transform: 'rotate(180deg)',
  },
  index: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 650,
    minWidth: '16px',
    textAlign: 'center',
  },
  match: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: '5px',
  },
  tag: {
    maxWidth: '160px',
    overflow: 'hidden',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    paddingTop: '3px',
    paddingBottom: '3px',
    paddingLeft: '7px',
    paddingRight: '7px',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 620,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  tagAny: {
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-faint)',
    fontWeight: 560,
  },
  summaryTail: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: '6px',
    overflow: 'hidden',
  },
  chips: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: '4px',
    overflow: 'hidden',
  },
  chip: {
    display: 'inline-flex',
    maxWidth: '170px',
    minWidth: 0,
    alignItems: 'center',
    gap: '4px',
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingTop: '2px',
    paddingBottom: '2px',
    paddingLeft: '6px',
    paddingRight: '6px',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
  },
  chipDrop: {
    borderStyle: 'dashed',
    color: 'var(--color-text-faint)',
  },
  chipMark: {
    color: 'var(--color-danger)',
    fontWeight: 700,
  },
  chipValue: {
    overflow: 'hidden',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    textOverflow: 'ellipsis',
  },
  chipMore: {
    flexShrink: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 650,
  },
  tools: {
    display: 'flex',
    flexShrink: 0,
    alignItems: 'center',
    gap: '2px',
    paddingRight: '6px',
  },
  toolsGap: {
    width: '8px',
  },
  dangerIcon: {
    color: 'var(--color-danger)',
  },
  ruleBody: {
    display: 'grid',
    gap: 'var(--space-3)',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-3)',
    paddingBottom: 'var(--space-3)',
    paddingLeft: '12px',
    paddingRight: '12px',
  },
  field: {
    display: 'grid',
    gap: '6px',
  },
  fieldParams: {},
  fieldLabel: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
  },
  matchInputs: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 170px) minmax(0, 1fr) auto',
    alignItems: 'start',
    gap: 'var(--space-2)',
  },
  protocolSelect: {
    minWidth: 0,
  },
  modelCell: {
    display: 'grid',
    minWidth: 0,
    gap: '4px',
  },
  count: {
    alignSelf: 'center',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  },
  countZero: {
    color: 'var(--color-warning)',
    fontWeight: 600,
  },
  paramsStack: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  params: {
    display: 'grid',
    gap: '6px',
  },
  param: {
    display: 'grid',
    gridTemplateColumns: 'auto minmax(0, 1fr) auto minmax(0, 1fr) auto',
    alignItems: 'start',
    gap: '6px',
  },
  paramDrop: {
    gridTemplateColumns: 'auto minmax(0, 1fr) auto',
  },
  op: {
    minWidth: '96px',
  },
  opStatic: {
    alignSelf: 'center',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
    paddingTop: '5px',
    paddingBottom: '5px',
  },
  pathCell: {
    display: 'grid',
    minWidth: 0,
    gap: '4px',
  },
  kind: {
    minWidth: '92px',
  },
  bool: {
    minWidth: '92px',
  },
  valueNone: {
    alignSelf: 'center',
    color: 'var(--color-text-faint)',
    paddingLeft: '6px',
  },
  input: {
    width: '100%',
    minHeight: 'var(--control-sm)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingLeft: '8px',
    paddingRight: '8px',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
  },
  inputInvalid: {
    borderColor: 'var(--color-danger)',
  },
  path: {},
  valueInput: {},
  textarea: {
    minHeight: '96px',
    paddingTop: '7px',
    paddingBottom: '7px',
    resize: 'vertical',
    lineHeight: 1.55,
  },
  errorText: {
    color: 'var(--color-danger)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 'var(--line-normal)',
  },
  paramsFoot: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
  },
  paramsActions: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  linkButton: {
    paddingLeft: 0,
    paddingRight: 0,
    color: 'var(--color-action)',
    fontWeight: 560,
  },
  hint: {
    display: 'grid',
    width: '18px',
    height: '18px',
    placeItems: 'center',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: '50%',
    backgroundColor: 'transparent',
    color: 'var(--color-text-faint)',
    fontSize: '11px',
    fontWeight: 650,
    cursor: 'help',
    padding: 0,
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
