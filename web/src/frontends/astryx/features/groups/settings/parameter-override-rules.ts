import type {
  AccessProtocol,
  ParameterJSONValue,
  ParameterOverrideRuleDto,
} from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
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
  ParameterValueError,
  tryValueFromText,
  valueFromText,
  valueKind,
  type ParameterValueKind,
} from '@shared/lib/parameter-values'

import type { TValues } from '../../../app/i18n'

export type Translate = (id: MessageId, values?: TValues) => string

export type ParamOp = 'set' | 'remove'

/** 一行参数动作。行序只是录入次序，不参与语义。 */
export interface ParamRow {
  key: number
  op: ParamOp
  path: string
  valueText: string
  kind: ParameterValueKind
  /** 用户动过类型列之后就不再跟随输入推断，否则一打字又被改回去。 */
  kindPinned: boolean
}

export interface RuleRow {
  key: number
  open: boolean
  protocol: string
  model: string
  /** true 时 set 用整段 JSON 编辑，remove 仍是行。 */
  json: boolean
  params: ParamRow[]
  setText: string
}

export interface RuleErrors {
  model?: string
  set?: string
  action?: string
  params: Map<number, string>
}

export interface SummaryChip {
  key: string
  op: ParamOp
  path: string
  value: string
}

export interface RuleSummary {
  chips: SummaryChip[]
  overflow: number
}

/** 折叠行放不下太多摘要；超出的用 +N 兜住，避免静默裁切成看不见。 */
export const summaryLimit = 3

export const forbiddenRootFields = ['model', 'stream', 'store']

export function createSetParam({ path, value }: ParameterPathEntry, key: number): ParamRow {
  return {
    key,
    op: 'set',
    path,
    valueText: formatValueText(value),
    kind: valueKind(value),
    kindPinned: false,
  }
}

/** Seeded rows get deterministic negative keys so render-time seeding stays pure. */
export function createRows(source: ParameterOverrideRuleDto[]): RuleRow[] {
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

export function parseSetText(text: string): Record<string, ParameterJSONValue> | undefined {
  const trimmed = text.trim()
  if (!trimmed) return {}
  const parsed: unknown = JSON.parse(trimmed)
  assertJSONNumbersRoundTrip(trimmed)
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) return undefined
  if (hasEmptyParameterKey(parsed)) throw new ParameterValueError('empty-key')
  return parsed as Record<string, ParameterJSONValue>
}

export function tryParseSetText(text: string): Record<string, ParameterJSONValue> | undefined {
  try {
    return parseSetText(text)
  } catch {
    return undefined
  }
}

/** 刚加出来还没填的行：不标红、不阻止保存，序列化时直接忽略。 */
export function isBlankParam(param: ParamRow): boolean {
  if (param.path.trim()) return false
  return param.op === 'remove' || param.kind === 'null' || !param.valueText.trim()
}

export function setEntries(row: RuleRow): ParameterPathEntry[] | undefined {
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

export function removePointers(row: RuleRow): string[] {
  return row.params
    .filter((param) => param.op === 'remove' && !isBlankParam(param))
    .map(({ path }) => toParameterPointer(path))
}

export function pathError(row: ParamRow, siblings: ParamRow[], t: Translate): string {
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

export function valueError(row: ParamRow, t: Translate): string {
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
    return t('group.settings.parameterOverrides.errors.valueJSON', { brace: '{', bracket: '[' })
  }
}

export function ruleErrors(row: RuleRow, t: Translate): RuleErrors {
  const errors: RuleErrors = { params: new Map() }
  const model = row.model.trim()
  if (model.includes('*') && !/^[^*]+\*$/u.test(model)) {
    errors.model = t('group.settings.parameterOverrides.errors.modelPattern')
  }

  const visible = row.json ? row.params.filter(({ op }) => op === 'remove') : row.params
  const filled = visible.filter((param) => !isBlankParam(param))
  for (const param of filled) {
    const error = pathError(param, filled, t) || (param.op === 'set' ? valueError(param, t) : '')
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

export function ruleInvalid(errors: RuleErrors | undefined): boolean {
  return Boolean(errors && (errors.model || errors.set || errors.action || errors.params.size > 0))
}

/** 设置与删除只在路径相交时才互相影响，这时后端固定先删后设。 */
export function pathsCrossing(row: RuleRow): boolean {
  const removes = row.params.filter(({ op }) => op === 'remove').map(({ path }) => path)
  if (removes.length === 0) return false
  const sets = (setEntries(row) ?? []).map(({ path }) => path)
  return removes.some((left) => sets.some((right) => parameterPathsCross(left, right)))
}

export function previewValue(value: ParameterJSONValue): string {
  return typeof value === 'string' ? value : formatValueText(value)
}

export function buildSummary(row: RuleRow): SummaryChip[] {
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

export function summarizeRules(rows: RuleRow[]): Map<number, RuleSummary> {
  return new Map(
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
}

export function collectErrors(rows: RuleRow[], t: Translate): Map<number, RuleErrors> {
  return new Map(rows.map((row) => [row.key, ruleErrors(row, t)] as const))
}

export function rulesValid(errorsByRule: Map<number, RuleErrors>): boolean {
  return [...errorsByRule.values()].every((errors) => !ruleInvalid(errors))
}

export function serializeRows(rows: RuleRow[]): ParameterOverrideRuleDto[] {
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

/**
 * The rows↔JSON toggle is lossy in both directions, so it stays disabled the
 * moment either side cannot round-trip — a silent conversion would drop values.
 */
export function switchBlockedByRule(
  rows: RuleRow[],
  errorsByRule: Map<number, RuleErrors>,
): Map<number, boolean> {
  return new Map(
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
}

export function matchCount(model: string, suggestions: readonly string[]): number | undefined {
  const text = model.trim()
  if (!text || (text.includes('*') && !/^[^*]+\*$/u.test(text))) return undefined
  const prefix = text.endsWith('*') ? text.slice(0, -1) : undefined
  return suggestions.filter((candidate) =>
    prefix === undefined ? candidate === text : candidate.startsWith(prefix),
  ).length
}
