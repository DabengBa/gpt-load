import {
  findModelNameConflicts,
  normalizedModels,
  sameModels,
  type ModelDraftItem,
  type ModelNameConflict,
} from '@shared/domain/groups/models/model-diff'
import {
  indexesWithInvalidPriorities,
  indexesWithInvalidWeights,
  indexesWithZeroShare,
} from '@shared/domain/models/model-draft'
import type { MessageId } from '@shared/i18n/message-ids'

import type { TValues } from '../../../app/i18n'

type Translate = (id: MessageId, values?: TValues) => string

export interface ModelEditorAnalysis {
  conflicts: ModelNameConflict[]
  emptyAliasIndexes: Set<number>
  emptyIDIndexes: Set<number>
  invalidWeightIndexes: Set<number>
  invalidPriorityIndexes: Set<number>
  zeroShareIndexes: Set<number>
  invalidRowCount: number
  validationSummary: string
  saveBarError: string
  dirty: boolean
  empty: boolean
  canSave: boolean
}

export function analyzeModelDraft(args: {
  saved: ModelDraftItem[]
  draft: ModelDraftItem[]
  serverConflicts: ModelNameConflict[]
  saveError: string
  operationBlocked: boolean
  t: Translate
}): ModelEditorAnalysis {
  const { saved, draft, serverConflicts, saveError, operationBlocked, t } = args
  const conflicts = serverConflicts.length
    ? serverConflicts
    : findModelNameConflicts(normalizedModels(draft))
  const emptyAliasIndexes = new Set(
    draft.flatMap((item, index) => (item.alias_enabled && !item.alias.trim() ? [index] : [])),
  )
  const emptyIDIndexes = new Set(draft.flatMap((item, index) => (!item.id.trim() ? [index] : [])))
  const invalidWeightIndexes = indexesWithInvalidWeights(draft)
  const invalidPriorityIndexes = indexesWithInvalidPriorities(draft)
  const zeroShareIndexes = indexesWithZeroShare(draft)
  const invalidRowCount = new Set([
    ...conflicts.flatMap((item) => item.indexes),
    ...emptyAliasIndexes,
    ...emptyIDIndexes,
    ...invalidWeightIndexes,
    ...invalidPriorityIndexes,
    ...zeroShareIndexes,
  ]).size
  const validationSummary = [
    conflicts.length ? t('group.modelEditor.conflictSummary') : '',
    emptyIDIndexes.size ? t('group.modelEditor.manualIdRequired') : '',
    emptyAliasIndexes.size ? t('group.modelEditor.emptyAliasSummary') : '',
    zeroShareIndexes.size ? t('group.modelEditor.zeroShareSummary') : '',
    invalidWeightIndexes.size ? t('group.modelEditor.invalidWeight') : '',
    invalidPriorityIndexes.size ? t('group.modelEditor.invalidPriority') : '',
  ]
    .filter(Boolean)
    .join(' · ')
  const dirty =
    !sameModels(saved, draft) || draft.some((item) => item.editable_id && !item.id.trim())
  const canSave =
    dirty &&
    !operationBlocked &&
    conflicts.length === 0 &&
    emptyIDIndexes.size === 0 &&
    emptyAliasIndexes.size === 0 &&
    invalidWeightIndexes.size === 0 &&
    invalidPriorityIndexes.size === 0 &&
    zeroShareIndexes.size === 0

  return {
    conflicts,
    emptyAliasIndexes,
    emptyIDIndexes,
    invalidWeightIndexes,
    invalidPriorityIndexes,
    zeroShareIndexes,
    invalidRowCount,
    validationSummary,
    saveBarError: validationSummary || saveError,
    dirty,
    empty: normalizedModels(draft).length === 0,
    canSave,
  }
}
