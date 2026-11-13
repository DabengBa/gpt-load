import type { ModelAliasEditorLabels } from '@shared/domain/models/model-draft'
import type { MessageId } from '@shared/i18n/message-ids'

import type { TValues } from '../../../app/i18n'

type Translate = (id: MessageId, values?: TValues) => string

export function buildAliasEditorLabels(
  t: Translate,
  pendingPricingCount: number,
): ModelAliasEditorLabels {
  return {
    tableLabel: t('group.modelEditor.tableLabel'),
    id: t('group.modelEditor.id'),
    alias: t('group.modelEditor.alias'),
    thirdColumn: t('group.modelEditor.schedulingPricingActions'),
    actions: t('group.modelEditor.actions'),
    search: t('group.modelEditor.searchPlaceholder'),
    searchLabel: t('group.modelEditor.searchLabel'),
    clearSearch: t('group.modelEditor.searchClear'),
    aliasFor: (id) => t('group.modelEditor.aliasFor', { id }),
    aliasPlaceholder: t('group.modelEditor.aliasPlaceholder'),
    removeFor: (id) => t('group.modelEditor.removeFor', { id }),
    manualId: t('group.modelEditor.manualId'),
    manualIdRequired: t('group.modelEditor.manualIdRequired'),
    add: t('group.modelEditor.add'),
    addInline: t('group.modelEditor.addInline'),
    count: (count) =>
      pendingPricingCount
        ? t('group.modelEditor.summary', { count, pending: pendingPricingCount })
        : t('group.modelEditor.total', { count }),
    empty: t('group.modelEditor.empty'),
    noMatches: t('group.modelEditor.noMatches'),
    nameConflict: (name) => t('group.modelEditor.nameConflict', { name }),
    weight: t('group.modelEditor.weight'),
    priority: t('group.modelEditor.priority'),
    weightDisabled: t('group.modelEditor.weightDisabled'),
    invalidWeight: t('group.modelEditor.invalidWeight'),
    invalidPriority: t('group.modelEditor.invalidPriority'),
    zeroShare: t('group.modelEditor.zeroShareSummary'),
  }
}
