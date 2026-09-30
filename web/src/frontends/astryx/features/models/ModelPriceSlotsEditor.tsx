import * as stylex from '@stylexjs/stylex'
import { TextInput } from '@astryxdesign/core'

import {
  modelPriceFields,
  type ModelPriceSlotDraft,
  type ModelPriceSlotErrors,
} from '@shared/domain/model-prices/model-price-form'

import { useT } from '../../app/i18n'
import { decimalInputAttrs } from '../../components/input-attrs'

const styles = stylex.create({
  slots: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'repeat(4, minmax(0, 1fr))',
      '@media (max-width: 560px)': 'repeat(2, minmax(0, 1fr))',
    },
    gap: 'var(--space-1-75) var(--space-2)',
  },
  field: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  error: {
    gridColumn: '1 / -1',
    margin: 0,
    color: 'var(--color-danger)',
    fontSize: 'var(--text-label-xs)',
  },
})

export function ModelPriceSlotsEditor({
  draft,
  onDraftChange,
  errors,
  pending,
  idPrefix = 'model-price-slots',
}: {
  draft: ModelPriceSlotDraft
  onDraftChange: (next: ModelPriceSlotDraft) => void
  errors: ModelPriceSlotErrors
  pending: boolean
  idPrefix?: string
}) {
  const t = useT()

  const errorMessages = [
    ...new Set(
      modelPriceFields
        .map((field) =>
          errors[field] ? t(`modelPrices.matrix.errors.${errors[field]}`) : undefined,
        )
        .filter((message): message is string => Boolean(message)),
    ),
  ]
  const errorID = errorMessages.length > 0 ? `${idPrefix}-errors` : undefined

  return (
    <div {...stylex.props(styles.slots)}>
      {modelPriceFields.map((field) => (
        <label key={field} {...stylex.props(styles.field)} htmlFor={`${idPrefix}-${field}`}>
          <span>{t(`modelPrices.fields.${field}`)}</span>
          <TextInput
            id={`${idPrefix}-${field}`}
            label={t(`modelPrices.fields.${field}`)}
            isLabelHidden
            size="sm"
            value={draft[field]}
            isDisabled={pending}
            status={errors[field] ? { type: 'error' } : undefined}
            aria-describedby={errorID}
            {...decimalInputAttrs}
            onChange={(value) => onDraftChange({ ...draft, [field]: value })}
          />
        </label>
      ))}
      {errorMessages.length > 0 && (
        <p {...stylex.props(styles.error)} id={errorID} role="alert">
          {errorMessages.join(' · ')}
        </p>
      )}
    </div>
  )
}
