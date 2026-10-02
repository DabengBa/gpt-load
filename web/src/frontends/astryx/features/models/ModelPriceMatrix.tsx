import * as stylex from '@stylexjs/stylex'
import {
  Button,
  Dialog,
  DialogHeader,
  IconButton,
  Layout,
  LayoutContent,
  LayoutFooter,
  TextInput,
} from '@astryxdesign/core'
import { Plus, TriangleAlert, X } from 'lucide-react'
import { useState } from 'react'
import { useIntl } from 'react-intl'

import {
  modelPriceFields,
  tierDisplayOrder,
  type ModelPriceField,
  type ModelPriceScheduleDraft,
  type ModelPriceScheduleErrors,
} from '@shared/domain/model-prices/model-price-form'
import { formatInteger } from '@shared/lib/format'

import { useT } from '../../app/i18n'
import { decimalInputAttrs, numericInputAttrs } from '../../components/input-attrs'

const MATRIX_NARROW = '@media (max-width: 560px)'

const styles = stylex.create({
  matrix: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-2-5)',
  },
  grid: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1-75) var(--space-2)',
  },
  row: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) var(--control-xs, 32px)',
      [MATRIX_NARROW]: 'minmax(0, 1fr) auto',
    },
    alignItems: 'center',
    gap: 'var(--space-1-75)',
  },
  rowTiered: {
    gridTemplateColumns: {
      default: '76px minmax(0, 1fr) calc(var(--control-xs, 32px) * 2)',
      [MATRIX_NARROW]: 'minmax(0, 1fr) auto',
    },
  },
  headerRow: {
    display: { default: 'grid', [MATRIX_NARROW]: 'none' },
    alignItems: 'end',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  headerCells: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, minmax(0, 1fr))',
    paddingInlineStart: 'calc(var(--space-2) + 1px)',
  },
  group: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'repeat(4, minmax(0, 1fr))',
      [MATRIX_NARROW]: 'repeat(2, minmax(0, 1fr))',
    },
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
  },
  groupFocused: {
    borderColor: 'var(--color-focus)',
    boxShadow: 'var(--focus-ring)',
  },
  groupThreshold: {
    gridTemplateColumns: 'minmax(0, 1fr)',
  },
  groupCell: {
    gridColumn: { default: 'auto', [MATRIX_NARROW]: '1 / -1' },
    gridRow: { default: 'auto', [MATRIX_NARROW]: '2' },
  },
  thresholdCell: {
    gridColumn: { default: 'auto', [MATRIX_NARROW]: '1' },
    gridRow: { default: 'auto', [MATRIX_NARROW]: '1' },
  },
  actionsCell: {
    gridColumn: { default: 'auto', [MATRIX_NARROW]: '2' },
    gridRow: { default: 'auto', [MATRIX_NARROW]: '1' },
  },
  actions: {
    display: 'grid',
    gridAutoFlow: 'column',
    gridAutoColumns: 'var(--control-xs, 32px)',
    alignItems: 'center',
    justifyContent: 'start',
  },
  cellInput: {
    borderWidth: 0,
    borderRadius: 0,
    backgroundColor: 'transparent',
  },
  field: {
    display: 'grid',
    minWidth: 0,
  },
  fieldLabel: {
    display: { default: 'none', [MATRIX_NARROW]: 'block' },
    padding: 'var(--space-1) var(--space-2) 0',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  cellInputInvalid: {
    backgroundColor: 'var(--color-danger-bg)',
  },
  cellInputDivider: {
    borderInlineStartWidth: 1,
    borderInlineStartStyle: 'solid',
    borderInlineStartColor: 'var(--color-border-subtle)',
  },
  error: {
    margin: 0,
    color: 'var(--color-danger)',
    fontSize: 'var(--text-label-xs)',
  },
  rules: {
    display: 'grid',
    gap: 'var(--space-1)',
    fontSize: 'var(--text-label-xs)',
  },
  cacheRule: {
    margin: 0,
    color: 'var(--color-text-muted)',
  },
  tierRules: {
    display: 'grid',
    gap: 'var(--space-0-75)',
    margin: 0,
    padding: 0,
    color: 'var(--color-text-faint)',
    listStyle: 'none',
    lineHeight: 1.55,
  },
  failureBanner: {
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-danger)',
    backgroundColor:
      'var(--color-danger-bg, color-mix(in srgb, var(--color-danger) 10%, transparent))',
    padding: '9px 12px',
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    fontSize: 'var(--text-meta)',
  },
  dialogWarning: {
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    backgroundColor:
      'var(--color-warning-bg, color-mix(in srgb, var(--color-warning) 12%, transparent))',
    padding: '9px 12px',
    fontSize: 'var(--text-meta)',
  },
})

function dedupe(messages: (string | undefined)[]): string[] {
  return [...new Set(messages.filter((message): message is string => Boolean(message)))]
}

/** 组合控件统一由外框表达焦点,子输入仅保留单元格分隔线(focus-within 等价)。 */
function PriceInputGroup({
  children,
  label,
  xstyles,
}: {
  children: React.ReactNode
  label: string
  xstyles?: (stylex.StyleXStyles | null | undefined | false)[]
}) {
  const [focused, setFocused] = useState(false)
  return (
    <div
      {...stylex.props(styles.group, focused && styles.groupFocused, ...(xstyles ?? []))}
      role="group"
      aria-label={label}
      onFocusCapture={() => setFocused(true)}
      onBlurCapture={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
          setFocused(false)
        }
      }}
    >
      {children}
    </div>
  )
}

export function ModelPriceMatrix({
  modelId,
  draft,
  onDraftChange,
  errors,
  pending,
  failure,
  unpricedConfirmOpen,
  onUnpricedConfirmOpenChange,
  onAddTier,
  onRemoveTier,
  onConfirmUnpriced,
}: {
  modelId: string
  draft: ModelPriceScheduleDraft
  onDraftChange: (next: ModelPriceScheduleDraft) => void
  errors: ModelPriceScheduleErrors
  pending: boolean
  failure: string
  unpricedConfirmOpen: boolean
  onUnpricedConfirmOpenChange: (open: boolean) => void
  onAddTier: () => void
  onRemoveTier: (key: string) => void
  onConfirmUnpriced: () => void
}) {
  const t = useT()
  const intl = useIntl()
  const tiered = draft.tiers.length > 0

  /**
   * 新增档位固定追加到末尾,用户填入的阈值可能比上面已有档位更小,
   * 导致编辑区行序与下方按阈值排序的派生说明相互矛盾。失焦时才重排,
   * 避免每次按键都重排导致正在编辑的行跳动。
   */
  function reorderTiers(): void {
    onDraftChange({
      ...draft,
      tiers: [...draft.tiers].sort(
        (left, right) => tierDisplayOrder(left.threshold) - tierDisplayOrder(right.threshold),
      ),
    })
  }

  const baseFieldError = (field: ModelPriceField): string | undefined => {
    const code = errors.base[field]
    return code ? t(`modelPrices.matrix.errors.${code}`) : undefined
  }
  const tierThresholdError = (key: string): string | undefined => {
    const code = errors.tiers[key]?.threshold
    return code ? t(`modelPrices.matrix.errors.threshold_${code}`) : undefined
  }
  const tierSlotError = (key: string, field: ModelPriceField): string | undefined => {
    const code = errors.tiers[key]?.slots?.[field]
    return code ? t(`modelPrices.matrix.errors.${code}`) : undefined
  }

  const baseErrors = dedupe(modelPriceFields.map((field) => baseFieldError(field)))

  const tierErrors = (key: string): string[] => {
    const messages = [
      tierThresholdError(key),
      ...modelPriceFields.map((field) => tierSlotError(key, field)),
    ]
    if (errors.tiers[key]?.emptyTier === true) {
      messages.push(t('modelPrices.matrix.errors.tier_empty'))
    }
    return dedupe(messages)
  }

  const thresholdDescription = (value: string): string => {
    const threshold = tierDisplayOrder(value)
    if (Number.isFinite(threshold)) return formatInteger(threshold, intl.locale)
    return value.trim() || t('modelPrices.matrix.thresholdNotSet')
  }

  const tierRules = draft.tiers.map((tier, index) => ({
    key: tier.key,
    description: t('modelPrices.matrix.tierRule', {
      threshold: thresholdDescription(tier.threshold),
      tier: index + 1,
    }),
  }))

  return (
    <div {...stylex.props(styles.matrix)}>
      {failure && (
        <div {...stylex.props(styles.failureBanner)} role="alert">
          <TriangleAlert size={13} aria-hidden />
          <span>{failure}</span>
        </div>
      )}

      <div {...stylex.props(styles.grid)}>
        {/* 纯视觉列标签;每个输入的可访问名称由输入自身的 label 承担。 */}
        <div
          {...stylex.props(styles.row, tiered && styles.rowTiered, styles.headerRow)}
          aria-hidden
        >
          {tiered && <span>{t('modelPrices.matrix.thresholdColumn')}</span>}
          <div {...stylex.props(styles.headerCells)}>
            {modelPriceFields.map((field) => (
              <span key={field}>{t(`modelPrices.fields.${field}`)}</span>
            ))}
          </div>
          {tiered && <span />}
        </div>

        <div {...stylex.props(styles.row, tiered && styles.rowTiered)}>
          {tiered && (
            <PriceInputGroup
              label={t('modelPrices.matrix.baseRow')}
              xstyles={[styles.groupThreshold, styles.thresholdCell]}
            >
              <TextInput
                id="model-price-default"
                label={t('modelPrices.matrix.baseRow')}
                isLabelHidden
                size="sm"
                value={t('modelPrices.matrix.baseRow')}
                isReadOnly
                xstyle={styles.cellInput}
              />
            </PriceInputGroup>
          )}
          {/* 一横排为一个输入框组:共享外框,内部竖线分隔,无间隙。 */}
          <PriceInputGroup
            label={t('modelPrices.matrix.baseRow')}
            xstyles={[tiered && styles.groupCell]}
          >
            {modelPriceFields.map((field, index) => (
              <label
                key={field}
                {...stylex.props(styles.field)}
                htmlFor={`model-price-base-${field}`}
              >
                <span {...stylex.props(styles.fieldLabel)}>{t(`modelPrices.fields.${field}`)}</span>
                <TextInput
                  id={`model-price-base-${field}`}
                  label={t(`modelPrices.fields.${field}`)}
                  isLabelHidden
                  size="sm"
                  value={draft.base[field]}
                  isDisabled={pending}
                  status={baseFieldError(field) ? { type: 'error' } : undefined}
                  xstyle={[
                    styles.cellInput,
                    index > 0 && styles.cellInputDivider,
                    !!baseFieldError(field) && styles.cellInputInvalid,
                  ]}
                  {...decimalInputAttrs}
                  onChange={(value) =>
                    onDraftChange({ ...draft, base: { ...draft.base, [field]: value } })
                  }
                />
              </label>
            ))}
          </PriceInputGroup>
          {/* 增删都落在同一列:无档位时基础行就是最后一行,由它承载加号。 */}
          <div {...stylex.props(styles.actions, styles.actionsCell)}>
            {!tiered && (
              <IconButton
                variant="ghost"
                size="sm"
                isDisabled={pending}
                label={t('modelPrices.matrix.addTier')}
                icon={<Plus size={14} aria-hidden />}
                onClick={onAddTier}
              />
            )}
          </div>
        </div>
        {baseErrors.length > 0 && (
          <p {...stylex.props(styles.error)} role="alert">
            {baseErrors.join(' · ')}
          </p>
        )}

        {draft.tiers.map((tier, index) => (
          <div key={tier.key}>
            <div {...stylex.props(styles.row, styles.rowTiered)}>
              <PriceInputGroup
                label={t('modelPrices.matrix.thresholdColumn')}
                xstyles={[styles.groupThreshold, styles.thresholdCell]}
              >
                <TextInput
                  id={`model-price-tier-${tier.key}-threshold`}
                  label={t('modelPrices.matrix.thresholdColumn')}
                  isLabelHidden
                  size="sm"
                  value={tier.threshold}
                  isDisabled={pending}
                  status={tierThresholdError(tier.key) ? { type: 'error' } : undefined}
                  xstyle={[
                    styles.cellInput,
                    !!tierThresholdError(tier.key) && styles.cellInputInvalid,
                  ]}
                  {...numericInputAttrs}
                  onChange={(value) =>
                    onDraftChange({
                      ...draft,
                      tiers: draft.tiers.map((item) =>
                        item.key === tier.key ? { ...item, threshold: value } : item,
                      ),
                    })
                  }
                  onBlur={reorderTiers}
                />
              </PriceInputGroup>
              <PriceInputGroup
                label={tier.threshold.trim() || t('modelPrices.matrix.thresholdColumn')}
                xstyles={[styles.groupCell]}
              >
                {modelPriceFields.map((field, fieldIndex) => (
                  <label
                    key={field}
                    {...stylex.props(styles.field)}
                    htmlFor={`model-price-tier-${tier.key}-${field}`}
                  >
                    <span {...stylex.props(styles.fieldLabel)}>
                      {t(`modelPrices.fields.${field}`)}
                    </span>
                    <TextInput
                      id={`model-price-tier-${tier.key}-${field}`}
                      label={t(`modelPrices.fields.${field}`)}
                      isLabelHidden
                      size="sm"
                      value={tier.slots[field]}
                      isDisabled={pending}
                      status={tierSlotError(tier.key, field) ? { type: 'error' } : undefined}
                      xstyle={[
                        styles.cellInput,
                        fieldIndex > 0 && styles.cellInputDivider,
                        !!tierSlotError(tier.key, field) && styles.cellInputInvalid,
                      ]}
                      {...decimalInputAttrs}
                      onChange={(value) =>
                        onDraftChange({
                          ...draft,
                          tiers: draft.tiers.map((item) =>
                            item.key === tier.key
                              ? { ...item, slots: { ...item.slots, [field]: value } }
                              : item,
                          ),
                        })
                      }
                    />
                  </label>
                ))}
              </PriceInputGroup>
              <div {...stylex.props(styles.actions, styles.actionsCell)}>
                <IconButton
                  variant="ghost"
                  size="sm"
                  isDisabled={pending}
                  label={t('modelPrices.matrix.removeTier')}
                  icon={<X size={14} aria-hidden />}
                  onClick={() => onRemoveTier(tier.key)}
                />
                {index === draft.tiers.length - 1 && (
                  <IconButton
                    variant="ghost"
                    size="sm"
                    isDisabled={pending}
                    label={t('modelPrices.matrix.addTier')}
                    icon={<Plus size={14} aria-hidden />}
                    onClick={onAddTier}
                  />
                )}
              </div>
            </div>
            {tierErrors(tier.key).length > 0 && (
              <p {...stylex.props(styles.error)} role="alert">
                {tierErrors(tier.key).join(' · ')}
              </p>
            )}
          </div>
        ))}
      </div>

      <div {...stylex.props(styles.rules)}>
        <p {...stylex.props(styles.cacheRule)}>{t('modelPrices.matrix.oneHourRule')}</p>
        {tierRules.length > 0 && (
          <ul {...stylex.props(styles.tierRules)}>
            {tierRules.map((rule) => (
              <li key={rule.key}>{rule.description}</li>
            ))}
          </ul>
        )}
      </div>

      <Dialog isOpen={unpricedConfirmOpen} onOpenChange={onUnpricedConfirmOpenChange} width={440}>
        <Layout
          header={
            <DialogHeader
              title={t('modelPrices.matrix.unpricedConfirm.title')}
              subtitle={t('modelPrices.matrix.unpricedConfirm.description', { model: modelId })}
              onOpenChange={onUnpricedConfirmOpenChange}
              hasDivider
            />
          }
          content={
            <LayoutContent>
              <div {...stylex.props(styles.dialogWarning)} role="status">
                {t('modelPrices.matrix.unpricedConfirm.warning')}
              </div>
            </LayoutContent>
          }
          footer={
            <LayoutFooter hasDivider>
              <Button
                variant="secondary"
                size="sm"
                label={t('common.cancel')}
                onClick={() => onUnpricedConfirmOpenChange(false)}
              />
              <Button
                variant="destructive"
                size="sm"
                isLoading={pending}
                label={t('modelPrices.matrix.unpricedConfirm.confirm')}
                onClick={onConfirmUnpriced}
              />
            </LayoutFooter>
          }
        />
      </Dialog>
    </div>
  )
}
