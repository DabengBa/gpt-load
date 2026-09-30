import * as stylex from '@stylexjs/stylex'
import { IconButton, Selector, Switch, TextInput } from '@astryxdesign/core'
import { Plus, X } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import type {
  AccessKeyCostLimitRuleStatusDto,
  AccessKeyCostLimitStatusDto,
} from '@shared/control/types'
import type { AccessKeyCostLimitRuleDraft } from '@shared/domain/access-keys/access-key-patch'
import type { MessageId } from '@shared/i18n/message-ids'
import { quotaProgressTone, type QuotaProgressTone } from '@shared/lib/quota-progress'
import { createUUID } from '@shared/lib/uuid'

import { useT } from '../../app/i18n'
import { decimalInputAttrs, numericInputAttrs } from '../../components/input-attrs'
import { AccessKeyCostLimitWindowTime } from './AccessKeyCostLimitWindowTime'

const COMPACT = '@media (max-width: 560px)'
const NARROW = '@media (max-width: 860px)'

const styles = stylex.create({
  editor: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-4)',
  },
  section: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-3)',
  },
  // `.cost-limit-section + .cost-limit-section` — StyleX has no sibling
  // selectors, so the divider rides on the second section.
  sectionSibling: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-4)',
  },
  sectionHeader: {
    display: 'flex',
    alignItems: { default: 'center', [COMPACT]: 'flex-start' },
    justifyContent: 'space-between',
    gap: 'var(--space-3)',
  },
  sectionIntro: {
    minWidth: 0,
  },
  sectionTitle: {
    margin: 0,
    color: 'var(--color-text)',
    fontSize: 'var(--text-body-sm)',
    fontWeight: 700,
  },
  sectionDescription: {
    margin: 0,
    marginTop: 'var(--space-1)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.45,
  },
  sectionToggle: {
    display: 'flex',
    alignItems: 'center',
    flex: '0 0 auto',
  },
  sectionContent: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-2)',
  },
  ruleList: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-2)',
  },
  rule: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-2)',
  },
  ruleRow: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) 56px',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  ruleActions: {
    display: 'grid',
    width: '56px',
    gridAutoFlow: 'column',
    gridAutoColumns: '28px',
    alignItems: 'center',
  },
  // `.cost-limit-rule__remove/__add` — ghost icon tint. `xstyle` arrays reject
  // `false` entries, and StyleX pseudos can't express `:hover:not(:disabled)`,
  // so the enabled/disabled variants are two whole style objects.
  ruleActionTint: {
    color: 'var(--color-text-faint)',
  },
  ruleRemoveTint: {
    color: {
      default: 'var(--color-text-faint)',
      ':hover': 'var(--color-danger)',
    },
  },
  ruleAddTint: {
    color: {
      default: 'var(--color-text-faint)',
      ':hover': 'var(--color-action)',
    },
  },
  fieldsHeader: {
    display: 'grid',
    minWidth: 0,
    alignItems: 'end',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  fieldsHeaderTotal: {
    gridTemplateColumns: 'minmax(0, 1fr)',
  },
  fieldsHeaderPeriodic: {
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) 76px 96px 64px',
      [COMPACT]: 'minmax(0, 1fr) 58px 76px 64px',
    },
  },
  inputGroup: {
    display: 'grid',
    width: '100%',
    minWidth: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
  },
  inputGroupTotal: {
    gridTemplateColumns: 'minmax(0, 1fr)',
  },
  inputGroupPeriodic: {
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) 76px 96px',
      [COMPACT]: 'minmax(0, 1fr) 58px 76px',
    },
  },
  // JS-driven `:focus-within` equivalent — the group frame owns the focus ring
  // while embedded controls stay borderless (same idiom as ModelPriceMatrix).
  inputGroupFocused: {
    borderColor: 'var(--color-focus)',
    boxShadow: 'var(--focus-ring)',
  },
  embeddedCell: {
    borderWidth: 0,
    borderRadius: 0,
    backgroundColor: 'transparent',
    height: { default: 'var(--control-xs)', [NARROW]: 'var(--touch-target)' },
  },
  // Classic: `:deep(.app-text-input) { padding-left: var(--space-2) }` — the
  // selector trigger keeps its own padding, so this rides on text inputs only.
  embeddedTextInput: {
    paddingInlineStart: 'var(--space-2)',
  },
  embeddedDivider: {
    borderInlineStartWidth: 1,
    borderInlineStartStyle: 'solid',
    borderInlineStartColor: 'var(--color-border-subtle)',
  },
  embeddedSelector: {
    alignSelf: 'stretch',
    minHeight: { default: 'var(--control-xs)', [NARROW]: 'var(--touch-target)' },
    fontSize: 'var(--text-meta)',
  },
  runtime: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto minmax(54px, auto)',
    minHeight: '18px',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  runtimeValue: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 650,
    whiteSpace: 'nowrap',
  },
  // `.cost-limit-rule__runtime > :last-child` — applied to whichever element
  // closes the row (the runtime value when no tail cell renders).
  runtimeTail: {
    minWidth: 0,
    overflow: 'hidden',
    textAlign: 'right',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
})

const quotaStyles = stylex.create({
  quota: {
    position: 'relative',
    display: 'block',
    width: '100%',
    minWidth: 0,
    height: '8px',
    overflow: 'hidden',
    borderRadius: '999px',
    backgroundColor: 'light-dark(#e9edf1, #29313a)',
  },
  quotaCompact: {
    height: '6px',
  },
  quotaSuccess: {
    backgroundColor: 'light-dark(#dff6e6, #1b3c29)',
  },
  quotaWarning: {
    backgroundColor: 'light-dark(#fff4cc, #3b310f)',
  },
  quotaDanger: {
    backgroundColor: 'light-dark(#ffe5e8, #421d25)',
  },
  quotaUnknown: {
    backgroundImage:
      'repeating-linear-gradient(135deg, var(--color-border-subtle), var(--color-border-subtle) 6px, var(--color-surface-sunken) 6px, var(--color-surface-sunken) 12px)',
  },
  quotaFill: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 0,
    right: 'auto',
    borderRadius: 'inherit',
    backgroundColor: '#42be65',
  },
  quotaFillWarning: {
    backgroundColor: '#f1c21b',
  },
  quotaFillDanger: {
    backgroundColor: '#fa4d56',
  },
})

/** Port of classic `QuotaProgressBar` — fixed-tone strip with ARIA progress semantics. */
function QuotaBar({
  value,
  tone = 'success',
  label,
  valueText,
  compact = false,
}: {
  value?: number
  tone?: QuotaProgressTone
  label: string
  valueText: string
  compact?: boolean
}) {
  const normalized =
    value === undefined || !Number.isFinite(value) ? undefined : Math.max(0, Math.min(100, value))
  const toneStyle =
    tone === 'warning'
      ? quotaStyles.quotaWarning
      : tone === 'danger'
        ? quotaStyles.quotaDanger
        : quotaStyles.quotaSuccess
  const fillToneStyle =
    tone === 'warning'
      ? quotaStyles.quotaFillWarning
      : tone === 'danger'
        ? quotaStyles.quotaFillDanger
        : undefined

  if (normalized === undefined) {
    return (
      <span
        {...stylex.props(
          quotaStyles.quota,
          quotaStyles.quotaUnknown,
          compact && quotaStyles.quotaCompact,
        )}
        role="img"
        aria-label={`${label}: ${valueText}`}
      />
    )
  }
  return (
    <span
      {...stylex.props(quotaStyles.quota, toneStyle, compact && quotaStyles.quotaCompact)}
      role="progressbar"
      aria-label={label}
      aria-valuenow={normalized}
      aria-valuetext={valueText}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <span
        {...stylex.props(quotaStyles.quotaFill, fillToneStyle)}
        style={{ width: `${normalized}%` }}
      />
    </span>
  )
}

/** Bordered multi-control box that owns the focus ring for its cells. */
function InputGroup({
  label,
  periodic,
  children,
}: {
  label: string
  periodic: boolean
  children: ReactNode
}) {
  const [focused, setFocused] = useState(false)
  return (
    <div
      {...stylex.props(
        styles.inputGroup,
        periodic ? styles.inputGroupPeriodic : styles.inputGroupTotal,
        focused && styles.inputGroupFocused,
      )}
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

type PeriodUnit = 'seconds' | 'minutes' | 'hours' | 'days'

const periodUnits: ReadonlyArray<{ value: PeriodUnit; seconds: number }> = [
  { value: 'seconds', seconds: 1 },
  { value: 'minutes', seconds: 60 },
  { value: 'hours', seconds: 3_600 },
  { value: 'days', seconds: 86_400 },
]

function periodUnit(seconds: number | undefined): PeriodUnit {
  const value = seconds ?? 0
  for (const unit of [...periodUnits].reverse()) {
    if (value > 0 && value % unit.seconds === 0) return unit.value
  }
  return 'seconds'
}

function unitSeconds(unit: PeriodUnit): number {
  return periodUnits.find((candidate) => candidate.value === unit)?.seconds ?? 1
}

function periodValue(seconds: number | undefined): number {
  const unit = periodUnit(seconds)
  return Math.max(1, Math.floor((seconds ?? 0) / unitSeconds(unit)))
}

function remainingPercent(runtime: AccessKeyCostLimitRuleStatusDto): number {
  if (runtime.status === 'inactive') return 100
  const limit = Number(runtime.limit_usd)
  const remaining = Number(runtime.remaining_usd)
  if (!Number.isFinite(limit) || !Number.isFinite(remaining) || limit <= 0) return 0
  return Math.round(Math.max(0, Math.min(100, (remaining / limit) * 100)))
}

/** Runtime row: remaining-quota bar + percent text + trailing status cell. */
function RuleRuntime({
  runtime,
  label,
  tail,
}: {
  runtime: AccessKeyCostLimitRuleStatusDto
  label: string
  tail?: ReactNode
}) {
  const t = useT()
  const intl = useIntl()
  const percent = remainingPercent(runtime)
  const valueText = t('accessKeys.costLimits.remainingPercent', {
    value: intl.formatNumber(percent),
  })
  return (
    <div {...stylex.props(styles.runtime)}>
      <QuotaBar
        value={percent}
        tone={quotaProgressTone(percent, runtime.status === 'exhausted')}
        label={label}
        valueText={valueText}
        compact
      />
      <strong {...stylex.props(styles.runtimeValue, tail === undefined && styles.runtimeTail)}>
        {valueText}
      </strong>
      {tail !== undefined && <span {...stylex.props(styles.runtimeTail)}>{tail}</span>}
    </div>
  )
}

/**
 * Port of classic `AccessKeyCostLimitEditor.vue` — `modelValue` maps to
 * `value`/`onChange`. The enable switches materialize as "add one rule when
 * enabling an empty section, drop all rules of the kind when disabling".
 */
export function AccessKeyCostLimitEditor({
  value,
  runtimeStatus,
  disabled,
  onChange,
}: {
  value: AccessKeyCostLimitRuleDraft[]
  runtimeStatus: AccessKeyCostLimitStatusDto | null
  disabled: boolean
  onChange: (value: AccessKeyCostLimitRuleDraft[]) => void
}) {
  const t = useT()

  const totalRules = value.filter((rule) => rule.kind === 'total')
  const periodicRules = value.filter((rule) => rule.kind === 'periodic')
  const totalCount = totalRules.length
  const periodicCount = periodicRules.length
  const runtimeByID = new Map((runtimeStatus?.rules ?? []).map((rule) => [rule.id, rule]))
  const runtimeFor = (rule: AccessKeyCostLimitRuleDraft) =>
    rule.id === undefined ? undefined : runtimeByID.get(rule.id)

  const periodUnitOptions = periodUnits.map((unit) => ({
    value: unit.value,
    label: t(`accessKeys.drawer.costLimits.units.${unit.value}` as MessageId),
  }))

  const updateRule = (clientKey: string, patch: Partial<AccessKeyCostLimitRuleDraft>): void => {
    onChange(value.map((rule) => (rule.clientKey === clientKey ? { ...rule, ...patch } : rule)))
  }

  const removeRule = (clientKey: string): void => {
    onChange(value.filter((rule) => rule.clientKey !== clientKey))
  }

  const removeRules = (kind: 'total' | 'periodic'): void => {
    onChange(value.filter((rule) => rule.kind !== kind))
  }

  const addRule = (kind: 'total' | 'periodic'): void => {
    if (disabled || (kind === 'total' ? totalCount >= 1 : periodicCount >= 10)) return
    onChange([
      ...value,
      {
        clientKey: createUUID(),
        kind,
        limit_usd: kind === 'total' ? '100' : '20',
        ...(kind === 'periodic' ? { period_seconds: 18_000 } : {}),
      },
    ])
  }

  // `totalEnabled`/`periodicEnabled` computed setters in classic.
  const setKindEnabled = (kind: 'total' | 'periodic', enabled: boolean): void => {
    if (disabled) return
    if (enabled) {
      if ((kind === 'total' ? totalCount : periodicCount) === 0) addRule(kind)
      return
    }
    removeRules(kind)
  }

  const updatePeriodValue = (rule: AccessKeyCostLimitRuleDraft, raw: string): void => {
    const parsed = Number(raw)
    updateRule(rule.clientKey, {
      period_seconds: Number.isFinite(parsed)
        ? parsed * unitSeconds(periodUnit(rule.period_seconds))
        : 0,
    })
  }

  const updatePeriodUnit = (rule: AccessKeyCostLimitRuleDraft, unit: PeriodUnit): void => {
    updateRule(rule.clientKey, {
      period_seconds: periodValue(rule.period_seconds) * unitSeconds(unit),
    })
  }

  const amountLabel = t('accessKeys.drawer.costLimits.amount')
  const periodLabel = t('accessKeys.drawer.costLimits.period')
  const unitLabel = t('accessKeys.drawer.costLimits.unit')
  const totalLabel = t('accessKeys.drawer.costLimits.total')
  const periodicLabel = t('accessKeys.drawer.costLimits.periodic')

  return (
    <div {...stylex.props(styles.editor)}>
      <section {...stylex.props(styles.section)}>
        <header {...stylex.props(styles.sectionHeader)}>
          <div {...stylex.props(styles.sectionIntro)}>
            <h3 {...stylex.props(styles.sectionTitle)}>{totalLabel}</h3>
            <p {...stylex.props(styles.sectionDescription)}>
              {t('accessKeys.drawer.costLimits.totalDescription')}
            </p>
          </div>
          <div {...stylex.props(styles.sectionToggle)}>
            <Switch
              value={totalCount > 0}
              isDisabled={disabled}
              label={t('accessKeys.drawer.costLimits.enableTotal')}
              isLabelHidden
              size="sm"
              onChange={(enabled) => setKindEnabled('total', enabled)}
            />
          </div>
        </header>

        {totalCount > 0 && (
          <div {...stylex.props(styles.sectionContent)}>
            <div {...stylex.props(styles.fieldsHeader, styles.fieldsHeaderTotal)}>
              <span>{amountLabel}</span>
            </div>
            {totalRules.map((rule) => {
              const runtime = runtimeFor(rule)
              return (
                <article key={rule.clientKey} {...stylex.props(styles.rule)}>
                  <InputGroup label={totalLabel} periodic={false}>
                    <TextInput
                      id={`cost-limit-amount-${rule.clientKey}`}
                      label={amountLabel}
                      isLabelHidden
                      size="sm"
                      value={rule.limit_usd}
                      isDisabled={disabled}
                      startIcon={<span aria-hidden="true">$</span>}
                      xstyle={[styles.embeddedCell, styles.embeddedTextInput]}
                      data-gptload-mono
                      {...decimalInputAttrs}
                      onChange={(next) => updateRule(rule.clientKey, { limit_usd: next })}
                    />
                  </InputGroup>
                  {runtime !== undefined && (
                    <RuleRuntime
                      runtime={runtime}
                      label={totalLabel}
                      tail={
                        runtime.status === 'exhausted'
                          ? t('accessKeys.costLimits.notAutomatic')
                          : undefined
                      }
                    />
                  )}
                </article>
              )
            })}
          </div>
        )}
      </section>

      <section {...stylex.props(styles.section, styles.sectionSibling)}>
        <header {...stylex.props(styles.sectionHeader)}>
          <div {...stylex.props(styles.sectionIntro)}>
            <h3 {...stylex.props(styles.sectionTitle)}>{periodicLabel}</h3>
            <p {...stylex.props(styles.sectionDescription)}>
              {t('accessKeys.drawer.costLimits.periodicDescription')}
            </p>
          </div>
          <div {...stylex.props(styles.sectionToggle)}>
            <Switch
              value={periodicCount > 0}
              isDisabled={disabled}
              label={t('accessKeys.drawer.costLimits.enablePeriodic')}
              isLabelHidden
              size="sm"
              onChange={(enabled) => setKindEnabled('periodic', enabled)}
            />
          </div>
        </header>

        {periodicCount > 0 && (
          <div {...stylex.props(styles.sectionContent)}>
            <div {...stylex.props(styles.fieldsHeader, styles.fieldsHeaderPeriodic)}>
              <span>{amountLabel}</span>
              <span>{periodLabel}</span>
              <span>{unitLabel}</span>
              <span aria-hidden="true" />
            </div>
            <div {...stylex.props(styles.ruleList)}>
              {periodicRules.map((rule, index) => {
                const runtime = runtimeFor(rule)
                return (
                  <article key={rule.clientKey} {...stylex.props(styles.rule)}>
                    <div {...stylex.props(styles.ruleRow)}>
                      <InputGroup label={periodicLabel} periodic>
                        <TextInput
                          id={`cost-limit-amount-${rule.clientKey}`}
                          label={amountLabel}
                          isLabelHidden
                          size="sm"
                          value={rule.limit_usd}
                          isDisabled={disabled}
                          startIcon={<span aria-hidden="true">$</span>}
                          xstyle={[styles.embeddedCell, styles.embeddedTextInput]}
                          data-gptload-mono
                          {...decimalInputAttrs}
                          onChange={(next) => updateRule(rule.clientKey, { limit_usd: next })}
                        />
                        <TextInput
                          id={`cost-limit-period-${rule.clientKey}`}
                          label={periodLabel}
                          isLabelHidden
                          size="sm"
                          value={String(periodValue(rule.period_seconds))}
                          isDisabled={disabled}
                          xstyle={[
                            styles.embeddedCell,
                            styles.embeddedTextInput,
                            styles.embeddedDivider,
                          ]}
                          data-gptload-mono
                          {...numericInputAttrs}
                          onChange={(raw) => updatePeriodValue(rule, raw)}
                        />
                        <Selector
                          label={unitLabel}
                          isLabelHidden
                          variant="ghost"
                          size="sm"
                          width="100%"
                          options={periodUnitOptions}
                          value={periodUnit(rule.period_seconds)}
                          isDisabled={disabled}
                          xstyle={[
                            styles.embeddedCell,
                            styles.embeddedDivider,
                            styles.embeddedSelector,
                          ]}
                          onChange={(unit) => {
                            const periodUnitValue = periodUnits.find(
                              (candidate) => candidate.value === unit,
                            )
                            if (periodUnitValue) updatePeriodUnit(rule, periodUnitValue.value)
                          }}
                        />
                      </InputGroup>
                      <div {...stylex.props(styles.ruleActions)}>
                        {index > 0 ? (
                          <IconButton
                            variant="ghost"
                            size="sm"
                            label={t('accessKeys.drawer.costLimits.remove')}
                            icon={<X size={14} aria-hidden="true" />}
                            isDisabled={disabled}
                            xstyle={disabled ? styles.ruleActionTint : styles.ruleRemoveTint}
                            onClick={() => removeRule(rule.clientKey)}
                          />
                        ) : (
                          <span aria-hidden="true" />
                        )}
                        {index === periodicRules.length - 1 ? (
                          <IconButton
                            variant="ghost"
                            size="sm"
                            label={t('accessKeys.drawer.costLimits.addPeriodic')}
                            icon={<Plus size={14} aria-hidden="true" />}
                            isDisabled={disabled || periodicCount >= 10}
                            xstyle={
                              disabled || periodicCount >= 10
                                ? styles.ruleActionTint
                                : styles.ruleAddTint
                            }
                            onClick={() => addRule('periodic')}
                          />
                        ) : (
                          <span aria-hidden="true" />
                        )}
                      </div>
                    </div>
                    {runtime !== undefined && (
                      <RuleRuntime
                        runtime={runtime}
                        label={periodicLabel}
                        tail={<AccessKeyCostLimitWindowTime rule={runtime} />}
                      />
                    )}
                  </article>
                )
              })}
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
