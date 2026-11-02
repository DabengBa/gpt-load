import { IconButton } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { ArrowDown, ArrowUp, ChevronDown, Copy, Trash2 } from 'lucide-react'

import {
  ParameterOverrideRuleParams,
  type ParameterRuleHandlers,
  type ParameterRuleOptions,
} from './ParameterOverrideRuleParams'
import {
  ruleInvalid,
  type RuleErrors,
  type RuleRow,
  type RuleSummary,
  type Translate,
} from './parameter-override-rules'

export interface ParameterOverrideRuleCardProps extends ParameterRuleHandlers {
  row: RuleRow
  index: number
  count: number
  instanceId: string
  disabled: boolean
  errors: RuleErrors | undefined
  summary: RuleSummary | undefined
  matches: number | undefined
  switchBlocked: boolean
  options: ParameterRuleOptions
  tr: Translate
  onToggleOpen(row: RuleRow): void
  onMove(index: number, offset: -1 | 1): void
  onCopy(index: number): void
  onRemove(index: number): void
}

/**
 * One override rule: a collapsed summary row (match tags + value chips) that
 * expands into the full editor. Collapsing is the default because an operator
 * usually scans which parameters a group overrides before editing any of them.
 */
export function ParameterOverrideRuleCard({
  row,
  index,
  count,
  instanceId,
  disabled,
  errors,
  summary,
  matches,
  switchBlocked,
  options,
  tr,
  onToggleOpen,
  onMove,
  onCopy,
  onRemove,
  ...handlers
}: ParameterOverrideRuleCardProps) {
  function protocolLabel(value: string): string {
    return value || tr('group.settings.parameterOverrides.allProtocols')
  }

  function modelLabel(target: RuleRow): string {
    return target.model.trim() || tr('group.settings.parameterOverrides.allModels')
  }

  return (
    <article
      key={row.key}
      {...stylex.props(styles.rule, ruleInvalid(errors) && styles.ruleInvalid)}
    >
      <div {...stylex.props(styles.ruleHead)}>
        <button
          type="button"
          {...stylex.props(styles.summaryButton)}
          aria-expanded={row.open}
          aria-controls={`${instanceId}-rule-${row.key}`}
          onClick={() => onToggleOpen(row)}
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
            label={tr('group.settings.parameterOverrides.moveUp')}
            isDisabled={disabled || index === 0}
            onClick={() => onMove(index, -1)}
            icon={<ArrowUp size={15} aria-hidden="true" />}
          />
          <IconButton
            variant="ghost"
            size="sm"
            label={tr('group.settings.parameterOverrides.moveDown')}
            isDisabled={disabled || index === count - 1}
            onClick={() => onMove(index, 1)}
            icon={<ArrowDown size={15} aria-hidden="true" />}
          />
          <span {...stylex.props(styles.toolsGap)} aria-hidden="true" />
          <IconButton
            variant="ghost"
            size="sm"
            label={tr('group.settings.parameterOverrides.copy')}
            isDisabled={disabled}
            onClick={() => onCopy(index)}
            icon={<Copy size={15} aria-hidden="true" />}
          />
          <IconButton
            variant="ghost"
            size="sm"
            label={tr('group.settings.parameterOverrides.deleteRule')}
            isDisabled={disabled}
            onClick={() => onRemove(index)}
            icon={<Trash2 size={15} aria-hidden="true" />}
            xstyle={styles.dangerIcon}
          />
        </div>
      </div>

      {row.open && (
        <ParameterOverrideRuleParams
          row={row}
          errors={errors}
          options={options}
          matches={matches}
          instanceId={instanceId}
          disabled={disabled}
          switchBlocked={switchBlocked}
          tr={tr}
          {...handlers}
        />
      )}
    </article>
  )
}

const styles = stylex.create({
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
})
