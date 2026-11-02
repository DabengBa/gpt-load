import { Button, IconButton, Selector, Tooltip } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { Plus, X } from 'lucide-react'

import { InlineNotice } from '../../../components/InlineNotice'
import {
  pathsCrossing as pathsCross,
  type ParamRow,
  type RuleErrors,
  type RuleRow,
  type Translate,
} from './parameter-override-rules'

export interface ParameterRuleOption {
  value: string
  label: string
}

export interface ParameterRuleOptions {
  protocols: ParameterRuleOption[]
  operations: ParameterRuleOption[]
  kinds: ParameterRuleOption[]
  booleans: ParameterRuleOption[]
}

export interface ParameterRuleHandlers {
  onUpdateRule(key: number, patch: Partial<RuleRow>): void
  onUpdateParam(rowKey: number, paramKey: number, patch: Partial<ParamRow>): void
  onSetParamOp(row: RuleRow, param: ParamRow, value: string): void
  onSetParamKind(row: RuleRow, param: ParamRow, value: string): void
  onSetParamValue(row: RuleRow, param: ParamRow, text: string): void
  onRemoveParam(row: RuleRow, key: number): void
  onAddParam(row: RuleRow): void
  onToggleJSON(row: RuleRow): void
  onFormatSetText(row: RuleRow): void
}

export interface ParameterOverrideRuleParamsProps extends ParameterRuleHandlers {
  row: RuleRow
  errors: RuleErrors | undefined
  options: ParameterRuleOptions
  matches: number | undefined
  instanceId: string
  disabled: boolean
  switchBlocked: boolean
  tr: Translate
}

/**
 * The expanded half of a rule: match selectors plus the set/remove body. It is a
 * sibling of the rule card rather than a child so the collapsed summary stays a
 * cheap render while this only mounts when a rule is open.
 */
export function ParameterOverrideRuleParams({
  row,
  errors,
  options,
  matches,
  instanceId,
  disabled,
  switchBlocked,
  tr,
  onUpdateRule,
  onUpdateParam,
  onSetParamOp,
  onSetParamKind,
  onSetParamValue,
  onRemoveParam,
  onAddParam,
  onToggleJSON,
  onFormatSetText,
}: ParameterOverrideRuleParamsProps) {
  const protocolOptions = options.protocols
  const opOptions = options.operations
  const kindOptions = options.kinds
  const booleanOptions = options.booleans
  const pathsCrossing = pathsCross(row)

  return (
    <div id={`${instanceId}-rule-${row.key}`} {...stylex.props(styles.ruleBody)}>
      <div {...stylex.props(styles.field)}>
        <span {...stylex.props(styles.fieldLabel)}>
          {tr('group.settings.parameterOverrides.match')}
        </span>
        <div {...stylex.props(styles.matchInputs)}>
          <Selector
            label={tr('group.settings.parameterOverrides.protocol')}
            isLabelHidden
            options={protocolOptions}
            value={row.protocol}
            size="sm"
            isDisabled={disabled}
            onChange={(value) => onUpdateRule(row.key, { protocol: value })}
            controlXstyle={styles.protocolSelect}
          />
          <span {...stylex.props(styles.modelCell)}>
            <input
              {...stylex.props(styles.input, errors?.model !== undefined && styles.inputInvalid)}
              id={`${instanceId}-model-${row.key}`}
              value={row.model}
              list={`${instanceId}-models`}
              placeholder={tr('group.settings.parameterOverrides.allModels')}
              aria-label={tr('group.settings.parameterOverrides.model')}
              aria-invalid={errors?.model !== undefined ? true : undefined}
              disabled={disabled}
              autoComplete="off"
              onChange={(event) => onUpdateRule(row.key, { model: event.target.value })}
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
                ? tr('group.settings.parameterOverrides.modelNoMatch')
                : tr('group.settings.parameterOverrides.modelMatch', {
                    count: matches,
                  })}
            </span>
          )}
        </div>
      </div>

      <div {...stylex.props(styles.field, styles.fieldParams)}>
        <span {...stylex.props(styles.fieldLabel)}>
          {tr('group.settings.parameterOverrides.params')}
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
                aria-label={tr('group.settings.parameterOverrides.setLabel')}
                aria-invalid={errors?.set !== undefined ? true : undefined}
                disabled={disabled}
                spellCheck={false}
                onChange={(event) => onUpdateRule(row.key, { setText: event.target.value })}
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
                      {...stylex.props(styles.param, param.op === 'remove' && styles.paramDrop)}
                    >
                      {/* JSON 视图下 set 归 textarea，行只可能是删除，下拉就成了死控件。 */}
                      {row.json ? (
                        <span {...stylex.props(styles.opStatic)}>
                          {tr('group.settings.parameterOverrides.opRemove')}
                        </span>
                      ) : (
                        <span {...stylex.props(styles.op)}>
                          <Selector
                            label={tr('group.settings.parameterOverrides.paramAction')}
                            isLabelHidden
                            options={opOptions}
                            value={param.op}
                            size="sm"
                            isDisabled={disabled}
                            onChange={(value) => onSetParamOp(row, param, value)}
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
                            param.op === 'remove' ? 'generationConfig/topP' : 'thinking/type'
                          }
                          aria-label={tr('group.settings.parameterOverrides.paramPath')}
                          aria-invalid={paramError !== undefined ? true : undefined}
                          disabled={disabled}
                          spellCheck={false}
                          onChange={(event) =>
                            onUpdateParam(row.key, param.key, {
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
                              label={tr('group.settings.parameterOverrides.paramKind')}
                              isLabelHidden
                              options={kindOptions}
                              value={param.kind}
                              size="sm"
                              isDisabled={disabled}
                              onChange={(value) => onSetParamKind(row, param, value)}
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
                                label={tr('group.settings.parameterOverrides.paramValue')}
                                isLabelHidden
                                options={booleanOptions}
                                value={param.valueText.trim() === 'false' ? 'false' : 'true'}
                                size="sm"
                                isDisabled={disabled}
                                onChange={(value) =>
                                  onUpdateParam(row.key, param.key, {
                                    valueText: value,
                                  })
                                }
                              />
                            </span>
                          ) : (
                            <input
                              {...stylex.props(styles.input, styles.valueInput)}
                              value={param.valueText}
                              placeholder={param.kind === 'json' ? '[&quot;\n\n&quot;]' : '0.7'}
                              aria-label={tr('group.settings.parameterOverrides.paramValue')}
                              disabled={disabled}
                              spellCheck={false}
                              onChange={(event) =>
                                onUpdateParam(row.key, param.key, {
                                  valueText: event.target.value,
                                })
                              }
                              onBlur={(event) => onSetParamValue(row, param, event.target.value)}
                            />
                          )}
                        </>
                      )}
                      <IconButton
                        variant="ghost"
                        size="sm"
                        label={tr('group.settings.parameterOverrides.deleteParam')}
                        isDisabled={disabled}
                        onClick={() => onRemoveParam(row, param.key)}
                        icon={<X size={15} aria-hidden="true" />}
                      />
                    </div>
                  )
                })}
            </div>
          )}

          {pathsCrossing && (
            <InlineNotice tone="warning" appearance="ledger-hint">
              {tr('group.settings.parameterOverrides.pathsCross')}
            </InlineNotice>
          )}

          <div {...stylex.props(styles.paramsFoot)}>
            <Button
              variant="ghost"
              size="sm"
              isDisabled={disabled}
              onClick={() => onAddParam(row)}
              icon={<Plus size={14} aria-hidden="true" />}
              label={
                row.json
                  ? tr('group.settings.parameterOverrides.addRemovePath')
                  : tr('group.settings.parameterOverrides.addParam')
              }
              xstyle={styles.linkButton}
            />
            <span {...stylex.props(styles.paramsActions)}>
              {row.json && (
                <Button
                  variant="ghost"
                  size="sm"
                  isDisabled={disabled || !row.setText.trim()}
                  onClick={() => onFormatSetText(row)}
                  label={tr('group.settings.parameterOverrides.formatJSON')}
                  xstyle={styles.linkButton}
                />
              )}
              <Button
                variant="ghost"
                size="sm"
                isDisabled={disabled || switchBlocked}
                onClick={() => onToggleJSON(row)}
                label={
                  row.json
                    ? tr('group.settings.parameterOverrides.toRows')
                    : tr('group.settings.parameterOverrides.toJSON')
                }
                xstyle={styles.linkButton}
              />
              <Tooltip
                content={`${tr('group.settings.parameterOverrides.pathHint')}　${tr('group.settings.parameterOverrides.mergeHint')}`}
              >
                <button
                  {...stylex.props(styles.hint)}
                  type="button"
                  aria-label={tr('group.settings.parameterOverrides.pathHint')}
                >
                  ?
                </button>
              </Tooltip>
            </span>
          </div>
        </div>
      </div>

      {errors?.action !== undefined && <InlineNotice tone="danger">{errors.action}</InlineNotice>}
    </div>
  )
}

const styles = stylex.create({
  bool: {
    minWidth: '92px',
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
  errorText: {
    color: 'var(--color-danger)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 'var(--line-normal)',
  },
  field: {
    display: 'grid',
    gap: '6px',
  },
  fieldLabel: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
  },
  fieldParams: {},
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
  kind: {
    minWidth: '92px',
  },
  linkButton: {
    paddingLeft: 0,
    paddingRight: 0,
    color: 'var(--color-action)',
    fontWeight: 560,
  },
  matchInputs: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 170px) minmax(0, 1fr) auto',
    alignItems: 'start',
    gap: 'var(--space-2)',
  },
  modelCell: {
    display: 'grid',
    minWidth: 0,
    gap: '4px',
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
  param: {
    display: 'grid',
    gridTemplateColumns: 'auto minmax(0, 1fr) auto minmax(0, 1fr) auto',
    alignItems: 'start',
    gap: '6px',
  },
  paramDrop: {
    gridTemplateColumns: 'auto minmax(0, 1fr) auto',
  },
  params: {
    display: 'grid',
    gap: '6px',
  },
  paramsActions: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  paramsFoot: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
  },
  paramsStack: {
    display: 'grid',
    gap: 'var(--space-2)',
  },
  path: {},
  pathCell: {
    display: 'grid',
    minWidth: 0,
    gap: '4px',
  },
  protocolSelect: {
    minWidth: 0,
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
  textarea: {
    minHeight: '96px',
    paddingTop: '7px',
    paddingBottom: '7px',
    resize: 'vertical',
    lineHeight: 1.55,
  },
  valueInput: {},
  valueNone: {
    alignSelf: 'center',
    color: 'var(--color-text-faint)',
    paddingLeft: '6px',
  },
})
