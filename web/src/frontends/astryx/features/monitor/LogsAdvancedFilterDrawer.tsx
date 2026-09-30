import * as stylex from '@stylexjs/stylex'
import { Button, Selector, TextInput } from '@astryxdesign/core'
import type { FormEvent, JSX } from 'react'

import { enabledDataProtocols } from '@shared/control/protocols'
import type { ChannelDto } from '@shared/control/resources/channels'
import type { AccessKeyOptionDto } from '@shared/control/types'
import {
  requestLogCostStates,
  requestLogFailureCategories,
  requestLogModelConsistencies,
  requestLogOperations,
  requestLogPricingCompleteness,
  requestLogRetryStates,
  requestLogUsageStates,
  type LogFilterDraft,
  type LogFilterErrors,
} from '@shared/domain/monitor/log-filters'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { DetailPanel } from '../../components/DetailPanel'
import {
  decimalInputAttrs,
  numericInputAttrs,
  plainTextInputAttrs,
} from '../../components/input-attrs'

// Classic breakpoints: the two-column field grid collapses to one column at
// <=520px, and the footer action pair switches to a 2-up grid there.
const NARROW = '@media (max-width: 520px)'

const styles = stylex.create({
  form: {
    display: 'grid',
    minWidth: 0,
    gap: 0,
  },
  section: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-3)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBlock: 'var(--space-4)',
  },
  // Classic .logs-advanced__section:last-child — the trailing section drops
  // its divider.
  sectionFlush: {
    borderBottomWidth: 0,
  },
  sectionTitle: {
    margin: 0,
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
    fontWeight: 650,
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      [NARROW]: 'minmax(0, 1fr)',
    },
    columnGap: 10,
    rowGap: 12,
  },
  // Classic ledger drawer footer: space-between row that stacks at <=520px.
  footer: {
    display: 'flex',
    width: '100%',
    alignItems: {
      default: 'center',
      [NARROW]: 'stretch',
    },
    flexDirection: {
      default: 'row',
      [NARROW]: 'column',
    },
    justifyContent: 'space-between',
    gap: 14,
  },
  footerActions: {
    display: {
      default: 'inline-flex',
      [NARROW]: 'grid',
    },
    gridTemplateColumns: {
      default: 'none',
      [NARROW]: 'repeat(2, minmax(0, 1fr))',
    },
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
})

interface SelectorOption {
  value: string
  label: string
}

// Numeric range fields rendered in the classic drawer's "ranges" section, in
// template order.
const rangeFields = [
  'first_response_min_ms',
  'first_response_max_ms',
  'duration_min_ms',
  'duration_max_ms',
  'input_tokens_min',
  'input_tokens_max',
  'output_tokens_min',
  'output_tokens_max',
  'cost_min_usd',
  'cost_max_usd',
] as const satisfies readonly (keyof LogFilterDraft)[]

export interface LogsAdvancedFilterDrawerProps {
  open: boolean
  draft: LogFilterDraft
  errors: LogFilterErrors
  accessKeys: AccessKeyOptionDto[]
  channels: ChannelDto[]
  accessKeysFailed: boolean
  channelsFailed: boolean
  onOpenChange: (open: boolean) => void
  onUpdateField: (field: keyof LogFilterDraft, value: string) => void
  onApply: () => void
  onReset: () => void
  /**
   * Classic `selfScoped` (access-key session view): hides the access-key and
   * affinity-key fields plus the whole retry-attempt section, which a scoped
   * principal is not allowed to filter on.
   */
  selfScoped?: boolean
}

/**
 * Advanced request-log filter drawer — classic LogsAdvancedFilterDrawer.vue
 * with AppDrawer(ledger) + FormField + AppSelect/raw inputs. Maps to
 * DetailPanel + Selector/TextInput (which compose Field internally, so
 * `status` replaces the FormField error plumbing); emits become
 * `onOpenChange`, `onUpdateField`, `onApply`, `onReset`. The classic `apply`
 * emit only notified the parent — the parent owns `open` and decides whether
 * the drawer closes, so Apply here never touches `onOpenChange`.
 */
export function LogsAdvancedFilterDrawer({
  open,
  draft,
  errors,
  accessKeys,
  channels,
  accessKeysFailed,
  channelsFailed,
  onOpenChange,
  onUpdateField,
  onApply,
  onReset,
  selfScoped,
}: LogsAdvancedFilterDrawerProps): JSX.Element {
  const t = useT()

  const option = (value: string, label: string): SelectorOption => ({ value, label })

  function booleanOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.any')),
      option('true', t('monitor.logs.yes')),
      option('false', t('monitor.logs.no')),
    ]
  }

  function protocolOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.anyProtocol')),
      ...enabledDataProtocols.map((value) => option(value, value)),
    ]
  }

  function operationOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.anyOperation')),
      ...requestLogOperations.map((value) =>
        option(value, t(`monitor.logs.operation.${value}` as MessageId)),
      ),
    ]
  }

  function usageOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.any')),
      ...requestLogUsageStates.map((value) =>
        option(value, t(`monitor.logs.filters.usageState.${value}` as MessageId)),
      ),
    ]
  }

  function costOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.any')),
      ...requestLogCostStates.map((value) =>
        option(value, t(`monitor.logs.filters.costState.${value}` as MessageId)),
      ),
    ]
  }

  function completenessOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.any')),
      ...requestLogPricingCompleteness.map((value) =>
        option(value, t(`monitor.logs.filters.completeness.${value}` as MessageId)),
      ),
    ]
  }

  function failureOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.any')),
      ...requestLogFailureCategories.map((value) =>
        option(value, t(`monitor.logs.failureCategory.${value}` as MessageId)),
      ),
    ]
  }

  function retryOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.any')),
      ...requestLogRetryStates.map((value) =>
        option(value, t(`monitor.logs.filters.retryState.${value}` as MessageId)),
      ),
    ]
  }

  function modelConsistencyOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.any')),
      ...requestLogModelConsistencies.map((value) =>
        option(value, t(`monitor.logs.filters.modelConsistency.${value}` as MessageId)),
      ),
    ]
  }

  function accessKeyOptions(): SelectorOption[] {
    return [
      option('', t('monitor.logs.filters.anyAccessKey')),
      ...accessKeys.map((key) => option(String(key.id), `${key.name} · #${key.id}`)),
    ]
  }

  function channelOptions(): SelectorOption[] {
    const options = [option('', t('monitor.logs.filters.anyChannel'))]
    if (
      draft.channel_id !== '' &&
      !channels.some((channel) => channel.channel_id === draft.channel_id)
    ) {
      options.push(option(draft.channel_id, draft.channel_id))
    }
    return [...options, ...channels.map((channel) => option(channel.channel_id, channel.name))]
  }

  function error(field: keyof LogFilterDraft): { type: 'error'; message: string } | undefined {
    const key = errors[field]
    return key === undefined || key === ''
      ? undefined
      : { type: 'error', message: t(key as MessageId) }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    onApply()
  }

  const channelLabel = t('monitor.logs.filters.channel')

  return (
    <DetailPanel
      isOpen={open}
      onOpenChange={onOpenChange}
      title={t('monitor.logs.filters.advancedTitle')}
      subtitle={t('monitor.logs.filters.advancedDescription')}
      footer={
        <div {...stylex.props(styles.footer)}>
          <Button
            variant="secondary"
            size="sm"
            label={t('monitor.logs.filters.reset')}
            onClick={onReset}
          />
          <span {...stylex.props(styles.footerActions)}>
            <Button
              variant="secondary"
              size="sm"
              label={t('common.cancel')}
              onClick={() => onOpenChange(false)}
            />
            <Button
              variant="primary"
              size="sm"
              label={t('monitor.logs.filters.apply')}
              onClick={onApply}
            />
          </span>
        </div>
      }
    >
      <form {...stylex.props(styles.form)} onSubmit={submit}>
        <section {...stylex.props(styles.section)}>
          <h3 {...stylex.props(styles.sectionTitle)}>
            {t('monitor.logs.filters.sections.request')}
          </h3>
          <div {...stylex.props(styles.grid)}>
            {selfScoped !== true && (
              <Selector
                id="logs-access-key"
                label={t('monitor.logs.filters.accessKey')}
                size="sm"
                options={accessKeyOptions()}
                value={draft.access_key_id}
                onChange={(value) => onUpdateField('access_key_id', value)}
                isDisabled={accessKeysFailed}
                status={error('access_key_id')}
              />
            )}
            {selfScoped !== true && (
              <TextInput
                id="logs-affinity-key"
                label={t('monitor.logs.filters.affinityKey')}
                size="sm"
                value={draft.affinity_key}
                onChange={(value) => onUpdateField('affinity_key', value)}
                status={error('affinity_key')}
                autoComplete="off"
                data-gptload-mono
                {...plainTextInputAttrs}
              />
            )}
            <Selector
              id="logs-protocol"
              label={t('monitor.logs.filters.protocol')}
              size="sm"
              options={protocolOptions()}
              value={draft.protocol}
              onChange={(value) => onUpdateField('protocol', value)}
            />
            <Selector
              id="logs-operation"
              label={t('monitor.logs.filters.operation')}
              size="sm"
              options={operationOptions()}
              value={draft.operation}
              onChange={(value) => onUpdateField('operation', value)}
            />
            <TextInput
              id="logs-request-id"
              label={t('monitor.logs.filters.requestId')}
              size="sm"
              value={draft.request_id}
              onChange={(value) => onUpdateField('request_id', value)}
              status={error('request_id')}
              autoComplete="off"
              data-gptload-mono
              {...plainTextInputAttrs}
            />
            <Selector
              id="logs-stream"
              label={t('monitor.logs.filters.stream')}
              size="sm"
              options={booleanOptions()}
              value={draft.stream}
              onChange={(value) => onUpdateField('stream', value)}
            />
          </div>
        </section>

        {selfScoped !== true && (
          <section {...stylex.props(styles.section)}>
            <h3 {...stylex.props(styles.sectionTitle)}>
              {t('monitor.logs.filters.sections.attempt')}
            </h3>
            <div {...stylex.props(styles.grid)}>
              {channelsFailed ? (
                <TextInput
                  id="logs-channel"
                  label={channelLabel}
                  size="sm"
                  value={draft.channel_id}
                  onChange={(value) => onUpdateField('channel_id', value)}
                  status={error('channel_id')}
                  autoComplete="off"
                  {...plainTextInputAttrs}
                />
              ) : (
                <Selector
                  id="logs-channel"
                  label={channelLabel}
                  size="sm"
                  options={channelOptions()}
                  value={draft.channel_id}
                  onChange={(value) => onUpdateField('channel_id', value)}
                  status={error('channel_id')}
                />
              )}
              <TextInput
                id="logs-credential"
                label={t('monitor.logs.filters.credential')}
                size="sm"
                value={draft.credential_id}
                onChange={(value) => onUpdateField('credential_id', value)}
                status={error('credential_id')}
                autoComplete="off"
                {...numericInputAttrs}
              />
              <TextInput
                id="logs-upstream-model"
                label={t('monitor.logs.filters.upstreamModel')}
                size="sm"
                value={draft.upstream_model}
                onChange={(value) => onUpdateField('upstream_model', value)}
                status={error('upstream_model')}
                autoComplete="off"
                {...plainTextInputAttrs}
              />
              <Selector
                id="logs-model-consistency"
                label={t('monitor.logs.filters.modelConsistencyLabel')}
                size="sm"
                options={modelConsistencyOptions()}
                value={draft.model_consistency}
                onChange={(value) => onUpdateField('model_consistency', value)}
              />
              <Selector
                id="logs-retry-state"
                label={t('monitor.logs.filters.retryStateLabel')}
                size="sm"
                options={retryOptions()}
                value={draft.retry_state}
                onChange={(value) => onUpdateField('retry_state', value)}
              />
              <TextInput
                id="logs-retry-min"
                label={t('monitor.logs.filters.retryMin')}
                size="sm"
                value={draft.retry_count_min}
                onChange={(value) => onUpdateField('retry_count_min', value)}
                status={error('retry_count_min')}
                autoComplete="off"
                {...numericInputAttrs}
              />
              <TextInput
                id="logs-retry-max"
                label={t('monitor.logs.filters.retryMax')}
                size="sm"
                value={draft.retry_count_max}
                onChange={(value) => onUpdateField('retry_count_max', value)}
                status={error('retry_count_max')}
                autoComplete="off"
                {...numericInputAttrs}
              />
              <TextInput
                id="logs-attempt-code"
                label={t('monitor.logs.filters.attemptStatusCode')}
                size="sm"
                value={draft.attempt_status_code}
                onChange={(value) => onUpdateField('attempt_status_code', value)}
                status={error('attempt_status_code')}
                autoComplete="off"
                {...numericInputAttrs}
              />
              <Selector
                id="logs-failure"
                label={t('monitor.logs.filters.failureCategory')}
                size="sm"
                options={failureOptions()}
                value={draft.failure_category}
                onChange={(value) => onUpdateField('failure_category', value)}
              />
              <TextInput
                id="logs-error-code"
                label={t('monitor.logs.filters.errorCode')}
                size="sm"
                value={draft.error_code}
                onChange={(value) => onUpdateField('error_code', value)}
                status={error('error_code')}
                autoComplete="off"
                {...plainTextInputAttrs}
              />
            </div>
          </section>
        )}

        <section {...stylex.props(styles.section)}>
          <h3 {...stylex.props(styles.sectionTitle)}>
            {t('monitor.logs.filters.sections.result')}
          </h3>
          <div {...stylex.props(styles.grid)}>
            <TextInput
              id="logs-final-code"
              label={t('monitor.logs.filters.finalStatusCode')}
              size="sm"
              value={draft.final_status_code}
              onChange={(value) => onUpdateField('final_status_code', value)}
              status={error('final_status_code')}
              autoComplete="off"
              {...numericInputAttrs}
            />
            <Selector
              id="logs-usage-state"
              label={t('monitor.logs.filters.usageStateLabel')}
              size="sm"
              options={usageOptions()}
              value={draft.usage_state}
              onChange={(value) => onUpdateField('usage_state', value)}
            />
            <Selector
              id="logs-cost-state"
              label={t('monitor.logs.filters.costStateLabel')}
              size="sm"
              options={costOptions()}
              value={draft.cost_state}
              onChange={(value) => onUpdateField('cost_state', value)}
            />
            <Selector
              id="logs-completeness"
              label={t('monitor.logs.filters.completenessLabel')}
              size="sm"
              options={completenessOptions()}
              value={draft.pricing_completeness}
              onChange={(value) => onUpdateField('pricing_completeness', value)}
            />
            <Selector
              id="logs-cache"
              label={t('monitor.logs.filters.cachePresent')}
              size="sm"
              options={booleanOptions()}
              value={draft.cache_present}
              onChange={(value) => onUpdateField('cache_present', value)}
            />
          </div>
        </section>

        <section {...stylex.props(styles.section, styles.sectionFlush)}>
          <h3 {...stylex.props(styles.sectionTitle)}>
            {t('monitor.logs.filters.sections.ranges')}
          </h3>
          <div {...stylex.props(styles.grid)}>
            {rangeFields.map((field) => (
              <TextInput
                key={field}
                id={`logs-${field}`}
                label={t(`monitor.logs.filters.rangeFields.${field}` as MessageId)}
                size="sm"
                value={draft[field]}
                onChange={(value) => onUpdateField(field, value)}
                status={error(field)}
                autoComplete="off"
                {...decimalInputAttrs}
              />
            ))}
          </div>
        </section>
      </form>
    </DetailPanel>
  )
}
