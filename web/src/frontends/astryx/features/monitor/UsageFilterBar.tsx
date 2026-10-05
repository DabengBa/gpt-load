import * as stylex from '@stylexjs/stylex'
import { Button, Selector, TextInput } from '@astryxdesign/core'
import { useMemo, type FormEvent, type JSX } from 'react'

import type { GroupOptionDto } from '@shared/control/types'
import { usageRanges } from '@shared/control/resources/usage'
import type { UsageFilterDraft, UsageFilterErrors } from '@shared/domain/monitor/usage-filters'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { numericInputAttrs, plainTextInputAttrs } from '../../components/input-attrs'

// LogsFilterForm-style inline bar: sunken card shell, single control row that
// wraps and drops to a full-width grid at 560px. The usage surface has three
// filters only — range, group (admin principals), upstream model.

const NARROW = '@media (max-width: 560px)'

const styles = stylex.create({
  form: {
    display: 'grid',
    minWidth: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface-sunken)',
  },
  row: {
    display: {
      default: 'flex',
      [NARROW]: 'grid',
    },
    gridTemplateColumns: {
      default: 'none',
      [NARROW]: 'repeat(2, minmax(0, 1fr))',
    },
    minWidth: 0,
    alignItems: {
      default: 'center',
      [NARROW]: 'stretch',
    },
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    padding: 'var(--space-2-5)',
  },
  field: {
    display: 'block',
    minWidth: 0,
    width: {
      default: 150,
      [NARROW]: '100%',
    },
    gridColumn: {
      default: 'auto',
      [NARROW]: '1 / -1',
    },
  },
  fieldModel: {
    display: 'block',
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: 180,
    minWidth: {
      default: 130,
      [NARROW]: 0,
    },
    width: {
      default: 'auto',
      [NARROW]: '100%',
    },
    gridColumn: {
      default: 'auto',
      [NARROW]: '1 / -1',
    },
  },
  error: {
    marginTop: -2,
    marginBottom: 'var(--space-2)',
    marginInline: 'var(--space-2-5)',
    color: 'var(--color-danger)',
    fontSize: 'var(--text-label-xs)',
  },
})

export interface UsageFilterBarProps {
  draft: UsageFilterDraft
  errors: UsageFilterErrors
  groups: GroupOptionDto[]
  groupsFailed: boolean
  onFieldChange(field: keyof UsageFilterDraft, value: string): void
  onApply(): void
  onReset(): void
  /**
   * Access-key session view: hides the Group selector and swaps the
   * model Selector for a free-text input, matching LogsFilterForm.
   */
  selfScoped?: boolean
}

export function UsageFilterBar({
  draft,
  errors,
  groups,
  groupsFailed,
  onFieldChange,
  onApply,
  onReset,
  selfScoped,
}: UsageFilterBarProps): JSX.Element {
  const t = useT()

  const rangeOptions = useMemo(
    () =>
      usageRanges.map((value) => ({
        value,
        label: t(`monitor.usage.filters.ranges.${value}` as MessageId),
      })),
    [t],
  )

  function groupOptions() {
    const options = [{ value: '', label: t('monitor.usage.filters.anyGroup') }]
    if (draft.group_id !== '' && !groups.some((group) => String(group.id) === draft.group_id)) {
      options.push({
        value: draft.group_id,
        label: t('monitor.usage.filters.deletedOrUnknownGroup'),
      })
    }
    return [...options, ...groups.map((group) => ({ value: String(group.id), label: group.name }))]
  }

  // Upstream-model options mirror the logs client-model rule: declared models
  // of the group catalog, exact match, pending values appended so they stay
  // selectable.
  function modelOptions() {
    const models = [...new Set(groups.flatMap((group) => group.models))].sort((a, b) =>
      a.localeCompare(b),
    )
    const options = [
      { value: '', label: t('monitor.usage.filters.anyModel') },
      ...models.map((model) => ({ value: model, label: model })),
    ]
    if (draft.upstream_model !== '' && !models.includes(draft.upstream_model)) {
      options.push({ value: draft.upstream_model, label: draft.upstream_model })
    }
    return options
  }

  function fieldError(field: keyof UsageFilterErrors): { type: 'error' } | undefined {
    return errors[field] === undefined || errors[field] === '' ? undefined : { type: 'error' }
  }

  const firstError = Object.values(errors).find((value) => value !== undefined && value !== '')

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    onApply()
  }

  const modelLabel = t('monitor.usage.filters.model')
  const groupLabel = t('monitor.usage.filters.group')
  const noMatches = t('monitor.usage.filters.noMatches')

  return (
    <form
      {...stylex.props(styles.form)}
      aria-label={t('monitor.usage.filters.label')}
      onSubmit={submit}
    >
      <div {...stylex.props(styles.row)}>
        <span {...stylex.props(styles.field)}>
          <Selector
            label={t('monitor.usage.filters.range')}
            isLabelHidden
            size="sm"
            width="100%"
            options={rangeOptions}
            value={draft.range}
            onChange={(value) => onFieldChange('range', String(value))}
          />
        </span>
        {selfScoped !== true && (
          <span {...stylex.props(styles.field)}>
            {groupsFailed ? (
              <TextInput
                label={groupLabel}
                isLabelHidden
                size="sm"
                width="100%"
                value={draft.group_id}
                onChange={(value) => onFieldChange('group_id', value)}
                placeholder={t('monitor.usage.filters.groupPlaceholder')}
                status={fieldError('group_id')}
                aria-describedby={
                  fieldError('group_id') !== undefined ? 'usage-filter-error' : undefined
                }
                {...numericInputAttrs}
              />
            ) : (
              <Selector
                label={groupLabel}
                isLabelHidden
                size="sm"
                width="100%"
                hasSearch
                searchPlaceholder={t('monitor.usage.filters.searchGroups')}
                emptyText={noMatches}
                emptySearchText={noMatches}
                options={groupOptions()}
                value={draft.group_id}
                onChange={(value) => onFieldChange('group_id', String(value))}
                status={fieldError('group_id')}
              />
            )}
          </span>
        )}
        <span {...stylex.props(styles.fieldModel)}>
          {selfScoped === true ? (
            <TextInput
              label={modelLabel}
              isLabelHidden
              size="sm"
              width="100%"
              value={draft.upstream_model}
              onChange={(value) => onFieldChange('upstream_model', value)}
              placeholder={t('monitor.usage.filters.modelPlaceholder')}
              status={fieldError('upstream_model')}
              aria-describedby={
                fieldError('upstream_model') !== undefined ? 'usage-filter-error' : undefined
              }
              {...plainTextInputAttrs}
            />
          ) : (
            <Selector
              label={modelLabel}
              isLabelHidden
              size="sm"
              width="100%"
              hasSearch
              searchPlaceholder={t('monitor.usage.filters.searchModels')}
              emptyText={noMatches}
              emptySearchText={noMatches}
              options={modelOptions()}
              value={draft.upstream_model}
              onChange={(value) => onFieldChange('upstream_model', String(value))}
              isDisabled={groupsFailed}
              status={fieldError('upstream_model')}
            />
          )}
        </span>
        <Button
          type="submit"
          variant="primary"
          size="sm"
          label={t('monitor.usage.filters.apply')}
        />
        <Button
          variant="secondary"
          size="sm"
          label={t('monitor.usage.filters.reset')}
          onClick={onReset}
        />
      </div>
      {firstError !== undefined && (
        <p id="usage-filter-error" {...stylex.props(styles.error)} role="alert">
          {t(firstError as MessageId)}
        </p>
      )}
    </form>
  )
}
