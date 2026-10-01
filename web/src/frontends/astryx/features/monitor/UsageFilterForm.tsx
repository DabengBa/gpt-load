import * as stylex from '@stylexjs/stylex'
import { Button, Selector, TextInput } from '@astryxdesign/core'
import type { FormEvent } from 'react'

import type { GroupOptionDto } from '@shared/control/types'
import type { ChannelDto } from '@shared/control/resources/channels'
import type { UsageFilterDraft, UsageFilterErrors } from '@shared/domain/monitor/usage-filters'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { DetailPanel } from '../../components/DetailPanel'
import { numericInputAttrs, plainTextInputAttrs } from '../../components/input-attrs'

const NARROW = '@media (max-width: 520px)'

const styles = stylex.create({
  form: {
    minWidth: 0,
    paddingBlock: 16,
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

/**
 * Controlled usage/cost filter drawer — classic UsageFilterForm.vue with
 * AppDrawer(ledger) + FormField + AppSelect/AppButton. Maps to DetailPanel +
 * Selector/TextInput/Button; emits become `onOpenChange`, `onFieldChange`,
 * `onApply`, `onReset` props.
 */
export function UsageFilterForm({
  open,
  draft,
  errors,
  groups,
  channels,
  groupsFailed,
  channelsFailed,
  selfScoped,
  onOpenChange,
  onFieldChange,
  onApply,
  onReset,
}: {
  open: boolean
  draft: UsageFilterDraft
  errors: UsageFilterErrors
  groups: GroupOptionDto[]
  channels: ChannelDto[]
  groupsFailed: boolean
  channelsFailed: boolean
  selfScoped?: boolean
  onOpenChange(open: boolean): void
  onFieldChange(field: keyof UsageFilterDraft, value: string): void
  onApply(): void
  onReset(): void
}) {
  const t = useT()

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

  function channelOptions() {
    const options = [{ value: '', label: t('monitor.usage.filters.anyChannel') }]
    if (
      draft.channel_id !== '' &&
      !channels.some((channel) => channel.channel_id === draft.channel_id)
    ) {
      options.push({ value: draft.channel_id, label: draft.channel_id })
    }
    return [
      ...options,
      ...channels.map((channel) => ({ value: channel.channel_id, label: channel.name })),
    ]
  }

  function error(field: keyof UsageFilterErrors): { type: 'error'; message: string } | undefined {
    const key = errors[field]
    return key === undefined || key === ''
      ? undefined
      : { type: 'error', message: t(key as MessageId) }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    onApply()
  }

  const groupLabel = t('monitor.usage.filters.group')
  const channelLabel = t('monitor.usage.filters.channel')

  return (
    <DetailPanel
      isOpen={open}
      onOpenChange={onOpenChange}
      title={t('monitor.usage.filters.title')}
      subtitle={t('monitor.usage.filters.description')}
      footer={
        <div {...stylex.props(styles.footer)}>
          <Button
            variant="secondary"
            size="sm"
            label={t('monitor.usage.filters.reset')}
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
              label={t('monitor.usage.filters.apply')}
              onClick={onApply}
            />
          </span>
        </div>
      }
    >
      <form {...stylex.props(styles.form)} onSubmit={submit}>
        <div {...stylex.props(styles.grid)}>
          {selfScoped !== true &&
            (groupsFailed ? (
              <TextInput
                id="usage-group"
                label={groupLabel}
                size="sm"
                value={draft.group_id}
                onChange={(value) => onFieldChange('group_id', value)}
                status={error('group_id')}
                autoComplete="off"
                {...numericInputAttrs}
              />
            ) : (
              <Selector
                id="usage-group"
                label={groupLabel}
                size="sm"
                options={groupOptions()}
                value={draft.group_id}
                onChange={(value) => onFieldChange('group_id', value)}
                status={error('group_id')}
              />
            ))}
          {selfScoped !== true &&
            (channelsFailed ? (
              <TextInput
                id="usage-channel"
                label={channelLabel}
                size="sm"
                value={draft.channel_id}
                onChange={(value) => onFieldChange('channel_id', value)}
                status={error('channel_id')}
                autoComplete="off"
                {...plainTextInputAttrs}
              />
            ) : (
              <Selector
                id="usage-channel"
                label={channelLabel}
                size="sm"
                options={channelOptions()}
                value={draft.channel_id}
                onChange={(value) => onFieldChange('channel_id', value)}
                status={error('channel_id')}
              />
            ))}
          {selfScoped !== true && (
            <TextInput
              id="usage-credential"
              label={t('monitor.usage.filters.credential', { name: '' })}
              size="sm"
              value={draft.credential_id}
              onChange={(value) => onFieldChange('credential_id', value)}
              status={error('credential_id')}
              autoComplete="off"
              {...numericInputAttrs}
            />
          )}
          <TextInput
            id="usage-model"
            label={t('monitor.usage.filters.model')}
            size="sm"
            value={draft.upstream_model}
            onChange={(value) => onFieldChange('upstream_model', value)}
            status={error('upstream_model')}
            placeholder={t('monitor.usage.filters.modelPlaceholder')}
            autoComplete="off"
            {...plainTextInputAttrs}
          />
        </div>
      </form>
    </DetailPanel>
  )
}
