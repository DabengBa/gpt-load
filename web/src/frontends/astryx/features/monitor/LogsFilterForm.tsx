import * as stylex from '@stylexjs/stylex'
import {
  Button,
  DateTimeInput,
  Selector,
  TextInput,
  type ISODateTimeString,
} from '@astryxdesign/core'
import { ListFilter, X } from 'lucide-react'
import { useMemo, type FormEvent, type JSX } from 'react'

import type { ChannelDto } from '@shared/control/resources/channels'
import type { AccessKeyOptionDto, GroupOptionDto } from '@shared/control/types'
import {
  requestLogStatuses,
  type LogFilterDraft,
  type LogFilterErrors,
} from '@shared/domain/monitor/log-filters'
import type { MessageId } from '@shared/i18n/message-ids'
import {
  currentTimeZone,
  dateTimePresets,
  localDateTimeInput,
  resolveDateTimePreset,
  type DateTimePreset,
} from '@shared/lib/time'

import { useT } from '../../app/i18n'
import { plainTextInputAttrs } from '../../components/input-attrs'
import { LogsAdvancedFilterDrawer } from './LogsAdvancedFilterDrawer'

// Classic breakpoints: the control row stays single-line until 1120px, wraps
// below it, and drops to a two-column grid (controls span full width) at
// 560px. The <=860px touch-target bump on buttons is covered theme-wide by
// the gptload theme adaptation (width below md=861).
const WRAP = '@media (max-width: 1120px)'
const NARROW = '@media (max-width: 560px)'

const styles = stylex.create({
  // Classic .logs-filter — sunken card shell around the bar.
  form: {
    display: 'grid',
    minWidth: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface-sunken)',
  },
  chips: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBlock: 'var(--space-2)',
    paddingInline: 'var(--space-2-5)',
  },
  chipsLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  // Classic .logs-filter__chip — pill button with a trailing remove icon.
  chip: {
    display: 'inline-flex',
    minHeight: 24,
    alignItems: 'center',
    gap: 5,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: {
      default: 'var(--color-border-control)',
      ':hover': 'var(--color-text-faint)',
    },
    borderRadius: 999,
    backgroundColor: 'var(--color-surface)',
    color: {
      default: 'var(--color-text-muted)',
      ':hover': 'var(--color-text)',
    },
    paddingBlock: 2,
    paddingInline: 'var(--space-2)',
    fontSize: 'var(--text-label-xs)',
    cursor: 'pointer',
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
    flexWrap: {
      default: 'nowrap',
      [WRAP]: 'wrap',
    },
    gap: 'var(--space-2)',
    padding: 'var(--space-2-5)',
  },
  // The classic AppDateTimeRangePicker trigger becomes a from/to DateTimeInput
  // pair (the spike LogsView mapping) wrapped in a labelled group.
  timeRange: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'flex-start',
    gap: 'var(--space-2)',
    gridColumn: {
      default: 'auto',
      [NARROW]: '1 / -1',
    },
  },
  timeField: {
    display: 'block',
    minWidth: 0,
    width: {
      default: 236,
      [NARROW]: '100%',
    },
  },
  // Field wrappers carry the classic fixed/flex widths; controls fill them
  // via width="100%". At <=560px each spans the full two-column grid row.
  fieldGroup: {
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
  fieldStatus: {
    display: 'block',
    minWidth: 0,
    width: {
      default: 108,
      [NARROW]: '100%',
    },
    gridColumn: {
      default: 'auto',
      [NARROW]: '1 / -1',
    },
  },
  // Classic .logs-filter__count — the badge inside the "More filters" button.
  count: {
    display: 'inline-flex',
    minWidth: 18,
    height: 18,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 999,
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    fontSize: 11,
  },
  // The classic picker's shortcut strip: one horizontally scrolling row of
  // ghost chips under the control row, with the timezone note pinned at the
  // trailing edge like the classic popover footer.
  presets: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'nowrap',
    gap: 2,
    overflowX: 'auto',
    scrollbarWidth: 'thin',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingBlock: 'var(--space-1-75)',
    paddingInline: 'var(--space-2-5)',
  },
  timezone: {
    marginInlineStart: 'auto',
    paddingInlineStart: 'var(--space-2)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
  },
  error: {
    marginTop: -2,
    marginBottom: 'var(--space-2)',
    marginInline: 'var(--space-2-5)',
    color: 'var(--color-danger)',
    fontSize: 'var(--text-label-xs)',
  },
})

// Classic picker shortcut behaviour: presets only update the draft — Apply
// still validates and commits. Two field writes mirror the classic
// `update:from`/`update:to` emit pair (the parent applies them through
// functional state updates). Module scope keeps the `Date.now()` read out of
// render scope (react-hooks/purity).
function selectPreset(
  preset: DateTimePreset,
  updateField: (field: keyof LogFilterDraft, value: string) => void,
): void {
  const range = resolveDateTimePreset(preset, Math.floor(Date.now() / 1000) * 1000)
  updateField('from', localDateTimeInput(range.from_ms))
  updateField('to', localDateTimeInput(range.to_ms))
}

export interface AppliedChip {
  key: string
  label: string
}

export interface LogsFilterFormProps {
  draft: LogFilterDraft
  errors: LogFilterErrors
  groups: GroupOptionDto[]
  channels: ChannelDto[]
  accessKeys: AccessKeyOptionDto[]
  groupsFailed: boolean
  channelsFailed: boolean
  accessKeysFailed: boolean
  appliedChips: AppliedChip[]
  advancedCount: number
  advancedOpen: boolean
  onAdvancedOpenChange: (open: boolean) => void
  onUpdateField: (field: keyof LogFilterDraft, value: string) => void
  onRemoveFilter: (key: string) => void
  onApply: () => void
  onReset: () => void
  /**
   * Classic `selfScoped` (access-key session view): hides the Group selector
   * and swaps the client-model Selector for a free-text input; also forwarded
   * to the advanced drawer, which drops the fields a scoped principal cannot
   * filter on.
   */
  selfScoped?: boolean
}

/**
 * Quick request-log filter bar — classic LogsFilterForm.vue. The
 * AppDateTimeRangePicker popover (calendar + time fields + preset shortcuts +
 * timezone note) maps onto a from/to DateTimeInput pair plus a preset chip
 * row, the same substitution the spike LogsView made. SearchableSelect maps
 * to Selector hasSearch; AppSelect to Selector; compact size to `sm`. DS
 * field labels stay hidden (`isLabelHidden`) to keep the classic single-bar
 * density — each `label` still supplies the accessible name. Failure flags
 * only disable their inputs (the single host-level InlineFeedback carries the
 * notice, same as classic) and the first draft error keeps the classic
 * `role="alert"` summary line.
 */
export function LogsFilterForm({
  draft,
  errors,
  groups,
  channels,
  accessKeys,
  groupsFailed,
  channelsFailed,
  accessKeysFailed,
  appliedChips,
  advancedCount,
  advancedOpen,
  onAdvancedOpenChange,
  onUpdateField,
  onRemoveFilter,
  onApply,
  onReset,
  selfScoped,
}: LogsFilterFormProps): JSX.Element {
  const t = useT()
  const timezone = useMemo(() => currentTimeZone(), [])

  function groupOptions() {
    return [
      { value: '', label: t('monitor.logs.filters.anyGroup') },
      ...groups.map((group) => ({
        value: String(group.id),
        label: group.name,
      })),
    ]
  }

  // Client-model filtering keeps the classic semantics: options come from the
  // declared models of the group catalog, exact match; a draft value absent
  // from the catalog is appended so it stays selectable.
  function clientModelOptions() {
    const models = [...new Set(groups.flatMap((group) => group.models))].sort((a, b) =>
      a.localeCompare(b),
    )
    const options = [
      { value: '', label: t('monitor.logs.filters.anyClientModel') },
      ...models.map((model) => ({ value: model, label: model })),
    ]
    const pending = draft.client_model
    if (pending !== '' && !models.includes(pending)) {
      options.push({ value: pending, label: pending })
    }
    return options
  }

  function statusOptions() {
    return [
      { value: '', label: t('monitor.logs.filters.anyStatus') },
      ...requestLogStatuses.map((value) => ({
        value,
        label: t(`monitor.logs.status.${value}` as MessageId),
      })),
    ]
  }

  const firstErrorKey = Object.values(errors)[0]

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    onApply()
  }

  const dateTimeInputProps = {
    hasSeconds: true,
    hourFormat: '24h' as const,
    hasClear: true,
    isLabelHidden: true,
    size: 'sm' as const,
    width: '100%' as const,
  }

  return (
    <>
      <form
        {...stylex.props(styles.form)}
        aria-label={t('monitor.logs.filters.label')}
        onSubmit={submit}
      >
        {appliedChips.length > 0 && (
          <div {...stylex.props(styles.chips)}>
            <span {...stylex.props(styles.chipsLabel)}>{t('monitor.logs.filters.applied')}</span>
            {appliedChips.map((chip) => (
              <button
                key={chip.key}
                type="button"
                {...stylex.props(styles.chip)}
                aria-label={t('monitor.logs.filters.remove', { value: chip.label })}
                onClick={() => onRemoveFilter(chip.key)}
              >
                <span>{chip.label}</span>
                <X size={12} aria-hidden="true" />
              </button>
            ))}
          </div>
        )}

        <div {...stylex.props(styles.row)}>
          <div
            {...stylex.props(styles.timeRange)}
            role="group"
            aria-label={t('monitor.logs.filters.timeRange')}
          >
            <span {...stylex.props(styles.timeField)}>
              <DateTimeInput
                {...dateTimeInputProps}
                label={t('monitor.logs.filters.from')}
                value={(draft.from || undefined) as ISODateTimeString | undefined}
                onChange={(value) => onUpdateField('from', value ?? '')}
                status={
                  errors.from !== undefined && errors.from !== ''
                    ? { type: 'error', message: t(errors.from as MessageId) }
                    : undefined
                }
              />
            </span>
            <span {...stylex.props(styles.timeField)}>
              <DateTimeInput
                {...dateTimeInputProps}
                label={t('monitor.logs.filters.to')}
                value={(draft.to || undefined) as ISODateTimeString | undefined}
                onChange={(value) => onUpdateField('to', value ?? '')}
                status={
                  errors.to !== undefined && errors.to !== ''
                    ? { type: 'error', message: t(errors.to as MessageId) }
                    : undefined
                }
              />
            </span>
          </div>

          {selfScoped !== true && (
            <span {...stylex.props(styles.fieldGroup)}>
              <Selector
                label={t('monitor.logs.filters.group')}
                isLabelHidden
                size="sm"
                width="100%"
                hasSearch
                searchPlaceholder={t('monitor.logs.filters.searchGroups')}
                emptyText={t('monitor.logs.filters.noMatches')}
                emptySearchText={t('monitor.logs.filters.noMatches')}
                options={groupOptions()}
                value={draft.group_id}
                onChange={(value) => onUpdateField('group_id', value)}
                isDisabled={groupsFailed}
              />
            </span>
          )}
          <span {...stylex.props(styles.fieldModel)}>
            {selfScoped === true ? (
              <TextInput
                label={t('monitor.logs.filters.clientModel')}
                isLabelHidden
                size="sm"
                width="100%"
                value={draft.client_model}
                onChange={(value) => onUpdateField('client_model', value)}
                placeholder={t('monitor.logs.filters.clientModel')}
                status={
                  errors.client_model !== undefined && errors.client_model !== ''
                    ? { type: 'error' }
                    : undefined
                }
                aria-describedby={
                  errors.client_model !== undefined && errors.client_model !== ''
                    ? 'logs-filter-error'
                    : undefined
                }
                data-1p-ignore="true"
                data-lpignore="true"
                {...plainTextInputAttrs}
              />
            ) : (
              <Selector
                label={t('monitor.logs.filters.clientModel')}
                isLabelHidden
                size="sm"
                width="100%"
                hasSearch
                searchPlaceholder={t('monitor.logs.filters.searchClientModels')}
                emptyText={t('monitor.logs.filters.noMatches')}
                emptySearchText={t('monitor.logs.filters.noMatches')}
                options={clientModelOptions()}
                value={draft.client_model}
                onChange={(value) => onUpdateField('client_model', value)}
                isDisabled={groupsFailed}
                status={
                  errors.client_model !== undefined && errors.client_model !== ''
                    ? { type: 'error' }
                    : undefined
                }
              />
            )}
          </span>
          <span {...stylex.props(styles.fieldStatus)}>
            <Selector
              label={t('monitor.logs.filters.status')}
              isLabelHidden
              size="sm"
              width="100%"
              options={statusOptions()}
              value={draft.status}
              onChange={(value) => onUpdateField('status', value)}
            />
          </span>
          <Button
            variant="secondary"
            size="sm"
            icon={<ListFilter size={14} aria-hidden="true" />}
            label={t('monitor.logs.filters.more')}
            onClick={() => onAdvancedOpenChange(true)}
            endContent={
              advancedCount > 0 ? (
                <span {...stylex.props(styles.count)}>{advancedCount}</span>
              ) : undefined
            }
          />
          <Button
            type="submit"
            variant="primary"
            size="sm"
            label={t('monitor.logs.filters.apply')}
          />
          <Button
            variant="secondary"
            size="sm"
            label={t('monitor.logs.filters.reset')}
            onClick={onReset}
          />
        </div>

        <div
          {...stylex.props(styles.presets)}
          role="group"
          aria-label={t('monitor.logs.filters.quickRanges')}
        >
          {dateTimePresets.map((preset) => (
            <Button
              key={preset}
              type="button"
              variant="ghost"
              size="sm"
              label={t(`monitor.logs.filters.quick.${preset}` as MessageId)}
              onClick={() => selectPreset(preset, onUpdateField)}
            />
          ))}
          <span {...stylex.props(styles.timezone)}>
            {t('monitor.logs.filters.timezone')} · {timezone}
          </span>
        </div>

        {firstErrorKey !== undefined && firstErrorKey !== '' && (
          <p id="logs-filter-error" {...stylex.props(styles.error)} role="alert">
            {t(firstErrorKey as MessageId)}
          </p>
        )}
      </form>

      <LogsAdvancedFilterDrawer
        open={advancedOpen}
        draft={draft}
        errors={errors}
        accessKeys={accessKeys}
        channels={channels}
        accessKeysFailed={accessKeysFailed}
        channelsFailed={channelsFailed}
        onOpenChange={onAdvancedOpenChange}
        onUpdateField={onUpdateField}
        onApply={onApply}
        onReset={onReset}
        selfScoped={selfScoped}
      />
    </>
  )
}
