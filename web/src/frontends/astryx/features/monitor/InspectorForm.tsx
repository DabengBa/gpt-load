import * as stylex from '@stylexjs/stylex'
import { Banner, Button, Selector, Spinner, VisuallyHidden } from '@astryxdesign/core'
import { Info } from 'lucide-react'
import type { FormEvent, JSX } from 'react'

import type { MessageId } from '@shared/i18n/message-ids'

import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { MonitorSectionHeading } from './MonitorSectionHeading'

// Classic breakpoints: <=1120px inlines the three selects with the submit
// button, <=760px drops to two columns with a full-width submit, <=520px
// stacks to a single column. Narrower queries are declared later so they win.
const TABLET = '@media (max-width: 1120px)'
const TIGHT = '@media (max-width: 760px)'
const NARROW = '@media (max-width: 520px)'

const styles = stylex.create({
  panel: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-3)',
  },
  body: {
    display: 'grid',
    minWidth: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
    gap: { default: 'var(--space-4)', [TABLET]: 'var(--space-3)' },
    paddingBlock: 18,
    paddingInline: { default: 18, [NARROW]: 14 },
  },
  // Classic AsyncRefreshIndicator: a subtle debounced "loading" affordance —
  // here a faint inline spinner line instead of the absolute 2px top bar.
  pending: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  form: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(0, 1fr)',
      [TABLET]: 'repeat(3, minmax(0, 1fr)) auto',
      [TIGHT]: 'repeat(2, minmax(0, 1fr))',
      [NARROW]: 'minmax(0, 1fr)',
    },
    alignItems: { default: 'stretch', [TABLET]: 'end' },
    gap: 'var(--space-3)',
  },
  submit: {
    width: { default: '100%', [TABLET]: 'auto', [TIGHT]: '100%' },
    marginTop: { default: 'var(--space-1)', [TABLET]: 0 },
  },
  // Classic sets monospace on the protocol/model select values.
  monoValue: {
    fontFamily: 'var(--font-mono)',
  },
  // Classic .inspector-inline-error.
  inlineError: {
    margin: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'color-mix(in srgb, var(--color-danger) 34%, var(--color-border-subtle))',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
    paddingBlock: 9,
    paddingInline: 10,
    fontSize: 'var(--text-sm)',
  },
  // Classic InlineFeedback tone="neutral" appearance="ledger".
  boundary: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'flex-start',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    paddingBlock: 9,
    paddingInline: 10,
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  boundaryGlyph: {
    display: 'inline-flex',
    flex: '0 0 auto',
    alignItems: 'center',
    height: '1em',
  },
})

export interface InspectorFormOption {
  value: string
  label: string
}

type InspectorField = 'protocol' | 'externalModel' | 'accessKey'

export interface InspectorFormProps {
  protocol: string
  model: string
  accessKeyId: string
  protocolOptions: InspectorFormOption[]
  modelOptions: InspectorFormOption[]
  accessKeyOptions: InspectorFormOption[]
  errors: Partial<Record<InspectorField, MessageId>>
  optionsPending: boolean
  optionsFailed: boolean
  missingAccessKey: boolean
  submitPending: boolean
  onProtocolChange: (value: string) => void
  onModelChange: (value: string) => void
  onAccessKeyIdChange: (value: string) => void
  onSubmit: () => void
  onRetryOptions: () => void
}

/**
 * Route-inspector input panel — classic InspectorForm.vue with FormField +
 * AppSelect. DS Selector already composes Field internally (label, required
 * marker, status message, aria-invalid/aria-describedby on the trigger), so
 * `status` replaces the classic FormField error plumbing; QueryFeedback maps
 * to Banner + retry Button and the sr-only error summary to VisuallyHidden.
 */
export function InspectorForm({
  protocol,
  model,
  accessKeyId,
  protocolOptions,
  modelOptions,
  accessKeyOptions,
  errors,
  optionsPending,
  optionsFailed,
  missingAccessKey,
  submitPending,
  onProtocolChange,
  onModelChange,
  onAccessKeyIdChange,
  onSubmit,
  onRetryOptions,
}: InspectorFormProps): JSX.Element {
  const t = useT()
  const optionsLoading = useStableLoading(optionsPending)
  const hasValidationError = Object.keys(errors).length > 0

  function fieldError(field: InspectorField): { type: 'error'; message: string } | undefined {
    const key = errors[field]
    return key === undefined ? undefined : { type: 'error', message: t(key) }
  }

  function submit(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault()
    onSubmit()
  }

  const optionsLoadingLabel = t('monitor.inspector.options.loading')

  return (
    <aside {...stylex.props(styles.panel)} aria-labelledby="inspector-form-title">
      <MonitorSectionHeading id="inspector-form-title" title={t('monitor.inspector.form.title')} />

      <div {...stylex.props(styles.body)}>
        {optionsLoading && (
          <span {...stylex.props(styles.pending)}>
            <Spinner size="sm" shade="subtle" aria-label={optionsLoadingLabel} />
            <span aria-hidden="true">{optionsLoadingLabel}</span>
          </span>
        )}
        {optionsFailed && (
          <Banner
            status="error"
            title={t('monitor.inspector.options.failed')}
            endContent={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={onRetryOptions}
              />
            }
          />
        )}

        <form
          {...stylex.props(styles.form)}
          aria-label={t('monitor.inspector.form.label')}
          onSubmit={submit}
        >
          <Selector
            id="inspector-access-key"
            label={t('monitor.inspector.form.accessKey')}
            isRequired
            size="sm"
            options={accessKeyOptions}
            value={accessKeyId}
            onChange={onAccessKeyIdChange}
            status={fieldError('accessKey')}
          />
          <Selector
            id="inspector-protocol"
            label={t('monitor.inspector.form.protocol')}
            isRequired
            size="sm"
            options={protocolOptions}
            value={protocol}
            onChange={onProtocolChange}
            status={fieldError('protocol')}
            renderValue={(option) => (
              <span {...stylex.props(styles.monoValue)}>{option.label}</span>
            )}
          />
          <Selector
            id="inspector-model"
            label={t('monitor.inspector.form.model')}
            isRequired
            size="sm"
            options={modelOptions}
            value={model}
            onChange={onModelChange}
            status={fieldError('externalModel')}
            renderValue={(option) => (
              <span {...stylex.props(styles.monoValue)}>{option.label}</span>
            )}
          />
          <Button
            type="submit"
            variant="primary"
            size="sm"
            isLoading={submitPending}
            isDisabled={optionsPending || optionsFailed}
            label={t('monitor.inspector.form.submit')}
            xstyle={styles.submit}
          />
        </form>

        {missingAccessKey && (
          <p {...stylex.props(styles.inlineError)} role="alert">
            {t('monitor.inspector.errors.missingDeepLinkAccessKey', { id: accessKeyId })}
          </p>
        )}
        {hasValidationError && (
          <VisuallyHidden as="p" role="alert">
            {t('monitor.inspector.errors.summary')}
          </VisuallyHidden>
        )}

        <div {...stylex.props(styles.boundary)} role="status">
          <span {...stylex.props(styles.boundaryGlyph)}>
            <Info size={14} aria-hidden="true" />
          </span>
          <span>{t('monitor.inspector.boundary')}</span>
        </div>
      </div>
    </aside>
  )
}
