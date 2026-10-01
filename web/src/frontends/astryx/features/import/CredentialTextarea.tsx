import * as stylex from '@stylexjs/stylex'
import { Field } from '@astryxdesign/core'

import type { ChannelDto } from '@shared/control/resources/channels'
import { analyzeCredentials } from '@shared/domain/import/credential-analysis'

import { useT } from '../../app/i18n'
import { InlineNotice } from '../../components/InlineNotice'

const styles = stylex.create({
  entry: {
    display: 'grid',
    gap: 0,
    minWidth: 0,
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingTop: '22px',
    paddingBottom: 'var(--space-6)',
  },
  entryCompact: {
    borderBottomWidth: 0,
    paddingTop: 0,
    paddingBottom: 0,
  },
  header: {
    marginBottom: 'var(--space-3)',
  },
  heading: {
    margin: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
    letterSpacing: '-0.01em',
  },
  headerDescription: {
    margin: 0,
    marginTop: '3px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  textarea: {
    width: '100%',
    minHeight: '124px',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    paddingBlock: '9px',
    paddingInline: '10px',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    lineHeight: 1.65,
  },
  counters: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '8px',
    minHeight: '27px',
    marginTop: '14px',
  },
  countersEmpty: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  pill: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '5px',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: '4px',
    paddingInline: '10px',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  pillValue: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    fontWeight: 640,
  },
  format: {
    marginTop: '12px',
  },
  warning: {
    marginTop: '14px',
  },
  note: {
    marginTop: '18px',
  },
})

export interface CredentialTextareaProps {
  value: string
  channel?: ChannelDto | null
  disabled?: boolean
  showHeaderDescription?: boolean
  storageDescription?: string
  duplicateLabel?: string
  showCredentialNotice?: boolean
  hideHeader?: boolean
  compact?: boolean
  rows?: number
  onChange(value: string): void
}

export function CredentialTextarea({
  value,
  channel = null,
  disabled = false,
  showHeaderDescription = true,
  storageDescription,
  duplicateLabel,
  showCredentialNotice = true,
  hideHeader = false,
  compact = false,
  rows = 6,
  onChange,
}: CredentialTextareaProps) {
  const t = useT()
  const analysis = analyzeCredentials(value, channel?.channel_id)
  const structured =
    channel !== null &&
    channel !== undefined &&
    (channel.credential_fields.length !== 1 || channel.credential_fields[0]?.key !== 'api_key')
  const title = structured ? t('import.credentials.structuredTitle') : t('import.credentials.title')
  const description = structured
    ? t('import.credentials.structuredDescription')
    : t('import.credentials.description')
  const label = structured ? t('import.credentials.structuredLabel') : t('import.credentials.label')
  const placeholder = (() => {
    switch (channel?.channel_id) {
      case 'azure_openai':
        return t('import.credentials.placeholders.azure', {
          example: '{"api_key":"..."}',
          exampleAlt: '{"client_id":"...","client_secret":"...","tenant_id":"..."}',
        })
      case 'aws_bedrock':
        return t('import.credentials.placeholders.bedrock', {
          example: '{"api_key":"..."}',
          exampleAlt: '{"access_key":"...","secret_key":"..."}',
        })
      case 'google_vertex':
        return t('import.credentials.placeholders.vertex', {
          example: '{"service_account_json":"..."}',
        })
      default:
        return t('import.credentials.placeholder')
    }
  })()
  const fieldSummary =
    channel?.credential_fields.map(({ label: fieldLabel }) => fieldLabel).join(' · ') ?? ''
  const error = analysis.tooManyCredentials ? t('import.credentials.tooMany') : ''
  // Only the metrics worth a second look become pills; a likely-AccessKey
  // count already gets its own warning banner below, so it is not repeated
  // here.
  const counters = [
    { value: analysis.nonEmptyCount, label: t('import.credentials.counters.nonEmpty') },
    { value: analysis.emptyLineCount, label: t('import.credentials.counters.empty') },
    {
      value: analysis.duplicateCount,
      label: duplicateLabel ?? t('import.credentials.counters.duplicates'),
    },
  ].filter(({ value: count }) => count > 0)
  const hasInput = value.trim().length > 0

  return (
    <section
      {...stylex.props(styles.entry, compact && styles.entryCompact)}
      aria-labelledby={hideHeader ? undefined : 'channel-credentials-heading'}
      aria-label={hideHeader ? title : undefined}
    >
      {!hideHeader && (
        <header {...stylex.props(styles.header)}>
          <h2 id="channel-credentials-heading" {...stylex.props(styles.heading)}>
            {title}
          </h2>
          {showHeaderDescription && (
            <p {...stylex.props(styles.headerDescription)}>{description}</p>
          )}
        </header>
      )}

      <Field
        label={label}
        description={storageDescription ?? t('import.credentials.storageNotice')}
        inputID="channel-credentials"
        isRequired
        status={error ? { type: 'error', message: error } : undefined}
      >
        <textarea
          id="channel-credentials"
          {...stylex.props(styles.textarea)}
          rows={rows}
          value={value}
          disabled={disabled}
          aria-invalid={error ? true : undefined}
          autoComplete="off"
          autoCapitalize="none"
          spellCheck={false}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
        />
      </Field>

      {structured && fieldSummary && (
        <div {...stylex.props(styles.format)}>
          <InlineNotice tone="neutral" appearance="hint">
            {t('import.credentials.structuredHint', { fields: fieldSummary })}
          </InlineNotice>
        </div>
      )}

      <div
        {...stylex.props(styles.counters)}
        aria-label={t('import.credentials.analysisLabel')}
        aria-live="polite"
      >
        {!hasInput && (
          <span {...stylex.props(styles.countersEmpty)}>{t('import.credentials.noInput')}</span>
        )}
        {counters.map((counter) => (
          <span key={counter.label} {...stylex.props(styles.pill)}>
            <strong {...stylex.props(styles.pillValue)}>{counter.value}</strong>
            {counter.label}
          </span>
        ))}
      </div>

      {analysis.likelyAccessKeyCount > 0 && (
        <div {...stylex.props(styles.warning)}>
          <InlineNotice tone="warning" appearance="hint">
            {t('import.credentials.accessKeyWarning', { count: analysis.likelyAccessKeyCount })}
          </InlineNotice>
        </div>
      )}
      {showCredentialNotice && (
        <div {...stylex.props(styles.note)}>
          <InlineNotice tone="neutral" appearance="ledger-hint" glyph="i">
            {t('import.credentials.channelCredentialNotice')}
          </InlineNotice>
        </div>
      )}
    </section>
  )
}
