import { Switch, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { useEffect, useState } from 'react'

import { HeaderRulesEditor } from '../../components/HeaderRulesEditor'
import { numericInputAttrs } from '../../components/input-attrs'
import { SettingBlock } from '../../components/setting-chrome'
import { useT } from '../../app/i18n'
import type { HeaderRulesDto } from '@shared/control/types'
import { isValidCORSConfig, setSettingsOverride } from '@shared/domain/settings/settings-patch'
import { SettingsSectionFrame, useSettingTools, type SettingsSectionProps } from './section-tools'

type CORSListKey = 'allowed_origins' | 'allowed_methods' | 'allowed_headers' | 'exposed_headers'
type ToggleableKey = 'header_rules' | 'cors' | 'response_header_rules'

// `min`/`step` are omitted from TextInput's prop surface but forward to the
// underlying input via `...rest`.
const maxAgeBounds = { min: '0', step: '1' } as const

const narrow = '@media (max-width: 800px)'

const styles = stylex.create({
  blocks: {
    display: 'grid',
    gap: 'var(--space-5)',
  },
  corsForm: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      [narrow]: '1fr',
    },
    gap: 'var(--space-3) var(--space-4)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: 'var(--space-3)',
    paddingInline: 'var(--space-4)',
  },
  field: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
  },
  fieldLabel: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  wide: {
    gridColumn: { default: '1 / -1', [narrow]: 'auto' },
  },
  switchField: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    alignItems: 'center',
    gap: 'var(--space-3)',
  },
  switchText: {
    display: 'grid',
    gap: 'var(--space-1)',
  },
  switchTitle: {
    fontSize: 'var(--text-meta)',
    fontWeight: 600,
  },
  switchHint: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  notice: {
    margin: 0,
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'color-mix(in srgb, var(--color-warning) 34%, var(--color-border-subtle))',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
    paddingBlock: '10px',
    paddingInline: '12px',
    fontSize: 'var(--text-sm)',
    lineHeight: 1.5,
    gridColumn: { default: '1 / -1', [narrow]: 'auto' },
  },
  summary: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
})

function splitList(value: string): string[] {
  if (value.trim() === '') return []
  return value.split(',').map((entry) => entry.trim())
}

function unique(values: string[], caseInsensitive = false): boolean {
  const normalized = values.map((value) => (caseInsensitive ? value.toLowerCase() : value))
  return new Set(normalized).size === normalized.length
}

function validHTTPToken(value: string): boolean {
  return value.length > 0 && /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/u.test(value)
}

function validOrigin(value: string): boolean {
  if (value === '*' || value === 'null') return true
  return (
    value === value.trim() &&
    !value.includes('@') &&
    /^[A-Za-z][A-Za-z0-9+.-]*:\/\/[^/?#\s,]+$/u.test(value)
  )
}

function validHeaderList(values: string[], required: boolean): boolean {
  if (required && values.length === 0) return false
  return (
    unique(values, true) &&
    values.every((value) => value === '*' || validHTTPToken(value)) &&
    (!values.includes('*') || values.length === 1)
  )
}

export interface BrowserAccessSectionProps extends SettingsSectionProps {
  resetKey: number
  onValidChange: (value: boolean) => void
  onHeaderRulesValidChange: (value: boolean) => void
  onCorsValidChange: (value: boolean) => void
  onResponseRulesValidChange: (value: boolean) => void
  onHeaderRulesInvalidEditsChange: (value: boolean) => void
  onResponseRulesInvalidEditsChange: (value: boolean) => void
}

export function BrowserAccessSection({
  resetKey,
  onValidChange,
  onHeaderRulesValidChange,
  onCorsValidChange,
  onResponseRulesValidChange,
  onHeaderRulesInvalidEditsChange,
  onResponseRulesInvalidEditsChange,
  ...props
}: BrowserAccessSectionProps) {
  const { base, draft, disabled } = props
  const t = useT()
  const tools = useSettingTools(props)

  const [headerRulesRawValid, setHeaderRulesRawValid] = useState(true)
  const [headerRulesInvalidEdits, setHeaderRulesInvalidEdits] = useState(false)
  const [headerRulesEditorResetKey, setHeaderRulesEditorResetKey] = useState(0)
  const [responseRulesValid, setResponseRulesValid] = useState(true)
  const [responseRulesInvalidEdits, setResponseRulesInvalidEdits] = useState(false)
  const [responseEditorResetKey, setResponseEditorResetKey] = useState(0)

  const headerRulesOverridden = draft.overrides.has('header_rules')
  const corsOverridden = draft.overrides.has('cors')
  const responseRulesOverridden = draft.overrides.has('response_header_rules')
  const headerRulesPendingRestore =
    !headerRulesOverridden && base.settings.overrides.includes('header_rules')
  const corsPendingRestore = !corsOverridden && base.settings.overrides.includes('cors')
  const responseRulesPendingRestore =
    !responseRulesOverridden && base.settings.overrides.includes('response_header_rules')

  const headerRules =
    headerRulesOverridden || headerRulesPendingRestore
      ? draft.values.header_rules
      : base.settings.values.header_rules
  const headerRuleCount = Object.keys(headerRules.set).length + headerRules.remove.length
  const cors = corsOverridden ? draft.values.cors : base.settings.values.cors
  const responseRules =
    responseRulesOverridden || responseRulesPendingRestore
      ? draft.values.response_header_rules
      : base.settings.values.response_header_rules
  const responseRuleCount = Object.keys(responseRules.set).length + responseRules.remove.length

  const effectiveHeaderRulesValid = !headerRulesOverridden || headerRulesRawValid
  const corsValid = !corsOverridden || isValidCORSConfig(draft.values.cors)
  const effectiveResponseRulesValid = !responseRulesOverridden || responseRulesValid
  const valid = effectiveHeaderRulesValid && corsValid && effectiveResponseRulesValid

  // Mirrors the classic `update:*` watchers — fire on mount and on change.
  useEffect(() => onValidChange(valid), [onValidChange, valid])
  useEffect(
    () => onHeaderRulesValidChange(headerRulesRawValid),
    [onHeaderRulesValidChange, headerRulesRawValid],
  )
  useEffect(() => onCorsValidChange(corsValid), [onCorsValidChange, corsValid])
  useEffect(
    () => onResponseRulesValidChange(effectiveResponseRulesValid),
    [onResponseRulesValidChange, effectiveResponseRulesValid],
  )
  useEffect(
    () => onHeaderRulesInvalidEditsChange(headerRulesInvalidEdits),
    [onHeaderRulesInvalidEditsChange, headerRulesInvalidEdits],
  )
  useEffect(
    () => onResponseRulesInvalidEditsChange(responseRulesInvalidEdits),
    [onResponseRulesInvalidEditsChange, responseRulesInvalidEdits],
  )

  // Classic resetKey watch: restore flags and force the editors to rebuild.
  const [seenResetKey, setSeenResetKey] = useState(resetKey)
  if (seenResetKey !== resetKey) {
    setSeenResetKey(resetKey)
    setHeaderRulesRawValid(true)
    setHeaderRulesInvalidEdits(false)
    setHeaderRulesEditorResetKey((key) => key + 1)
    setResponseRulesValid(true)
    setResponseRulesInvalidEdits(false)
    setResponseEditorResetKey((key) => key + 1)
  }

  // Leaving the override clears stale editor flags (classic watcher).
  const [seenHeaderOverridden, setSeenHeaderOverridden] = useState(headerRulesOverridden)
  if (seenHeaderOverridden !== headerRulesOverridden) {
    setSeenHeaderOverridden(headerRulesOverridden)
    if (!headerRulesOverridden) {
      setHeaderRulesRawValid(true)
      setHeaderRulesInvalidEdits(false)
    }
  }
  const [seenResponseOverridden, setSeenResponseOverridden] = useState(responseRulesOverridden)
  if (seenResponseOverridden !== responseRulesOverridden) {
    setSeenResponseOverridden(responseRulesOverridden)
    if (!responseRulesOverridden) {
      setResponseRulesValid(true)
      setResponseRulesInvalidEdits(false)
    }
  }

  const toggleOverride = (key: ToggleableKey): void => {
    props.publish(key, setSettingsOverride(base.settings, draft, key, !draft.overrides.has(key)))
    if (key === 'header_rules') {
      setHeaderRulesRawValid(true)
      setHeaderRulesInvalidEdits(false)
      setHeaderRulesEditorResetKey((value) => value + 1)
    }
    if (key === 'response_header_rules') {
      setResponseRulesValid(true)
      setResponseRulesInvalidEdits(false)
      setResponseEditorResetKey((value) => value + 1)
    }
  }

  const sourceLabel = (overridden: boolean, pendingRestore: boolean): string => {
    if (overridden) return t('settings.runtime.overrideSource')
    if (pendingRestore) return t('settings.runtime.pendingRestoreSource')
    return t('settings.runtime.defaultSource')
  }

  const originsError = (() => {
    const values = cors.allowed_origins
    if (
      (cors.enabled && values.length === 0) ||
      !unique(values) ||
      values.some((value) => !validOrigin(value)) ||
      (values.includes('*') && values.length > 1) ||
      (values.includes('*') && cors.allow_credentials)
    ) {
      return t('settings.browserAccess.errors.origins')
    }
    return undefined
  })()
  const methodsError = (() => {
    const values = cors.allowed_methods
    return (cors.enabled && values.length === 0) ||
      !unique(values, true) ||
      !values.every((method) => method !== '*' && validHTTPToken(method))
      ? t('settings.browserAccess.errors.methods')
      : undefined
  })()
  const allowedHeadersError = !validHeaderList(cors.allowed_headers, cors.enabled)
    ? t('settings.browserAccess.errors.headers')
    : undefined
  const exposedHeadersError =
    !validHeaderList(cors.exposed_headers, false) ||
    (cors.allow_credentials && cors.exposed_headers.includes('*'))
      ? t('settings.browserAccess.errors.headers')
      : undefined
  const maxAgeError =
    Number.isSafeInteger(cors.max_age) && cors.max_age >= 0
      ? undefined
      : t('settings.browserAccess.errors.maxAge')

  const corsTextField = (
    id: string,
    listKey: CORSListKey,
    label: string,
    error: string | undefined,
    placeholder?: string,
    wide = false,
  ) => (
    <div key={id} {...stylex.props(styles.field, wide && styles.wide)}>
      <span {...stylex.props(styles.fieldLabel)}>{label}</span>
      <TextInput
        id={id}
        data-gptload-mono
        value={cors[listKey].join(', ')}
        label={label}
        isLabelHidden
        placeholder={placeholder}
        size="sm"
        isDisabled={disabled}
        status={error === undefined ? undefined : { type: 'error', message: error }}
        statusVariant="tooltip"
        onChange={(value) =>
          tools.update('cors', (next) => {
            next.values.cors[listKey] = splitList(value)
          })
        }
      />
    </div>
  )

  return (
    <SettingsSectionFrame
      id="settings-browser-access"
      title={t('settings.browserAccess.title')}
      description={t('settings.browserAccess.description')}
    >
      <div {...stylex.props(styles.blocks)}>
        <SettingBlock
          title={t('settings.browserAccess.cors.title')}
          help={t('settings.browserAccess.cors.description')}
          sourceLabel={sourceLabel(corsOverridden, corsPendingRestore)}
          actionLabel={
            corsOverridden ? t('settings.runtime.restoreDefault') : t('settings.runtime.override')
          }
          overridden={corsOverridden}
          pendingRestore={corsPendingRestore}
          disabled={disabled}
          onToggle={() => toggleOverride('cors')}
        >
          {corsOverridden ? (
            <div {...stylex.props(styles.corsForm)}>
              <div {...stylex.props(styles.switchField, styles.wide)}>
                <div {...stylex.props(styles.switchText)}>
                  <strong {...stylex.props(styles.switchTitle)}>
                    {t('settings.browserAccess.cors.enabled')}
                  </strong>
                  <small {...stylex.props(styles.switchHint)}>
                    {t('settings.browserAccess.cors.enabledHelp')}
                  </small>
                </div>
                <Switch
                  value={cors.enabled}
                  isDisabled={disabled}
                  label={t('settings.browserAccess.cors.enabled')}
                  isLabelHidden
                  size="sm"
                  onChange={(value) =>
                    tools.update('cors', (next) => {
                      next.values.cors.enabled = value
                    })
                  }
                />
              </div>

              {corsTextField(
                'settings-value-cors-origins',
                'allowed_origins',
                t('settings.browserAccess.cors.allowedOrigins'),
                originsError,
                t('settings.browserAccess.cors.allowedOriginsPlaceholder'),
                true,
              )}
              {corsTextField(
                'settings-value-cors-methods',
                'allowed_methods',
                t('settings.browserAccess.cors.allowedMethods'),
                methodsError,
              )}
              {corsTextField(
                'settings-value-cors-headers',
                'allowed_headers',
                t('settings.browserAccess.cors.allowedHeaders'),
                allowedHeadersError,
              )}
              {corsTextField(
                'settings-value-cors-exposed',
                'exposed_headers',
                t('settings.browserAccess.cors.exposedHeaders'),
                exposedHeadersError,
              )}

              <div {...stylex.props(styles.field)}>
                <span {...stylex.props(styles.fieldLabel)}>
                  {t('settings.browserAccess.cors.maxAge')}
                </span>
                <TextInput
                  id="settings-value-cors-max-age"
                  type="text"
                  {...numericInputAttrs}
                  {...maxAgeBounds}
                  data-gptload-mono
                  value={Number.isNaN(cors.max_age) ? '' : String(cors.max_age)}
                  label={t('settings.browserAccess.cors.maxAge')}
                  isLabelHidden
                  size="sm"
                  isDisabled={disabled}
                  status={
                    maxAgeError === undefined ? undefined : { type: 'error', message: maxAgeError }
                  }
                  statusVariant="tooltip"
                  onChange={(value) =>
                    tools.update('cors', (next) => {
                      next.values.cors.max_age = value.trim() === '' ? Number.NaN : Number(value)
                    })
                  }
                />
              </div>

              <div {...stylex.props(styles.switchField)}>
                <div {...stylex.props(styles.switchText)}>
                  <strong {...stylex.props(styles.switchTitle)}>
                    {t('settings.browserAccess.cors.allowCredentials')}
                  </strong>
                  <small {...stylex.props(styles.switchHint)}>
                    {t('settings.browserAccess.cors.allowCredentialsHelp')}
                  </small>
                </div>
                <Switch
                  value={cors.allow_credentials}
                  isDisabled={disabled}
                  label={t('settings.browserAccess.cors.allowCredentials')}
                  isLabelHidden
                  size="sm"
                  onChange={(value) =>
                    tools.update('cors', (next) => {
                      next.values.cors.allow_credentials = value
                    })
                  }
                />
              </div>

              <p {...stylex.props(styles.notice)} role="note">
                {t('settings.browserAccess.cors.securityNotice')}
              </p>
            </div>
          ) : (
            <p {...stylex.props(styles.summary)}>
              {cors.enabled
                ? t('settings.browserAccess.cors.enabledSummary', {
                    count: cors.allowed_origins.length,
                  })
                : t('settings.browserAccess.cors.disabledSummary')}
            </p>
          )}
        </SettingBlock>

        <SettingBlock
          title={t('settings.headers.blockTitle')}
          help={t('settings.headers.description')}
          meta={t('settings.headers.ruleCount', { count: headerRuleCount })}
          sourceLabel={sourceLabel(headerRulesOverridden, headerRulesPendingRestore)}
          actionLabel={
            headerRulesOverridden
              ? t('settings.runtime.restoreDefault')
              : t('settings.runtime.override')
          }
          overridden={headerRulesOverridden}
          pendingRestore={headerRulesPendingRestore}
          disabled={disabled}
          onToggle={() => toggleOverride('header_rules')}
        >
          <HeaderRulesEditor
            value={headerRules}
            disabled={disabled || !headerRulesOverridden}
            resetKey={headerRulesEditorResetKey}
            showAdd={headerRulesOverridden}
            onChange={(value: HeaderRulesDto) =>
              tools.update('header_rules', (next) => {
                next.values.header_rules = value
              })
            }
            onValidChange={setHeaderRulesRawValid}
            onInvalidEditsChange={setHeaderRulesInvalidEdits}
          />
        </SettingBlock>

        <SettingBlock
          title={t('settings.browserAccess.responseHeaders.title')}
          help={t('settings.browserAccess.responseHeaders.description')}
          meta={t('settings.headers.ruleCount', { count: responseRuleCount })}
          sourceLabel={sourceLabel(responseRulesOverridden, responseRulesPendingRestore)}
          actionLabel={
            responseRulesOverridden
              ? t('settings.runtime.restoreDefault')
              : t('settings.runtime.override')
          }
          overridden={responseRulesOverridden}
          pendingRestore={responseRulesPendingRestore}
          disabled={disabled}
          onToggle={() => toggleOverride('response_header_rules')}
        >
          <HeaderRulesEditor
            value={responseRules}
            disabled={disabled || !responseRulesOverridden}
            resetKey={responseEditorResetKey}
            showAdd={responseRulesOverridden}
            validationPolicy="response"
            removeHint={t('settings.browserAccess.responseHeaders.removeHint')}
            onChange={(value: HeaderRulesDto) =>
              tools.update('response_header_rules', (next) => {
                next.values.response_header_rules = value
              })
            }
            onValidChange={setResponseRulesValid}
            onInvalidEditsChange={setResponseRulesInvalidEdits}
          />
        </SettingBlock>
      </div>
    </SettingsSectionFrame>
  )
}
