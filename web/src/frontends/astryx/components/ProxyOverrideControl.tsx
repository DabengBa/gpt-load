import { SegmentedControl, SegmentedControlItem, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { useId } from 'react'

import { useT } from '../app/i18n'
import { plainTextInputAttrs } from './input-attrs'
import type { ProxyConfiguredMode, ProxyViewDto } from '@shared/control/types'
import { proxyDraftState, proxyPlaceholderURL } from '@shared/control/resources/proxy'

const styles = stylex.create({
  root: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  input: {
    flexGrow: 1,
    flexShrink: 1,
    flexBasis: '190px',
    minWidth: 0,
  },
})

export function ProxyOverrideControl({
  base,
  mode,
  endpoint,
  disabled = false,
  onModeChange,
  onEndpointChange,
}: {
  /** Saved baseline — derives the placeholder URL and validity state. */
  base: ProxyViewDto
  mode: ProxyConfiguredMode
  endpoint: string
  disabled?: boolean
  onModeChange: (value: ProxyConfiguredMode) => void
  onEndpointChange: (value: string) => void
}) {
  const t = useT()
  const inputId = `${useId()}-proxy-url`

  // `inherit` is expressed by the outer override/restore toggle; only the two
  // explicit override modes appear here.
  const error = proxyDraftState(base, mode, endpoint).invalid
    ? t('common.proxy.invalid')
    : undefined
  const placeholder = proxyPlaceholderURL(base) ?? t('common.proxy.placeholder')

  return (
    <div {...stylex.props(styles.root)}>
      <SegmentedControl
        value={mode === 'custom' ? 'custom' : 'direct'}
        label={t('common.proxy.modeLabel')}
        size="sm"
        isDisabled={disabled}
        onChange={(value) => {
          if (!disabled && (value === 'direct' || value === 'custom')) onModeChange(value)
        }}
      >
        <SegmentedControlItem value="direct" label={t('common.proxy.mode.direct')} />
        <SegmentedControlItem value="custom" label={t('common.proxy.mode.custom')} />
      </SegmentedControl>

      {mode === 'custom' && (
        <TextInput
          id={inputId}
          xstyle={styles.input}
          data-gptload-mono
          value={endpoint}
          label={t('common.proxy.urlLabel')}
          isLabelHidden
          placeholder={placeholder}
          size="sm"
          autoComplete="off"
          {...plainTextInputAttrs}
          isDisabled={disabled}
          status={error === undefined ? undefined : { type: 'error', message: error }}
          statusVariant="tooltip"
          onChange={onEndpointChange}
        />
      )}
    </div>
  )
}
