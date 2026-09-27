import * as stylex from '@stylexjs/stylex'
import { Switch, TextInput } from '@astryxdesign/core'
import { useImperativeHandle, useRef, type Ref } from 'react'

import type { AccessKeyDto } from '@shared/control/types'
import { isValidPriceMultiplier } from '@shared/lib/price-multiplier'

import { useT } from '../../app/i18n'
import { decimalInputAttrs, numericInputAttrs } from '../../components/input-attrs'

const styles = stylex.create({
  stack: {
    display: 'grid',
    gap: 12,
  },
})

export interface AccessKeyFormFieldsHandle {
  focusName(): void
}

// Port of classic AccessKeyFormFields.vue: name / price-multiplier / enabled
// switch / RPM fields for the access-key drawer. v-model pairs become
// value + onXChange props; focusName is exposed on the ref handle.
export function AccessKeyFormFields({
  name,
  status,
  rpmLimit,
  priceMultiplier,
  disabled,
  onNameChange,
  onStatusChange,
  onRpmLimitChange,
  onPriceMultiplierChange,
  ref,
}: {
  name: string
  status: AccessKeyDto['status']
  rpmLimit: number
  priceMultiplier: string
  disabled: boolean
  onNameChange(value: string): void
  onStatusChange(value: AccessKeyDto['status']): void
  onRpmLimitChange(value: number): void
  onPriceMultiplierChange(value: string): void
  ref?: Ref<AccessKeyFormFieldsHandle>
}) {
  const t = useT()
  const nameInputRef = useRef<HTMLInputElement | null>(null)

  useImperativeHandle(
    ref,
    () => ({ focusName: () => nameInputRef.current?.focus() }),
    [],
  )

  const priceMultiplierValid = isValidPriceMultiplier(priceMultiplier)

  return (
    <div {...stylex.props(styles.stack)}>
      <TextInput
        ref={nameInputRef}
        label={t('accessKeys.drawer.name')}
        isRequired
        size="sm"
        value={name}
        isDisabled={disabled}
        onChange={onNameChange}
        autoComplete="off"
      />

      <TextInput
        label={t('common.priceMultiplier.label')}
        size="sm"
        value={priceMultiplier}
        isDisabled={disabled}
        onChange={onPriceMultiplierChange}
        autoComplete="off"
        {...decimalInputAttrs}
        description={
          priceMultiplierValid ? t('common.priceMultiplier.accessKeyHelp') : undefined
        }
        status={
          priceMultiplierValid
            ? undefined
            : { type: 'error', message: t('common.priceMultiplier.invalid') }
        }
      />

      <Switch
        label={t('accessKeys.drawer.enabled')}
        description={t('accessKeys.drawer.enabledDescription')}
        labelPosition="start"
        labelSpacing="spread"
        width="100%"
        size="sm"
        value={status === 'active'}
        isDisabled={disabled}
        onChange={(checked) => onStatusChange(checked ? 'active' : 'disabled')}
      />

      <TextInput
        label={t('accessKeys.drawer.rpm')}
        isOptional
        size="sm"
        value={rpmLimit === 0 ? '' : String(rpmLimit)}
        placeholder={t('accessKeys.drawer.rpmPlaceholder')}
        description={t('accessKeys.drawer.rpmDescription')}
        isDisabled={disabled}
        onChange={(value) => onRpmLimitChange(value === '' ? 0 : Number(value))}
        autoComplete="off"
        {...numericInputAttrs}
      />
    </div>
  )
}
