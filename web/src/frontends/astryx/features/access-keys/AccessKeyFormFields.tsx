import * as stylex from '@stylexjs/stylex'
import { Switch, TextInput } from '@astryxdesign/core'
import { useImperativeHandle, useRef, type Ref } from 'react'

import type { AccessKeyDto } from '@shared/control/types'

import { useT } from '../../app/i18n'
import { numericInputAttrs } from '../../components/input-attrs'

const styles = stylex.create({
  stack: {
    display: 'grid',
    gap: 12,
  },
})

export interface AccessKeyFormFieldsHandle {
  focusName(): void
}

// Port of classic AccessKeyFormFields.vue: name / enabled
// switch / RPM fields for the access-key drawer. v-model pairs become
// value + onXChange props; focusName is exposed on the ref handle.
export function AccessKeyFormFields({
  name,
  status,
  rpmLimit,

  disabled,
  onNameChange,
  onStatusChange,
  onRpmLimitChange,

  ref,
}: {
  name: string
  status: AccessKeyDto['status']
  rpmLimit: number

  disabled: boolean
  onNameChange(value: string): void
  onStatusChange(value: AccessKeyDto['status']): void
  onRpmLimitChange(value: number): void

  ref?: Ref<AccessKeyFormFieldsHandle>
}) {
  const t = useT()
  const nameInputRef = useRef<HTMLInputElement | null>(null)

  useImperativeHandle(ref, () => ({ focusName: () => nameInputRef.current?.focus() }), [])

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
