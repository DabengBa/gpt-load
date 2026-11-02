import type { GroupRuntimeConfigDto, HeaderRulesDto } from '@shared/control/resources/groups'
import type { ChannelDto } from '@shared/control/resources/channels'
import type { GroupSettingsDto, ParameterOverrideRuleDto } from '@shared/control/types'
import {
  preserveChannelParams,
  setGroupConfigOverride,
  setGroupPolicyCountOverride,
  type GroupPolicyCountKey,
  type GroupSettingsDraft,
  type GroupTimeoutKey,
} from '@shared/domain/groups/settings/group-settings-patch'

import type { useT } from '../../../app/i18n'

type Translate = ReturnType<typeof useT>

export type SetGroupDraft = (
  update: (current: GroupSettingsDraft | undefined) => GroupSettingsDraft | undefined,
) => void

type OverrideKey = keyof GroupRuntimeConfigDto

function withOverride(
  current: GroupSettingsDraft,
  key: OverrideKey,
  enabled: boolean,
  effective: unknown,
): GroupSettingsDraft {
  const overrides: Record<string, unknown> = { ...current.overrides }
  if (enabled) overrides[key] = effective
  else delete overrides[key]
  return { ...current, overrides: overrides as GroupRuntimeConfigDto }
}

export function useGroupSettingsFieldsMutators(args: {
  saved: GroupSettingsDto | undefined
  draft: GroupSettingsDraft | undefined
  setDraft: SetGroupDraft
  isMutationPending(): boolean
  t: Translate
}) {
  const { saved, draft, setDraft, isMutationPending, t } = args

  function updateParam(key: string, value: string | null): void {
    setDraft((current) => {
      if (!current) return current
      const params = { ...current.params }
      if (value === null) delete params[key]
      else params[key] = value
      return { ...current, params }
    })
  }

  function selectChannel(channel: ChannelDto): void {
    if (isMutationPending()) return
    setDraft((current) => {
      if (!current) return current
      const params = preserveChannelParams(current.params, channel.param_fields)
      return { ...current, channel_id: channel.channel_id, params }
    })
  }

  function setTimeoutOverride(key: GroupTimeoutKey, enabled: boolean): void {
    if (!saved) return
    const effective = saved.effective[key]
    setDraft((current) =>
      current ? setGroupConfigOverride(current, key, enabled, effective) : current,
    )
  }

  function setTimeoutValue(key: GroupTimeoutKey, value: string): void {
    setDraft((current) => (current ? withOverride(current, key, true, Number(value)) : current))
  }

  function setPolicyCountOverride(key: GroupPolicyCountKey, enabled: boolean): void {
    if (!saved) return
    const effective = saved.effective[key]
    setDraft((current) =>
      current ? setGroupPolicyCountOverride(current, key, enabled, effective) : current,
    )
  }

  function setPolicyCountValue(key: GroupPolicyCountKey, value: string): void {
    setDraft((current) => (current ? withOverride(current, key, true, Number(value)) : current))
  }

  function policyCountError(key: GroupPolicyCountKey): string | undefined {
    const value = draft?.overrides[key]
    return value !== undefined && (!Number.isSafeInteger(value) || value < 0)
      ? t('group.settings.runtime.nonNegativeIntegerError')
      : undefined
  }

  return {
    updateParam,
    selectChannel,
    setTimeoutOverride,
    setTimeoutValue,
    setPolicyCountOverride,
    setPolicyCountValue,
    policyCountError,
  }
}

type BooleanOverride = {
  key:
    'affinity_enabled' | 'responses_websocket_enabled' | 'responses_reasoning_status_filter_enabled'
}

export function useGroupSettingsOverrideMutators(args: {
  saved: GroupSettingsDto | undefined
  setDraft: SetGroupDraft
  overridden: {
    headerRules: boolean
    affinity: boolean
    websocket: boolean
    reasoningStatusFilter: boolean
  }
  onHeaderRulesReset(): void
}) {
  const { saved, setDraft, overridden, onHeaderRulesReset } = args
  const overriddenByKey: Record<BooleanOverride['key'], boolean> = {
    affinity_enabled: overridden.affinity,
    responses_websocket_enabled: overridden.websocket,
    responses_reasoning_status_filter_enabled: overridden.reasoningStatusFilter,
  }

  function updateHeaderRules(value: HeaderRulesDto): void {
    setDraft((current) =>
      current
        ? withOverride(current, 'header_rules', true, {
            set: { ...value.set },
            remove: [...value.remove],
          })
        : current,
    )
  }

  function toggleHeaderRulesOverride(): void {
    if (!saved) return
    const inherited = saved.effective.header_rules
    setDraft((current) =>
      current
        ? withOverride(current, 'header_rules', !overridden.headerRules, {
            set: { ...inherited.set },
            remove: [...inherited.remove],
          })
        : current,
    )
    onHeaderRulesReset()
  }

  function updateParameterOverrides(value: ParameterOverrideRuleDto[]): void {
    setDraft((current) =>
      current ? withOverride(current, 'parameter_overrides', value.length > 0, value) : current,
    )
  }

  function toggler(key: BooleanOverride['key']): () => void {
    return () => {
      if (!saved) return
      const effective = saved.effective[key]
      setDraft((current) =>
        current ? withOverride(current, key, !overriddenByKey[key], effective) : current,
      )
    }
  }

  function setter(key: BooleanOverride['key']): (value: boolean) => void {
    return (value) =>
      setDraft((current) => (current ? withOverride(current, key, true, value) : current))
  }

  return {
    updateHeaderRules,
    toggleHeaderRulesOverride,
    updateParameterOverrides,
    toggleAffinityOverride: toggler('affinity_enabled'),
    setAffinityValue: setter('affinity_enabled'),
    toggleWebsocketOverride: toggler('responses_websocket_enabled'),
    setWebsocketValue: setter('responses_websocket_enabled'),
    toggleReasoningStatusFilterOverride: toggler('responses_reasoning_status_filter_enabled'),
    setReasoningStatusFilterValue: setter('responses_reasoning_status_filter_enabled'),
  }
}
