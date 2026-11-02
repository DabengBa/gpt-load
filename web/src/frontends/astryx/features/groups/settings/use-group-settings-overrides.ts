import { useState } from 'react'

import type {
  GroupPolicyCountKey,
  GroupSettingsDraft,
  GroupTimeoutKey,
} from '@shared/domain/groups/settings/group-settings-patch'
import type { GroupSettingsDto, HeaderRulesDto, ProxyConfiguredMode } from '@shared/control/types'
import { proxyDraftState, proxyOverrideToggleMode } from '@shared/control/resources/proxy'

import type { useT } from '../../../app/i18n'

type Translate = ReturnType<typeof useT>

export function useGroupProxyDraft(args: {
  saved: GroupSettingsDto | undefined
  proxySupported: boolean
  t: Translate
}) {
  const { saved, proxySupported, t } = args
  const [proxyMode, setProxyMode] = useState<ProxyConfiguredMode>('inherit')
  const [proxyEndpoint, setProxyEndpoint] = useState('')

  const proxyState = saved
    ? proxyDraftState(saved.proxy, proxyMode, proxyEndpoint)
    : { dirty: false, invalid: false, value: undefined }
  // 代理沿用其它设置项的覆盖语义：inherit 即“继承全局”，direct/custom 即“本分组覆盖”。
  const proxyOverridden = proxyMode !== 'inherit'
  const proxyPendingRestore = saved?.proxy.configured_mode !== 'inherit' && proxyMode === 'inherit'
  const proxyEffectiveLabel = (() => {
    const view = saved?.proxy
    if (!view) return ''
    return view.display_url ?? t(`common.proxy.mode.${view.effective_mode}`)
  })()
  const proxyValue = (() => {
    if (!proxySupported) return t('common.proxy.unsupported')
    if (proxyPendingRestore) return t('group.settings.runtime.resetPending')
    return proxyEffectiveLabel
  })()

  function toggleProxyOverride(): void {
    const base = saved?.proxy
    if (base) setProxyMode(proxyOverrideToggleMode(base, proxyOverridden))
    setProxyEndpoint('')
  }

  return {
    proxyMode,
    proxyEndpoint,
    setProxyEndpoint,
    proxyState,
    proxyOverridden,
    proxyPendingRestore,
    proxyValue,
    toggleProxyOverride,
    resetProxyDraft(mode: ProxyConfiguredMode): void {
      setProxyMode(mode)
      setProxyEndpoint('')
    },
  }
}

const enabledLabel = (t: Translate, value: boolean | undefined): string =>
  value ? t('group.settings.runtime.enabledValue') : t('group.settings.runtime.disabledValue')

export function useGroupOverrideFlags(
  saved: GroupSettingsDto | undefined,
  draft: GroupSettingsDraft | undefined,
  t: Translate,
) {
  const isPendingRestore = (key: GroupTimeoutKey | GroupPolicyCountKey): boolean =>
    draft?.overrides[key] === undefined && saved?.overrides[key] !== undefined

  const headerRulesOverridden = draft?.overrides.header_rules !== undefined
  const headerRulesPendingRestore =
    !headerRulesOverridden && saved?.overrides.header_rules !== undefined
  const displayedHeaderRules: HeaderRulesDto = (() => {
    if (draft?.overrides.header_rules !== undefined) return draft.overrides.header_rules
    if (headerRulesPendingRestore) return { set: {}, remove: [] }
    return saved?.effective.header_rules ?? { set: {}, remove: [] }
  })()

  const affinityOverridden = draft?.overrides.affinity_enabled !== undefined
  const websocketOverridden = draft?.overrides.responses_websocket_enabled !== undefined
  const reasoningOverridden =
    draft?.overrides.responses_reasoning_status_filter_enabled !== undefined

  return {
    isPendingRestore,
    headerRulesOverridden,
    headerRulesPendingRestore,
    displayedHeaderRules,
    affinity: {
      overridden: affinityOverridden,
      pendingRestore: !affinityOverridden && saved?.overrides.affinity_enabled !== undefined,
      label: enabledLabel(t, saved?.effective.affinity_enabled),
    },
    websocket: {
      overridden: websocketOverridden,
      pendingRestore:
        !websocketOverridden && saved?.overrides.responses_websocket_enabled !== undefined,
      label: enabledLabel(t, saved?.effective.responses_websocket_enabled),
    },
    reasoningStatusFilter: {
      overridden: reasoningOverridden,
      label: enabledLabel(t, saved?.effective.responses_reasoning_status_filter_enabled),
    },
  }
}
