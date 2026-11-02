import { Switch, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'

import type {
  AccessProtocol,
  GroupModelItemDto,
  GroupSettingsDto,
  HeaderRulesDto,
  ParameterOverrideRuleDto,
  ProxyConfiguredMode,
} from '@shared/control/types'

import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../../app/i18n'
import { HeaderRulesEditor } from '../../../components/HeaderRulesEditor'
import { ProxyOverrideControl } from '../../../components/ProxyOverrideControl'
import { SettingBlock, SettingRow } from '../../../components/setting-chrome'
import { ParameterOverrideRulesEditor } from './ParameterOverrideRulesEditor'
import {
  groupTimeoutKeys,
  type GroupPolicyCountKey,
  type GroupSettingsDraft,
  type GroupTimeoutKey,
} from '@shared/domain/groups/settings/group-settings-patch'

const policyRows = [
  {
    key: 'blacklist_threshold',
    helpKey: 'blacklistThresholdHelp',
  },
] as const

export interface GroupSettingsAdvancedProps {
  saved: GroupSettingsDto
  draft: GroupSettingsDraft
  pending: boolean
  proxy: {
    mode: ProxyConfiguredMode
    endpoint: string
    supported: boolean
    overridden: boolean
    pendingRestore: boolean
    value: string
    disabled: boolean
  }
  onProxyModeChange(mode: ProxyConfiguredMode): void
  onProxyEndpointChange(value: string): void
  onToggleProxy(): void
  websocket: {
    overridden: boolean
    pendingRestore: boolean
    label: string
  }
  onToggleWebsocket(): void
  onSetWebsocket(value: boolean): void
  reasoningStatusFilter: {
    overridden: boolean
    label: string
  }
  onToggleReasoningStatusFilter(): void
  onSetReasoningStatusFilter(value: boolean): void
  affinity: {
    overridden: boolean
    pendingRestore: boolean
    label: string
  }
  onToggleAffinity(): void
  onSetAffinity(value: boolean): void
  isPendingRestore(key: GroupTimeoutKey | GroupPolicyCountKey): boolean
  onSetTimeoutOverride(key: GroupTimeoutKey, enabled: boolean): void
  onSetTimeoutValue(key: GroupTimeoutKey, value: string): void
  onSetPolicyCountOverride(key: GroupPolicyCountKey, enabled: boolean): void
  onSetPolicyCountValue(key: GroupPolicyCountKey, value: string): void
  onPolicyCountError(key: GroupPolicyCountKey): string | undefined
  parameterOverrides: {
    revision: number
    protocols: AccessProtocol[]
    models: GroupModelItemDto[]
    onValidChange(value: boolean): void
    onInvalidEditsChange(value: boolean): void
    onChange(value: ParameterOverrideRuleDto[]): void
  }
  headerRules: {
    overridden: boolean
    pendingRestore: boolean
    revision: number
    value: HeaderRulesDto
    summary: string
    onToggle(): void
    onValidChange(value: boolean): void
    onInvalidEditsChange(value: boolean): void
    onChange(value: HeaderRulesDto): void
  }
}

export function GroupSettingsAdvanced({
  saved,
  draft,
  pending,
  proxy,
  onProxyModeChange,
  onProxyEndpointChange,
  onToggleProxy,
  websocket,
  onToggleWebsocket,
  onSetWebsocket,
  reasoningStatusFilter,
  onToggleReasoningStatusFilter,
  onSetReasoningStatusFilter,
  affinity,
  onToggleAffinity,
  onSetAffinity,
  isPendingRestore,
  onSetTimeoutOverride,
  onSetTimeoutValue,
  onSetPolicyCountOverride,
  onSetPolicyCountValue,
  onPolicyCountError,
  parameterOverrides,
  headerRules,
}: GroupSettingsAdvancedProps) {
  const t = useT()

  return (
    <details {...stylex.props(styles.advanced)}>
      <summary {...stylex.props(styles.advancedSummary)}>{t('group.settings.advanced')}</summary>
      <div {...stylex.props(styles.advancedContent)}>
        <div {...stylex.props(styles.advancedSections)}>
          <section id="settings-runtime" {...stylex.props(styles.section)}>
            <header>
              <h3 {...stylex.props(styles.sectionTitle)}>{t('group.settings.sections.runtime')}</h3>
              <p {...stylex.props(styles.sectionDescription)}>
                {t('group.settings.runtime.description')}
              </p>
            </header>
            <div {...stylex.props(styles.runtime)}>
              <SettingRow
                label={t('group.settings.runtime.responses_websocket_enabled')}
                value={
                  websocket.pendingRestore
                    ? t('group.settings.runtime.resetPending')
                    : websocket.label
                }
                help={t('group.settings.runtime.websocketHelp')}
                sourceLabel={
                  websocket.overridden
                    ? t('group.settings.runtime.override')
                    : websocket.pendingRestore
                      ? t('group.settings.runtime.pendingRestoreSource')
                      : t('group.settings.runtime.inherited')
                }
                actionLabel={
                  websocket.overridden
                    ? t('group.settings.runtime.useInherited')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={websocket.overridden}
                pendingRestore={websocket.pendingRestore}
                disabled={pending}
                onToggle={onToggleWebsocket}
                control={
                  <Switch
                    size="sm"
                    value={draft.overrides.responses_websocket_enabled ?? false}
                    isDisabled={pending}
                    label={t('group.settings.runtime.responses_websocket_enabled')}
                    isLabelHidden
                    onChange={onSetWebsocket}
                  />
                }
              />
              <SettingRow
                label={t('group.settings.runtime.responses_reasoning_status_filter_enabled')}
                value={reasoningStatusFilter.label}
                help={t('group.settings.runtime.reasoningStatusFilterHelp')}
                sourceLabel={
                  reasoningStatusFilter.overridden
                    ? t('group.settings.runtime.override')
                    : t('group.settings.runtime.groupDefault')
                }
                actionLabel={
                  reasoningStatusFilter.overridden
                    ? t('group.settings.runtime.useDefault')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={reasoningStatusFilter.overridden}
                disabled={pending}
                onToggle={onToggleReasoningStatusFilter}
                control={
                  <Switch
                    size="sm"
                    value={draft.overrides.responses_reasoning_status_filter_enabled ?? false}
                    isDisabled={pending}
                    label={t('group.settings.runtime.responses_reasoning_status_filter_enabled')}
                    isLabelHidden
                    onChange={onSetReasoningStatusFilter}
                  />
                }
              />
              <SettingRow
                label={t('common.proxy.title')}
                value={proxy.value}
                help={proxy.supported ? undefined : t('common.proxy.unsupportedHelp')}
                sourceLabel={
                  !proxy.supported
                    ? t('common.proxy.unsupportedBadge')
                    : proxy.overridden
                      ? t('group.settings.runtime.override')
                      : proxy.pendingRestore
                        ? t('group.settings.runtime.pendingRestoreSource')
                        : t('group.settings.runtime.inherited')
                }
                actionLabel={
                  proxy.overridden
                    ? t('group.settings.runtime.useInherited')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={proxy.supported && proxy.overridden}
                pendingRestore={proxy.supported && proxy.pendingRestore}
                locked={!proxy.supported}
                disabled={proxy.disabled}
                onToggle={onToggleProxy}
                control={
                  <ProxyOverrideControl
                    base={saved.proxy}
                    mode={proxy.mode}
                    endpoint={proxy.endpoint}
                    disabled={pending}
                    onModeChange={onProxyModeChange}
                    onEndpointChange={onProxyEndpointChange}
                  />
                }
              />
              {groupTimeoutKeys.map((key) => (
                <SettingRow
                  key={key}
                  label={t(`group.settings.runtime.${key}` as MessageId)}
                  value={
                    isPendingRestore(key)
                      ? t('group.settings.runtime.resetPending')
                      : t('group.settings.runtime.effective', {
                          value: saved.effective[key],
                        })
                  }
                  sourceLabel={
                    draft.overrides[key] !== undefined
                      ? t('group.settings.runtime.override')
                      : isPendingRestore(key)
                        ? t('group.settings.runtime.pendingRestoreSource')
                        : t('group.settings.runtime.inherited')
                  }
                  actionLabel={
                    draft.overrides[key] === undefined
                      ? t('group.settings.runtime.useOverride')
                      : t('group.settings.runtime.useInherited')
                  }
                  overridden={draft.overrides[key] !== undefined}
                  pendingRestore={isPendingRestore(key)}
                  disabled={pending}
                  onToggle={() => onSetTimeoutOverride(key, draft.overrides[key] === undefined)}
                  control={
                    <div {...stylex.props(styles.runtimeInput)}>
                      <TextInput
                        type="text"
                        size="sm"
                        value={String(draft.overrides[key])}
                        label={t('group.settings.runtime.valueFor', {
                          field: t(`group.settings.runtime.${key}` as MessageId),
                        })}
                        isLabelHidden
                        data-gptload-mono
                        isDisabled={pending}
                        onChange={(value) => onSetTimeoutValue(key, value)}
                      />
                      <span aria-hidden="true" {...stylex.props(styles.runtimeUnit)}>
                        {t('group.settings.runtime.seconds')}
                      </span>
                    </div>
                  }
                />
              ))}
              {policyRows.map((policy) => (
                <SettingRow
                  key={policy.key}
                  label={t(`group.settings.runtime.${policy.key}` as MessageId)}
                  value={
                    isPendingRestore(policy.key)
                      ? t('group.settings.runtime.resetPending')
                      : t('group.settings.runtime.effectiveCount', {
                          value: saved.effective[policy.key],
                        })
                  }
                  help={t(`group.settings.runtime.${policy.helpKey}` as MessageId)}
                  sourceLabel={
                    draft.overrides[policy.key] !== undefined
                      ? t('group.settings.runtime.override')
                      : isPendingRestore(policy.key)
                        ? t('group.settings.runtime.pendingRestoreSource')
                        : t('group.settings.runtime.inherited')
                  }
                  actionLabel={
                    draft.overrides[policy.key] === undefined
                      ? t('group.settings.runtime.useOverride')
                      : t('group.settings.runtime.useInherited')
                  }
                  overridden={draft.overrides[policy.key] !== undefined}
                  pendingRestore={isPendingRestore(policy.key)}
                  disabled={pending}
                  onToggle={() =>
                    onSetPolicyCountOverride(policy.key, draft.overrides[policy.key] === undefined)
                  }
                  control={
                    <div {...stylex.props(styles.runtimeInput)}>
                      <TextInput
                        id={`group-settings-${policy.key}`}
                        type="text"
                        size="sm"
                        value={String(draft.overrides[policy.key])}
                        label={t('group.settings.runtime.valueFor', {
                          field: t(`group.settings.runtime.${policy.key}` as MessageId),
                        })}
                        isLabelHidden
                        data-gptload-mono
                        isDisabled={pending}
                        status={
                          onPolicyCountError(policy.key) === undefined
                            ? undefined
                            : {
                                type: 'error',
                                message: onPolicyCountError(policy.key) ?? '',
                              }
                        }
                        statusVariant="tooltip"
                        onChange={(value) => onSetPolicyCountValue(policy.key, value)}
                      />
                      <span aria-hidden="true" {...stylex.props(styles.runtimeUnit)}>
                        {t('group.settings.runtime.countUnit')}
                      </span>
                    </div>
                  }
                />
              ))}
              <SettingRow
                label={t('group.settings.runtime.affinity_enabled')}
                value={
                  affinity.pendingRestore
                    ? t('group.settings.runtime.resetPending')
                    : affinity.label
                }
                help={t('group.settings.runtime.affinityHelp')}
                sourceLabel={
                  affinity.overridden
                    ? t('group.settings.runtime.override')
                    : affinity.pendingRestore
                      ? t('group.settings.runtime.pendingRestoreSource')
                      : t('group.settings.runtime.inherited')
                }
                actionLabel={
                  affinity.overridden
                    ? t('group.settings.runtime.useInherited')
                    : t('group.settings.runtime.useOverride')
                }
                overridden={affinity.overridden}
                pendingRestore={affinity.pendingRestore}
                divided={false}
                disabled={pending}
                onToggle={onToggleAffinity}
                control={
                  <Switch
                    size="sm"
                    value={draft.overrides.affinity_enabled ?? false}
                    isDisabled={pending}
                    label={t('group.settings.runtime.affinity_enabled')}
                    isLabelHidden
                    onChange={onSetAffinity}
                  />
                }
              />
            </div>
          </section>
          <section id="settings-parameters" {...stylex.props(styles.section)}>
            <header>
              <h3 {...stylex.props(styles.sectionTitle)}>
                {t('group.settings.sections.parameters')}
              </h3>
              <p {...stylex.props(styles.sectionDescription)}>
                {t('group.settings.parameterOverrides.description')}
              </p>
            </header>
            <ParameterOverrideRulesEditor
              resetKey={parameterOverrides.revision}
              value={draft.overrides.parameter_overrides ?? []}
              protocols={parameterOverrides.protocols}
              models={parameterOverrides.models}
              disabled={pending}
              onValidChange={parameterOverrides.onValidChange}
              onInvalidEditsChange={parameterOverrides.onInvalidEditsChange}
              onChange={parameterOverrides.onChange}
            />
          </section>

          <section id="settings-headers" {...stylex.props(styles.section)}>
            <SettingBlock
              title={t('group.settings.sections.headers')}
              help={t('group.settings.headers.description')}
              meta={headerRules.summary}
              sourceLabel={
                headerRules.overridden
                  ? t('group.settings.runtime.override')
                  : headerRules.pendingRestore
                    ? t('group.settings.runtime.pendingRestoreSource')
                    : t('group.settings.runtime.inherited')
              }
              actionLabel={
                headerRules.overridden
                  ? t('group.settings.runtime.useInherited')
                  : t('group.settings.runtime.useOverride')
              }
              overridden={headerRules.overridden}
              pendingRestore={headerRules.pendingRestore}
              disabled={pending}
              onToggle={() => void headerRules.onToggle()}
            >
              <HeaderRulesEditor
                resetKey={headerRules.revision}
                value={headerRules.value}
                disabled={pending || !headerRules.overridden}
                showAdd={headerRules.overridden}
                removeLabel={t('group.settings.runtime.headerRemove')}
                removeHint={t('group.settings.runtime.headerRemoveHint')}
                onValidChange={(value) => headerRules.onValidChange(value)}
                onInvalidEditsChange={(value) => headerRules.onInvalidEditsChange(value)}
                onChange={headerRules.onChange}
              />
            </SettingBlock>
          </section>
        </div>
      </div>
    </details>
  )
}

const styles = stylex.create({
  advanced: {
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-3)',
  },
  advancedSummary: {
    color: 'var(--color-text-muted)',
    cursor: 'pointer',
    fontSize: 'var(--text-sm)',
    fontWeight: 600,
  },
  advancedContent: {
    display: 'grid',
    gridTemplateColumns: '1fr',
    alignItems: 'start',
    gap: {
      default: 'var(--space-3)',
      '@media (max-width: 860px)': 'var(--space-2)',
    },
    marginTop: 'var(--space-3)',
  },
  advancedSections: {
    display: 'grid',
    minWidth: 0,
    gap: {
      default: 'var(--space-4)',
      '@media (max-width: 860px)': 'var(--space-3)',
    },
  },
  section: {
    display: 'grid',
    gap: '15px',
    scrollMarginTop: '76px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: '17px',
  },
  sectionTitle: {
    margin: 0,
    fontSize: 'var(--text-body)',
    fontWeight: 650,
  },
  sectionDescription: {
    maxWidth: '580px',
    marginTop: '3px',
    marginBottom: 0,
    marginLeft: 0,
    marginRight: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  runtime: {
    display: 'grid',
    gap: 'var(--space-1)',
  },
  runtimeInput: {
    display: 'flex',
    width: {
      default: 'min(100%, 190px)',
      '@media (max-width: 800px)': 'min(100%, 220px)',
    },
    minWidth: 0,
    alignItems: 'center',
    gap: '7px',
  },
  runtimeUnit: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: '11px',
    whiteSpace: 'nowrap',
  },
})
