import type { ProtocolValue } from './protocols'

export type GroupProtocol = ProtocolValue
export type AccessProtocol = ProtocolValue
export const routeStrategies = ['native_first', 'weighted_mix'] as const
export type RouteStrategy = (typeof routeStrategies)[number]
export type FailureCategory =
  | 'ok'
  | 'rate_limited'
  | 'model_unavailable'
  | 'invalid_key'
  | 'billing'
  | 'upstream_host_error'
  | 'client_error'
  | 'downstream_cancel'
  | 'authentication_required'
  | 'ambiguous'

export type GroupCollectionStatus = 'available' | 'unavailable' | 'disabled'
export type GroupUnavailableReason = 'no_available_credentials' | 'no_models'
export type GroupCollectionSort = 'recent' | 'status' | 'name' | 'created'
export type ModelPricingStatus = 'pending' | 'configured'
export type ChannelParamsDto = Record<string, string>
export type ProxyConfiguredMode = 'inherit' | 'direct' | 'custom'
export type ProxyEffectiveMode = 'direct' | 'environment' | 'custom'
export type ProxyEffectiveSource = 'credential' | 'group' | 'global' | 'environment' | 'default'

export interface ProxyViewDto {
  configured_mode: ProxyConfiguredMode
  effective_mode: ProxyEffectiveMode
  effective_source: ProxyEffectiveSource
  display_url?: string
  has_auth: boolean
}

export type ProxyConfigInput = { mode: 'direct' } | { mode: 'custom'; url: string }
export type ProxyMutation = ProxyConfigInput | null

export interface GroupCollectionFilters {
  q?: string
  status?: GroupCollectionStatus
  sort: GroupCollectionSort
  page: number
  page_size: 100
}

export interface GroupCollectionSummaryDto {
  total: number
  available: number
  unavailable: number
  disabled: number
}

export interface GroupCollectionItemDto {
  id: number
  name: string

  channel_id: string
  params: ChannelParamsDto
  provider_url: string | null
  status: GroupCollectionStatus
  model_count: number
  client_model_count: number
  credential_configured: boolean
  credential_status: CredentialStatus | null
}

export interface GroupCollectionPaginationDto {
  page: number
  page_size: 100
  total_items: number
  total_pages: number
}

export interface GroupCollectionResponseDto {
  observed_at_ms: number
  summary: GroupCollectionSummaryDto
  items: GroupCollectionItemDto[]
  pagination: GroupCollectionPaginationDto
}

export interface GroupSummaryDto {
  id: number
  name: string

  channel_id: string
  params: ChannelParamsDto
  provider_url: string | null
  service_status: GroupCollectionStatus
  service_status_reason: GroupUnavailableReason | null
  credential_configured: boolean
  credential_status: CredentialStatus | null
  model_count: number
}

export interface HeaderRulesDto {
  set: Record<string, string>
  remove: string[]
}

export type ParameterJSONValue =
  null | boolean | number | string | unknown[] | Record<string, unknown>

export interface ParameterOverrideMatchDto {
  protocol?: AccessProtocol
  model?: string
}

export interface ParameterOverrideRuleDto {
  match: ParameterOverrideMatchDto
  set?: Record<string, ParameterJSONValue>
  remove?: string[]
}

export interface GroupRuntimeConfigDto {
  first_byte_timeout?: number
  request_timeout?: number
  stream_idle_timeout?: number
  blacklist_threshold?: number
  header_rules?: HeaderRulesDto
  affinity_enabled?: boolean
  responses_websocket_enabled?: boolean
  responses_reasoning_status_filter_enabled?: boolean
  parameter_overrides?: ParameterOverrideRuleDto[]
}

export type ReasoningEffortDto = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'

export interface GroupEffectiveConfigDto {
  first_byte_timeout: number
  request_timeout: number
  stream_idle_timeout: number
  blacklist_threshold: number
  header_rules: HeaderRulesDto
  affinity_enabled: boolean
  responses_websocket_enabled: boolean
  responses_reasoning_status_filter_enabled: boolean
}

export interface GroupSettingsDto {
  name: string

  channel_id: string
  params: ChannelParamsDto
  provider_url: string | null
  enabled: boolean
  overrides: GroupRuntimeConfigDto
  effective: GroupEffectiveConfigDto
  proxy: ProxyViewDto
}

export interface EntryCircuitBreakerDto {
  blacklist_threshold?: number | null
  cooldown_seconds?: number | null
}

export interface GroupModelItemDto {
  concurrency_limit: number
  id: string
  alias: string
  alias_enabled: boolean
  client_model: string
  test_alias: string
  /** Present after the server lazily backfills the route-entry identity. */
  entry_id?: string
  weight: number | null
  priority: number | null
  /** Optional for compatibility with responses produced before breaker exposure. */
  circuit_breaker?: EntryCircuitBreakerDto | null
  pricing_status: ModelPricingStatus
  /** Present when a price record exists for this model; absent means no record yet. */
  price_id?: number
}

export interface GroupModelsDto {
  items: GroupModelItemDto[]
  total: number
  pending: number
}

export type CredentialStatus = 'available' | 'cooldown' | 'blacklisted' | 'disabled'
export type CredentialRecoveryMode = 'none' | 'cooldown' | 'scheduled_release' | 'manual'
export interface CredentialRecoveryDto {
  mode: CredentialRecoveryMode
  automatic: boolean
  at_ms: number | null
}

export interface CredentialItemDto {
  credential_id: number
  secret_version: number
  mask: string
  effective_status: CredentialStatus
  recent_success_count: number
  recent_failure_count: number
  consecutive_failure_count: number
  last_failure_category: FailureCategory
  last_failure_summary?: string
  last_status_code: number | null
  cooldown_until_ms: number | null
  recovery: CredentialRecoveryDto
}

export interface CredentialDetailDto {
  credential: CredentialItemDto | null
}

export interface CredentialRevealDto {
  credential_id: number
  credential: Record<string, string>
  revealed_at_ms: number
}

export type CredentialTestOutcome = 'passed' | 'failed' | 'inconclusive'

export type CredentialTestReason =
  | 'invalid_credential'
  | 'insufficient_balance'
  | 'model_unavailable'
  | 'rate_limited'
  | 'timeout'
  | 'upstream_error'
  | 'probe_incompatible'
  | 'no_answer'
  | 'invalid_response'
  | 'unknown'

export interface CredentialTestResultDto {
  outcome: CredentialTestOutcome
  model: string
  protocol: ProtocolValue
  latency_ms: number
  reason: CredentialTestReason | null
  recovered: boolean
  log_id: string | null
  tested_at_ms: number
}

export interface GroupOptionDto {
  id: number
  name: string
  channel_id: string
  params: ChannelParamsDto
  provider_url: string | null
  enabled: boolean
  models: string[]
}

export interface CredentialCounts {
  total: number
  available: number
  cooldown: number
  blacklisted: number
  disabled: number
}

export interface HealthCredentialCountsDto {
  credentials: number
  available: number
  cooldown: number
  blacklisted: number
}

export interface HealthGroupDto {
  id: number
  name: string
  enabled: boolean
  counts: HealthCredentialCountsDto
}

export interface HealthRecoveryDto {
  automatic: boolean
  mode: 'cooldown_expiry' | 'scheduled_release'
  at_ms: number | null
}

export interface HealthProblemCredentialDto {
  credential_id: number
  group_id: number
  group_name: string
  cooldown_until_ms: number | null
  failure_count: number
  recent_success_count: number
  recent_problem_count: number
  consecutive_problem_count: number
  recovery: HealthRecoveryDto
  /** API 密钥掩码。 */
  identity: string
  last_failure_category: Exclude<FailureCategory, 'ok'>
  last_status_code: number | null
}

export interface RequestLogHealthDto {
  enqueued_total: number
  persisted_total: number
  dropped_not_running_total: number
  dropped_queue_full_total: number
  dropped_stopping_total: number
  dropped_persist_failed_total: number
  dropped_shutdown_total: number
  dropped_total: number
  write_failure_total: number
  access_quota_checkpoint_write_failure_total: number
  access_quota_checkpoint_degraded: boolean
  retention_delete_failure_total: number
  queue_depth: number
  queue_capacity: number
  last_write_failure_at_ms: number | null
  last_access_quota_checkpoint_write_failure_at_ms: number | null
  last_retention_failure_at_ms: number | null
}

export interface RuntimeDebugCaptureHealthDto {
  enabled: boolean
  running: boolean
  retention_seconds: number
  active: number
  completed: number
  failed: number
  sweep_total: number
  removed_total: number
  sweep_failure_total: number
  error: string
  last_sweep_at_ms: number | null
  last_failure_at_ms: number | null
}

export interface RuntimeHealthDto {
  observed_at_ms: number
  version: string
  uptime_seconds: number
  snapshot_revision: number
  stats_window_seconds: number
  counts: HealthCredentialCountsDto
  groups: HealthGroupDto[]
  cooldown_credentials: HealthProblemCredentialDto[]
  blacklisted_credentials: HealthProblemCredentialDto[]
  blocked_access_keys: HealthAccessKeyCostLimitDto[]
  request_log: RequestLogHealthDto
  debug_capture: RuntimeDebugCaptureHealthDto
}

export interface AccessKeyFiltersDto {
  groups: number[]
  protocols: AccessProtocol[]
  models: string[]
  allowed_cidrs: string[]
}

export type AccessKeyCostLimitKind = 'total' | 'periodic'
export type AccessKeyCostLimitRuleState = 'available' | 'inactive' | 'exhausted'

export interface AccessKeyCostLimitRuleDto {
  id: number
  kind: AccessKeyCostLimitKind
  limit_usd: string
  period_seconds: number
}

export interface AccessKeyCostLimitRuleStatusDto extends AccessKeyCostLimitRuleDto {
  used_usd: string
  remaining_usd: string
  status: AccessKeyCostLimitRuleState
  window_started_at_ms: number | null
  window_ends_at_ms: number | null
}

export interface AccessKeyCostLimitStatusDto {
  observed_at_ms: number
  allowed: boolean
  recoverable: boolean
  next_available_at_ms: number | null
  rules: AccessKeyCostLimitRuleStatusDto[]
}

export interface HealthAccessKeyCostLimitDto {
  access_key_id: number
  name: string
  masked_key: string
  recoverable: boolean
  next_available_at_ms: number | null
  blocking_rules: AccessKeyCostLimitRuleStatusDto[]
}

export interface AccessKeyDto {
  id: number
  name: string
  masked_key: string
  status: 'active' | 'disabled'
  filters: AccessKeyFiltersDto
  expires_at_ms: number | null
  rpm_limit: number
  cost_limit_rules: AccessKeyCostLimitRuleDto[]
  cost_limit_status: AccessKeyCostLimitStatusDto | null
  created_at_ms: number
  updated_at_ms: number
}

export type AccessKeyCollectionStatus = AccessKeyDto['status']

export interface AccessKeyCollectionFilters {
  q?: string
  status?: AccessKeyCollectionStatus
  page: number
  page_size: 20
}

export interface AccessKeyCollectionSummaryDto {
  total: number
  active: number
  disabled: number
}

export interface AccessKeyCollectionItemDto extends AccessKeyDto {
  expired: boolean
  last_request_at_ms: number | null
}

export interface AccessKeyCollectionPaginationDto {
  page: number
  page_size: 20
  total_items: number
  total_pages: number
}

export interface AccessKeyCollectionResponseDto {
  summary: AccessKeyCollectionSummaryDto
  items: AccessKeyCollectionItemDto[]
  pagination: AccessKeyCollectionPaginationDto
}

export interface AccessKeyOptionDto {
  id: number
  name: string
  status: AccessKeyDto['status']
}

export interface AccessKeyCreateResultDto extends AccessKeyDto {
  key?: string
  replayed: boolean
}

export interface AccessKeyRotateResultDto extends AccessKeyDto {
  key?: string
  replayed: boolean
}

export interface AccessKeyRevealDto {
  id: number
  key: string
  revealed_at_ms: number
}

export type AgentCredentialScope = 'diagnostics:read' | 'changes:propose' | 'changes:apply'

export interface AgentCredentialDto {
  id: number
  name: string
  scopes: AgentCredentialScope[]
  status: 'active' | 'disabled'
  expires_at_ms: number | null
  disabled_at_ms: number | null
  created_at_ms: number
  updated_at_ms: number
}

export interface AgentCredentialListDto {
  items: AgentCredentialDto[]
}

export interface AgentCredentialCreateResultDto extends AgentCredentialDto {
  secret?: string
  replayed: boolean
  operation_id: string
}
