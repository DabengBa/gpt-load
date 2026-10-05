export type RouteInspectReasonCode =
  | 'access_key_disabled'
  | 'access_key_expired'
  | 'protocol_filtered'
  | 'model_filtered'
  | 'model_required_by_filter'
  | 'operation_unsupported'
  | 'native_route_required'
  | 'no_route_target'
  | 'group_disabled'
  | 'group_filtered'
  | 'no_available_group'
  | 'no_credentials'
  | 'credential_blacklisted'
  | 'credential_cooldown'
  | 'credential_auth_unavailable'
  | 'credential_not_allowed'
  | 'no_available_credential'
  | 'entry_disabled'
  | 'entry_blacklisted'
  | 'entry_cooldown'
  | 'entry_weight_zero'
  | 'tier_demoted'

export type RouteInspectOperation =
  | 'chat_completion'
  | 'responses_create'
  | 'responses_retrieve'
  | 'responses_delete'
  | 'responses_cancel'
  | 'responses_input_items'
  | 'responses_compact'
  | 'responses_input_tokens'
  | 'count_tokens'
  | 'responses_passthrough'
  | 'images_generate'
  | 'images_edit'
  | 'embeddings_create'
  | 'rerank'
export type RouteInspectRequirement = 'any' | 'native'

export interface RouteInspectCredentialDto {
  credential_id: number
  available: boolean
  reason_code: RouteInspectReasonCode | null
  cooldown_until_ms: number | null
}

export const routeInspectOperations = [
  'chat_completion',
  'responses_create',
  'responses_retrieve',
  'responses_delete',
  'responses_cancel',
  'responses_input_items',
  'responses_compact',
  'responses_input_tokens',
  'count_tokens',
  'responses_passthrough',
  'images_generate',
  'images_edit',
  'embeddings_create',
  'rerank',
] as const
export const routeInspectRequirements = ['any', 'native'] as const
export const routeInspectReasonCodes = [
  'access_key_disabled',
  'access_key_expired',
  'protocol_filtered',
  'model_filtered',
  'model_required_by_filter',
  'operation_unsupported',
  'native_route_required',
  'no_route_target',
  'group_disabled',
  'group_filtered',
  'no_available_group',
  'no_credentials',
  'credential_blacklisted',
  'credential_cooldown',
  'credential_auth_unavailable',
  'credential_not_allowed',
  'no_available_credential',
  'entry_disabled',
  'entry_blacklisted',
  'entry_cooldown',
  'entry_weight_zero',
  'tier_demoted',
] as const
