export const channelOperations = [
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
  'list_models',
  'probe',
  'web_search',
] as const

export type ChannelOperation = (typeof channelOperations)[number]

export function sameStringMembers(left: readonly string[], right: readonly string[]): boolean {
  if (left.length !== right.length) return false
  const leftSet = new Set(left)
  const rightSet = new Set(right)
  return (
    leftSet.size === left.length &&
    rightSet.size === right.length &&
    [...leftSet].every((value) => rightSet.has(value))
  )
}
