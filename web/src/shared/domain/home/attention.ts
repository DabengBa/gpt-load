import type { HealthProblemCredentialDto } from '@shared/control/resources/health'

export type AttentionKind = 'billing' | 'blacklisted'

export interface AttentionItem {
  kind: AttentionKind
  groupID: number
  groupName: string
  /** 对应凭据数。 */
  value: number
  /** 仅 billing：最早一次自动重探（冷却结束）时刻 */
  cooldownUntilMS?: number
}

/** 首页最多渲染的条数；超过就退化成一行汇总，避免把接入面板挤到折叠线以下。 */
export const attentionRowLimit = 2

/**
 * 把健康响应压成首页「需要处理」要说的话。
 *
 * 冷却一般不报：几分钟内自愈，看完不需要做任何事。billing 是唯一例外——
 * 余额不足要人去充值，24h 冷却期内每次出现都必须浮出来；展示最早一次
 * 自动重探时刻，和凭据列表的恢复语义一致。
 *
 */
export function collectAttentionItems(
  blacklisted: readonly HealthProblemCredentialDto[],
  cooldown: readonly HealthProblemCredentialDto[],
): AttentionItem[] {
  const billingByGroup = new Map<number, AttentionItem>()
  for (const credential of cooldown) {
    if (credential.last_failure_category !== 'billing') continue
    const existing = billingByGroup.get(credential.group_id)
    if (existing) {
      existing.value += 1
      const until = credential.cooldown_until_ms
      if (
        until !== null &&
        (existing.cooldownUntilMS === undefined || until < existing.cooldownUntilMS)
      ) {
        existing.cooldownUntilMS = until
      }
      continue
    }
    billingByGroup.set(credential.group_id, {
      kind: 'billing',
      groupID: credential.group_id,
      groupName: credential.group_name,
      value: 1,
      cooldownUntilMS: credential.cooldown_until_ms ?? undefined,
    })
  }
  const billingItems = [...billingByGroup.values()].sort(
    (left, right) => (left.cooldownUntilMS ?? 0) - (right.cooldownUntilMS ?? 0),
  )

  const byGroup = new Map<number, AttentionItem>()
  for (const credential of blacklisted) {
    const existing = byGroup.get(credential.group_id)
    if (existing) {
      existing.value += 1
      continue
    }
    byGroup.set(credential.group_id, {
      kind: 'blacklisted',
      groupID: credential.group_id,
      groupName: credential.group_name,
      value: 1,
    })
  }

  const items = [...byGroup.values()].sort((left, right) => right.value - left.value)

  return [...billingItems, ...items]
}

/** 需要处理的问题总数，用于超限时的汇总文案。 */
export function attentionTotal(items: readonly AttentionItem[]): number {
  return items.reduce((total, item) => total + item.value, 0)
}
