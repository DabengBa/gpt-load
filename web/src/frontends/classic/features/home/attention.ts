import type {
  HealthExpiringResetCreditDto,
  HealthProblemCredentialDto,
  HealthQuotaCredentialDto,
} from '@/app/resources/health'

export type AttentionKind = 'billing' | 'blacklisted' | 'expiringResetCredit' | 'lowQuota'

export interface AttentionItem {
  kind: AttentionKind
  groupID: number
  groupName: string
  /** billing/blacklisted：对应凭据数；expiringResetCredit：临期卡数；lowQuota：剩余额度比例 0..1 */
  value: number
  /** 仅 lowQuota：额度窗口重置时刻 */
  resetAtMS?: number
  /** 仅 expiringResetCredit：最近一张卡的到期时刻 */
  expiresAtMS?: number
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
 * 也刻意不说「全部拉黑」：classifyHealthKey 把 auth_state != ready 的凭据归入 disabled 桶，
 * 且 disabled 不计入 counts.credentials，拿它当分母会算错。只报绝对条数。
 */
export function collectAttentionItems(
  blacklisted: readonly HealthProblemCredentialDto[],
  expiringResetCredits: readonly HealthExpiringResetCreditDto[],
  lowQuota: readonly HealthQuotaCredentialDto[],
  cooldown: readonly HealthProblemCredentialDto[],
): AttentionItem[] {
  const billingByGroup = new Map<number, AttentionItem>()
  for (const credential of cooldown) {
    if (credential.last_failure_category !== 'billing') continue
    const existing = billingByGroup.get(credential.group_id)
    if (existing) {
      existing.value += 1
      const until = credential.cooldown_until_ms
      if (until !== null && (existing.cooldownUntilMS === undefined || until < existing.cooldownUntilMS)) {
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

  const resetCreditByGroup = new Map<number, AttentionItem>()
  for (const credential of expiringResetCredits) {
    const existing = resetCreditByGroup.get(credential.group_id)
    if (existing) {
      existing.value += credential.count
      existing.expiresAtMS = Math.min(
        existing.expiresAtMS ?? credential.nearest_expires_at_ms,
        credential.nearest_expires_at_ms,
      )
      continue
    }
    resetCreditByGroup.set(credential.group_id, {
      kind: 'expiringResetCredit',
      groupID: credential.group_id,
      groupName: credential.group_name,
      value: credential.count,
      expiresAtMS: credential.nearest_expires_at_ms,
    })
  }
  const resetCreditItems = [...resetCreditByGroup.values()].sort(
    (left, right) => (left.expiresAtMS ?? 0) - (right.expiresAtMS ?? 0),
  )

  // 额度按剩余从少到多，最紧迫的先说。
  const quotaItems = [...lowQuota]
    .sort((left, right) => left.remaining - right.remaining)
    .map<AttentionItem>((credential) => ({
      kind: 'lowQuota',
      groupID: credential.group_id,
      groupName: credential.group_name,
      value: credential.remaining,
      resetAtMS: credential.reset_at_ms,
    }))

  return [...billingItems, ...items, ...resetCreditItems, ...quotaItems]
}

/** 需要处理的问题总数，用于超限时的汇总文案。 */
export function attentionTotal(items: readonly AttentionItem[]): number {
  return items.reduce((total, item) => total + (item.kind === 'lowQuota' ? 1 : item.value), 0)
}
