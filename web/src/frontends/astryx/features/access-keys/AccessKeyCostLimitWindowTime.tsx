import type { AccessKeyCostLimitRuleStatusDto } from '@shared/control/types'

import { CostLimitWindowTime } from '../home/CostLimitWindowTime'

/**
 * Port of classic `AccessKeyCostLimitWindowTime.vue`. The Home frontend needed
 * the same live/preview window rendering first, so it was ported there as
 * `CostLimitWindowTime` — this alias keeps the access-keys feature namespace
 * and prop surface (`rule`) identical to the Vue component without duplicating
 * the implementation.
 */
export function AccessKeyCostLimitWindowTime({
  rule,
}: {
  rule: AccessKeyCostLimitRuleStatusDto
}) {
  return <CostLimitWindowTime rule={rule} />
}
