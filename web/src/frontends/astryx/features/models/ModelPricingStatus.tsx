import { Badge } from '@astryxdesign/core'

import type { ModelPricingStatus as ModelPricingStatusValue } from '@shared/control/types'

/**
 * Compact pricing-state badge — the Astryx counterpart of classic
 * ModelPricingStatus.vue: `configured` reads success, `pending` reads
 * warning. Distinct from ModelPriceStatusBadge, which presents the
 * price-method/match detail on the models collection page.
 */
export function ModelPricingStatus({
  status,
  labels,
}: {
  status: ModelPricingStatusValue
  labels: Record<ModelPricingStatusValue, string>
}) {
  return <Badge variant={status === 'configured' ? 'success' : 'warning'} label={labels[status]} />
}
