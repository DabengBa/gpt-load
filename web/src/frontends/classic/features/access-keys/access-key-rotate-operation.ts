import type { AccessKeyDto } from '@shared/control/types'

export interface PendingAccessKeyRotateOperation {
  base: AccessKeyDto
  idempotencyKey: string
  state: 'indeterminate' | 'reconciling'
}
