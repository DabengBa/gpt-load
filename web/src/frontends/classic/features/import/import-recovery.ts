import { inject, type InjectionKey } from 'vue'

import type { ImportRecoveryService } from '@shared/controllers/import-recovery'

export * from '@shared/controllers/import-recovery'

export const importRecoveryKey: InjectionKey<ImportRecoveryService> = Symbol('import-recovery')

export function useImportRecovery(): ImportRecoveryService {
  const service = inject(importRecoveryKey)
  if (!service) throw new Error('IMPORT_RECOVERY_NOT_PROVIDED')
  return service
}
