import { useEffect, useState } from 'react'

import {
  createTransientFlag,
  type TransientFlagController,
} from '@shared/controllers/transient-flag'

export function useTransientFlag(
  durationMs: number,
): Pick<TransientFlagController, 'show' | 'clear'> & { value: boolean } {
  const [controller] = useState(() => createTransientFlag(durationMs))
  const [value, setValue] = useState(controller.getValue())

  useEffect(() => controller.subscribe(() => setValue(controller.getValue())), [controller])
  useEffect(() => () => controller.clear(), [controller])

  return { value, show: controller.show, clear: controller.clear }
}
