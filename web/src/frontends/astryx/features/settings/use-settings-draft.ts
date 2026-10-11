import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useReducer, useState } from 'react'

import type { SettingsResource } from '@shared/control/resources/settings'
import {
  createSettingsDraftController,
  type SettingsDraftController,
} from '@shared/controllers/settings-draft'
import type { ApiClient } from '@shared/http/client'

export function useSettingsDraftController(options: {
  client: ApiClient
  resource: SettingsResource | null
  locale: string
}): SettingsDraftController {
  const queryClient = useQueryClient()
  const [, bump] = useReducer((count: number) => count + 1, 0)

  const [controller] = useState(() =>
    createSettingsDraftController({
      client: options.client,
      queryClient,
      locale: options.locale,
    }),
  )
  // `locale` feeds only the query key at save/cache time — a post-commit push
  // keeps it current without capturing a ref.
  useEffect(() => controller.setLocale(options.locale), [controller, options.locale])

  useEffect(() => controller.subscribe(bump), [controller])
  useEffect(() => controller.setResource(options.resource), [controller, options.resource])
  useEffect(() => () => controller.dispose(), [controller])

  return controller
}
