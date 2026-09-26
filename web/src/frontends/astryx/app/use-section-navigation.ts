import { useEffect, useState } from 'react'

import {
  createSectionNavigationController,
  type SectionNavigationController,
} from '@shared/controllers/section-navigation'

export interface SectionNavigationOptions {
  ids: readonly string[]
  initialId?: string
  topOffset?: number
}

export function useSectionNavigation({
  ids,
  initialId,
  topOffset = 88,
}: SectionNavigationOptions): Pick<
  SectionNavigationController,
  'selectSection'
> & { activeSection: string } {
  const [controller] = useState(() =>
    createSectionNavigationController({ ids, initialId, topOffset }),
  )
  // The scrollspy reads the id list only from scroll/resize handlers and
  // notifyUpdated — pushing it in an effect keeps it fresh without refs.
  useEffect(() => controller.updateIds(ids), [controller, ids])
  const [activeSection, setActiveSection] = useState(controller.getActiveSection())

  useEffect(() => controller.subscribe(() => setActiveSection(controller.getActiveSection())), [
    controller,
  ])
  useEffect(() => controller.mount(), [controller])
  // Vue's onUpdated equivalent — re-sync the scrollspy after every render.
  useEffect(() => controller.notifyUpdated())

  return { activeSection, selectSection: controller.selectSection }
}
