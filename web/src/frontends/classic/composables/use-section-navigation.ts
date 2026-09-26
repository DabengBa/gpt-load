import { onBeforeUnmount, onMounted, onUpdated, ref, type Ref } from 'vue'

import { createSectionNavigationController } from '@shared/controllers/section-navigation'

export interface SectionNavigationOptions {
  ids: Readonly<Ref<readonly string[]>>
  initialId?: string
  topOffset?: number
}

export interface SectionNavigationController {
  activeSection: Ref<string>
  selectSection(id: string, behavior?: ScrollBehavior): void
}

export function useSectionNavigation({
  ids,
  initialId,
  topOffset = 88,
}: SectionNavigationOptions): SectionNavigationController {
  const controller = createSectionNavigationController({
    ids: ids.value,
    initialId,
    topOffset,
  })
  const activeSection = ref(controller.getActiveSection())
  const unsubscribe = controller.subscribe(() => {
    activeSection.value = controller.getActiveSection()
  })
  let unmount: (() => void) | undefined

  onMounted(() => {
    unmount = controller.mount()
  })

  onUpdated(() => {
    controller.updateIds(ids.value)
    controller.notifyUpdated()
  })

  onBeforeUnmount(() => {
    unmount?.()
    unsubscribe()
  })

  return { activeSection, selectSection: controller.selectSection }
}
