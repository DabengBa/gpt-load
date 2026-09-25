import { inject, onBeforeUnmount, onMounted, readonly, ref, type InjectionKey, type Ref } from 'vue'
import { onBeforeRouteLeave, onBeforeRouteUpdate, type RouteLocationNormalized } from 'vue-router'

import {
  createUnsavedChangesController as createSharedUnsavedChangesController,
} from '@shared/controllers/unsaved-changes'

export interface UnsavedChangesController {
  readonly dialogOpen: Readonly<Ref<boolean>>
  bypassNext(): void
  consumeBypass(): boolean
  requestConfirmation(): Promise<boolean>
  resolveConfirmation(confirmed: boolean): void
}

export interface UnsavedChangesOptions {
  blocked?: Readonly<Ref<boolean>>
  allowRouteUpdate?: (to: RouteLocationNormalized, from: RouteLocationNormalized) => boolean
}

export interface UnsavedChangesGuard {
  confirmDiscard(): Promise<boolean>
  runWithoutPrompt<T>(navigate: () => Promise<T>): Promise<T>
}

export function createUnsavedChangesController(): UnsavedChangesController {
  const core = createSharedUnsavedChangesController()
  const dialogOpen = ref(core.getDialogOpen())
  core.subscribe(() => {
    dialogOpen.value = core.getDialogOpen()
  })

  return {
    dialogOpen: readonly(dialogOpen),
    bypassNext: () => core.bypassNext(),
    consumeBypass: () => core.consumeBypass(),
    requestConfirmation: () => core.requestConfirmation(),
    resolveConfirmation: (confirmed) => core.resolveConfirmation(confirmed),
  }
}

export const unsavedChangesKey: InjectionKey<UnsavedChangesController> = Symbol('unsaved-changes')

export function useUnsavedChanges(
  dirty: Readonly<Ref<boolean>>,
  options: UnsavedChangesOptions = {},
): UnsavedChangesGuard {
  const controller = useUnsavedChangesController()

  const beforeUnload = (event: BeforeUnloadEvent | Event) => {
    if (!dirty.value && !options.blocked?.value) return
    event.preventDefault()
    if ('returnValue' in event) event.returnValue = ''
  }

  onMounted(() => window.addEventListener('beforeunload', beforeUnload))
  onBeforeUnmount(() => window.removeEventListener('beforeunload', beforeUnload))

  async function confirmDiscard(): Promise<boolean> {
    if (options.blocked?.value) return false
    return !dirty.value || controller.requestConfirmation()
  }

  const confirmNavigation = async () => {
    if (controller.consumeBypass()) return true
    return await confirmDiscard()
  }

  onBeforeRouteLeave(confirmNavigation)
  onBeforeRouteUpdate((to, from) => {
    if (options.allowRouteUpdate?.(to, from)) return true
    return confirmNavigation()
  })

  async function runWithoutPrompt<T>(navigate: () => Promise<T>): Promise<T> {
    controller.bypassNext()
    try {
      return await navigate()
    } finally {
      controller.consumeBypass()
    }
  }

  return { confirmDiscard, runWithoutPrompt }
}

export function useUnsavedChangesController(): UnsavedChangesController {
  const controller = inject(unsavedChangesKey)
  if (!controller) throw new Error('UNSAVED_CHANGES_NOT_PROVIDED')
  return controller
}
