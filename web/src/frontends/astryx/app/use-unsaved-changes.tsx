import { AlertDialog } from '@astryxdesign/core'
import { useBlocker } from '@tanstack/react-router'
import { useCallback, useEffect, useRef, useSyncExternalStore, type ReactNode } from 'react'

import type { UnsavedChangesController } from '@shared/controllers/unsaved-changes'

import { useT } from './i18n'
import { useAppServices } from './services'

export interface BlockerRouteLocation {
  routeId: string
  fullPath: string
  pathname: string
  params: Record<string, unknown>
  search: Record<string, unknown>
}

export interface UnsavedChangesOptions {
  dirty: boolean
  blocked?: boolean
  allowRouteUpdate?: (current: BlockerRouteLocation, next: BlockerRouteLocation) => boolean
  // The confirmation dialog reads the shared controller store, so every
  // mounted consumer would render a duplicate dialog. Consumers nested inside
  // a view that already renders its own `dialog` pass false here.
  renderDialog?: boolean
}

export interface UnsavedChangesGuard {
  confirmDiscard(): Promise<boolean>
  runWithoutPrompt<T>(navigate: () => Promise<T>): Promise<T>
  dialog: ReactNode
}

function UnsavedChangesAlert({ controller }: { controller: UnsavedChangesController }) {
  const t = useT()
  // useSyncExternalStore forces a synchronous re-render on store change. A
  // plain setState would inherit the in-flight navigation's transition scope
  // and never commit — the blocker promise only resolves after the user
  // answers, deadlocking the dialog.
  const subscribe = useCallback(
    (listener: () => void) => controller.subscribe(listener),
    [controller],
  )
  const open = useSyncExternalStore(subscribe, () => controller.getDialogOpen())
  return (
    <AlertDialog
      isOpen={open}
      onOpenChange={(open) => {
        if (!open) controller.resolveConfirmation(false)
      }}
      title={t('common.unsavedChangesDialog.title')}
      description={t('common.unsavedChangesDialog.description')}
      cancelLabel={t('common.unsavedChangesDialog.cancel')}
      actionLabel={t('common.unsavedChangesDialog.confirm')}
      onAction={() => controller.resolveConfirmation(true)}
    />
  )
}

export function useUnsavedChanges(options: UnsavedChangesOptions): UnsavedChangesGuard {
  const controller = useAppServices().unsavedChanges
  const optionsRef = useRef(options)
  useEffect(() => {
    optionsRef.current = options
  })

  useBlocker({
    shouldBlockFn: ({ current, next }) => {
      const { dirty, blocked, allowRouteUpdate } = optionsRef.current
      if (controller.consumeBypass()) return false
      if (allowRouteUpdate?.(current as BlockerRouteLocation, next as BlockerRouteLocation))
        return false
      if (blocked) return true
      if (!dirty) return false
      return controller.requestConfirmation().then((confirmed) => !confirmed)
    },
    enableBeforeUnload: () => optionsRef.current.dirty || optionsRef.current.blocked === true,
  })

  async function confirmDiscard(): Promise<boolean> {
    if (optionsRef.current.blocked) return false
    return !optionsRef.current.dirty || controller.requestConfirmation()
  }

  async function runWithoutPrompt<T>(navigate: () => Promise<T>): Promise<T> {
    controller.bypassNext()
    try {
      return await navigate()
    } finally {
      controller.consumeBypass()
    }
  }

  return {
    confirmDiscard,
    runWithoutPrompt,
    dialog: options.renderDialog === false ? null : <UnsavedChangesAlert controller={controller} />,
  }
}
