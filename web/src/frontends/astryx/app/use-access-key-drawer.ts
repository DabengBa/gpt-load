import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

import { createUUID } from '@shared/lib/uuid'
import {
  createAccessKeyDrawerController,
  type AccessKeyDrawerSnapshot,
} from '@shared/controllers/access-key-drawer'
import type { PendingAccessKeyCreateOperation } from '@shared/domain/access-keys/access-key-create-operation'
import type { PendingAccessKeyEditOperation } from '@shared/domain/access-keys/access-key-edit-operation'

import { useAppServices } from './services'

export interface AccessKeyDrawerControllerOptions {
  isOpen(): boolean
  onSaved(kind: 'created' | 'updated', name: string): void
  onDeleted(name: string): void
  onCreateOperation(operation: PendingAccessKeyCreateOperation | null): void
  onEditOperation(operation: PendingAccessKeyEditOperation | null): void
}

/**
 * Subscribes the drawer to the shared mutation-state controller. All render
 * code must read `snapshot.*` — the controller object itself is stable, so
 * React Compiler would otherwise freeze `controller.get*()` results at their
 * first-computed values (the Phase 2 upstream-drawer failure mode).
 */
export function useAccessKeyDrawerController(options: AccessKeyDrawerControllerOptions) {
  const { apiClient, queryClient } = useAppServices()

  const [controller] = useState(() =>
    createAccessKeyDrawerController({
      client: apiClient,
      queryClient,
      generateOperationID: createUUID,
    }),
  )

  // Per-render runtime push (no deps array): the callbacks close over current
  // props/state, so the controller always sees the freshest isOpen/saved.
  useEffect(() => {
    controller.setRuntime({
      isOpen: () => options.isOpen(),
      onSaved: (kind, name) => options.onSaved(kind, name),
      onDeleted: (name) => options.onDeleted(name),
      setCreateOperation: (operation) => options.onCreateOperation(operation),
      setEditOperation: (operation) => options.onEditOperation(operation),
    })
  })

  useEffect(() => () => controller.dispose(), [controller])

  const subscribe = useCallback(
    (listener: () => void) => controller.subscribe(listener),
    [controller],
  )
  const snapshot = useSyncExternalStore(subscribe, () => controller.getSnapshot())

  return { controller, snapshot }
}

export type { AccessKeyDrawerSnapshot }
