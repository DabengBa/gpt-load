import { useCallback, useEffect, useState, useSyncExternalStore } from 'react'

import type { ModelPriceDto } from '@shared/control/resources/model-prices'
import {
  createModelPriceEditorController,
  type ModelPriceEditorController,
  type ModelPriceEditorSnapshot,
} from '@shared/controllers/model-price-editor'

import { useT } from './i18n'
import { useAppServices } from './services'
import { useUnsavedChanges, type UnsavedChangesGuard } from './use-unsaved-changes'

export interface ModelPriceEditorHandle {
  controller: ModelPriceEditorController
  /**
   * Memoized controller state. Always read render-time values from here —
   * React Compiler can cache `controller.*()` calls on the stable reference
   * and serve stale data; the snapshot object changes identity per notify so
   * downstream reads stay fresh even inside a transition-mounted drawer.
   */
  snapshot: ModelPriceEditorSnapshot
  confirmDiscardSwitch: UnsavedChangesGuard['confirmDiscard']
  dialog: UnsavedChangesGuard['dialog']
}

export function useModelPriceEditor(row: ModelPriceDto): ModelPriceEditorHandle {
  const { apiClient, queryClient } = useAppServices()
  const t = useT()

  const [controller] = useState(() =>
    createModelPriceEditorController(
      {
        client: apiClient,
        queryClient,
        failureMessage: () => t('modelPrices.matrix.saveFailed'),
      },
      row,
    ),
  )

  // Sync-store subscription: draft updates must commit even while a router
  // transition lane is still open (the drawer mounts inside the
  // `?selected_price_id` navigation) — a plain setState bump can be swallowed
  // there, same trap as the unsaved-changes dialog.
  const snapshot = useSyncExternalStore(
    useCallback((listener) => controller.subscribe(listener), [controller]),
    () => controller.getSnapshot(),
  )

  // Row-watch parity: the controller decides whether the refreshed row can
  // replace the draft baseline (id change always resets; updated_at_ms only
  // resets a clean draft). Reconciled during render — an effect here can be
  // swallowed while the drawer mounts inside a still-open navigation
  // transition, leaving the committed DOM stuck on the placeholder row.
  const [reconciledRow, setReconciledRow] = useState(row)
  if (reconciledRow !== row) {
    controller.setRow(row)
    setReconciledRow(row)
  }

  useEffect(() => () => controller.dispose(), [controller])

  const unsaved = useUnsavedChanges({
    dirty: snapshot.changed,
    blocked: snapshot.pending,
  })

  return {
    controller,
    snapshot,
    confirmDiscardSwitch: unsaved.confirmDiscard,
    dialog: unsaved.dialog,
  }
}
