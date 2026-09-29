/**
 * Imperative contract between GroupDetailView's unified save bar and the
 * settings/models editor tabs — the React counterpart of the classic
 * `ref="settingsEditor"`/`ref="modelsEditor"` + `defineExpose` pairing.
 * Both tabs accept `ref` as a regular prop (React 19) and publish this via
 * useImperativeHandle.
 */
export interface GroupEditorHandle {
  requestSave(): void
  discard(): void
}

export interface GroupModelsEditorHandle extends GroupEditorHandle {
  focusFirstInvalid(): Promise<void>
}

/** Aggregated editor state each tab reports upward via `onStateChange` —
 *  same shape as the classic `@state` emit payload. */
export interface GroupEditorState {
  dirty: boolean
  pending: boolean
  error: string
  saved: boolean
  invalid?: boolean
  invalidRowCount?: number
}
