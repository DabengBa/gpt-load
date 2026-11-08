/**
 * Imperative contract between GroupDetailView's unified save bar and the
 * settings/models editor tabs — the React counterpart of the classic
 * `ref="settingsEditor"`/`ref="modelsEditor"` + `defineExpose` pairing.
 * Both tabs accept `ref` as a regular prop (React 19) and publish this via
 * useImperativeHandle.
 */
import type {
  GroupModelUpdateDto,
  GroupModelsDto,
  GroupSettingsUpdateRequest,
} from '@shared/control/resources/groups'
import type { GroupSettingsDto } from '@shared/control/types'

export interface GroupEditorHandle {
  requestSave(): void
  discard(): void
}

export interface GroupSettingsEditorHandle extends GroupEditorHandle {
  prepareCombined(): GroupSettingsUpdateRequest | undefined
  acceptCombined(result: GroupSettingsDto): Promise<void>
  failCombined(cause: unknown): void
  recoverCombined(): Promise<void>
}

export interface GroupModelsEditorHandle extends GroupEditorHandle {
  focusFirstInvalid(): Promise<void>
  prepareCombined(): GroupModelUpdateDto[] | undefined
  acceptCombined(result: GroupModelsDto): Promise<void>
  recoverCombined(): Promise<void>
  confirmCombinedEmpty(action: () => void): void
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
  channelId?: string
}
