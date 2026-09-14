<script setup lang="ts">
import { useQueryClient } from '@tanstack/vue-query'
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import type { CredentialItemDto } from '@/api/control/types'
import { useApiClient } from '@/api/client-context'
import {
  cacheCredentialItem,
  revealCredential,
  updateCredential,
} from '@/app/resources/credentials'
import { invalidateGroupSettingsDependents } from '@/app/resources/groups'
import { useAbortControllerPool } from '@/app/use-abort-controller-pool'
import AppButton from '@/components/ui/AppButton.vue'
import CopyChip from '@/components/ui/CopyChip.vue'
import InlineFeedback from '@/components/ui/InlineFeedback.vue'

const props = withDefaults(
  defineProps<{
    groupId: number
    credential: CredentialItemDto
    disabled?: boolean
  }>(),
  { disabled: false },
)
const { t } = useI18n()
const client = useApiClient()
const queryClient = useQueryClient()
const copyControllers = useAbortControllerPool()
const editing = ref(false)
const nextCredential = ref('')
const pending = ref(false)
const error = ref('')
const saved = ref(false)
let updateController: AbortController | undefined

const isAPIKey = computed(() => props.credential.connection_type === 'api_key')
const inputLabel = computed(() => t('group.credentials.update.inputLabel'))

async function resolveCopyValue(): Promise<string> {
  const controller = copyControllers.create()
  try {
    const result = await revealCredential(
      client,
      props.groupId,
      props.credential.credential_id,
      controller.signal,
    )
    const values = Object.values(result.credential)
    return values.length === 1 ? values[0] : JSON.stringify(result.credential)
  } finally {
    copyControllers.release(controller)
  }
}

function startEditing(): void {
  if (props.disabled || !isAPIKey.value) return
  error.value = ''
  saved.value = false
  editing.value = true
}

function stopEditing(): void {
  if (pending.value) return
  editing.value = false
  nextCredential.value = ''
  error.value = ''
}

async function submit(): Promise<void> {
  const value = nextCredential.value.trim()
  if (pending.value || props.disabled || !isAPIKey.value) return
  if (!value) {
    error.value = t('group.credentials.update.required')
    saved.value = false
    return
  }

  const controller = new AbortController()
  updateController = controller
  pending.value = true
  error.value = ''
  saved.value = false
  try {
    const result = await updateCredential(
      client,
      props.groupId,
      props.credential.credential_id,
      { credentials: value },
      controller.signal,
    )
    if (updateController !== controller) return
    nextCredential.value = ''
    try {
      await cacheCredentialItem(queryClient, props.groupId, result)
    } catch {
      error.value = t('group.credentials.reconcileFailed')
      return
    }
    saved.value = true
    try {
      await invalidateGroupSettingsDependents(queryClient, props.groupId)
    } catch {
      error.value = t('group.credentials.reconcileFailed')
    }
  } catch {
    if (controller.signal.aborted || updateController !== controller) return
    error.value = t('group.credentials.updateFailed')
  } finally {
    if (updateController === controller) {
      updateController = undefined
      pending.value = false
    }
  }
}

watch(
  () => [props.groupId, props.credential.secret_version],
  () => {
    updateController?.abort()
    updateController = undefined
    pending.value = false
    copyControllers.abortAll()
    editing.value = false
    nextCredential.value = ''
    saved.value = false
  },
)

onBeforeUnmount(() => {
  updateController?.abort()
  copyControllers.abortAll()
})
</script>

<template>
  <div v-if="isAPIKey" class="group-api-key-editor">
    <div class="group-api-key-editor__summary">
      <CopyChip
        :key="credential.secret_version"
        :value="credential.mask"
        :label="t('group.credentials.copy')"
        :success-label="t('common.copied')"
        :failure-label="t('common.copyFailed')"
        :resolve-value="resolveCopyValue"
      />
      <AppButton
        variant="ghost"
        size="compact"
        :disabled="disabled || pending"
        :aria-expanded="editing"
        @click="editing ? stopEditing() : startEditing()"
      >
        {{ editing ? t('group.credentials.update.cancel') : t('group.credentials.update.action') }}
      </AppButton>
    </div>
    <form v-if="editing" class="group-api-key-editor__form" @submit.prevent="submit">
      <label class="group-api-key-editor__field">
        <span>{{ inputLabel }}</span>
        <input
          v-model="nextCredential"
          type="password"
          :placeholder="t('group.credentials.update.placeholder')"
          :disabled="pending"
          :aria-invalid="error ? 'true' : undefined"
          autocomplete="new-password"
          spellcheck="false"
        />
      </label>
      <AppButton type="submit" size="compact" :busy="pending" :disabled="!nextCredential.trim()">
        {{ pending ? t('group.credentials.update.saving') : t('group.credentials.update.submit') }}
      </AppButton>
    </form>
    <InlineFeedback v-if="error" tone="danger" appearance="ledger">{{ error }}</InlineFeedback>
    <InlineFeedback v-else-if="saved" tone="success" appearance="ledger">
      {{ t('group.credentials.update.succeeded') }}
    </InlineFeedback>
  </div>
</template>

<style scoped>
.group-api-key-editor {
  display: grid;
  min-width: 0;
  gap: 7px;
}
.group-api-key-editor__summary {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}
.group-api-key-editor__form {
  display: flex;
  min-width: 0;
  flex-wrap: wrap;
  align-items: end;
  gap: 8px;
  border-top: 1px solid var(--color-border-subtle);
  padding-top: 8px;
}
.group-api-key-editor__field {
  display: grid;
  min-width: min(260px, 100%);
  flex: 1 1 280px;
  gap: 4px;
}
.group-api-key-editor__field > span {
  color: var(--color-text-muted);
  font-size: var(--text-label-xs);
}
.group-api-key-editor__field input {
  width: 100%;
  min-height: var(--control-sm);
  border: 1px solid var(--color-border-control);
  border-radius: var(--radius-control);
  background: var(--color-surface);
  color: var(--color-text);
  padding: 0 10px;
  font: inherit;
  font-family: var(--font-mono);
}
.group-api-key-editor__form :deep(.app-button) {
  flex: none;
}
.group-api-key-editor :deep(.inline-feedback) {
  overflow-wrap: anywhere;
}
@media (max-width: 520px) {
  .group-api-key-editor__form {
    align-items: stretch;
  }
  .group-api-key-editor__form :deep(.app-button) {
    width: 100%;
  }
}
</style>
