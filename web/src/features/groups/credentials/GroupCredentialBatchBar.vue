<script setup lang="ts">
import { ListChecks, RefreshCw, Trash2 } from '@lucide/vue'
import { useI18n } from 'vue-i18n'
import AppButton from '@/components/ui/AppButton.vue'

defineProps<{
  selectedCount: number
  allVisibleSelected: boolean
  canSelectAll: boolean
  pending?: boolean
  canSync?: boolean
  canDownload?: boolean
}>()
const emit = defineEmits<{ 'toggle-select': []; sync: []; download: []; remove: [] }>()
const { n, t } = useI18n()
</script>
<template>
  <div class="group-credential-batch">
    <AppButton
      variant="secondary"
      size="compact"
      :busy="pending"
      :disabled="!canSelectAll"
      @click="emit('toggle-select')"
      ><ListChecks :size="15" aria-hidden="true" />{{
        allVisibleSelected
          ? t('group.credentials.batch.clearAll')
          : t('group.credentials.batch.selectAll')
      }}<span v-if="selectedCount > 0">{{ n(selectedCount) }}</span></AppButton
    >
    <AppButton
      v-if="canSync"
      variant="secondary"
      size="compact"
      :busy="pending"
      :disabled="selectedCount === 0"
      @click="emit('sync')"
      ><RefreshCw :size="15" aria-hidden="true" />{{ t('group.credentials.batch.sync') }}</AppButton
    >
    <AppButton
      v-if="canDownload"
      variant="secondary"
      size="compact"
      :busy="pending"
      :disabled="selectedCount === 0"
      @click="emit('download')"
      >{{ t('group.credentials.batch.download') }}</AppButton
    >
    <AppButton
      variant="secondary"
      tone="danger"
      size="compact"
      :busy="pending"
      :disabled="selectedCount === 0"
      @click="emit('remove')"
      ><Trash2 :size="15" aria-hidden="true" />{{ t('group.credentials.batch.delete') }}</AppButton
    >
  </div>
</template>
<style scoped>
.group-credential-batch {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}
</style>
