<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import AppButton from '@/components/ui/AppButton.vue'
import AppDialog from '@/components/ui/AppDialog.vue'

// A batch that mixes enabled and disabled groups must ask before spending an
// upstream call on a group that is not serving traffic. Dismissing the dialog
// aborts the probe; the two explicit choices keep the scope unambiguous.
const props = defineProps<{
  open: boolean
  total: number
  disabledCount: number
}>()
const emit = defineEmits<{
  'update:open': [open: boolean]
  'probe-all': []
  'probe-enabled': []
}>()
const { t } = useI18n()

const enabledCount = computed(() => props.total - props.disabledCount)

function setOpen(open: boolean): void {
  emit('update:open', open)
}
</script>

<template>
  <AppDialog
    appearance="ledger"
    :open="open"
    :title="t('monitor.modelProbe.disabledScope.title')"
    :description="
      t('monitor.modelProbe.disabledScope.description', { total, disabled: disabledCount })
    "
    :close-label="t('monitor.modelProbe.close')"
    @update:open="setOpen"
  >
    <template #footer>
      <AppButton variant="secondary" size="compact" @click="setOpen(false)">
        {{ t('monitor.modelProbe.disabledScope.cancel') }}
      </AppButton>
      <AppButton
        variant="secondary"
        size="compact"
        :disabled="enabledCount === 0"
        @click="emit('probe-enabled')"
      >
        {{ t('monitor.modelProbe.disabledScope.enabledOnly', { count: enabledCount }) }}
      </AppButton>
      <AppButton variant="primary" size="compact" @click="emit('probe-all')">
        {{ t('monitor.modelProbe.disabledScope.includeDisabled', { count: total }) }}
      </AppButton>
    </template>
  </AppDialog>
</template>
