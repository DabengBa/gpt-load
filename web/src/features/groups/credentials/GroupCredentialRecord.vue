<script setup lang="ts">
import { Activity, ChevronDown, Ellipsis, RotateCcw, Trash2 } from '@lucide/vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import type { CredentialItemDto } from '@/api/control/types'
import AppPopover from '@/components/ui/AppPopover.vue'
import CopyChip from '@/components/ui/CopyChip.vue'
import IconButton from '@/components/ui/IconButton.vue'
import StatusBadge from '@/components/ui/StatusBadge.vue'
import { formatLocalInstant } from '@/lib/format'
import { presentCredentialFailureCategory } from './credential-failure-presenter'

const props = defineProps<{
  item: CredentialItemDto
  rowIndex: number
  selected: boolean
  busy: boolean
  expanded: boolean
  resolveCopyValue: (id: number) => Promise<string>
}>()
const emit = defineEmits<{
  'update:selected': [selected: boolean]
  'update:expanded': [expanded: boolean]
  test: [item: CredentialItemDto]
  restore: [item: CredentialItemDto]
  remove: [item: CredentialItemDto]
}>()
const { locale, n, t } = useI18n()
const menuOpen = ref(false)
const detailId = computed(() => `group-credential-details-${props.item.credential_id}`)
const isProblem = computed(
  () => props.item.effective_status === 'cooldown' || props.item.effective_status === 'blacklisted',
)
const recentLabel = computed(() =>
  props.item.recent_failure_count === 0
    ? t('group.credentials.recentSuccessOnly', { success: n(props.item.recent_success_count) })
    : t('group.credentials.recent', {
        success: n(props.item.recent_success_count),
        failure: n(props.item.recent_failure_count),
      }),
)
function setSelected(event: Event): void {
  emit('update:selected', (event.target as HTMLInputElement).checked)
}

function testCredential(): void {
  menuOpen.value = false
  emit('test', props.item)
}

function restoreCredential(): void {
  menuOpen.value = false
  emit('restore', props.item)
}

function removeCredential(): void {
  menuOpen.value = false
  emit('remove', props.item)
}
</script>

<template>
  <article
    class="ledger-record-list__record group-credential-record"
    role="row"
    :aria-rowindex="rowIndex"
  >
    <div class="group-credential-record__summary" role="presentation">
      <div class="ledger-record-list__cell group-credential-record__select" role="cell">
        <label
          ><span class="sr-only">{{
            t('group.credentials.selectCredential', { mask: item.mask })
          }}</span
          ><input type="checkbox" :checked="selected" :disabled="busy" @change="setSelected"
        /></label>
      </div>
      <div class="ledger-record-list__cell group-credential-record__mask" role="cell">
        <span class="group-credential-record__mobile-label">{{
          t('group.credentials.columns.credential')
        }}</span>
        <CopyChip
          :value="item.mask"
          :label="t('group.credentials.copy')"
          :success-label="t('common.copied')"
          :failure-label="t('common.copyFailed')"
          :resolve-value="() => resolveCopyValue(item.credential_id)"
        />
      </div>
      <div class="ledger-record-list__cell" role="cell">
        <span class="group-credential-record__mobile-label">{{
          t('group.credentials.columns.status')
        }}</span>
        <StatusBadge :status="item.effective_status" size="compact">{{
          t(`group.credentials.effective.${item.effective_status}`)
        }}</StatusBadge>
      </div>
      <div class="ledger-record-list__cell group-credential-record__recent" role="cell">
        <span class="group-credential-record__mobile-label">{{
          t('group.credentials.columns.recent')
        }}</span
        >{{ recentLabel }}
      </div>
      <div class="ledger-record-list__cell group-credential-record__actions" role="cell">
        <AppPopover v-model:open="menuOpen" align="end">
          <template #trigger
            ><IconButton
              variant="ghost"
              size="compact"
              :label="t('group.credentials.moreActions')"
              :disabled="busy"
              ><Ellipsis :size="16" aria-hidden="true" /></IconButton
          ></template>
          <div class="group-credential-record__menu">
            <button type="button" :disabled="busy" @click="testCredential">
              <Activity :size="15" aria-hidden="true" />{{ t('group.credentials.test.action') }}
            </button>
            <button v-if="isProblem" type="button" :disabled="busy" @click="restoreCredential">
              <RotateCcw :size="15" aria-hidden="true" />{{ t('group.credentials.restore') }}
            </button>
            <button
              type="button"
              class="group-credential-record__menu-danger"
              :disabled="busy"
              @click="removeCredential"
            >
              <Trash2 :size="15" aria-hidden="true" />{{ t('group.credentials.delete') }}
            </button>
          </div>
        </AppPopover>
        <IconButton
          variant="ghost"
          size="compact"
          :label="expanded ? t('group.credentials.collapse') : t('group.credentials.expand')"
          :aria-expanded="expanded"
          :aria-controls="detailId"
          @click="emit('update:expanded', !expanded)"
          ><ChevronDown :size="16" aria-hidden="true"
        /></IconButton>
      </div>
    </div>
    <div v-if="expanded" :id="detailId" class="group-credential-record__details" role="cell">
      <dl>
        <div>
          <dt>{{ t('group.credentials.detailsFailure') }}</dt>
          <dd>{{ presentCredentialFailureCategory(t, item.last_failure_category) }}</dd>
        </div>
        <div>
          <dt>{{ t('group.credentials.detailsRecovery') }}</dt>
          <dd>
            {{
              item.recovery.at_ms
                ? formatLocalInstant(item.recovery.at_ms, locale)
                : t(`group.credentials.recovery.${item.recovery.mode}`)
            }}
          </dd>
        </div>
        <div>
          <dt>{{ t('group.credentials.detailsConsecutive') }}</dt>
          <dd>{{ n(item.consecutive_failure_count) }}</dd>
        </div>
      </dl>
    </div>
  </article>
</template>

<style scoped>
.group-credential-record {
  display: grid;
  grid-template-columns: 1fr;
  padding: 0;
}
.group-credential-record__summary {
  display: grid;
  grid-template-columns: 34px minmax(150px, 1.4fr) minmax(110px, 0.8fr) minmax(130px, 1fr) 90px;
  align-items: center;
  min-height: 56px;
  gap: 12px;
}
.group-credential-record__select {
  display: flex;
  justify-content: center;
}
.group-credential-record__select input {
  width: 16px;
  height: 16px;
  accent-color: var(--color-action);
}
.group-credential-record__mask,
.group-credential-record__recent {
  min-width: 0;
  font-family: var(--font-mono);
  font-variant-numeric: tabular-nums;
}
.group-credential-record__actions {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
}
.group-credential-record__menu {
  display: grid;
  min-width: 160px;
  gap: 2px;
}
.group-credential-record__menu button {
  display: flex;
  align-items: center;
  gap: 8px;
  border: 0;
  background: transparent;
  padding: 8px;
  text-align: left;
}
.group-credential-record__menu button:hover {
  background: var(--color-surface-sunken);
}
.group-credential-record__menu-danger {
  color: var(--color-danger);
}
.group-credential-record__details {
  margin: 0 14px 8px;
  border-radius: var(--radius-control);
  background: var(--color-surface-sunken);
  padding: 12px 14px;
}
.group-credential-record__details dl {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 16px;
  margin: 0;
}
.group-credential-record__details dt,
.group-credential-record__mobile-label {
  color: var(--color-text-muted);
  font-size: var(--text-meta);
}
.group-credential-record__details dd {
  margin: 3px 0 0;
}
.group-credential-record__mobile-label {
  display: none;
}
@media (max-width: 700px) {
  .group-credential-record__summary {
    grid-template-columns: 28px minmax(0, 1fr) 86px;
    padding: 8px 0;
  }
  .group-credential-record__summary > :nth-child(3),
  .group-credential-record__summary > :nth-child(4) {
    display: none;
  }
  .group-credential-record__mobile-label {
    display: block;
  }
  .group-credential-record__details dl {
    grid-template-columns: 1fr;
  }
}
</style>
