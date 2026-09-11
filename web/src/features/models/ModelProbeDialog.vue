<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { ModelProbeResultDto } from '@/app/resources/model-probe'
import AppButton from '@/components/ui/AppButton.vue'
import AppDialog from '@/components/ui/AppDialog.vue'
import CopyChip from '@/components/ui/CopyChip.vue'
import InlineFeedback from '@/components/ui/InlineFeedback.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'
import { formatLocalInstant } from '@/lib/format'

const props = defineProps<{
  open: boolean
  pending: boolean
  failed: boolean
  stopped: boolean
  results: readonly ModelProbeResultDto[]
  disabledGroupIds: readonly number[]
  completed: number
  total: number
}>()
const emit = defineEmits<{
  'update:open': [open: boolean]
  stop: []
  'view-log': [logId: string]
}>()
const { locale, n, t } = useI18n()

const detail = computed(() => (props.results.length === 1 ? props.results[0] : undefined))
const summary = computed(() => {
  const value = { passed: 0, failed: 0, inconclusive: 0 }
  for (const result of props.results) value[result.outcome] += 1
  return value
})

function setOpen(open: boolean): void {
  if (!open && props.pending) return
  emit('update:open', open)
}

function tone(result: ModelProbeResultDto): 'success' | 'danger' | 'warning' {
  if (result.outcome === 'passed') return 'success'
  if (result.outcome === 'failed') return 'danger'
  return 'warning'
}

function outcomeLabel(result: ModelProbeResultDto): string {
  return t(`monitor.modelProbe.outcome.${result.outcome}`)
}

function reasonLabel(result: ModelProbeResultDto): string {
  return result.reason === null ? '' : t(`monitor.modelProbe.reason.${result.reason}`)
}

function routeModeLabel(result: ModelProbeResultDto): string {
  return result.route_mode === null
    ? t('monitor.modelProbe.unknownValue')
    : t(`monitor.modelProbe.routeMode.${result.route_mode}`)
}

function groupLabel(result: ModelProbeResultDto): string {
  return result.group_name === '' ? `#${result.group_id}` : result.group_name
}

const disabledGroupIdSet = computed(() => new Set(props.disabledGroupIds))

function isDisabledGroup(result: ModelProbeResultDto): boolean {
  return disabledGroupIdSet.value.has(result.group_id)
}
</script>

<template>
  <AppDialog
    appearance="ledger"
    :open="open"
    :title="t('monitor.modelProbe.title')"
    :description="t('monitor.modelProbe.description')"
    :close-label="t('monitor.modelProbe.close')"
    :dismissible="!pending"
    @update:open="setOpen"
  >
    <template #body>
      <div class="model-probe-dialog">
        <QueryFeedback
          v-if="pending && results.length === 0"
          state="loading"
          :message="t('monitor.modelProbe.loading')"
        />
        <InlineFeedback v-if="failed" tone="danger" appearance="ledger">
          {{ t('monitor.modelProbe.requestFailed') }}
        </InlineFeedback>
        <InlineFeedback v-if="stopped" tone="warning" appearance="ledger">
          {{ t('monitor.modelProbe.stopped') }}
        </InlineFeedback>
        <InlineFeedback v-if="total > 1" tone="warning" appearance="ledger">
          {{ t('monitor.modelProbe.progress', { completed, total }) }}
        </InlineFeedback>
        <InlineFeedback v-if="total > 1 && results.length > 0" tone="warning" appearance="ledger">
          {{
            t('monitor.modelProbe.summary', {
              passed: summary.passed,
              failed: summary.failed,
              inconclusive: summary.inconclusive,
            })
          }}
        </InlineFeedback>

        <dl v-if="detail" class="model-probe-dialog__details">
          <dt>{{ t('monitor.modelProbe.fields.group') }}</dt>
          <dd>
            {{ groupLabel(detail) }}
            <span v-if="isDisabledGroup(detail)" class="model-probe-dialog__disabled">
              {{ t('monitor.modelProbe.disabledBadge') }}
            </span>
          </dd>
          <dt>{{ t('monitor.modelProbe.fields.model') }}</dt>
          <dd>{{ detail.model }}</dd>
          <dt>{{ t('monitor.modelProbe.fields.outcome') }}</dt>
          <dd>{{ outcomeLabel(detail) }}</dd>
          <dt v-if="detail.reason">{{ t('monitor.modelProbe.fields.reason') }}</dt>
          <dd v-if="detail.reason">{{ reasonLabel(detail) }}</dd>
          <dt v-if="detail.protocol">{{ t('monitor.modelProbe.fields.protocol') }}</dt>
          <dd v-if="detail.protocol">{{ detail.protocol }}</dd>
          <dt>{{ t('monitor.modelProbe.fields.routeMode') }}</dt>
          <dd>{{ routeModeLabel(detail) }}</dd>
          <dt v-if="detail.status_code !== null">
            {{ t('monitor.modelProbe.fields.statusCode') }}
          </dt>
          <dd v-if="detail.status_code !== null">{{ detail.status_code }}</dd>
          <dt v-if="detail.latency_ms !== null">{{ t('monitor.modelProbe.fields.latency') }}</dt>
          <dd v-if="detail.latency_ms !== null">
            {{ t('monitor.modelProbe.latency', { value: n(detail.latency_ms) }) }}
          </dd>
          <dt v-if="detail.credential_label">{{ t('monitor.modelProbe.fields.credential') }}</dt>
          <dd v-if="detail.credential_label">{{ detail.credential_label }}</dd>
          <dt>{{ t('monitor.modelProbe.fields.logId') }}</dt>
          <dd>
            <CopyChip
              v-if="detail.log_id"
              :value="detail.log_id"
              :label="t('monitor.modelProbe.fields.logId')"
              :success-label="t('common.copied')"
              :failure-label="t('common.copyFailed')"
            />
            <AppButton
              v-if="detail.log_id"
              variant="link"
              size="inline"
              @click="emit('view-log', detail.log_id)"
            >
              {{ t('monitor.modelProbe.viewLog') }}
            </AppButton>
            <template v-if="!detail.log_id">{{ t('monitor.modelProbe.notExecuted') }}</template>
          </dd>
          <dt>{{ t('monitor.modelProbe.fields.testedAt') }}</dt>
          <dd>{{ formatLocalInstant(detail.tested_at_ms, locale) }}</dd>
        </dl>

        <ul v-else-if="results.length > 1" class="model-probe-dialog__list">
          <li v-for="result in results" :key="`${result.group_id}:${result.model}`">
            <div class="model-probe-dialog__row">
              <span class="model-probe-dialog__identity">
                {{ groupLabel(result) }} · {{ result.model }}
                <span v-if="isDisabledGroup(result)" class="model-probe-dialog__disabled">
                  {{ t('monitor.modelProbe.disabledBadge') }}
                </span>
              </span>
              <span class="model-probe-dialog__outcome" :class="`tone-${tone(result)}`">
                {{ outcomeLabel(result) }}
              </span>
            </div>
            <div class="model-probe-dialog__meta">
              <span v-if="result.reason">{{ reasonLabel(result) }}</span>
              <span v-if="result.credential_label">
                {{ t('monitor.modelProbe.credential', { credential: result.credential_label }) }}
              </span>
              <template v-if="result.log_id">
                <CopyChip
                  :value="result.log_id"
                  layout="trailing"
                  :label="t('monitor.modelProbe.fields.logId')"
                  :success-label="t('common.copied')"
                  :failure-label="t('common.copyFailed')"
                />
                <AppButton
                  variant="link"
                  size="inline"
                  @click="emit('view-log', result.log_id as string)"
                >
                  {{ t('monitor.modelProbe.viewLog') }}
                </AppButton>
              </template>
            </div>
          </li>
        </ul>
      </div>
    </template>

    <template #footer>
      <AppButton v-if="pending" variant="secondary" size="compact" @click="emit('stop')">
        {{ t('monitor.modelProbe.stop') }}
      </AppButton>
      <AppButton variant="secondary" size="compact" :disabled="pending" @click="setOpen(false)">
        {{ t('monitor.modelProbe.close') }}
      </AppButton>
    </template>
  </AppDialog>
</template>

<style scoped>
.model-probe-dialog {
  display: grid;
  gap: var(--space-3);
}

.model-probe-dialog__details {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  gap: 8px var(--space-3);
  margin: 0;
  font-size: var(--text-sm);
  line-height: var(--line-normal);
}

.model-probe-dialog__details dt {
  color: var(--color-text-muted);
}

.model-probe-dialog__details dd {
  display: flex;
  min-width: 0;
  margin: 0;
  align-items: center;
  gap: var(--space-2);
  color: var(--color-text);
  font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}

.model-probe-dialog__list {
  display: grid;
  max-height: 60vh;
  gap: var(--space-3);
  margin: 0;
  padding: 0;
  overflow-y: auto;
  list-style: none;
}

.model-probe-dialog__row {
  display: flex;
  min-width: 0;
  align-items: baseline;
  justify-content: space-between;
  gap: var(--space-2);
}

.model-probe-dialog__identity {
  min-width: 0;
  color: var(--color-text);
  font-weight: 600;
  overflow-wrap: anywhere;
}

.model-probe-dialog__outcome {
  flex: none;
  font-size: var(--text-sm);
}

.model-probe-dialog__outcome.tone-success {
  color: var(--color-text-success, var(--color-text));
}

.model-probe-dialog__outcome.tone-danger {
  color: var(--color-text-danger, var(--color-text));
}

.model-probe-dialog__outcome.tone-warning {
  color: var(--color-text-warning, var(--color-text));
}

.model-probe-dialog__disabled {
  margin-left: var(--space-2);
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-control);
  padding: 0 var(--space-1);
  color: var(--color-text-muted);
  font-size: var(--text-sm);
  font-weight: 500;
  white-space: nowrap;
}

.model-probe-dialog__meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--space-2);
  color: var(--color-text-muted);
  font-size: var(--text-sm);
}

@media (max-width: 480px) {
  .model-probe-dialog__details {
    grid-template-columns: 1fr;
    gap: var(--space-1);
  }

  .model-probe-dialog__details dd + dt {
    margin-top: var(--space-2);
  }
}
</style>
