<script setup lang="ts">
import { RefreshCw } from '@lucide/vue'

import type { ModelRouteScheduleIndexItemDto } from '@/app/resources/model-route-schedule'
import AppButton from '@/components/ui/AppButton.vue'
import QueryFeedback from '@/components/ui/QueryFeedback.vue'

export interface SchedulePanelIndexLabels {
  title: string
  refresh: string
  loading: string
  failed: string
  empty: string
  candidates: string
  groups: string
  fallback: string
  cooldown: string
  blacklist: string
  select: string
  eyebrow?: string
  routeExceptions?: string
}

const props = withDefaults(
  defineProps<{
    items: ModelRouteScheduleIndexItemDto[]
    selectedModel?: string
    loading: boolean
    refreshing?: boolean
    error?: string
    labels?: Partial<SchedulePanelIndexLabels>
  }>(),
  {
    selectedModel: '',
    refreshing: false,
    error: '',
    labels: () => ({}),
  },
)
const emit = defineEmits<{
  selectModel: [model: string]
  refresh: []
}>()

const text = (key: keyof SchedulePanelIndexLabels): string => props.labels[key] ?? ''
</script>

<template>
  <section class="schedule-index" aria-labelledby="schedule-index-title">
    <header class="schedule-index__header">
      <div>
        <p class="schedule-index__eyebrow">{{ text('eyebrow') }}</p>
        <h2 id="schedule-index-title">{{ text('title') }}</h2>
      </div>
      <AppButton
        variant="secondary"
        size="compact"
        :busy="refreshing"
        :aria-label="text('refresh')"
        @click="emit('refresh')"
      >
        <RefreshCw :size="14" aria-hidden="true" />
        {{ text('refresh') }}
      </AppButton>
    </header>

    <QueryFeedback v-if="loading" state="loading" :message="text('loading')" />
    <QueryFeedback
      v-else-if="error"
      state="error"
      :message="error"
      :retry-label="text('refresh')"
      @retry="emit('refresh')"
    />
    <div v-else-if="items.length === 0" class="schedule-index__empty" role="status">
      {{ text('empty') }}
    </div>
    <ul v-else class="schedule-index__list" :aria-label="text('title')">
      <li v-for="item in items" :key="item.external_model">
        <button
          type="button"
          class="schedule-index__item"
          :class="{ 'schedule-index__item--selected': selectedModel === item.external_model }"
          :aria-pressed="selectedModel === item.external_model"
          @click="emit('selectModel', item.external_model)"
        >
          <span class="schedule-index__model">{{ item.external_model }}</span>
          <span class="schedule-index__stats">
            <span>{{ item.candidate_count }} {{ text('candidates') }}</span>
            <span>{{ item.group_count }} {{ text('groups') }}</span>
            <span v-if="item.has_fallback" class="schedule-index__fallback">
              {{ text('fallback') }}
            </span>
          </span>
          <span class="schedule-index__alerts" :aria-label="text('routeExceptions')">
            <span
              v-if="item.cooled_candidates"
              class="schedule-index__alert schedule-index__alert--cooldown"
            >
              {{ item.cooled_candidates }} {{ text('cooldown') }}
            </span>
            <span
              v-if="item.blacklisted_candidates"
              class="schedule-index__alert schedule-index__alert--blacklist"
            >
              {{ item.blacklisted_candidates }} {{ text('blacklist') }}
            </span>
          </span>
          <span class="schedule-index__select">{{ text('select') }}</span>
        </button>
      </li>
    </ul>
  </section>
</template>

<style scoped>
.schedule-index {
  display: grid;
  min-width: 0;
  gap: var(--space-3);
}
.schedule-index__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
}
.schedule-index__eyebrow {
  margin: 0 0 4px;
  color: var(--color-text-faint);
  font-family: var(--font-mono);
  font-size: 10px;
  letter-spacing: 0.1em;
}
.schedule-index h2 {
  margin: 0;
  color: var(--color-text);
  font-size: var(--text-lg);
}
.schedule-index__list {
  display: grid;
  min-width: 0;
  gap: 1px;
  overflow: hidden;
  margin: 0;
  border: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-card);
  background: var(--color-border-subtle);
  padding: 0;
  list-style: none;
}
.schedule-index__item {
  display: grid;
  grid-template-columns: minmax(140px, 1.5fr) minmax(180px, 2fr) minmax(120px, 1fr) auto;
  min-width: 0;
  align-items: center;
  gap: var(--space-3);
  border: 0;
  background: var(--color-surface);
  color: var(--color-text);
  padding: 14px 16px;
  text-align: left;
  cursor: pointer;
  transition: background-color var(--duration-fast) var(--easing-standard);
}
.schedule-index__item:hover,
.schedule-index__item--selected {
  background: var(--color-action-soft);
}
.schedule-index__model {
  min-width: 0;
  overflow: hidden;
  font-family: var(--font-mono);
  font-size: var(--text-sm);
  font-weight: 650;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.schedule-index__stats,
.schedule-index__alerts {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 12px;
  color: var(--color-text-muted);
  font-size: var(--text-meta);
}
.schedule-index__fallback {
  color: var(--color-action);
}
.schedule-index__alerts {
  justify-content: flex-start;
}
.schedule-index__alert {
  border-radius: 999px;
  padding: 2px 7px;
  font-size: 10px;
  white-space: nowrap;
}
.schedule-index__alert--cooldown {
  background: var(--color-warning-bg);
  color: var(--color-warning);
}
.schedule-index__alert--blacklist {
  background: var(--color-danger-bg);
  color: var(--color-danger);
}
.schedule-index__select {
  color: var(--color-action);
  font-size: var(--text-meta);
  font-weight: 650;
}
.schedule-index__empty {
  border: 1px dashed var(--color-border-control);
  border-radius: var(--radius-card);
  color: var(--color-text-muted);
  padding: 32px 20px;
  text-align: center;
}
@media (max-width: 760px) {
  .schedule-index__item {
    grid-template-columns: minmax(0, 1fr) auto;
  }
  .schedule-index__stats,
  .schedule-index__alerts {
    grid-column: 1 / -1;
  }
  .schedule-index__select {
    grid-column: 2;
    grid-row: 1;
  }
}
</style>
