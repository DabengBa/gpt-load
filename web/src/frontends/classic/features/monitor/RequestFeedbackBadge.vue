<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { RequestLogFeedbackDto } from '@/app/resources/request-logs'
import AppTooltip from '@/components/ui/AppTooltip.vue'
import StatusBadge from '@/components/ui/StatusBadge.vue'
import type { StatusTone } from '@/components/ui/status-presenter'

import { formatLogDuration, formatProviderOutputRate } from './log-format'

const props = defineProps<{
  feedback: RequestLogFeedbackDto
  testId: string
}>()

const { locale, t } = useI18n()

// 反馈只是性能判定，与请求成功/失败、失败分类各自独立，互不覆盖。
const status = computed(() => props.feedback.feedback_status ?? 'unassessed')
const tone = computed<StatusTone>(() => {
  switch (status.value) {
    case 'normal':
      return 'success'
    case 'slow':
      return 'warning'
    case 'faulty':
      return 'danger'
    default:
      return 'neutral'
  }
})
const label = computed(() => t(`monitor.logs.feedback.status.${status.value}`))

const measurements = computed(() =>
  [
    props.feedback.provider_first_response_ms === null
      ? ''
      : t('monitor.logs.feedback.providerFirstResponseValue', {
          value: formatLogDuration(props.feedback.provider_first_response_ms),
        }),
    props.feedback.provider_tokens_per_second === null
      ? ''
      : t('monitor.logs.feedback.outputRateValue', {
          value: formatProviderOutputRate(props.feedback.provider_tokens_per_second, locale.value),
        }),
  ].filter(Boolean),
)
const reason = computed(() => {
  const value = props.feedback.feedback_reason
  if (value === null) return ''
  return `${t('monitor.logs.feedback.reasonLabel')}: ${t(`monitor.logs.feedback.reason.${value}`)}`
})
const description = computed(() =>
  [`${t(`monitor.logs.feedback.status.${status.value}`)}`, ...measurements.value, reason.value]
    .filter(Boolean)
    .join('\n'),
)
</script>

<template>
  <AppTooltip :content="description">
    <span
      class="request-feedback"
      :data-feedback-status="status"
      :data-testid="testId"
      tabindex="0"
      :aria-label="description"
    >
      <StatusBadge :tone="tone" size="compact">{{ label }}</StatusBadge>
    </span>
  </AppTooltip>
</template>

<style scoped>
.request-feedback {
  display: inline-flex;
  min-width: 0;
}
</style>
