<script setup lang="ts">
import LedgerSheet from '@/components/layout/LedgerSheet.vue'
import PageFrame from '@/components/layout/PageFrame.vue'
import PageHeader from '@/components/ui/PageHeader.vue'
import { useI18n } from 'vue-i18n'

import LogsTab from '@/features/monitor/LogsTab.vue'

const { t } = useI18n()
</script>

<template>
  <PageFrame wide aria-labelledby="logs-title">
    <LedgerSheet class="logs-page">
      <PageHeader id="logs-title" :title="t('shell.logs')" />
      <LogsTab />
    </LedgerSheet>
  </PageFrame>
</template>

<style scoped>
.logs-page {
  display: grid;
  min-height: 760px;
  min-width: 0;
  align-content: start;
  gap: 0;
}

@media (max-width: 800px) {
  .logs-page {
    min-height: 0;
  }
}
</style>
