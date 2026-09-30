import * as stylex from '@stylexjs/stylex'
import { Banner, Button, Selector, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useState } from 'react'

import type { ModelProbeTargetDto } from '@shared/control/resources/model-probe'
import {
  modelRouteScheduleDetailQueryOptions,
  modelRouteScheduleIndexQueryOptions,
  type ModelRouteScheduleIndexItemDto,
} from '@shared/control/resources/model-route-schedule'
import { pagePath } from '@shared/routing/page-routes'
import type { ScheduleDrafts } from '@shared/routing/monitor-route'

import { useAppServices } from '../../app/services'
import { ModelProbeDialog } from '../models/ModelProbeDialog'
import { useModelProbe } from '../models/use-model-probe'
import { SchedulePanelDetail, type SchedulePanelDetailLabels } from './SchedulePanelDetail'

export interface SchedulePanelLabels {
  model: string
  selectModel: string
  loadingOptions: string
  contextRequired: string
  kicker?: string
  contextReady?: string
  context?: string
  retry?: string
  indexFailed?: string
  detailFailed?: string
  detail?: Partial<SchedulePanelDetailLabels>
}

const styles = stylex.create({
  panel: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-4)',
  },
  filters: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(150px, 0.5fr) minmax(220px, 1fr)',
      '@media (max-width: 620px)': '1fr',
    },
    gap: 'var(--space-3)',
    maxWidth: '620px',
  },
  field: {
    display: 'grid',
    minWidth: 0,
    gap: '5px',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontWeight: 650,
  },
})

export interface SchedulePanelProps {
  externalModel?: string
  selectedRow?: string
  sourceGroupId?: number
  drafts?: ScheduleDrafts
  labels?: Partial<SchedulePanelLabels>
  locale?: string
  onChangeContext(context: { externalModel?: string }): void
  onDraftChange(drafts: ScheduleDrafts): void
  onRowChange(row: string | undefined): void
  onRefresh(): void
  onSaved(snapshotRevision: number): void
  onRecovered(groupId: number, entryId: string): void
}

export function SchedulePanel({
  externalModel = '',
  selectedRow,
  sourceGroupId,
  drafts = {},
  labels = {},
  locale = 'en-US',
  onChangeContext,
  onDraftChange,
  onRowChange,
  onRefresh,
  onSaved,
  onRecovered,
}: SchedulePanelProps) {
  const { apiClient } = useAppServices()
  const navigate = useNavigate()

  const [selectedModel, setSelectedModel] = useState(externalModel)
  // Classic watch(props.externalModel): URL-driven context changes re-seed the
  // selector draft (render-time adjustment, not an effect).
  const [syncedModel, setSyncedModel] = useState(externalModel)
  if (syncedModel !== externalModel) {
    setSyncedModel(externalModel)
    setSelectedModel(externalModel)
  }

  const indexQuery = useQuery(modelRouteScheduleIndexQueryOptions(apiClient))
  const indexItems: ModelRouteScheduleIndexItemDto[] = indexQuery.data?.items ?? []
  const selectedIndexItem = indexItems.find((item) => item.external_model === selectedModel)
  const detailRequest = selectedIndexItem
    ? {
        protocol: selectedIndexItem.protocol,
        external_model: selectedModel,
        operation: selectedIndexItem.operation,
      }
    : undefined
  const detailQuery = useQuery(modelRouteScheduleDetailQueryOptions(apiClient, detailRequest))

  const modelOptions = [
    { value: '', label: labels.selectModel ?? '' },
    ...indexItems.map((item) => ({ value: item.external_model, label: item.external_model })),
  ]
  const indexError = indexQuery.error
    ? indexQuery.error instanceof Error && indexQuery.error.message
      ? indexQuery.error.message
      : (labels.indexFailed ?? '')
    : ''
  const detailError = detailQuery.error
    ? detailQuery.error instanceof Error && detailQuery.error.message
      ? detailQuery.error.message
      : (labels.detailFailed ?? '')
    : ''

  function setModel(value: string): void {
    setSelectedModel(value)
    onChangeContext({ externalModel: value || undefined })
  }

  async function refreshIndex(): Promise<void> {
    await indexQuery.refetch()
    onRefresh()
  }

  async function refreshDetail(): Promise<void> {
    if (detailRequest) await detailQuery.refetch()
    await indexQuery.refetch()
    onRefresh()
  }

  async function refreshAll(): Promise<void> {
    await Promise.all([indexQuery.refetch(), detailQuery.refetch()])
    onRefresh()
  }

  async function handleSaved(revision: number): Promise<void> {
    onSaved(revision)
    await refreshAll()
  }

  async function handleRecovered(groupID: number, entryID: string): Promise<void> {
    onRecovered(groupID, entryID)
    await refreshAll()
  }

  const probe = useModelProbe()

  function probeRow(groupID: number, modelID: string, disabled: boolean): void {
    void probe.start([{ group_id: groupID, model: modelID }], {
      disabledGroupIds: disabled ? [groupID] : [],
    })
  }

  function probeRows(targets: ModelProbeTargetDto[], disabledGroupIds: number[]): void {
    void probe.start(targets, { disabledGroupIds })
  }

  function viewProbeLog(logID: string): void {
    void navigate({
      to: pagePath('logs'),
      search: { selected_request_id: logID },
    })
  }

  return (
    <section
      {...stylex.props(styles.panel)}
      data-testid="schedule-panel"
      aria-label={labels.model ?? ''}
    >
      <div {...stylex.props(styles.filters)} aria-label={labels.context ?? ''}>
        <label {...stylex.props(styles.field)}>
          <span>{labels.model}</span>
          <Selector
            label={labels.model ?? ''}
            isLabelHidden
            size="sm"
            value={selectedModel}
            options={modelOptions}
            isDisabled={indexQuery.isPending || indexQuery.isError}
            onChange={setModel}
          />
        </label>
      </div>

      {indexQuery.isPending ? (
        <div role="status" aria-label={labels.loadingOptions ?? ''}>
          <Skeleton height={44} radius={2} />
        </div>
      ) : indexError ? (
        <Banner
          status="error"
          title={indexError}
          endContent={
            <Button
              variant="secondary"
              size="sm"
              label={labels.retry ?? ''}
              onClick={() => void refreshIndex()}
            />
          }
        />
      ) : null}

      <SchedulePanelDetail
        detail={detailQuery.data}
        loading={detailQuery.isPending && detailQuery.data === undefined && Boolean(detailRequest)}
        error={detailRequest && detailQuery.data === undefined ? detailError : ''}
        stale={detailQuery.isRefetchError}
        locale={locale}
        selectedRow={selectedRow}
        sourceGroupId={sourceGroupId}
        drafts={drafts}
        labels={labels.detail}
        onRefresh={() => void refreshDetail()}
        onSaved={(revision) => void handleSaved(revision)}
        onRecovered={(groupId, entryId) => void handleRecovered(groupId, entryId)}
        onProbe={probeRow}
        onProbeAll={probeRows}
        onDraftChange={onDraftChange}
        onRowChange={onRowChange}
      />
      <ModelProbeDialog
        open={probe.open}
        pending={probe.pending}
        failed={probe.failed}
        stopped={probe.stopped}
        results={probe.results}
        disabledGroupIds={probe.disabledGroupIds}
        total={probe.total}
        completed={probe.completed}
        applying={true}
        hideGroupControls={true}
        onOpenChange={(open) => {
          if (!open) probe.close()
        }}
        onStop={probe.stop}
        onViewLog={viewProbeLog}
      />
    </section>
  )
}
