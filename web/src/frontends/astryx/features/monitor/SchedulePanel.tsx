import * as stylex from '@stylexjs/stylex'
import './SchedulePanel.css'
import { Banner, Button, TextInput, EmptyState, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { useEffect, useLayoutEffect, useRef, useState } from 'react'

import type { ModelProbeTargetDto } from '@shared/control/resources/model-probe'
import {
  modelRouteScheduleDetailQueryOptions,
  modelRouteScheduleIndexQueryOptions,
  type ModelRouteScheduleIndexItemDto,
} from '@shared/control/resources/model-route-schedule'
import { pagePath } from '@shared/routing/page-routes'
import type { ScheduleDrafts } from '@shared/routing/monitor-route'
import { resolveSchedulePriceID } from '@shared/domain/monitor/schedule-price'
import { listModels } from '@shared/control/resources/models'

import { useAppServices } from '../../app/services'
import { ModelProbeDialog } from '../models/ModelProbeDialog'
import { useModelProbe } from '../models/use-model-probe'
import { SchedulePanelDetail, type SchedulePanelDetailLabels } from './SchedulePanelDetail'

export interface SchedulePanelLabels {
  model: string
  selectModel: string
  noMatchingModels?: string
  noModels?: string
  importModels?: string
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
    gridTemplateColumns: {
      default: 'minmax(180px, 240px) minmax(0, 1fr)',
      '@media (max-width: 800px)': 'minmax(0, 1fr)',
    },
    minWidth: 0,
    gap: '24px',
  },
  filters: {
    display: 'grid',
    minWidth: 0,
    alignContent: 'start',
    gridTemplateColumns: {
      default: 'minmax(0, 1fr)',
      '@media (max-width: 620px)': '1fr',
    },
    gap: 'var(--space-3)',
    width: '100%',
  },
  field: {
    display: 'grid',
    minWidth: 0,
    gap: '5px',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontWeight: 650,
  },
  detail: { minWidth: 0, display: 'grid', alignContent: 'start', gap: '16px' },
  modelButton: { minWidth: 0, maxWidth: '100%', whiteSpace: 'normal', overflowWrap: 'anywhere' },
  searchControl: { minWidth: 0, width: '100%', boxSizing: 'border-box' },
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
  onSaved(snapshotRevision: number, clearDrafts: boolean): void
  onRecovered(groupId: number, entryId: string): void
  onOpenPrice(priceID: number): void
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
  onOpenPrice,
}: SchedulePanelProps) {
  const { apiClient } = useAppServices()
  const navigate = useNavigate()

  const [selectedModel, setSelectedModel] = useState(externalModel)
  const [search, setSearch] = useState('')
  const [priceError, setPriceError] = useState('')
  const [priceTarget, setPriceTarget] = useState<{ groupID: number; modelID: string }>()
  const priceGeneration = useRef(0)
  useEffect(
    () => () => {
      priceGeneration.current += 1
    },
    [],
  )
  const latestModel = useRef(externalModel)
  useLayoutEffect(() => {
    latestModel.current = externalModel
  }, [externalModel])
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

  const filteredModels = indexItems.filter((item) =>
    item.external_model.toLowerCase().includes(search.trim().toLowerCase()),
  )
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
    priceGeneration.current += 1
    onChangeContext({ externalModel: value || undefined })
  }

  async function openPrice(groupID: number, modelID: string): Promise<void> {
    const generation = ++priceGeneration.current
    const requestedModel = selectedModel
    setPriceTarget({ groupID, modelID })
    setPriceError('')
    try {
      const group = detailQuery.data?.groups.find((candidate) => candidate.group_id === groupID)
      if (!group || selectedModel === '') return
      const priceID = await resolveSchedulePriceID(
        { clientModel: selectedModel, channelID: group.channel_id, upstreamModelID: modelID },
        async (page) => {
          const result = await listModels(apiClient, {
            group_status: 'all',
            pricing_status: 'all',
            q: selectedModel,
            page,
            page_size: 10,
          })
          return {
            items: result.items,
            pagination: {
              page: result.pagination.page,
              total_pages: result.pagination.total_pages,
            },
          }
        },
      )
      if (generation !== priceGeneration.current || latestModel.current !== requestedModel) return
      if (priceID === undefined) throw new Error(labels.detailFailed ?? '')
      onOpenPrice(priceID)
    } catch (error) {
      if (generation === priceGeneration.current && latestModel.current === requestedModel) {
        setPriceError(error instanceof Error ? error.message : (labels.detailFailed ?? ''))
      }
    }
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

  async function handleSaved(revision: number, clearDrafts: boolean): Promise<void> {
    onSaved(revision, clearDrafts)
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
        <div {...stylex.props(styles.field)}>
          <div className="schedule-model-search">
            <TextInput
              label={labels.model ?? ''}
              width="100%"
              xstyle={styles.searchControl}
              size="sm"
              value={search}
              onChange={setSearch}
            />
          </div>
          {filteredModels.map((item) => (
            <Button
              key={item.external_model}
              variant={selectedModel === item.external_model ? 'primary' : 'ghost'}
              size="sm"
              label={item.external_model}
              xstyle={styles.modelButton}
              onClick={() => setModel(item.external_model)}
            />
          ))}
          {!indexQuery.isPending && !indexQuery.isError && filteredModels.length === 0 && (
            <EmptyState
              title={
                indexItems.length === 0 ? (labels.noModels ?? '') : (labels.noMatchingModels ?? '')
              }
            />
          )}
          {!indexQuery.isPending && !indexQuery.isError && indexItems.length === 0 && (
            <Button
              label={labels.importModels ?? ''}
              onClick={() => void navigate({ to: pagePath('import') })}
            />
          )}
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
      </div>

      <div {...stylex.props(styles.detail)}>
        {priceError && (
          <Banner
            status="error"
            title={priceError}
            endContent={
              <Button
                label={labels.retry ?? ''}
                onClick={() => {
                  if (priceTarget) void openPrice(priceTarget.groupID, priceTarget.modelID)
                }}
              />
            }
          />
        )}
        {selectedModel === '' ? (
          indexItems.length > 0 ? (
            <EmptyState title={labels.selectModel ?? ''} />
          ) : null
        ) : (
          <SchedulePanelDetail
            detail={detailQuery.data}
            loading={
              detailQuery.isPending && detailQuery.data === undefined && Boolean(detailRequest)
            }
            error={detailRequest && detailQuery.data === undefined ? detailError : ''}
            stale={detailQuery.isRefetchError}
            locale={locale}
            selectedRow={selectedRow}
            sourceGroupId={sourceGroupId}
            drafts={drafts}
            labels={labels.detail}
            onRefresh={() => void refreshDetail()}
            onSaved={(revision, clearDrafts) => void handleSaved(revision, clearDrafts)}
            onRecovered={(groupId, entryId) => void handleRecovered(groupId, entryId)}
            onProbe={probeRow}
            onProbeAll={probeRows}
            onDraftChange={onDraftChange}
            onRowChange={onRowChange}
            onOpenPrice={(groupID, modelID) => void openPrice(groupID, modelID)}
          />
        )}
      </div>
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
