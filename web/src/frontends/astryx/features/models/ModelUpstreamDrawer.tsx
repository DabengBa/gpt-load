import * as stylex from '@stylexjs/stylex'
import { Button, EmptyState, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { TriangleAlert, Zap } from 'lucide-react'
import { useImperativeHandle, type Ref } from 'react'
import { useIntl } from 'react-intl'

import { controlQueryKeys } from '@shared/control/query-keys'
import {
  getUpstreamModelDetail,
  type UpstreamModelDetailDto,
} from '@shared/control/resources/models'
import { formatLocalInstant } from '@shared/lib/format'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { useAppServices } from '../../app/services'
import { useStableLoading } from '../../app/collection-loading'
import { useModelPriceEditor } from '../../app/use-model-price-editor'
import { ChannelIcon } from '../../components/ChannelIcon'
import { CopyButton } from '../../components/CopyButton'
import { DetailPanel } from '../../components/DetailPanel'
import { ModelPriceMatrix } from './ModelPriceMatrix'
import { ModelPriceResetDialog } from './ModelPriceResetDialog'
import { ModelPriceSlotsEditor } from './ModelPriceSlotsEditor'
import { ModelPriceStatusBadge } from './ModelPriceStatusBadge'
import { ModelSpecSheet } from './ModelSpecSheet'

const DRAWER_NARROW = '@media (max-width: 620px)'

const styles = stylex.create({
  body: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-4)',
    paddingBlock: 'var(--space-3-5)',
  },
  meta: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-1) var(--space-2-5)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: 'var(--space-2)',
    paddingInline: 'var(--space-2-5)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  identity: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      [DRAWER_NARROW]: 'minmax(0, 1fr)',
    },
    gap: 'var(--space-2)',
    margin: 0,
  },
  identityItem: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    borderInlineStartWidth: 2,
    borderInlineStartStyle: 'solid',
    borderInlineStartColor: 'var(--color-border-control)',
    paddingInlineStart: 'var(--space-2)',
  },
  identityTerm: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  identityValue: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'baseline',
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  identityCode: {
    color: 'var(--color-text)',
    fontSize: 'var(--text-label-xs)',
    overflowWrap: 'anywhere',
  },
  channelIcon: {
    alignSelf: 'center',
    flex: 'none',
    fontSize: 'var(--text-body)',
  },
  channelName: {
    color: 'var(--color-text)',
    fontSize: 'var(--text-body)',
  },
  faint: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  warningBanner: {
    borderRadius: 'var(--radius-control, 6px)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    backgroundColor:
      'var(--color-warning-bg, color-mix(in srgb, var(--color-warning) 12%, transparent))',
    padding: '9px 12px',
    fontSize: 'var(--text-meta)',
  },
  section: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-2-5)',
  },
  sectionTitle: {
    display: 'flex',
    alignItems: 'baseline',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    margin: 0,
    fontSize: 'var(--text-meta)',
    fontWeight: 650,
  },
  eyebrow: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 400,
    letterSpacing: '0.03em',
  },
  modeHeading: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
  },
  modeHeadingTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-1)',
    margin: 0,
    fontSize: 'var(--text-meta)',
    fontWeight: 650,
  },
  modeHeadingIcon: {
    display: 'inline-flex',
    color: 'var(--color-text-faint)',
  },
  associations: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    margin: 0,
    padding: 0,
    listStyle: 'none',
  },
  association: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) minmax(120px, auto)',
      [DRAWER_NARROW]: 'minmax(0, 1fr)',
    },
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    padding: 'var(--space-2)',
  },
  mapping: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
    color: 'var(--color-text-muted)',
  },
  mappingCode: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    overflowWrap: 'anywhere',
  },
  group: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
    justifyContent: { default: 'flex-end', [DRAWER_NARROW]: 'flex-start' },
  },
  groupLink: {
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'currentcolor',
    color: 'var(--color-action)',
    fontSize: 'var(--text-meta)',
    fontWeight: 560,
    overflowWrap: 'anywhere',
    textDecoration: 'none',
  },
  groupTag: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    whiteSpace: 'nowrap',
  },
  footer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    width: '100%',
  },
  footerActions: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  skeletonStack: {
    display: 'grid',
    gap: '12px',
    paddingBlock: 'var(--space-3-5)',
  },
})

export interface ModelUpstreamDrawerHandle {
  requestClose(): Promise<void>
  confirmDiscardSwitch(): Promise<boolean>
  discardChanges(): void
  hasUnsavedChanges(): boolean
}

const placeholderPrice: UpstreamModelDetailDto['price'] = {
  id: 0,
  channel_id: '',
  channel_name: '',
  channel_mark: '',
  channel_icon: '',
  model_id: '',
  prices: { input: null, output: null, cache_read: null, cache_write: null },
  mode_schedules: {},
  pricing_status: 'pending',
  method: null,
  matched_provider_id: null,
  match_source: null,
  referenced: false,
  reference_count: 0,
  reference_group_count: 0,
  context_tiers: [],
  updated_at_ms: 0,
  can_reset: false,
  can_delete: false,
}

export function ModelUpstreamDrawer({
  isOpen,
  priceId,
  onClose,
  ref,
}: {
  isOpen: boolean
  priceId: number | null
  onClose: () => void
  ref?: Ref<ModelUpstreamDrawerHandle>
}) {
  const t = useT()
  const intl = useIntl()
  const { apiClient } = useAppServices()

  const detailQuery = useQuery({
    queryKey: controlQueryKeys.models.detail(priceId ?? 0),
    queryFn: ({ signal }) => getUpstreamModelDetail(apiClient, priceId as number, signal),
    enabled: isOpen && priceId !== null,
  })
  const initialLoading = useStableLoading(isOpen && detailQuery.isPending)
  const detail = detailQuery.data

  /**
   * 编辑器需要一个稳定的价格引用;详情未就绪时用占位行,
   * 待数据到达后控制器内部的 setRow 会按 id 重建草稿。
   */
  const price = detail?.price ?? placeholderPrice
  const editor = useModelPriceEditor(price)
  const controller = editor.controller
  const snapshot = editor.snapshot
  const draft = snapshot.draft
  const errors = snapshot.errors
  const pending = snapshot.pending
  const failure = snapshot.failure
  const changed = snapshot.changed

  const hasFastSchedule = Boolean(price.mode_schedules.fast)
  const fastDraft = draft.modeSchedules.fast?.base ?? {
    input: '',
    output: '',
    cache_read: '',
    cache_write: '',
  }
  const fastErrors = errors.modeSchedules.fast?.base ?? {}

  const channelName = price.channel_name.trim() || price.channel_id || '—'

  async function requestClose(): Promise<void> {
    if (!(await editor.confirmDiscardSwitch())) return
    controller.cancel()
    onClose()
  }

  useImperativeHandle(ref, () => ({
    requestClose,
    confirmDiscardSwitch: editor.confirmDiscardSwitch,
    discardChanges: () => controller.cancel(),
    hasUnsavedChanges: () => controller.hasChanged(),
  }))

  return (
    <DetailPanel
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) void requestClose()
      }}
      title={detail?.model_id ?? t('models.drawer.title')}
      subtitle={
        detail
          ? t('models.drawer.impact', {
              clients: detail.client_model_count,
              groups: detail.group_count,
            })
          : t('models.drawer.description')
      }
      titleAdornment={
        isOpen && detail ? (
          <CopyButton
            key={priceId ?? undefined}
            value={detail.model_id}
            label={t('models.drawer.copyModel', { model: detail.model_id })}
            successLabel={t('models.drawer.copySucceeded')}
            failureLabel={t('models.drawer.copyFailed')}
          />
        ) : undefined
      }
      footer={
        detail ? (
          <div {...stylex.props(styles.footer)}>
            {price.can_reset ? (
              <ModelPriceResetDialog
                row={price}
                action="reset"
                disabled={pending || changed}
                onCompleted={() => controller.cancel()}
              />
            ) : (
              <span />
            )}
            <span {...stylex.props(styles.footerActions)}>
              <Button
                variant="secondary"
                size="sm"
                isDisabled={pending || !changed}
                label={t('common.cancel')}
                onClick={() => controller.cancel()}
              />
              <Button
                size="sm"
                isLoading={pending}
                isDisabled={!snapshot.canSave}
                label={t('modelPrices.matrix.save')}
                onClick={() => controller.requestSave()}
              />
            </span>
          </div>
        ) : undefined
      }
    >
      {(isOpen && detailQuery.isPending) || initialLoading ? (
        <div {...stylex.props(styles.skeletonStack)} role="status" aria-label={t('models.drawer.loading')}>
          <Skeleton height={28} radius={2} />
          <Skeleton height={64} radius={2} />
          <Skeleton height={120} radius={2} />
          <Skeleton height={160} radius={2} />
          <Skeleton height={140} radius={2} />
        </div>
      ) : detailQuery.isError ? (
        <div {...stylex.props(styles.body)}>
          <EmptyState
            title={t('models.drawer.loadFailed')}
            icon={<TriangleAlert size={20} />}
            actions={
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void detailQuery.refetch()}
              />
            }
          />
        </div>
      ) : detail ? (
        <div {...stylex.props(styles.body)}>
          <div {...stylex.props(styles.meta)}>
            <ModelPriceStatusBadge
              price={price}
              providerName={detail.catalog_reference?.provider_name}
            />
            {price.updated_at_ms > 0 && (
              <span {...stylex.props(styles.faint)}>
                {t('models.drawer.updatedAt')}{' '}
                {formatLocalInstant(price.updated_at_ms, intl.locale)}
              </span>
            )}
          </div>

          <dl {...stylex.props(styles.identity)}>
            <div {...stylex.props(styles.identityItem)}>
              <dt {...stylex.props(styles.identityTerm)}>{t('models.drawer.pricingChannel')}</dt>
              <dd {...stylex.props(styles.identityValue)}>
                {(price.channel_icon || price.channel_mark) && (
                  <span {...stylex.props(styles.channelIcon)}>
                    <ChannelIcon icon={price.channel_icon} mark={price.channel_mark} />
                  </span>
                )}
                <span {...stylex.props(styles.channelName)}>{channelName}</span>
              </dd>
            </div>
            <div {...stylex.props(styles.identityItem)}>
              <dt {...stylex.props(styles.identityTerm)}>{t('models.drawer.upstreamModel')}</dt>
              <dd {...stylex.props(styles.identityValue)}>
                <code {...stylex.props(styles.identityCode)}>{detail.model_id}</code>
              </dd>
            </div>
          </dl>

          {price.reference_count > 1 && (
            <div {...stylex.props(styles.warningBanner)} role="status">
              {t('models.drawer.sharedImpact', {
                references: price.reference_count,
                clients: detail.client_model_count,
                groups: detail.group_count,
              })}
            </div>
          )}

          <section {...stylex.props(styles.section)}>
            <h3 {...stylex.props(styles.sectionTitle)}>{t('models.drawer.specs')}</h3>
            {detail.catalog_reference ? (
              <ModelSpecSheet reference={detail.catalog_reference} />
            ) : (
              <p {...stylex.props(styles.faint)}>{t('models.inspector.noCatalog')}</p>
            )}
          </section>

          <section {...stylex.props(styles.section)}>
            <h3 {...stylex.props(styles.sectionTitle)}>
              {t('models.drawer.prices')}
              <span {...stylex.props(styles.eyebrow)}>{t('modelPrices.matrix.unit')}</span>
            </h3>
            <ModelPriceMatrix
              modelId={detail.model_id}
              draft={draft}
              onDraftChange={(next) => controller.setScheduleDraft(undefined, next)}
              errors={errors}
              pending={pending}
              failure={failure}
              unpricedConfirmOpen={snapshot.unpricedConfirmOpen}
              onUnpricedConfirmOpenChange={(open) => controller.setUnpricedConfirmOpen(open)}
              onAddTier={() => controller.addTier()}
              onRemoveTier={(key) => controller.removeTier(key)}
              onConfirmUnpriced={() => controller.confirmUnpricedSave()}
            />
            {hasFastSchedule && (
              <div {...stylex.props(styles.modeHeading)}>
                <h3 {...stylex.props(styles.modeHeadingTitle)}>
                  <span {...stylex.props(styles.modeHeadingIcon)}>
                    <Zap size={13} aria-hidden />
                  </span>
                  {t('models.drawer.fastPrices')}
                </h3>
                <span {...stylex.props(styles.eyebrow)}>{t('modelPrices.matrix.unit')}</span>
              </div>
            )}
            {hasFastSchedule && (
              <ModelPriceSlotsEditor
                draft={fastDraft}
                onDraftChange={(next) => {
                  const schedule = draft.modeSchedules.fast
                  if (schedule) controller.setScheduleDraft('fast', { ...schedule, base: next })
                }}
                errors={fastErrors}
                pending={pending}
                idPrefix="model-price-fast"
              />
            )}
          </section>

          <section {...stylex.props(styles.section)}>
            <h3 {...stylex.props(styles.sectionTitle)}>{t('models.drawer.relationships')}</h3>
            <p {...stylex.props(styles.faint)}>
              {t('models.drawer.relationshipsHelp', { count: price.reference_count })}
            </p>
            <ul {...stylex.props(styles.associations)}>
              {detail.associations.map((association) => (
                <li
                  key={`${association.client_model}:${association.group.id}`}
                  {...stylex.props(styles.association)}
                >
                  <span
                    {...stylex.props(styles.mapping)}
                    aria-label={
                      association.alias_applied
                        ? t('models.drawer.aliasMapping', {
                            client: association.client_model,
                            upstream: detail.model_id,
                          })
                        : t('models.drawer.directMapping', {
                            model: association.client_model,
                          })
                    }
                  >
                    <code {...stylex.props(styles.mappingCode)}>{association.client_model}</code>
                    {association.alias_applied && (
                      <>
                        <span aria-hidden>→</span>
                        <code {...stylex.props(styles.mappingCode)}>{detail.model_id}</code>
                      </>
                    )}
                  </span>
                  <span {...stylex.props(styles.group)}>
                    <span {...stylex.props(styles.eyebrow)}>{t('models.drawer.group')}</span>
                    <RouteLink
                      to={`${pagePath('groups')}/${association.group.id}?tab=models`}
                      {...stylex.props(styles.groupLink)}
                    >
                      {association.group.name}
                    </RouteLink>
                    {!association.group.enabled && (
                      <span {...stylex.props(styles.groupTag)}>
                        {t('models.detail.groupDisabled')}
                      </span>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          </section>
        </div>
      ) : null}
      {editor.dialog}
    </DetailPanel>
  )
}
