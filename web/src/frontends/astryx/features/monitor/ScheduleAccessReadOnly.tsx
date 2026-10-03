import { useState } from 'react'
import * as stylex from '@stylexjs/stylex'
import { TextInput, Pagination, EmptyState, Skeleton, Button } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { modelCollectionQueryOptions } from '@shared/control/resources/models'
import { useAppServices } from '../../app/services'
import { useT } from '../../app/i18n'

const styles = stylex.create({
  panel: { display: 'grid', gap: '16px', minWidth: 0, paddingTop: '16px' },
  row: {
    display: 'grid',
    gap: '8px',
    paddingBlock: '12px',
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border)',
  },
  prices: { display: 'flex', flexWrap: 'wrap', gap: '12px' },
})

export function ScheduleAccessReadOnly({ externalModel }: { externalModel?: string }) {
  const { apiClient } = useAppServices()
  const t = useT()
  const [q, setQ] = useState(externalModel ?? '')
  const [page, setPage] = useState(1)
  const query = useQuery(
    modelCollectionQueryOptions(
      apiClient,
      {
        group_status: 'enabled',
        pricing_status: 'all',
        page,
        page_size: 10,
        q,
      },
      true,
    ),
  )
  return (
    <section {...stylex.props(styles.panel)} data-testid="schedule-read-only">
      <TextInput
        label={t('monitor.schedule.panel.model')}
        value={q}
        onChange={(value) => {
          setQ(value)
          setPage(1)
        }}
      />
      {query.isPending && <Skeleton height={44} radius={2} />}
      {query.isError && (
        <Button label={t('monitor.schedule.panel.retry')} onClick={() => void query.refetch()} />
      )}
      {query.data?.items.length === 0 && (
        <EmptyState title={t('monitor.schedule.detail.noEntries')} />
      )}
      {query.data?.items.map((item) => (
        <div key={item.client_model} {...stylex.props(styles.row)}>
          <strong>{item.client_model}</strong>
          <span>{item.protocols.join(', ')}</span>
          {item.upstream_models.map(({ price }) => (
            <div key={price.id} {...stylex.props(styles.prices)}>
              <span>
                {price.model_id} / {price.channel_name}
              </span>
              <span>
                {t('modelPrices.fields.input')}:{' '}
                {price.prices.input ?? t('models.filters.pricingStatus.pending')}
              </span>
              <span>
                {t('modelPrices.fields.output')}:{' '}
                {price.prices.output ?? t('models.filters.pricingStatus.pending')}
              </span>
              <span>{t('models.status.unit')}</span>
            </div>
          ))}
        </div>
      ))}
      {query.data && (
        <Pagination
          page={page}
          totalItems={query.data.pagination.total_items}
          totalPages={query.data.pagination.total_pages}
          pageSize={10}
          onChange={setPage}
          size="sm"
        />
      )}
    </section>
  )
}
