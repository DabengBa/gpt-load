import * as stylex from '@stylexjs/stylex'
import { Skeleton } from '@astryxdesign/core'
import { useIntl } from 'react-intl'

import type { HomeStatisticsDto } from '@shared/control/resources/home'
import { pagePath } from '@shared/routing/page-routes'
import { formatEstimatedCost, formatInteger, formatTokens } from '@shared/lib/format'

import { useT } from '../../app/i18n'
import { RouteLink } from '../../app/route-link'
import { HomeSectionHeading } from './home-chrome'

const NARROW = '@media (max-width: 560px)'

const styles = stylex.create({
  section: {
    marginTop: 36,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 20,
    paddingBottom: 4,
  },
  link: {
    color: 'var(--color-action)',
    fontSize: 'var(--text-meta)',
    fontWeight: 600,
    whiteSpace: 'nowrap',
    textDecoration: { default: 'none', ':hover': 'underline' },
  },
  value: {
    margin: '10px 0 16px',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--stat-value)',
    fontWeight: 550,
    lineHeight: 1.05,
    letterSpacing: '-0.03em',
    fontVariantNumeric: 'tabular-nums',
  },
  valueSkeleton: {
    marginTop: 10,
    marginBottom: 18,
  },
  tableWrap: {
    overflowX: 'auto',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: 'var(--text-sm)',
  },
  caption: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
  },
  headCell: {
    paddingTop: 7,
    paddingBottom: 7,
    paddingInline: 0,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 600,
    textAlign: 'left',
    whiteSpace: 'nowrap',
  },
  headCellNumber: {
    textAlign: 'right',
  },
  cell: {
    paddingTop: 7,
    paddingBottom: 7,
    paddingInline: 0,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    color: 'var(--color-text)',
    verticalAlign: 'middle',
  },
  number: {
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    textAlign: 'right',
    whiteSpace: 'nowrap',
  },
  model: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    maxWidth: {
      default: 'none',
      [NARROW]: '38vw',
    },
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  modelLink: {
    color: { default: 'inherit', ':hover': 'var(--color-action)' },
    textDecoration: { default: 'none', ':hover': 'underline' },
  },
  empty: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    paddingBlock: 'var(--space-3)',
  },
  lowPriority: {
    display: {
      default: 'table-cell',
      [NARROW]: 'none',
    },
  },
})

function modelUsageHref(model: string): string {
  const base = `${pagePath('monitor')}?range=30d`
  return model === '' ? base : `${base}&upstream_model=${encodeURIComponent(model)}`
}

const skeletonWidths = ['72%', '58%', '44%', '36%'] as const
const skeletonRows = [1, 2, 3, 4, 5] as const

export function HomeSpend({
  snapshot,
  loading = false,
}: {
  snapshot: HomeStatisticsDto | null
  loading?: boolean
}) {
  const intl = useIntl()
  const t = useT()

  const rows = snapshot?.rankings.models.slice(0, 5) ?? []
  const totalCost =
    snapshot === null
      ? '—'
      : formatEstimatedCost(snapshot.summary.estimated_cost_nano_usd, intl.locale)

  const modelName = (model: string): string => model || t('home.ledger.spend.unknownModel')

  return (
    <section {...stylex.props(styles.section)} aria-labelledby="home-spend-title">
      <HomeSectionHeading
        id="home-spend-title"
        title={t('home.ledger.spend.title')}
        actions={
          <RouteLink to={`${pagePath('monitor')}?range=30d`} {...stylex.props(styles.link)}>
            {t('home.ledger.spend.viewDetail')}
          </RouteLink>
        }
      />

      {loading ? (
        <div {...stylex.props(styles.valueSkeleton)}>
          <Skeleton width={140} height={36} />
        </div>
      ) : (
        <p {...stylex.props(styles.value)}>{totalCost}</p>
      )}

      <div {...stylex.props(styles.tableWrap)}>
        <table {...stylex.props(styles.table)} aria-busy={loading ? 'true' : undefined}>
          <caption {...stylex.props(styles.caption)}>{t('home.ledger.spend.caption')}</caption>
          <thead>
            <tr>
              <th scope="col" {...stylex.props(styles.headCell)}>
                {t('home.ledger.spend.columns.model')}
              </th>
              <th scope="col" {...stylex.props(styles.headCell, styles.headCellNumber)}>
                {t('home.ledger.spend.columns.requests')}
              </th>
              <th
                scope="col"
                {...stylex.props(styles.headCell, styles.headCellNumber, styles.lowPriority)}
              >
                {t('home.ledger.spend.columns.tokens')}
              </th>
              <th scope="col" {...stylex.props(styles.headCell, styles.headCellNumber)}>
                {t('home.ledger.spend.columns.cost')}
              </th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              skeletonRows.map((row) => (
                <tr key={row}>
                  {skeletonWidths.map((width, column) => (
                    <td key={column} {...stylex.props(styles.cell)}>
                      <Skeleton height={12} width={width} />
                    </td>
                  ))}
                </tr>
              ))
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={4} {...stylex.props(styles.cell, styles.empty)}>
                  {t('home.ledger.spend.empty')}
                </td>
              </tr>
            ) : (
              rows.map((row) => {
                const exactTokens = t('home.ledger.tokens', {
                  count: formatInteger(row.total_tokens, intl.locale),
                })
                return (
                  <tr key={row.model}>
                    <td {...stylex.props(styles.cell, styles.model)}>
                      <RouteLink
                        to={modelUsageHref(row.model)}
                        aria-label={t('home.ledger.spend.viewModel', {
                          model: modelName(row.model),
                        })}
                        {...stylex.props(styles.modelLink)}
                      >
                        {modelName(row.model)}
                      </RouteLink>
                    </td>
                    <td {...stylex.props(styles.cell, styles.number)}>
                      {formatInteger(row.request_count, intl.locale)}
                    </td>
                    <td
                      {...stylex.props(styles.cell, styles.number, styles.lowPriority)}
                      title={exactTokens}
                      aria-label={exactTokens}
                    >
                      {formatTokens(row.total_tokens, intl.locale)}
                    </td>
                    <td {...stylex.props(styles.cell, styles.number)}>
                      {formatEstimatedCost(row.estimated_cost_nano_usd, intl.locale)}
                    </td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </section>
  )
}
