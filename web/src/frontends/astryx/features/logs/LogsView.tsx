import * as stylex from '@stylexjs/stylex'
import { Badge, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { ArrowRight, TriangleAlert } from 'lucide-react'
import { useIntl } from 'react-intl'

import type { MessageId } from '@shared/i18n/message-ids'
import type {
  RequestLogDetailDto,
  RequestLogItemDto,
} from '@shared/control/resources/request-logs'
import {
  requestLogDetailQueryOptions,
  requestLogQueryOptions,
} from '@shared/control/resources/request-logs'
import { requestLogFirstScreen } from '@shared/domain/monitor/log-format'
import { pagePath } from '@shared/routing/page-routes'
import {
  parseSelectedRequestId,
  selectedRequestIdParam,
} from '@shared/routing/request-log-route'
import type { SharedRouteQuery } from '@shared/routing/route-query'

import { useT } from '../../app/i18n'
import { stringifySharedRouteSearch } from '../../app/search-codec'
import { useAppServices } from '../../app/services'
import { DetailPanel } from '../../components/DetailPanel'

// B11 spike(b) surface: a minimal real logs host — server-driven list plus
// the URL-driven detail panel the overlay decision is proven on. The full
// LogsTab parity (filters, cursor pagination, column density) is Phase 3
// work; this page only owns the spike's contract.

const statusTones: Record<RequestLogItemDto['status'], 'success' | 'error' | 'warning' | 'neutral'> = {
  success: 'success',
  error: 'error',
  incomplete: 'warning',
  canceled: 'neutral',
}

const styles = stylex.create({
  page: {
    width: '100%',
    paddingTop: 'var(--stage-padding-top)',
    paddingBottom: 'var(--stage-padding-bottom)',
    paddingInline: 'var(--stage-padding-inline)',
  },
  pageInner: {
    width: 'min(100%, 1240px)',
    marginInline: 'auto',
  },
  sheet: {
    minWidth: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-sheet)',
    backgroundColor: 'var(--color-surface)',
    boxShadow: 'var(--shadow-sheet)',
    paddingTop: 'var(--sheet-padding-top)',
    paddingBottom: 'var(--sheet-padding-bottom)',
    paddingInline: 'var(--sheet-padding-inline)',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-2-size, 20px)',
    fontWeight: 650,
  },
  list: {
    listStyle: 'none',
    margin: '14px 0 0',
    padding: 0,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    width: '100%',
    borderWidth: 0,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    padding: '10px 4px',
    backgroundColor: 'transparent',
    color: 'var(--color-text)',
    fontSize: 'var(--text-meta)',
    textAlign: 'start',
    cursor: 'pointer',
  },
  rowModel: {
    fontWeight: 600,
    minWidth: 0,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  rowMeta: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  },
  rowAction: {
    marginInlineStart: 'auto',
  },
  state: {
    paddingBlock: 24,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
  },
  detailGrid: {
    display: 'grid',
    gap: 14,
  },
  detailStatus: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
    fontVariantNumeric: 'tabular-nums',
  },
  detailSection: {
    display: 'grid',
    gap: 4,
  },
  detailLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  detailValue: {
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
  },
  detailMono: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    overflowWrap: 'anywhere',
  },
  attempt: {
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    paddingBlock: 6,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    fontSize: 'var(--text-meta)',
  },
})

function logTranslator(t: (id: MessageId, values?: Record<string, string | number>) => string) {
  return (key: string, named?: Record<string, string | number>) =>
    t(key as MessageId, named)
}

function LogRow({ log, onOpen }: { log: RequestLogItemDto; onOpen: () => void }) {
  const t = useT()
  const intl = useIntl()
  return (
    <li>
      <button
        type="button"
        id={`log-details-${log.request_id}`}
        {...stylex.props(styles.row)}
        aria-label={t('monitor.logs.details')}
        onClick={onOpen}
      >
        <Badge
          variant={statusTones[log.status]}
          label={t(`monitor.logs.status.${log.status}` as MessageId)}
        />
        <span {...stylex.props(styles.rowModel)}>
          {log.upstream_model ?? log.client_model ?? '—'}
        </span>
        <span {...stylex.props(styles.rowMeta)}>
          {intl.formatNumber(log.duration_ms)} ms
        </span>
        <span {...stylex.props(styles.rowMeta)}>{log.request_id.slice(0, 8)}</span>
        <span {...stylex.props(styles.rowAction)}>
          <ArrowRight size={15} aria-hidden />
        </span>
      </button>
    </li>
  )
}

function AttemptRow({ attempt }: { attempt: RequestLogDetailDto['attempts'][number] }) {
  const t = useT()
  const intl = useIntl()
  return (
    <div {...stylex.props(styles.attempt)}>
      <span>{t('monitor.logs.drawer.attempt', { sequence: attempt.sequence })}</span>
      <span {...stylex.props(styles.rowMeta)}>{attempt.status_code}</span>
      <span {...stylex.props(styles.rowModel)}>{attempt.group_name}</span>
      <span {...stylex.props(styles.rowMeta)}>
        {intl.formatNumber(attempt.duration_ms)} ms
      </span>
    </div>
  )
}

function LogDetailContent({ log }: { log: RequestLogDetailDto }) {
  const t = useT()
  const firstScreen = requestLogFirstScreen(log, logTranslator(t))
  return (
    <div {...stylex.props(styles.detailGrid)}>
      <div {...stylex.props(styles.detailStatus)}>
        <Badge
          variant={statusTones[log.status]}
          label={firstScreen.status}
        />
        <span {...stylex.props(styles.rowMeta)}>{firstScreen.status_code}</span>
        <span {...stylex.props(styles.rowMeta)}>
          {t('monitor.logs.attemptCount', { count: firstScreen.attempt_count })}
        </span>
      </div>
      {firstScreen.key_reason_label !== '' && (
        <div {...stylex.props(styles.detailSection)}>
          <span {...stylex.props(styles.detailLabel)}>{firstScreen.key_reason_label}</span>
          <span {...stylex.props(styles.detailValue)}>{firstScreen.key_reason_text}</span>
          {firstScreen.key_reason_code !== '' && (
            <span {...stylex.props(styles.detailMono)}>{firstScreen.key_reason_code}</span>
          )}
        </div>
      )}
      <div {...stylex.props(styles.detailSection)}>
        <span {...stylex.props(styles.detailLabel)}>
          {t('monitor.logs.drawer.upstreamModel')}
        </span>
        <span {...stylex.props(styles.detailValue)}>
          {log.upstream_model ??
            log.client_model ??
            t('monitor.logs.drawer.modelNotSpecified')}
        </span>
      </div>
      {firstScreen.request_error_code !== '' && (
        <div {...stylex.props(styles.detailSection)}>
          <span {...stylex.props(styles.detailLabel)}>
            {t('monitor.logs.drawer.requestErrorCode')}
          </span>
          <span {...stylex.props(styles.detailMono)}>{firstScreen.request_error_code}</span>
        </div>
      )}
      <div {...stylex.props(styles.detailSection)}>
        <span {...stylex.props(styles.detailLabel)}>{t('monitor.logs.drawer.attempts')}</span>
        {log.attempts.length === 0 ? (
          <span {...stylex.props(styles.detailValue)}>
            {t('monitor.logs.drawer.noAttempts')}
          </span>
        ) : (
          log.attempts.map((attempt) => (
            <AttemptRow key={attempt.sequence} attempt={attempt} />
          ))
        )}
      </div>
    </div>
  )
}

function LogDetailPanel({
  requestID,
  isOpen,
  onClose,
}: {
  requestID: string | undefined
  isOpen: boolean
  onClose: () => void
}) {
  const t = useT()
  const { apiClient } = useAppServices()
  const detailQuery = useQuery(requestLogDetailQueryOptions(apiClient, requestID))
  const log = detailQuery.data

  return (
    // isOpen toggles rather than unmounting: Dialog's close path runs the
    // invoker focus restore only when the element transitions to closed.
    <DetailPanel
      isOpen={isOpen}
      onOpenChange={(open) => {
        if (!open) onClose()
      }}
      title={t('monitor.logs.drawer.title')}
      subtitle={t('monitor.logs.drawer.description')}
    >
      {log === undefined ? (
        <div
          role="status"
          aria-label={t('monitor.logs.drawer.loading')}
          {...stylex.props(styles.detailGrid)}
        >
          <Skeleton height={20} radius={2} />
          <Skeleton height={56} radius={2} />
          <Skeleton height={120} radius={2} />
        </div>
      ) : detailQuery.isError ? (
        <div role="alert" {...stylex.props(styles.state)}>
          <TriangleAlert size={14} aria-hidden /> {t('monitor.logs.drawer.loadFailed')}
        </div>
      ) : (
        <LogDetailContent log={log} />
      )}
    </DetailPanel>
  )
}

export function LogsView() {
  const t = useT()
  const { apiClient } = useAppServices()
  const navigate = useNavigate()
  const rawSearch = useRouterState({
    select: (state) => state.location.search as SharedRouteQuery,
  })
  const selectedRequestID = parseSelectedRequestId(rawSearch)
  const logsPath = pagePath('logs')

  const logsQuery = useQuery(requestLogQueryOptions(apiClient, {}, undefined))

  function setSelected(requestID: string | undefined): void {
    // href navigation: the dynamic route tree keeps `search` loosely typed,
    // so the shared codec serializes the same vue-router-shaped query.
    void navigate({
      href: `${logsPath}${stringifySharedRouteSearch({
        ...rawSearch,
        [selectedRequestIdParam]: requestID,
      })}`,
    })
  }

  const items = logsQuery.data?.items

  return (
    <section {...stylex.props(styles.page)} aria-labelledby="logs-title">
      <div {...stylex.props(styles.pageInner)}>
        <div {...stylex.props(styles.sheet)}>
          <h1 id="logs-title" {...stylex.props(styles.title)}>
            {t('shell.logs')}
          </h1>
          {logsQuery.isPending ? (
            <div
              role="status"
              aria-label={t('monitor.logs.loading')}
              {...stylex.props(styles.detailGrid)}
            >
              {Array.from({ length: 5 }, (_, index) => (
                <Skeleton key={index} height={42} radius={2} />
              ))}
            </div>
          ) : logsQuery.isError ? (
            <div role="alert" {...stylex.props(styles.state)}>
              <TriangleAlert size={14} aria-hidden /> {t('monitor.logs.loadFailed')}
            </div>
          ) : items === undefined || items.length === 0 ? (
            <div {...stylex.props(styles.state)}>{t('monitor.logs.empty.title')}</div>
          ) : (
            <ul {...stylex.props(styles.list)} aria-label={t('monitor.logs.caption')}>
              {items.map((log) => (
                <LogRow
                  key={log.request_id}
                  log={log}
                  onOpen={() => setSelected(log.request_id)}
                />
              ))}
            </ul>
          )}
        </div>
      </div>
      <LogDetailPanel
        requestID={selectedRequestID}
        isOpen={selectedRequestID !== undefined}
        onClose={() => setSelected(undefined)}
      />
    </section>
  )
}
