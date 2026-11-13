import { Badge, Button, EmptyState, Selector, Skeleton, TextInput } from '@astryxdesign/core'
import * as stylex from '@stylexjs/stylex'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useNavigate } from '@tanstack/react-router'
import { Search, TriangleAlert } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useIntl } from 'react-intl'

import { announcementsQueryOptions } from '@shared/control/resources/announcements'
import { retryGroupSiteInfo } from '@shared/control/resources/groups'
import { controlQueryKeys } from '@shared/control/query-keys'
import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { ContentPanel } from '../../components/ContentPanel'
import { groupDetailHref } from '../groups/group-detail-href'

export function AnnouncementsView() {
  const t = useT()
  const navigate = useNavigate()
  const intl = useIntl()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const [site, setSite] = useState(
    () => new URLSearchParams(window.location.search).get('site') ?? '',
  )
  const [q, setQ] = useState('')
  useEffect(() => {
    const syncSite = () => setSite(new URLSearchParams(window.location.search).get('site') ?? '')
    window.addEventListener('popstate', syncSite)
    return () => window.removeEventListener('popstate', syncSite)
  }, [])
  const filters = { ...(site ? { site } : {}), ...(q.trim() ? { q: q.trim() } : {}) }
  const query = useQuery({
    ...announcementsQueryOptions(apiClient, filters),
    placeholderData: keepPreviousData,
  })
  const retry = useMutation({
    mutationFn: (groupId: number) => retryGroupSiteInfo(apiClient, groupId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['control', 'announcements'] }),
        queryClient.invalidateQueries({ queryKey: controlQueryKeys.groups.all }),
      ])
    },
  })
  const sites = query.data?.sites ?? []
  const sources = query.data?.sources ?? []
  const sourceSites = sources
    .filter((s) => !sites.some((option) => option.site_root_url === s.site_root_url))
    .filter(
      (s, index, all) =>
        all.findIndex((other) => other.site_root_url === s.site_root_url) === index,
    )
  const items = [...(query.data?.items ?? [])].sort(
    (a, b) => b.event_at_ms - a.event_at_ms || a.id.localeCompare(b.id),
  )
  return (
    <section {...stylex.props(styles.page)} aria-labelledby="announcements-title">
      <header {...stylex.props(styles.heading)}>
        <h1 id="announcements-title" {...stylex.props(styles.title)}>
          {t('nav.announcements')}
        </h1>
        {query.data && !query.isPlaceholderData && (
          <Badge variant="neutral" label={intl.formatNumber(items.length)} />
        )}
      </header>
      <ContentPanel>
        <div {...stylex.props(styles.toolbar)}>
          <Selector
            size="sm"
            variant="input"
            label={t('announcements.site')}
            options={[
              { value: '', label: t('announcements.allSites') },
              ...(site &&
              !sites.some((s) => s.site_root_url === site) &&
              !sourceSites.some((s) => s.site_root_url === site)
                ? [{ value: site, label: site }]
                : []),
              ...sites.map((s) => ({
                value: s.site_root_url,
                label: s.site_name || s.site_root_url,
              })),
              ...sourceSites.map((s) => ({
                value: s.site_root_url,
                label: s.site_name || s.site_root_url,
              })),
            ]}
            value={site}
            onChange={(value) => {
              setSite(value)
              const url = new URL(window.location.href)
              if (value) url.searchParams.set('site', value)
              else url.searchParams.delete('site')
              window.history.replaceState(window.history.state, '', url)
            }}
          />
          <TextInput
            size="sm"
            label={t('announcements.search')}
            placeholder={t('announcements.searchPlaceholder')}
            startIcon={<Search size={14} aria-hidden />}
            value={q}
            onChange={setQ}
          />
        </div>
        {query.isPending || query.isPlaceholderData ? (
          <div
            {...stylex.props(styles.loading)}
            role="status"
            aria-label={t('common.asyncLoading')}
          >
            <Skeleton height={80} radius={2} />
            <Skeleton height={80} radius={2} />
          </div>
        ) : query.isError && !query.data ? (
          <EmptyState
            title={t('announcements.error')}
            description={t('groups.collection.errorDescription')}
            headingLevel={2}
            actions={
              <Button
                size="sm"
                variant="secondary"
                label={t('announcements.retry')}
                onClick={() => void query.refetch()}
              />
            }
          />
        ) : (
          <>
            {query.isError && (
              <div {...stylex.props(styles.stale)} role="status">
                <TriangleAlert size={14} aria-hidden />
                {t('groups.collection.stale')}{' '}
                <Button
                  size="sm"
                  variant="secondary"
                  label={t('announcements.retry')}
                  onClick={() => void query.refetch()}
                />
              </div>
            )}
            {sources
              .filter((source) => !site || source.site_root_url === site)
              .map((source) => (
                <div
                  key={source.group.id}
                  {...stylex.props(
                    styles.source,
                    (source.announcement.status !== 'ok' || source.notice.status !== 'ok') &&
                      styles.sourceWarning,
                  )}
                >
                  <strong>
                    {source.site_name || source.site_root_url} · {source.group.name}
                  </strong>
                  <span>
                    {t('announcements.sourceAnnouncement')}:{' '}
                    {t(`announcements.sourceStatus.${source.announcement.status}`)}
                  </span>
                  <span>
                    {t('announcements.sourceNotice')}:{' '}
                    {t(`announcements.sourceStatus.${source.notice.status}`)}
                  </span>
                  {(source.announcement.status === 'error' || source.notice.status === 'error') && (
                    <>
                      {source.announcement.status === 'error' && (
                        <span {...stylex.props(styles.sourceHelp)}>
                          {t('announcements.sourceAnnouncement')}:{' '}
                          {source.announcement.last_error || t('announcements.sourceFailureHelp')}
                        </span>
                      )}
                      {source.notice.status === 'error' && (
                        <span {...stylex.props(styles.sourceHelp)}>
                          {t('announcements.sourceNotice')}:{' '}
                          {source.notice.last_error || t('announcements.sourceFailureHelp')}
                        </span>
                      )}
                      <Button
                        size="sm"
                        variant="secondary"
                        label={t('announcements.retry')}
                        isLoading={retry.isPending}
                        onClick={() => retry.mutate(source.group.id)}
                      />
                    </>
                  )}
                </div>
              ))}
            {items.length === 0 ? (
              <EmptyState
                headingLevel={2}
                title={q.trim() || site ? t('announcements.noMatches') : t('announcements.empty')}
                actions={
                  (q.trim() || site) && (
                    <Button
                      size="sm"
                      variant="secondary"
                      label={t('announcements.clearFilters')}
                      onClick={() => {
                        setQ('')
                        setSite('')
                        const url = new URL(window.location.href)
                        url.searchParams.delete('site')
                        window.history.replaceState(window.history.state, '', url)
                      }}
                    />
                  )
                }
              />
            ) : (
              <div {...stylex.props(styles.list)}>
                {items.map((item) => (
                  <article key={item.id} {...stylex.props(styles.item)}>
                    <div {...stylex.props(styles.meta)}>
                      <a
                        {...stylex.props(styles.siteLink)}
                        href={item.site_root_url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {item.site_name || item.site_root_url}
                      </a>
                      {item.groups.map((group) => (
                        <Button
                          key={group.id}
                          size="sm"
                          variant="ghost"
                          label={group.name}
                          onClick={() => void navigate({ to: groupDetailHref(group.id) })}
                        />
                      ))}
                      {item.level && (
                        <Badge
                          variant={
                            item.level === 'error'
                              ? 'error'
                              : item.level === 'success'
                                ? 'success'
                                : 'neutral'
                          }
                          label={item.level}
                        />
                      )}
                      <time
                        {...stylex.props(styles.time)}
                        dateTime={new Date(item.event_at_ms).toISOString()}
                      >
                        {intl.formatDate(item.event_at_ms, {
                          dateStyle: 'medium',
                          timeStyle: 'short',
                        })}
                      </time>
                    </div>
                    <p {...stylex.props(styles.content)}>{item.content}</p>
                    {item.extra && (
                      <p {...stylex.props(styles.content, styles.extra)}>{item.extra}</p>
                    )}
                  </article>
                ))}
              </div>
            )}
          </>
        )}
      </ContentPanel>
    </section>
  )
}
const styles = stylex.create({
  page: {
    minWidth: 0,
    padding: 'var(--stage-padding-top) var(--stage-padding-inline) var(--stage-padding-bottom)',
  },
  heading: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    paddingBottom: 'var(--space-2)',
  },
  title: { margin: 0, fontSize: 'var(--text-heading-2-size, 20px)', fontWeight: 650 },
  toolbar: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 680px)': 'minmax(0, 1fr)',
    },
    gap: 'var(--space-2)',
    marginBottom: 'var(--space-3)',
  },
  list: {
    minWidth: 0,
  },
  item: {
    minWidth: 0,
    paddingBlock: 'var(--space-3)',
    borderBottomWidth: { default: 1, ':last-child': 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
  },
  meta: {
    display: 'flex',
    gap: 'var(--space-1) var(--space-2)',
    alignItems: 'center',
    flexWrap: 'wrap',
    minWidth: 0,
  },
  siteLink: {
    color: 'var(--color-action)',
    fontWeight: 650,
    overflowWrap: 'anywhere',
    minWidth: 0,
    textDecorationLine: 'underline',
    textUnderlineOffset: 3,
    outlineColor: 'var(--color-focus)',
  },
  time: {
    marginLeft: { default: 'auto', '@media (max-width: 680px)': 0 },
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
  },
  content: {
    maxWidth: '75ch',
    whiteSpace: 'pre-wrap',
    overflowWrap: 'anywhere',
    marginTop: 'var(--space-2)',
    marginBottom: 0,
    lineHeight: 'var(--line-normal)',
  },
  extra: { color: 'var(--color-text-muted)' },
  loading: { display: 'grid', gap: 'var(--space-2)' },
  source: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    padding: 'var(--space-2) var(--space-3)',
    marginBottom: 'var(--space-2)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text)',
    fontSize: 'var(--text-meta)',
  },
  sourceWarning: {
    borderColor: 'var(--color-warning)',
    backgroundColor: 'var(--color-warning-bg)',
  },
  sourceHelp: { color: 'var(--color-text-muted)' },
  stale: {
    display: 'flex',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
    padding: 'var(--space-2) var(--space-3)',
    marginBottom: 'var(--space-3)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-text)',
    fontSize: 'var(--text-meta)',
  },
})
