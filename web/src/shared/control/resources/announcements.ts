import type { QueryFunctionContext } from '@tanstack/query-core'

import type { ApiClient } from '@shared/http/client'
import type {
  AnnouncementGroupRefDto,
  AnnouncementItemDto,
  AnnouncementSiteOptionDto,
  AnnouncementsResponseDto,
  AnnouncementSourceDto,
  AnnouncementSourceStateDto,
} from '@shared/control/types'
import { controlQueryKeys } from '@shared/control/query-keys'
import {
  assertNoSecretLikeFields,
  projectArray,
  projectEpochMilliseconds,
  projectNullableEpochMilliseconds,
  projectEnum,
  projectRecord,
  projectSafeInteger,
  projectString,
  projectUnixMilliseconds,
} from './projector'

function text(value: unknown): string {
  return projectString(value, { allowEmpty: true })
}
function group(value: unknown): AnnouncementGroupRefDto {
  const r = projectRecord(value)
  assertNoSecretLikeFields(r, ['id', 'name'])
  return { id: projectSafeInteger(r.id, { minimum: 1 }), name: projectString(r.name) }
}
function item(value: unknown): AnnouncementItemDto {
  const r = projectRecord(value)
  assertNoSecretLikeFields(r, [
    'id',
    'site_root_url',
    'site_name',
    'content',
    'extra',
    'level',
    'source',
    'event_at_ms',
    'fetched_at_ms',
    'groups',
  ])
  return {
    id: projectString(r.id),
    site_root_url: projectString(r.site_root_url),
    site_name: text(r.site_name),
    content: text(r.content),
    extra: text(r.extra),
    level: text(r.level),
    source: text(r.source),
    // RFC3339 event times before 1970 are legitimate negative epochs; a single clamped
    // item must not fail the whole page.
    event_at_ms: projectUnixMilliseconds(r.event_at_ms),
    fetched_at_ms: projectEpochMilliseconds(r.fetched_at_ms),
    groups: projectArray(r.groups, group),
  }
}
function site(value: unknown): AnnouncementSiteOptionDto {
  const r = projectRecord(value)
  assertNoSecretLikeFields(r, ['site_root_url', 'site_name', 'count'])
  return {
    site_root_url: projectString(r.site_root_url),
    site_name: text(r.site_name),
    count: projectSafeInteger(r.count, { minimum: 0 }),
  }
}
export function projectAnnouncementsResponse(value: unknown): AnnouncementsResponseDto {
  const r = projectRecord(value)
  assertNoSecretLikeFields(r, ['items', 'sites', 'sources'])
  return {
    items: projectArray(r.items, item),
    sites: projectArray(r.sites, site),
    sources: r.sources === undefined ? [] : projectArray(r.sources, source),
  }
}
function sourceState(value: unknown): AnnouncementSourceStateDto {
  const r = projectRecord(value)
  assertNoSecretLikeFields(r, ['status', 'fetched_at_ms', 'last_error'])
  return {
    status: projectEnum(r.status, ['pending', 'unknown', 'ok', 'error', 'stale'] as const),
    fetched_at_ms: projectNullableEpochMilliseconds(r.fetched_at_ms),
    last_error: r.last_error === undefined ? '' : text(r.last_error),
  }
}
function source(value: unknown): AnnouncementSourceDto {
  const r = projectRecord(value)
  assertNoSecretLikeFields(r, ['group', 'site_root_url', 'site_name', 'announcement', 'notice'])
  return {
    group: group(r.group),
    site_root_url: projectString(r.site_root_url),
    site_name: text(r.site_name),
    announcement: sourceState(r.announcement),
    notice: sourceState(r.notice),
  }
}
export async function getAnnouncements(
  client: ApiClient,
  filters: { site?: string; q?: string } = {},
  signal?: AbortSignal,
): Promise<AnnouncementsResponseDto> {
  const params = new URLSearchParams()
  params.set('include', 'source_status')
  if (filters.site) params.set('site', filters.site)
  if (filters.q) params.set('q', filters.q)
  const query = params.toString()
  return projectAnnouncementsResponse(
    await client.request(`/api/announcements${query ? `?${query}` : ''}`, {
      method: 'GET',
      signal,
    }),
  )
}

/**
 * The feed is filled by the background site sync (each group is refreshed on demand
 * and then once per hour) and the response carries no sync status, so an open page
 * cannot tell "sync still running" from "nothing to show". A bounded fast window
 * would park a still-empty page on the slow cadence and miss later sync results, so
 * poll on one steady cadence instead: an empty page picks up the first announcement,
 * and a populated page picks up new announcements from every periodic sync. The
 * interval only runs while the tab is focused; focus also refetches immediately.
 */
export const announcementsRefetchIntervalMs = 30_000

export function announcementsQueryOptions(
  client: ApiClient,
  filters: { site?: string; q?: string } = {},
) {
  return {
    queryKey: controlQueryKeys.announcements(filters),
    queryFn: ({ signal }: QueryFunctionContext) => getAnnouncements(client, filters, signal),
    // Returning to a permanently open page must not wait for the next poll.
    refetchOnWindowFocus: 'always' as const,
    refetchIntervalInBackground: false,
    refetchInterval: announcementsRefetchIntervalMs,
  }
}
