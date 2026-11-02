import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'

import { channelsQueryOptions, type ChannelDto } from '@shared/control/resources/channels'

import { useAppServices } from '../../app/services'

/**
 * Channel catalog keyed by `channel_id` so table rows can resolve an icon and
 * a display name without a per-row lookup over the whole list.
 */
export function useGroupChannels(): Record<string, ChannelDto> {
  const { apiClient } = useAppServices()
  const channelsQuery = useQuery(channelsQueryOptions(apiClient, ''))
  return useMemo<Record<string, ChannelDto>>(
    () =>
      Object.fromEntries(
        (channelsQuery.data?.items ?? []).map((channel) => [channel.channel_id, channel]),
      ),
    [channelsQuery.data],
  )
}
