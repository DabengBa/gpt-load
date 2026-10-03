// Schedule → price identity mapping. Fuzzy client-model search pages are
// narrowed to an exact client model, then the price is identified by
// channel_id + upstream model_id — never invented from a sibling match.

export interface SchedulePriceTarget {
  clientModel: string
  channelID: string
  upstreamModelID: string
}

export interface SchedulePricePage {
  items: ReadonlyArray<{
    client_model: string
    upstream_models: ReadonlyArray<{ model_id: string; price: { id: number; channel_id: string } }>
  }>
  pagination: { page: number; total_pages: number }
}

export type SchedulePricePageFetcher = (page: number) => Promise<SchedulePricePage>

export async function resolveSchedulePriceID(
  target: SchedulePriceTarget,
  fetchPage: SchedulePricePageFetcher,
): Promise<number | undefined> {
  const clientModel = target.clientModel.trim()
  if (clientModel === '') return undefined
  let page = 1
  for (;;) {
    const data = await fetchPage(page)
    const exact = data.items.find((entry) => entry.client_model === clientModel)
    if (exact !== undefined) {
      const upstream = exact.upstream_models.find(
        (candidate) =>
          candidate.model_id === target.upstreamModelID &&
          candidate.price.channel_id === target.channelID,
      )
      return upstream?.price.id
    }
    if (data.items.length === 0 || page >= data.pagination.total_pages) return undefined
    page += 1
  }
}
