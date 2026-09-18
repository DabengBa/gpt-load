import assert from 'node:assert/strict'

function fakeStorage() {
  const map = new Map()
  return {
    getItem: (key) => (map.has(key) ? map.get(key) : null),
    setItem: (key, value) => void map.set(key, String(value)),
    removeItem: (key) => void map.delete(key),
  }
}

const baseCollectionItem = {
  id: 7,
  name: 'alpha',
  price_multiplier: '1',
  channel_id: 'openai_compatible',
  connection_type: 'api_key',
  params: { base_url: 'https://alpha.example/v1' },
  status: 'available',
  model_count: 1,
  client_model_count: 1,
  credential_counts: { total: 1, available: 1, cooldown: 0, blacklisted: 0, disabled: 0 },
}

function collectionResponse(pageSize, items, totalItems) {
  return {
    observed_at_ms: 1_700_000_000_000,
    summary: { total: totalItems, available: totalItems, unavailable: 0, disabled: 0 },
    items,
    pagination: {
      page: 1,
      page_size: pageSize,
      total_items: totalItems,
      total_pages: totalItems === 0 ? 0 : Math.ceil(totalItems / pageSize),
    },
  }
}

function newImportDraft() {
  return {
    mode: 'new',
    channel_id: 'openai_compatible',
    connection_type: 'api_key',
    params: { base_url: 'https://alpha.example/v1' },
    proxy: { mode: 'inherit', url: '' },
    name: 'alpha',
    price_multiplier: '1',
    provider_url: 'https://provider.example',
    credentials: 'sk-one',
    staged_credentials: [],
    models: [
      {
        id: 'gpt-4o',
        name: 'GPT-4o',
        sources: ['catalog'],
        pricing_status: 'configured',
        alias: '',
        alias_enabled: false,
        key: 1,
      },
    ],
  }
}

export function runGroupCollectionContractTests({
  parseGroupCollectionRouteQuery,
  normalizeGroupCollectionFilters,
  projectGroupCollection,
  createImportRecoveryService,
  importRecoveryStorageKey,
  importRecoveryTtlMs,
}) {
  // 分组清单默认每页 100 条：路由解析、查询键归一化、响应投影三处一致。
  assert.equal(parseGroupCollectionRouteQuery({}).page_size, 100)
  assert.equal(
    normalizeGroupCollectionFilters({ sort: 'recent', page: 1, page_size: 100 }).page_size,
    100,
  )
  assert.equal(
    normalizeGroupCollectionFilters({ sort: 'recent', page: 2, page_size: 100, q: ' x ' })
      .page_size,
    100,
  )

  const linked = { ...baseCollectionItem, provider_url: 'https://provider.example' }
  const unlinked = { ...baseCollectionItem, id: 8, name: 'zulu', provider_url: null }
  const projected = projectGroupCollection(collectionResponse(100, [linked, unlinked], 2))
  assert.equal(projected.pagination.page_size, 100)
  assert.equal(projected.items[0].provider_url, 'https://provider.example')
  assert.equal(projected.items[1].provider_url, null)
  assert.throws(
    () =>
      projectGroupCollection(
        collectionResponse(100, [{ ...baseCollectionItem, provider_url: 'ftp://bad' }], 1),
      ),
    /INVALID_API_RESPONSE/u,
  )
  assert.throws(
    () => projectGroupCollection(collectionResponse(100, [{ ...baseCollectionItem }], 1)),
    /INVALID_API_RESPONSE/u,
    'collection items must carry an explicit provider_url field',
  )

  // 导入草稿恢复到 v9:new 模式草稿携带 provider_url,v8 记录迁移后补空串。
  const storage = fakeStorage()
  const service = createImportRecoveryService({
    storage,
    now: () => 1_000,
    setTimer: () => 0,
    clearTimer: () => {},
  })
  const draft = newImportDraft()
  service.register(() => draft)
  assert.equal(service.captureForUnauthorized(), 'stored')
  assert.deepEqual(service.consume(), draft)

  const legacyDraft = newImportDraft()
  delete legacyDraft.provider_url
  storage.setItem(
    importRecoveryStorageKey,
    JSON.stringify({ version: 8, expires_at: 1_000 + importRecoveryTtlMs, draft: legacyDraft }),
  )
  assert.deepEqual(service.consume(), { ...legacyDraft, provider_url: '' })
}
