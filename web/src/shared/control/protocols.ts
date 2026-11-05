export const protocolCatalog = [
  {
    value: 'openai-completions',
    supportsProtocolOnlyRouting: false,
  },
  {
    value: 'openai-responses',
    supportsProtocolOnlyRouting: true,
  },
  {
    value: 'openai-images',
    supportsProtocolOnlyRouting: false,
  },
  {
    value: 'openai-embeddings',
    supportsProtocolOnlyRouting: false,
  },
  {
    value: 'rerank',
    supportsProtocolOnlyRouting: false,
  },
  {
    value: 'anthropic',
    supportsProtocolOnlyRouting: false,
  },
  {
    value: 'gemini',
    supportsProtocolOnlyRouting: false,
  },
  {
    value: 'gemini-embeddings',
    supportsProtocolOnlyRouting: false,
  },
] as const

export type ProtocolValue = (typeof protocolCatalog)[number]['value']

export const knownAccessProtocols = protocolCatalog.map(({ value }) => value) as ProtocolValue[]
export const enabledDataProtocols = knownAccessProtocols
