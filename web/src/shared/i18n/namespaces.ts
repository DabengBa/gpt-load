// Lazy-loadable message namespaces shared by both frontends; the classic
// i18n runtime and the Astryx route-meta table both consume this list.
export const messageNamespaces = [
  'import',
  'group',
  'access-keys',
  'monitor',
  'models',
  'model-prices',
  'settings',
] as const

export type MessageNamespace = (typeof messageNamespaces)[number]
