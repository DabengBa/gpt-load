import type { MessageId } from '../i18n/message-ids'
import type { MessageNamespace } from '../i18n/namespaces'

// Per-route behavior table. The Astryx TanStack adapter reads it for
// beforeLoad guards and staticData. Keys are manifest route names (page_routes.json);
// coverage is asserted by scripts/astryx-routes.test.ts.
// `type` (not `interface`) so the value keeps an implicit index signature for
// the TanStack adapter's staticData record.
export type PageRouteMeta = {
  readonly titleKey?: MessageId
  readonly requiresAuth?: boolean
  readonly adminOnly?: boolean
  readonly primaryNav?: string
  readonly messageNamespaces?: readonly MessageNamespace[]
}

export const pageRouteMeta: Readonly<Record<string, PageRouteMeta>> = Object.freeze({
  home: {
    titleKey: 'home.ledger.title',
    requiresAuth: true,
    primaryNav: 'home',
    messageNamespaces: ['access-keys', 'group'],
  },
  login: {},
  import: {
    titleKey: 'shell.import',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'groups',
    // `group.*` — ImportConnectionSection shares the provider-website field
    // copy with the group settings form (classic does the same cross-ref).
    messageNamespaces: ['import', 'group'],
  },
  groups: {
    titleKey: 'groups.title',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'groups',
    messageNamespaces: ['group'],
  },
  // 模型 Tab 的「测活」入口与 ModelProbeDialog 用的是 monitor.modelProbe.* 文案，
  // 命名空间必须在这里声明；少一个就会在页面上渲染出原始 key。
  'group-detail': {
    titleKey: 'shell.groupDetail',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'groups',
    messageNamespaces: ['group', 'import', 'monitor'],
  },
  'access-keys': {
    titleKey: 'shell.accessKeys',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'access-keys',
    messageNamespaces: ['access-keys'],
  },
  // InspectorTab 的 route strategy 文案来自 settings.runtime.routeStrategies.*
  // （跨域引用 settings 命名空间），必须随 monitor 一起装载。
  monitor: {
    titleKey: 'shell.monitor',
    requiresAuth: true,
    primaryNav: 'monitor',
    messageNamespaces: ['monitor', 'settings'],
  },
  schedule: {
    titleKey: 'shell.schedule',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'schedule',
    messageNamespaces: ['monitor'],
  },
  logs: {
    titleKey: 'shell.logs',
    requiresAuth: true,
    primaryNav: 'logs',
    messageNamespaces: ['monitor'],
  },
  models: {
    titleKey: 'models.title',
    requiresAuth: true,
    primaryNav: 'models',
    messageNamespaces: ['models', 'model-prices'],
  },
  settings: {
    titleKey: 'shell.settings',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'settings',
    messageNamespaces: ['settings', 'model-prices', 'import'],
  },
})

export function pageRouteMetaFor(name: string): PageRouteMeta {
  const meta = pageRouteMeta[name]
  if (meta === undefined) {
    throw new Error(`page route meta missing for manifest route "${name}"`)
  }
  return meta
}
