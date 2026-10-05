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
  // Legacy route kept only as a redirect target into Settings → credentials;
  // the beforeLoad guard navigates away before any view or catalog loads.
  'access-keys': {
    titleKey: 'shell.settings',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'settings',
  },
  monitor: {
    titleKey: 'shell.monitor',
    requiresAuth: true,
    primaryNav: 'monitor',
    messageNamespaces: ['monitor'],
  },
  schedule: {
    titleKey: 'shell.schedule',
    requiresAuth: true,
    primaryNav: 'schedule',
    // Both principals reach the schedule page; the price deep link inside the
    // detail surfaces needs the model and price catalogs loaded up front.
    messageNamespaces: ['monitor', 'models', 'model-prices'],
  },
  logs: {
    titleKey: 'shell.logs',
    requiresAuth: true,
    primaryNav: 'logs',
    messageNamespaces: ['monitor'],
  },
  settings: {
    titleKey: 'shell.settings',
    requiresAuth: true,
    adminOnly: true,
    primaryNav: 'settings',
    // 'access-keys' — the credentials section reuses the access-key drawer and
    // collection strings.
    messageNamespaces: ['settings', 'model-prices', 'import', 'access-keys'],
  },
})

export function pageRouteMetaFor(name: string): PageRouteMeta {
  const meta = pageRouteMeta[name]
  if (meta === undefined) {
    throw new Error(`page route meta missing for manifest route "${name}"`)
  }
  return meta
}
