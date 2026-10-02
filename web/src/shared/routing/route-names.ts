import { pageRouteEntries } from './page-routes'

export const sharedPageRouteNames = {
  home: 'home',
  login: 'login',
  import: 'import',
  groups: 'groups',
  groupDetail: 'group-detail',
  accessKeys: 'access-keys',
  monitor: 'monitor',
  schedule: 'schedule',
  logs: 'logs',
  settings: 'settings',
} as const

function validateSharedPageRouteNames(): void {
  const manifestNames = new Set(pageRouteEntries.map((route) => route.name))
  const locationNames = Object.values(sharedPageRouteNames)
  if (
    manifestNames.size !== locationNames.length ||
    locationNames.some((name) => !manifestNames.has(name))
  ) {
    throw new Error('Page route locations must cover the shared page route manifest')
  }
}

validateSharedPageRouteNames()
