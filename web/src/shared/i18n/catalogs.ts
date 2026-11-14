import type { AppLocale } from '../preferences/locale'
import type { MessageNamespace } from './namespaces'

// Nested catalog trees as authored; the Astryx runtime flattens them into the
// react-intl flat dict — the key paths stay identical either way.
export type MessageTree = { [key: string]: string | MessageTree }
export type MessageLoader = () => Promise<{ default: MessageTree }>

export const coreLoaders: Record<AppLocale, MessageLoader> = {
  'zh-CN': () => import('./locales/zh-CN/core'),
  'en-US': () => import('./locales/en-US/core'),
}

export const namespaceLoaders: Record<MessageNamespace, Record<AppLocale, MessageLoader>> = {
  import: {
    'zh-CN': () => import('./locales/zh-CN/import'),
    'en-US': () => import('./locales/en-US/import'),
  },
  group: {
    'zh-CN': () => import('./locales/zh-CN/group'),
    'en-US': () => import('./locales/en-US/group'),
  },
  'access-keys': {
    'zh-CN': () => import('./locales/zh-CN/access-keys'),
    'en-US': () => import('./locales/en-US/access-keys'),
  },
  monitor: {
    'zh-CN': () => import('./locales/zh-CN/monitor'),
    'en-US': () => import('./locales/en-US/monitor'),
  },
  models: {
    'zh-CN': () => import('./locales/zh-CN/models'),
    'en-US': () => import('./locales/en-US/models'),
  },
  'model-prices': {
    'zh-CN': () => import('./locales/zh-CN/model-prices'),
    'en-US': () => import('./locales/en-US/model-prices'),
  },
  settings: {
    'zh-CN': () => import('./locales/zh-CN/settings'),
    'en-US': () => import('./locales/en-US/settings'),
  },
}

export function catalogLoader(
  locale: AppLocale,
  namespace: 'core' | MessageNamespace,
): MessageLoader {
  return namespace === 'core' ? coreLoaders[locale] : namespaceLoaders[namespace][locale]
}

// Flattens a nested catalog into react-intl's `messages` shape:
// { common: { retry: 'Retry' } } -> { 'common.retry': 'Retry' }.
export function flattenMessages(
  tree: MessageTree,
  prefix = '',
  out: Record<string, string> = {},
): Record<string, string> {
  for (const [key, value] of Object.entries(tree)) {
    const path = prefix === '' ? key : `${prefix}.${key}`
    if (typeof value === 'string') {
      out[path] = value
    } else {
      flattenMessages(value, path, out)
    }
  }
  return out
}
