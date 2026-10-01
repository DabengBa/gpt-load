// Message id union derived from the en-US catalogs' flattened key paths.
// `import type` + `typeof` keeps this fully erased — no catalog bytes enter
// any bundle through this module. en-US is the source of truth because the
// runtime overlays locale catalogs on top of en-US, so a key missing from
// zh-CN/ja-JP still resolves.
import type enAccessKeys from './locales/en-US/access-keys'
import type enCore from './locales/en-US/core'
import type enGroup from './locales/en-US/group'
import type enImport from './locales/en-US/import'
import type enModelPrices from './locales/en-US/model-prices'
import type enModels from './locales/en-US/models'
import type enMonitor from './locales/en-US/monitor'
import type enSettings from './locales/en-US/settings'

type FlatKeys<T> = {
  [K in keyof T & (string | number)]: T[K] extends string
    ? `${K}`
    : T[K] extends object
      ? `${K}.${FlatKeys<T[K]>}`
      : never
}[keyof T & (string | number)]

export type MessageId =
  | FlatKeys<typeof enCore>
  | FlatKeys<typeof enAccessKeys>
  | FlatKeys<typeof enGroup>
  | FlatKeys<typeof enImport>
  | FlatKeys<typeof enModelPrices>
  | FlatKeys<typeof enModels>
  | FlatKeys<typeof enMonitor>
  | FlatKeys<typeof enSettings>
