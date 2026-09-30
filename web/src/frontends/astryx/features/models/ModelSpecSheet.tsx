import * as stylex from '@stylexjs/stylex'
import { useIntl } from 'react-intl'

import type { ModelCatalogReferenceDto } from '@shared/control/resources/models'
import { formatInteger } from '@shared/lib/format'

import { useT } from '../../app/i18n'

const styles = stylex.create({
  root: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1-75)',
  },
  source: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  specs: {
    display: 'flex',
    alignItems: 'baseline',
    flexWrap: 'wrap',
    gap: 'var(--space-1) var(--space-3-5)',
    margin: 0,
    fontSize: 'var(--text-sm)',
  },
  specItem: {
    display: 'inline-flex',
    minWidth: 0,
    alignItems: 'baseline',
    gap: 'var(--space-1)',
  },
  specTerm: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  specValue: {
    margin: 0,
    fontFamily: 'var(--font-mono)',
  },
  capabilities: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
    margin: 0,
    padding: 0,
    listStyle: 'none',
  },
  capability: {
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-tag)',
    color: 'var(--color-text-muted)',
    paddingBlock: '2px',
    paddingInline: '7px',
    fontSize: 'var(--text-label-xs)',
  },
  status: {
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
  },
})

const knownModalities = new Set(['audio', 'embedding', 'image', 'pdf', 'text', 'video'])
const knownStatuses = new Set(['alpha', 'beta', 'deprecated'])

export function ModelSpecSheet({ reference }: { reference: ModelCatalogReferenceDto }) {
  const t = useT()
  const intl = useIntl()
  const model = reference.model

  const sourceLabel = t(`models.detail.catalogReference.${reference.source}`, {
    provider: reference.provider_name,
  })
  /** 目录来源与模型名同属一条溯源信息,合并成一行避免占两行。 */
  const sourceLine = `${sourceLabel} · ${model.name}`

  const capabilityKeys = [
    'reasoning',
    'tool_call',
    'structured_output',
    'attachment',
    'temperature',
  ] as const
  const capabilities = capabilityKeys
    .filter((capability) => model.capabilities[capability] === true)
    .map((capability) => t(`models.detail.capabilities.${capability}`))
    .concat(model.open_weights === true ? [t('models.detail.openWeights')] : [])

  /** 已知 Models.dev 枚举本地化;新值保留原文,避免目录扩展导致信息丢失。 */
  const catalogStatus =
    !model.status || !knownStatuses.has(model.status)
      ? model.status
      : t(`models.detail.status.${model.status as 'alpha' | 'beta' | 'deprecated'}`)

  const modalityLabel = (modality: string): string =>
    knownModalities.has(modality)
      ? t(
          `models.detail.modalities.${modality as 'audio' | 'embedding' | 'image' | 'pdf' | 'text' | 'video'}`,
        )
      : modality

  const formatModalities = (): string => {
    const { input, output } = model.modalities
    if (input.length === 0 && output.length === 0) return ''
    const inputLabel = input.map(modalityLabel).join(' / ')
    const outputLabel = output.map(modalityLabel).join(' / ')
    const arrow = outputLabel ? `→ ${outputLabel}` : ''
    return `${inputLabel} ${arrow}`.trim()
  }

  const specs: { key: string; label: string; value: string }[] = []
  const push = (key: 'context' | 'maxInput' | 'maxOutput' | 'modalities', value: string): void => {
    if (value) specs.push({ key, label: t(`models.detail.specs.${key}`), value })
  }
  push(
    'context',
    model.limits.context === null ? '' : formatInteger(model.limits.context, intl.locale),
  )
  push(
    'maxInput',
    model.limits.input === null ? '' : formatInteger(model.limits.input, intl.locale),
  )
  push(
    'maxOutput',
    model.limits.output === null ? '' : formatInteger(model.limits.output, intl.locale),
  )
  push('modalities', formatModalities())

  return (
    <div {...stylex.props(styles.root)}>
      <p {...stylex.props(styles.source)}>{sourceLine}</p>
      {specs.length > 0 && (
        <dl {...stylex.props(styles.specs)}>
          {specs.map((spec) => (
            <div key={spec.key} {...stylex.props(styles.specItem)}>
              <dt {...stylex.props(styles.specTerm)}>{spec.label}</dt>
              <dd {...stylex.props(styles.specValue)}>{spec.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {(capabilities.length > 0 || catalogStatus) && (
        <ul {...stylex.props(styles.capabilities)}>
          {capabilities.map((capability) => (
            <li key={capability} {...stylex.props(styles.capability)}>
              {capability}
            </li>
          ))}
          {catalogStatus && (
            <li {...stylex.props(styles.capability, styles.status)}>{catalogStatus}</li>
          )}
        </ul>
      )}
    </div>
  )
}
