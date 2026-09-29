import * as stylex from '@stylexjs/stylex'
import { KeyRound, TriangleAlert } from 'lucide-react'

import type { AccessKeyReferenceDto } from '@shared/control/resources/groups'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../../app/i18n'
import { RouteLink } from '../../../app/route-link'

const styles = stylex.create({
  root: {
    display: 'grid',
    gridTemplateColumns: 'auto minmax(0, 1fr)',
    gap: 'var(--space-3)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor:
      'color-mix(in srgb, var(--color-warning) 38%, var(--color-border-subtle))',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
    padding: 'var(--space-3)',
  },
  strong: { margin: 0 },
  paragraph: {
    margin: 0,
    marginTop: 'var(--space-1)',
  },
  list: {
    display: 'grid',
    gap: 'var(--space-2)',
    marginBlock: 'var(--space-3)',
    padding: 0,
    listStyle: 'none',
  },
  item: {
    display: 'flex',
    minHeight: '28px',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-text)',
  },
  code: {
    color: 'var(--color-text-muted)',
    fontFamily: 'var(--font-mono)',
  },
  manageLink: {
    minHeight: '44px',
    display: 'inline-flex',
    alignItems: 'center',
  },
})

export function GroupInUseFeedback({ references }: { references: AccessKeyReferenceDto[] }) {
  const t = useT()
  return (
    <div {...stylex.props(styles.root)} role="alert">
      <TriangleAlert size={18} aria-hidden="true" />
      <div>
        <strong {...stylex.props(styles.strong)}>
          {t('group.settings.delete.inUseTitle')}
        </strong>
        <p {...stylex.props(styles.paragraph)}>
          {t('group.settings.delete.inUseDescription')}
        </p>
        <ul {...stylex.props(styles.list)}>
          {references.map((reference) => (
            <li key={reference.id} {...stylex.props(styles.item)}>
              <KeyRound size={15} aria-hidden="true" />
              <span>{reference.name}</span>
              <code {...stylex.props(styles.code)}>#{reference.id}</code>
            </li>
          ))}
        </ul>
        <RouteLink to={pagePath('access-keys')} {...stylex.props(styles.manageLink)}>
          {t('group.settings.delete.manageAccessKeys')}
        </RouteLink>
      </div>
    </div>
  )
}
