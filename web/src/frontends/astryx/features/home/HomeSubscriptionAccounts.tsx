import * as stylex from '@stylexjs/stylex'

import type { HomeSubscriptionAccountsDto } from '@shared/control/resources/home'

import { useT } from '../../app/i18n'
import { HomeSectionHeading } from './home-chrome'
import { SubscriptionAccountMiniCard } from './SubscriptionAccountMiniCard'

const styles = stylex.create({
  section: {
    display: 'grid',
    gap: 'var(--space-3)',
    marginTop: 36,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 20,
  },
  // Backend caps at 4 accounts; min 232px per column fills a desktop row
  // exactly, auto-fill wraps smaller widths.
  row: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(232px, 1fr))',
    gap: 10,
  },
})

export function HomeSubscriptionAccounts({ accounts }: { accounts: HomeSubscriptionAccountsDto }) {
  const t = useT()
  if (accounts.items.length === 0) return null
  return (
    <section {...stylex.props(styles.section)} aria-labelledby="home-subscription-accounts-title">
      <HomeSectionHeading
        id="home-subscription-accounts-title"
        title={t('home.ledger.subscriptionAccounts.title')}
      />
      <div {...stylex.props(styles.row)}>
        {accounts.items.map((account) => (
          <SubscriptionAccountMiniCard
            key={`${account.channel_id}-${account.credential.credential_id}`}
            account={account}
          />
        ))}
      </div>
    </section>
  )
}
