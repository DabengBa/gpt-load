import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import { ArrowRight, CircleCheck, KeyRound } from 'lucide-react'

import type { HomeBaseDto } from '@shared/control/resources/home'
import type { ReleaseUpdateDto } from '@shared/control/resources/system-update'
import type { MessageId } from '@shared/i18n/message-ids'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../../app/i18n'
import { HomeReleaseUpdateLink } from './home-chrome'

const NARROW = '@media (max-width: 860px)'
const TIGHT = '@media (max-width: 560px)'

const styles = stylex.create({
  welcome: {
    minWidth: 0,
  },
  header: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(0, 1fr) auto',
      [NARROW]: '1fr',
    },
    minHeight: 72,
    alignItems: {
      default: 'center',
      [NARROW]: 'start',
    },
    gap: 36,
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    paddingBottom: 'var(--space-5)',
  },
  titleRow: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    gap: 'var(--space-1)',
  },
  title: {
    maxWidth: 'none',
    margin: 0,
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--title-lede)',
    fontWeight: 500,
    letterSpacing: '-0.015em',
    lineHeight: 'var(--line-compact)',
  },
  action: {
    whiteSpace: 'nowrap',
    justifySelf: {
      default: 'auto',
      [NARROW]: 'start',
    },
  },
  guide: {
    paddingTop: 28,
  },
  description: {
    maxWidth: '45rem',
    marginTop: 0,
    marginBottom: 24,
    marginInline: 0,
    color: 'var(--color-text-muted)',
    fontSize: 14,
    lineHeight: 'var(--line-relaxed)',
  },
  guideHeader: {
    display: 'flex',
    alignItems: {
      default: 'baseline',
      [TIGHT]: 'start',
    },
    flexDirection: {
      default: 'row',
      [TIGHT]: 'column',
    },
    justifyContent: 'space-between',
    gap: {
      default: 'var(--space-4)',
      [TIGHT]: 3,
    },
    marginBottom: 10,
  },
  guideTitle: {
    margin: 0,
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--title-section)',
    fontWeight: 500,
  },
  estimate: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
  },
  steps: {
    display: 'grid',
    margin: 0,
    padding: 0,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    listStyle: 'none',
  },
  step: {
    display: 'grid',
    gridTemplateColumns: {
      default: '46px minmax(0, 1fr) auto',
      [TIGHT]: '34px minmax(0, 1fr)',
    },
    gap: 14,
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingTop: 17,
    paddingBottom: 17,
    paddingInline: 4,
  },
  stepNumber: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
  },
  stepTitle: {
    margin: 0,
    fontSize: 'var(--text-md)',
    fontWeight: 600,
  },
  stepDescription: {
    margin: 0,
    marginTop: 3,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
    lineHeight: 1.6,
  },
  stepArrow: {
    color: 'var(--color-text-faint)',
    display: {
      default: 'block',
      [TIGHT]: 'none',
    },
  },
  note: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 'var(--space-2)',
    margin: '18px 0 0',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 8,
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text-muted)',
    paddingTop: 11,
    paddingBottom: 11,
    paddingInline: 13,
    fontSize: 'var(--text-sm)',
    lineHeight: 1.6,
  },
  noteIcon: {
    flex: 'none',
    marginTop: 1,
    color: 'var(--color-success)',
  },
})

const stepIds = [1, 2, 3] as const

export function HomeWelcome({
  base,
  update,
}: {
  base: HomeBaseDto
  update: ReleaseUpdateDto | null
}) {
  const t = useT()

  return (
    <section {...stylex.props(styles.welcome)} aria-labelledby="home-title">
      <header {...stylex.props(styles.header)}>
        <div {...stylex.props(styles.titleRow)}>
          <h1 id="home-title" {...stylex.props(styles.title)}>
            {t('home.ledger.welcomeTitle')}
          </h1>
          {update !== null && (
            <HomeReleaseUpdateLink currentVersion={base.version} update={update} />
          )}
        </div>
        <Button
          variant="primary"
          label={t('home.ledger.importCredentials')}
          icon={<KeyRound size={16} aria-hidden="true" />}
          xstyle={styles.action}
          onClick={() =>
            // The import route is classic-owned: document navigation so the
            // server-side frontend selection runs.
            window.location.assign(`${pagePath('import')}?mode=new`)
          }
        />
      </header>

      <section {...stylex.props(styles.guide)} aria-labelledby="home-welcome-guide-title">
        <p {...stylex.props(styles.description)}>{t('home.ledger.welcomeDescription')}</p>
        <div {...stylex.props(styles.guideHeader)}>
          <h2 id="home-welcome-guide-title" {...stylex.props(styles.guideTitle)}>
            {t('home.ledger.welcomeGuideTitle')}
          </h2>
          <span {...stylex.props(styles.estimate)}>{t('home.ledger.welcomeEstimatedTime')}</span>
        </div>
        <ol {...stylex.props(styles.steps)}>
          {stepIds.map((step) => (
            <li key={step} {...stylex.props(styles.step)}>
              <span {...stylex.props(styles.stepNumber)} aria-hidden="true">
                0{step}
              </span>
              <div>
                <h3 {...stylex.props(styles.stepTitle)}>
                  {t(`home.ledger.welcomeStep${step}Title` as MessageId)}
                </h3>
                <p {...stylex.props(styles.stepDescription)}>
                  {t(`home.ledger.welcomeStep${step}Description` as MessageId)}
                </p>
              </div>
              <ArrowRight {...stylex.props(styles.stepArrow)} size={16} aria-hidden="true" />
            </li>
          ))}
        </ol>
        <p {...stylex.props(styles.note)}>
          <CircleCheck {...stylex.props(styles.noteIcon)} size={15} aria-hidden="true" />
          {t('home.ledger.welcomeSecurityNote')}
        </p>
      </section>
    </section>
  )
}
