import * as stylex from '@stylexjs/stylex'
import { Badge, Button, Skeleton } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { TriangleAlert } from 'lucide-react'

import { systemInfoQueryOptions } from '@shared/control/resources/system-info'

import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { AgentCredentialsPanel } from './AgentCredentialsPanel'
import { AccessKeysPanel } from './AccessKeysPanel'
import { SettingsSectionFrame } from './section-tools'

const styles = stylex.create({
  subsections: {
    display: 'grid',
    gap: 'var(--space-5)',
  },
  divider: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 'var(--space-5)',
  },
  heading: {
    margin: 0,
    fontSize: 'var(--text-meta)',
    fontWeight: 650,
  },
  headingDescription: {
    margin: 0,
    marginTop: 'var(--space-1)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  adminBlock: {
    display: 'grid',
    gap: 'var(--space-3)',
  },
  adminRow: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  adminLabel: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  adminPath: {
    color: 'var(--color-code)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    overflowWrap: 'anywhere',
  },
  adminError: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-danger)',
    fontSize: 'var(--text-sm)',
  },
  adminNote: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    lineHeight: 1.55,
  },
})

// Administrator sign-in key status — deployment-managed, so this block only
// reports the configured source (and key-file path) from /api/system/info; the
// secret itself is never rendered.
function AdminKeyBlock() {
  const t = useT()
  const { apiClient } = useAppServices()
  const systemInfo = useQuery(systemInfoQueryOptions(apiClient))
  const authKey = systemInfo.data?.auth_key

  return (
    <div {...stylex.props(styles.adminBlock)}>
      <div>
        <h3 {...stylex.props(styles.heading)}>{t('settings.credentials.adminTitle')}</h3>
        <p {...stylex.props(styles.headingDescription)}>
          {t('settings.credentials.adminDescription')}
        </p>
      </div>
      {systemInfo.isPending ? (
        <Skeleton height={28} radius={2} />
      ) : systemInfo.isError || authKey === undefined ? (
        <div {...stylex.props(styles.adminError)} role="alert">
          <TriangleAlert size={15} aria-hidden />
          <span>{t('settings.credentials.adminLoadFailed')}</span>
          <Button
            variant="secondary"
            size="sm"
            label={t('common.retry')}
            onClick={() => void systemInfo.refetch()}
          />
        </div>
      ) : (
        <>
          <div {...stylex.props(styles.adminRow)}>
            <span {...stylex.props(styles.adminLabel)}>
              {t('settings.credentials.adminSource')}
            </span>
            <Badge
              variant="neutral"
              label={t(
                authKey.source === 'environment'
                  ? 'settings.system.sources.environment'
                  : 'settings.system.sources.key_file',
              )}
            />
            {authKey.path !== null && (
              <code {...stylex.props(styles.adminPath)}>{authKey.path}</code>
            )}
          </div>
          <p {...stylex.props(styles.adminNote)}>{t('settings.credentials.adminNote')}</p>
        </>
      )}
    </div>
  )
}

/**
 * "Keys and access" settings section: read-only administrator-key status, the
 * client access-key collection/drawer (moved off the standalone /access-keys
 * page), and Agent credential management. Mutations here are independent of
 * the runtime-settings draft — every action submits immediately.
 */
export function CredentialsSection() {
  const t = useT()
  return (
    <SettingsSectionFrame
      id="settings-credentials"
      title={t('settings.credentials.title')}
      description={t('settings.credentials.description')}
    >
      <div {...stylex.props(styles.subsections)}>
        <AdminKeyBlock />
        <div {...stylex.props(styles.divider)}>
          <AccessKeysPanel />
        </div>
        <div {...stylex.props(styles.divider)}>
          <AgentCredentialsPanel />
        </div>
      </div>
    </SettingsSectionFrame>
  )
}
