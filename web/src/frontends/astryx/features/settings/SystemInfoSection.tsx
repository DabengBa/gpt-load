import { Badge, Button, Skeleton } from '@astryxdesign/core'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import * as stylex from '@stylexjs/stylex'
import { RefreshCw, TriangleAlert } from 'lucide-react'
import { useState } from 'react'

import { CopyButton } from '../../components/CopyButton'
import { useT } from '../../app/i18n'
import { useAppServices } from '../../app/services'
import { useStableLoading } from '../../app/collection-loading'
import { controlQueryKeys } from '@shared/control/query-keys'
import {
  systemInfoQueryOptions,
  type DatabaseDriver,
  type SecretSource,
} from '@shared/control/resources/system-info'
import {
  getSystemUpdate,
  systemUpdateQueryOptions,
  type ReleaseUpdateDto,
} from '@shared/control/resources/system-update'
import { RequestCancelledError } from '@shared/http/errors'
import { SettingsSectionFrame } from './section-tools'

const narrow = '@media (max-width: 760px)'

const spin = stylex.keyframes({
  to: { transform: 'rotate(360deg)' },
})

const styles = stylex.create({
  panel: {
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    paddingBlock: 'var(--space-4)',
    paddingInline: 'var(--space-4)',
  },
  definition: {
    display: 'grid',
    margin: 0,
  },
  row: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'minmax(120px, 160px) minmax(0, 1fr)',
      [narrow]: '115px minmax(0, 1fr)',
    },
    columnGap: 'var(--space-4)',
  },
  term: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    borderBottomWidth: '1px',
    borderBottomStyle: 'dashed',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBlock: '10px',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 650,
  },
  termLast: {
    borderBottomWidth: 0,
  },
  detail: {
    display: 'flex',
    minWidth: 0,
    margin: 0,
    alignItems: 'center',
    borderBottomWidth: '1px',
    borderBottomStyle: 'dashed',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBlock: '10px',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
    overflowWrap: 'anywhere',
  },
  detailLast: {
    borderBottomWidth: 0,
  },
  inline: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  path: {
    display: 'grid',
    gridTemplateColumns: 'minmax(0, 1fr) auto',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  pathText: {
    overflowWrap: 'anywhere',
  },
  mono: {
    fontFamily: 'var(--font-mono)',
  },
  version: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-3)',
  },
  updateControls: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  updateResult: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 'var(--space-1)',
    minWidth: 0,
  },
  updateInfo: { color: 'var(--color-action)' },
  updateSuccess: { color: 'var(--color-success)' },
  updateWarning: { color: 'var(--color-warning)' },
  updateDanger: { color: 'var(--color-danger)' },
  updateLink: {
    color: 'inherit',
    fontWeight: 650,
    textDecoration: 'underline',
    textUnderlineOffset: '2px',
  },
  refreshIcon: {
    animationName: {
      default: spin,
      '@media (prefers-reduced-motion: reduce)': 'none',
    },
    animationDuration: '900ms',
    animationTimingFunction: 'linear',
    animationIterationCount: 'infinite',
  },
  errorBox: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-danger)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-danger-bg)',
    color: 'var(--color-danger)',
    paddingBlock: 'var(--space-3)',
    paddingInline: 'var(--space-4)',
    fontSize: 'var(--text-sm)',
  },
  staleBanner: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    color: 'var(--color-warning)',
    fontSize: 'var(--text-sm)',
  },
  securityNote: {
    margin: 0,
    marginTop: 'var(--space-1)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
})

type UpdateCheckState =
  | { kind: 'checking' }
  | { kind: 'latest' }
  | { kind: 'available'; update: ReleaseUpdateDto }
  | { kind: 'failed' }

export function SystemInfoSection() {
  const t = useT()
  const { apiClient } = useAppServices()
  const queryClient = useQueryClient()
  const infoQuery = useQuery(systemInfoQueryOptions(apiClient))
  const initialLoading = useStableLoading(infoQuery.isPending && infoQuery.data === undefined)
  const infoRefreshing = infoQuery.data !== undefined && infoQuery.isFetching
  const updateQuery = useQuery(systemUpdateQueryOptions(apiClient))
  const isDevelopmentBuild = /^v?\d+\.\d+\.\d+-dev(?:\.|$)/.test(infoQuery.data?.version ?? '')

  const [manualUpdateCheck, setManualUpdateCheck] = useState<UpdateCheckState | null>(null)
  const [updateCheckPending, setUpdateCheckPending] = useState(false)
  const updateCheck: UpdateCheckState | null =
    manualUpdateCheck ??
    (updateQuery.data?.update ? { kind: 'available', update: updateQuery.data.update } : null)
  const updateCheckTone =
    updateCheck?.kind === 'latest'
      ? 'success'
      : updateCheck?.kind === 'available'
        ? 'warning'
        : updateCheck?.kind === 'failed'
          ? 'danger'
          : 'info'

  const checkForUpdate = async (): Promise<void> => {
    if (updateCheckPending) return
    const previousCheck = updateCheck
    setUpdateCheckPending(true)
    setManualUpdateCheck({ kind: 'checking' })
    try {
      const result = await getSystemUpdate(apiClient, undefined, true)
      queryClient.setQueryData(controlQueryKeys.systemUpdate(), result)
      setManualUpdateCheck(
        result.update ? { kind: 'available', update: result.update } : { kind: 'latest' },
      )
    } catch (error) {
      if (error instanceof RequestCancelledError) {
        setManualUpdateCheck(previousCheck)
        return
      }
      setManualUpdateCheck(previousCheck ?? { kind: 'failed' })
    } finally {
      setUpdateCheckPending(false)
    }
  }

  const sourceLabel = (source: SecretSource): string => t(`settings.system.sources.${source}`)
  const databaseLabel = (database: DatabaseDriver): string =>
    t(`settings.system.databases.${database}`)

  const info = infoQuery.data

  return (
    <SettingsSectionFrame
      id="settings-system"
      title={t('settings.system.title')}
      description={t('settings.system.description')}
    >
      <span
        aria-live="polite"
        style={{
          position: 'absolute',
          width: 1,
          height: 1,
          overflow: 'hidden',
          clipPath: 'inset(50%)',
        }}
      >
        {infoRefreshing ? t('settings.system.loading') : ''}
      </span>

      {(infoQuery.isPending && info === undefined) || initialLoading ? (
        <div role="status" aria-label={t('settings.system.loading')} style={{ minHeight: 320 }}>
          <Skeleton height={320} radius={2} />
        </div>
      ) : infoQuery.isError && info === undefined ? (
        <div {...stylex.props(styles.errorBox)} role="alert">
          <TriangleAlert size={16} aria-hidden />
          <span>{t('settings.system.loadFailed')}</span>
          <Button
            variant="secondary"
            size="sm"
            label={t('common.retry')}
            onClick={() => void infoQuery.refetch()}
          />
        </div>
      ) : info !== undefined ? (
        <>
          {infoQuery.isError && (
            <div {...stylex.props(styles.staleBanner)} role="status">
              <TriangleAlert size={13} aria-hidden />
              <span>{t('settings.system.stale')}</span>
              <Button
                variant="secondary"
                size="sm"
                label={t('common.retry')}
                onClick={() => void infoQuery.refetch()}
              />
            </div>
          )}
          <div {...stylex.props(styles.panel)}>
            <dl {...stylex.props(styles.definition)}>
              {(
                [
                  {
                    term: t('settings.system.version'),
                    detail: (
                      <span {...stylex.props(styles.version)}>
                        <span {...stylex.props(styles.mono)}>{info.version}</span>
                        {!isDevelopmentBuild && (
                          <span {...stylex.props(styles.updateControls)}>
                            <Button
                              variant="secondary"
                              size="sm"
                              isLoading={updateCheckPending}
                              label={t('settings.system.checkUpdate')}
                              icon={
                                <RefreshCw
                                  size={14}
                                  aria-hidden
                                  {...(updateCheckPending ? stylex.props(styles.refreshIcon) : {})}
                                />
                              }
                              onClick={() => void checkForUpdate()}
                            />
                            {updateCheck && (
                              <span
                                {...stylex.props(
                                  styles.updateResult,
                                  updateCheckTone === 'success' && styles.updateSuccess,
                                  updateCheckTone === 'warning' && styles.updateWarning,
                                  updateCheckTone === 'danger' && styles.updateDanger,
                                  updateCheckTone === 'info' && styles.updateInfo,
                                )}
                                role={updateCheck.kind === 'failed' ? 'alert' : 'status'}
                                aria-live={updateCheck.kind === 'failed' ? 'assertive' : 'polite'}
                                aria-atomic="true"
                              >
                                {updateCheck.kind === 'checking' &&
                                  t('settings.system.checkingUpdate')}
                                {updateCheck.kind === 'latest' &&
                                  t('settings.system.latestVersion')}
                                {updateCheck.kind === 'available' && (
                                  <>
                                    {t('settings.system.updateAvailable', {
                                      version: updateCheck.update.version,
                                    })}
                                    <a
                                      {...stylex.props(styles.updateLink)}
                                      href={updateCheck.update.release_url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                    >
                                      {t('settings.system.viewRelease')}
                                    </a>
                                  </>
                                )}
                                {updateCheck.kind === 'failed' &&
                                  t('settings.system.checkUpdateFailed')}
                              </span>
                            )}
                          </span>
                        )}
                      </span>
                    ),
                  },
                  {
                    term: t('settings.system.deployment'),
                    detail: (
                      <span {...stylex.props(styles.inline)}>
                        <Badge label={t('settings.system.single')} />
                        <Badge label={databaseLabel(info.deployment.database)} />
                        <Badge label={t('settings.system.singleBinary')} />
                      </span>
                    ),
                  },
                  { term: t('settings.system.dataDir'), detail: info.data_dir, mono: true },
                  {
                    term: t('settings.system.authKey'),
                    detail: (
                      <span {...stylex.props(styles.inline)}>
                        <Badge label={sourceLabel(info.auth_key.source)} />
                        {info.auth_key.path && (
                          <span {...stylex.props(styles.path, styles.mono)}>
                            <span {...stylex.props(styles.pathText)}>{info.auth_key.path}</span>
                            <CopyButton
                              value={info.auth_key.path}
                              label={t('settings.system.copyPath')}
                              successLabel={t('common.copied')}
                              failureLabel={t('common.copyFailed')}
                            />
                          </span>
                        )}
                      </span>
                    ),
                  },
                  {
                    term: t('settings.system.encryption'),
                    detail: (
                      <span {...stylex.props(styles.inline)}>
                        <Badge variant="success" label={t('settings.system.enabled')} />
                        <Badge label={sourceLabel(info.encryption.source)} />
                        {info.encryption.path && (
                          <span {...stylex.props(styles.path, styles.mono)}>
                            <span {...stylex.props(styles.pathText)}>{info.encryption.path}</span>
                            <CopyButton
                              value={info.encryption.path}
                              label={t('settings.system.copyPath')}
                              successLabel={t('common.copied')}
                              failureLabel={t('common.copyFailed')}
                            />
                          </span>
                        )}
                      </span>
                    ),
                  },
                ] as const
              ).map((row, index, list) => (
                <div key={row.term} {...stylex.props(styles.row)}>
                  <dt {...stylex.props(styles.term, index === list.length - 1 && styles.termLast)}>
                    {row.term}
                  </dt>
                  <dd
                    {...stylex.props(styles.detail, index === list.length - 1 && styles.detailLast)}
                  >
                    {'mono' in row && row.mono ? (
                      <span {...stylex.props(styles.mono)}>{row.detail as string}</span>
                    ) : (
                      row.detail
                    )}
                  </dd>
                </div>
              ))}
            </dl>
          </div>
          <p {...stylex.props(styles.securityNote)}>{t('settings.system.securityNote')}</p>
        </>
      ) : null}
    </SettingsSectionFrame>
  )
}
