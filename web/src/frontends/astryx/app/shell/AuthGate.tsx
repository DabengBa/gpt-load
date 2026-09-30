import { Banner } from '@astryxdesign/core/Banner'
import { Button } from '@astryxdesign/core/Button'
import { Card } from '@astryxdesign/core/Card'
import * as stylex from '@stylexjs/stylex'
import { useRouter } from '@tanstack/react-router'
import { useEffect, useRef, useSyncExternalStore, type ReactNode } from 'react'

import { useT } from '../i18n'
import { useAppServices } from '../services'
import { useCountdown } from './use-countdown'

const styles = stylex.create({
  gate: {
    display: 'grid',
    minHeight: '60vh',
    placeItems: 'center',
    padding: '24px',
  },
  card: {
    display: 'grid',
    gap: '14px',
    width: 'min(420px, 100%)',
    padding: '24px',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-heading-3-size)',
    fontWeight: 600,
  },
  actions: {
    display: 'flex',
    gap: '8px',
  },
})

// Session-phase gate mirroring classic AuthGate: renders children only for a
// validated session (access_key principals never see adminOnly routes), and
// otherwise shows the validating/locked/network/invalid-response states with
// retry + change-key actions.
export function AuthGate({
  adminOnly = false,
  children,
}: {
  adminOnly?: boolean
  children: ReactNode
}) {
  const services = useAppServices()
  const session = services.authSession
  const state = useSyncExternalStore(session.subscribe, session.getState)
  const router = useRouter()
  const t = useT()
  const countdown = useCountdown(state.retryAfterSeconds)
  const gateRef = useRef<HTMLElement | null>(null)

  const canRenderRoute =
    state.phase === 'validated' && !(state.principalType === 'access_key' && adminOnly)

  useEffect(() => {
    if (state.phase !== 'validated') {
      void session.ensureValidated().catch(() => {})
    }
  }, [session, state.phase])

  useEffect(() => {
    if (state.phase === 'validated' && state.principalType === 'access_key' && adminOnly) {
      void router.navigate({ href: '/', replace: true })
    }
  }, [state.phase, state.principalType, adminOnly, router])

  // Classic moves focus to the retry action once the invalid-response card is
  // on screen.
  useEffect(() => {
    if (state.phase !== 'invalid-response') return
    const retry = gateRef.current?.querySelector<HTMLButtonElement>(
      'button.auth-gate-invalid-response-retry',
    )
    if (retry?.isConnected) retry.focus()
  }, [state.phase])

  if (canRenderRoute) return children
  if (state.phase === 'validated') return null

  const changeAuthKey = () => {
    services.importRecovery.clear()
    session.clear()
    void router.navigate({ href: '/login', replace: true })
  }
  const retryValidation = () => {
    void session.retryValidation().catch(() => {})
  }

  return (
    <main ref={gateRef} {...stylex.props(styles.gate)}>
      <Card {...stylex.props(styles.card)} aria-labelledby="auth-gate-title">
        <h1 id="auth-gate-title" {...stylex.props(styles.title)}>
          {t('common.appName')}
        </h1>

        {state.phase === 'validating' && <Banner status="info" title={t('auth.checking')} />}

        {state.phase === 'locked' && (
          <>
            <Banner status="warning" title={t('auth.locked', { seconds: countdown.seconds })} />
            <div {...stylex.props(styles.actions)}>
              <Button
                variant="secondary"
                label={t('common.retry')}
                isDisabled={countdown.active}
                onClick={retryValidation}
              />
              <Button variant="ghost" label={t('common.changeKey')} onClick={changeAuthKey} />
            </div>
          </>
        )}

        {state.phase === 'network-error' && (
          <>
            <Banner status="error" title={t('auth.network')} />
            <div {...stylex.props(styles.actions)}>
              <Button variant="secondary" label={t('common.retry')} onClick={retryValidation} />
            </div>
          </>
        )}

        {state.phase === 'invalid-response' && (
          <>
            <Banner status="error" title={t('auth.invalidResponse')} />
            <div {...stylex.props(styles.actions)}>
              <Button
                className="auth-gate-invalid-response-retry"
                variant="secondary"
                label={t('common.retry')}
                onClick={retryValidation}
              />
              <Button variant="ghost" label={t('common.changeKey')} onClick={changeAuthKey} />
            </div>
          </>
        )}
      </Card>
    </main>
  )
}
