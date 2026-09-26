import { Button } from '@astryxdesign/core/Button'
import { TextInput } from '@astryxdesign/core/TextInput'
import * as stylex from '@stylexjs/stylex'
import {
  Link,
  useLocation,
  useRouter,
  useSearch,
} from '@tanstack/react-router'
import { useState, type FormEvent } from 'react'

import { pageRouteMetaFor } from '@shared/routing/route-meta'
import { decodedPathSegments } from '@shared/routing/safe-redirect'
import { sharedPageRouteNames } from '@shared/routing/route-names'

import { useT } from './i18n'
import { useAppServices } from './services'
import { safeRedirect } from './safe-redirect'

const styles = stylex.create({
  page: {
    display: 'grid',
    gap: 'var(--space-4, 16px)',
    maxWidth: '560px',
  },
  title: {
    margin: 0,
    fontSize: 'var(--text-panel-title-size, 22px)',
    fontWeight: 600,
  },
  meta: {
    color: 'var(--color-text-secondary)',
    fontSize: 'var(--text-meta-size, 12px)',
  },
  form: {
    display: 'grid',
    gap: 'var(--space-3, 12px)',
    maxWidth: '320px',
  },
  error: {
    color: 'var(--color-text-danger, #d03b3b)',
    fontSize: 'var(--text-meta-size, 12px)',
  },
})

export function LoginPageStub() {
  const services = useAppServices()
  const router = useRouter()
  const t = useT()
  const search = useSearch({ strict: false }) as { redirect?: unknown }
  const redirect = typeof search.redirect === 'string' ? search.redirect : undefined
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)

  async function submit(event: FormEvent): Promise<void> {
    event.preventDefault()
    if (pending) return
    setPending(true)
    setError('')
    try {
      await services.authSession.login(key)
      router.navigate({ href: safeRedirect(router, redirect) })
    } catch {
      setError(t('auth.invalid'))
    } finally {
      setPending(false)
    }
  }

  return (
    <main {...stylex.props(styles.page)} data-route={sharedPageRouteNames.login}>
      <h1 {...stylex.props(styles.title)}>{t('auth.loginTitle')}</h1>
      <form {...stylex.props(styles.form)} onSubmit={submit}>
        <TextInput
          label={t('auth.keyLabel')}
          value={key}
          onChange={(value) => setKey(value)}
          type="password"
          autoComplete="current-password"
        />
        <Button type="submit" label={pending ? t('auth.submitting') : t('auth.submit')} />
        {error !== '' && <p {...stylex.props(styles.error)}>{error}</p>}
      </form>
    </main>
  )
}

export function RoutePageStub({ name }: { name: string }) {
  const t = useT()
  const titleKey = pageRouteMetaFor(name).titleKey
  return (
    <main {...stylex.props(styles.page)} data-route={name}>
      <h1 {...stylex.props(styles.title)}>{titleKey === undefined ? name : t(titleKey)}</h1>
      <p {...stylex.props(styles.meta)}>React/Astryx preview shell — this page is a stub.</p>
      <p {...stylex.props(styles.meta)}>
        <Link to="/">{t('notFound.backHome')}</Link>
      </p>
    </main>
  )
}

export function NotFoundPageStub() {
  const t = useT()
  const pathname = useLocation({ select: (location) => location.pathname })
  const segments = decodedPathSegments(pathname).join('/')
  return (
    <main {...stylex.props(styles.page)} data-route="not-found">
      <h1 {...stylex.props(styles.title)}>{t('notFound.title')}</h1>
      <p {...stylex.props(styles.meta)}>
        {t('notFound.requestedPath')}: /{segments}
      </p>
      <p {...stylex.props(styles.meta)}>
        <Link to="/">{t('notFound.backHome')}</Link>
      </p>
    </main>
  )
}
