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

import { decodedPathSegments } from '@shared/routing/safe-redirect'
import { sharedPageRouteNames } from '@shared/routing/route-names'

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
      setError('Authentication failed')
    } finally {
      setPending(false)
    }
  }

  return (
    <main {...stylex.props(styles.page)} data-route={sharedPageRouteNames.login}>
      <h1 {...stylex.props(styles.title)}>Sign in</h1>
      <form {...stylex.props(styles.form)} onSubmit={submit}>
        <TextInput
          label="Auth key"
          value={key}
          onChange={(value) => setKey(value)}
          type="password"
          autoComplete="current-password"
        />
        <Button type="submit" label={pending ? 'Signing in…' : 'Sign in'} />
        {error !== '' && <p {...stylex.props(styles.error)}>{error}</p>}
      </form>
    </main>
  )
}

export function RoutePageStub({ name }: { name: string }) {
  return (
    <main {...stylex.props(styles.page)} data-route={name}>
      <h1 {...stylex.props(styles.title)}>{name}</h1>
      <p {...stylex.props(styles.meta)}>React/Astryx preview shell — this page is a stub.</p>
      <p {...stylex.props(styles.meta)}>
        <Link to="/">Back to home</Link>
      </p>
    </main>
  )
}

export function NotFoundPageStub() {
  const pathname = useLocation({ select: (location) => location.pathname })
  const segments = decodedPathSegments(pathname).join('/')
  return (
    <main {...stylex.props(styles.page)} data-route="not-found">
      <h1 {...stylex.props(styles.title)}>This page does not exist</h1>
      <p {...stylex.props(styles.meta)}>Requested path: /{segments}</p>
      <p {...stylex.props(styles.meta)}>
        <Link to="/">Back to home</Link>
      </p>
    </main>
  )
}
