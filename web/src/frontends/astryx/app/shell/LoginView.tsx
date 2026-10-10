import { Banner } from '@astryxdesign/core/Banner'
import { Button } from '@astryxdesign/core/Button'
import { Collapsible } from '@astryxdesign/core/Collapsible'

import { IconButton } from '@astryxdesign/core/IconButton'
import { InputGroup } from '@astryxdesign/core/InputGroup'
import { TextInput } from '@astryxdesign/core/TextInput'
import * as stylex from '@stylexjs/stylex'
import { useRouter, useSearch } from '@tanstack/react-router'
import { Eye, EyeOff } from 'lucide-react'
import { useEffect, useRef, useState, type FormEvent } from 'react'

import { ApiError, NetworkError } from '@shared/http/errors'
import { pagePath } from '@shared/routing/page-routes'

import { useT } from '../i18n'
import { useAppServices } from '../services'
import { safeRedirect } from '../safe-redirect'
import { useCountdown } from './use-countdown'

type Feedback = 'invalid' | 'locked' | 'network' | 'invalid-response'

const compact = '@media (max-width: 860px)'
const collapse = '@media (max-width: 900px)'

const styles = stylex.create({
  frame: {
    display: 'grid',
    minHeight: 'calc(100vh - var(--topbar-height))',
    placeItems: 'center',
    padding: {
      default: 'var(--stage-padding-top) var(--stage-padding-inline) var(--stage-padding-bottom)',
      [compact]:
        'var(--stage-padding-top-compact) var(--stage-padding-inline-compact) var(--stage-padding-bottom-compact)',
    },
  },
  sheet: {
    display: 'grid',
    width: 'min(100%, var(--page-max))',
    minHeight: '500px',
    gridTemplateColumns: { default: 'minmax(0, 1fr) 420px', [collapse]: '1fr' },
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-page, 10px)',
    backgroundColor: 'var(--color-surface)',
  },
  intro: {
    display: { default: 'flex', [collapse]: 'none' },
    minWidth: 0,
    flexDirection: 'column',
    justifyContent: 'center',
    borderRightWidth: '1px',
    borderRightStyle: 'solid',
    borderRightColor: 'var(--color-border-subtle)',
    padding: '50px 56px 44px',
  },

  headline: {
    margin: 0,
    maxWidth: '20ch',
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--text-display-2-size)',
    fontWeight: 500,
    letterSpacing: '-0.03em',
    lineHeight: 1.15,
  },
  lead: {
    margin: '14px 0 0',
    maxWidth: '52ch',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-body-size)',
    lineHeight: 'var(--line-editorial)',
  },
  capabilities: {
    display: 'grid',
    gap: '18px',
    margin: '34px 0 0',
    padding: 0,
    listStyle: 'none',
  },
  capability: {
    display: 'flex',
    gap: '12px',
    alignItems: 'baseline',
  },
  capabilityIndex: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
  },
  capabilityTitle: {
    fontSize: 'var(--text-body-size)',
    fontWeight: 560,
  },
  capabilityDescription: {
    margin: '2px 0 0',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  auth: {
    display: 'flex',
    alignItems: 'center',
    padding: '40px 34px',
  },
  authInner: {
    width: '100%',
  },
  authTitle: {
    margin: 0,
    fontSize: 'var(--text-heading-3-size)',
    fontWeight: 600,
    letterSpacing: '-0.01em',
  },
  authDescription: {
    margin: '6px 0 0',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  form: {
    display: 'grid',
    gap: '14px',
    marginTop: '22px',
  },
  inputGroup: {
    width: '100%',
    boxSizing: 'border-box',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: {
      default: 'var(--color-border-emphasized)',
      ':focus-within': 'var(--color-accent)',
    },
    borderRadius: 'var(--radius-element)',
    backgroundColor: 'var(--color-background-surface)',
    height: { default: '34px', '@media (max-width: 860px)': '44px' },
  },
  groupedInput: {
    borderWidth: 0,
    boxShadow: { default: 'none', ':focus-within': 'none', ':hover': 'none' },
  },
  reveal: {
    alignSelf: 'center',
    flexShrink: 0,
    marginRight: '2px',
  },
  sessionNote: {
    margin: 0,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  code: {
    fontFamily: 'var(--font-mono)',
  },
  help: {
    marginTop: '4px',
  },
  helpSources: {
    display: 'grid',
    gap: '12px',
    paddingTop: '10px',
  },
  helpSourceTitle: {
    fontSize: 'var(--text-meta)',
    fontWeight: 560,
  },
  helpSourceBody: {
    margin: '2px 0 0',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  recovery: {
    margin: '4px 0 0',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
  recoveryAction: {
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-action)',
    padding: 0,
    fontFamily: 'inherit',
    fontSize: 'inherit',
    fontWeight: 'inherit',
    lineHeight: 'inherit',
    letterSpacing: 'inherit',
    cursor: 'pointer',
    textDecoration: 'underline',
  },
  instance: {
    margin: '18px 0 0',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-meta)',
    lineHeight: 'var(--line-normal)',
  },
})

export function LoginView() {
  const services = useAppServices()
  const session = services.authSession
  const router = useRouter()
  const t = useT()
  const search = useSearch({ strict: false }) as { redirect?: string; help?: string }
  const redirect = typeof search.redirect === 'string' ? search.redirect : undefined
  const helpOpen = search.help === 'auth'

  const [candidate, setCandidate] = useState('')
  const [visible, setVisible] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [fieldError, setFieldError] = useState('')
  const [feedback, setFeedback] = useState<Feedback>()
  const countdown = useCountdown(1)
  const inputRef = useRef<HTMLInputElement | null>(null)

  // TextInput's ref lands on the field wrapper; the focus target is the inner
  // input (classic focused the bare <input> directly).
  function focusKeyInput(): void {
    try {
      inputRef.current?.focus({ preventScroll: true })
    } catch {
      inputRef.current?.focus()
    }
  }

  const lockActive = feedback === 'locked' && countdown.active
  const controlsDisabled = submitting || lockActive
  const instanceOrigin = window.location.origin

  const feedbackMessage =
    fieldError !== ''
      ? fieldError
      : feedback === 'invalid'
        ? t('auth.invalid')
        : feedback === 'locked'
          ? t('auth.locked', { seconds: countdown.seconds })
          : feedback === 'network'
            ? t('auth.network')
            : feedback === 'invalid-response'
              ? t('auth.invalidResponse')
              : ''
  const feedbackTone = feedback === 'locked' ? 'warning' : 'error'

  // Classic clears the lock feedback and refocuses once the countdown ends.
  const wasLocked = useRef(false)
  useEffect(() => {
    if (wasLocked.current && !countdown.active && feedback === 'locked') {
      setFeedback(undefined)
      focusKeyInput()
    }
    wasLocked.current = countdown.active && feedback === 'locked'
  }, [countdown.active, feedback])

  useEffect(() => {
    focusKeyInput()
  }, [])

  function handleInput(): void {
    setFieldError('')
    if (feedback !== 'locked') setFeedback(undefined)
  }

  function setHelpOpen(open: boolean): void {
    if (open === helpOpen) return
    const params = new URLSearchParams()
    if (redirect !== undefined) params.set('redirect', redirect)
    if (open) params.set('help', 'auth')
    const qs = params.toString()
    void router.navigate({
      href: `${pagePath('login')}${qs === '' ? '' : `?${qs}`}`,
    })
  }

  async function retryFromHelp(): Promise<void> {
    if (candidate === '') {
      focusKeyInput()
      return
    }
    await submit()
  }

  async function submit(): Promise<void> {
    if (candidate === '') {
      setFieldError(t('auth.required'))
      focusKeyInput()
      return
    }
    if (/\s/u.test(candidate)) {
      setFieldError(t('auth.invalidFormat'))
      focusKeyInput()
      return
    }
    if (controlsDisabled) return

    setSubmitting(true)
    setFieldError('')
    setFeedback(undefined)
    try {
      await session.login(candidate)
      const target = safeRedirect(router, redirect)
      services.importRecovery.sweep()
      const preserveRecovery =
        session.getPrincipalType() === 'admin' && target.startsWith(pagePath('import'))
      if (!preserveRecovery) services.importRecovery.clear()
      // safeRedirect only returns manifest route paths, which the SPA owns —
      // replace keeps /login out of history.
      void router.navigate({ href: target, replace: true })
    } catch (error: unknown) {
      if (error instanceof ApiError && error.code === 'UNAUTHORIZED') {
        setFeedback('invalid')
      } else if (error instanceof ApiError && error.code === 'AUTH_LOCKED') {
        setFeedback('locked')
        countdown.reset(error.retryAfterSeconds ?? 1)
      } else if (error instanceof NetworkError) {
        setFeedback('network')
      } else {
        setFeedback('invalid-response')
      }
    } finally {
      setSubmitting(false)
    }
  }

  function onSubmit(event: FormEvent): void {
    event.preventDefault()
    void submit()
  }

  return (
    <main {...stylex.props(styles.frame)} aria-labelledby="login-intro-title">
      <div {...stylex.props(styles.sheet)}>
        <section {...stylex.props(styles.intro)} aria-labelledby="login-intro-title">
          <h1 id="login-intro-title" {...stylex.props(styles.headline)}>
            {t('auth.headline')}
          </h1>
          <p {...stylex.props(styles.lead)}>{t('auth.lead')}</p>

          <ol {...stylex.props(styles.capabilities)}>
            {([1, 2, 3] as const).map((number) => (
              <li key={number} {...stylex.props(styles.capability)}>
                <span aria-hidden {...stylex.props(styles.capabilityIndex)}>
                  {String(number).padStart(2, '0')}
                </span>
                <div>
                  <strong {...stylex.props(styles.capabilityTitle)}>
                    {t(`auth.capabilities.${number}.title`)}
                  </strong>
                  <p {...stylex.props(styles.capabilityDescription)}>
                    {t(`auth.capabilities.${number}.description`)}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </section>

        <section {...stylex.props(styles.auth)} aria-labelledby="login-title">
          <div {...stylex.props(styles.authInner)}>
            <h2 id="login-title" {...stylex.props(styles.authTitle)}>
              {t('auth.loginTitle')}
            </h2>
            <p {...stylex.props(styles.authDescription)}>{t('auth.loginDescription')}</p>

            <form {...stylex.props(styles.form)} noValidate onSubmit={onSubmit}>
              <InputGroup
                label={t('auth.keyLabel')}
                isDisabled={controlsDisabled}
                xstyle={styles.inputGroup}
              >
                <TextInput
                  label=""
                  xstyle={styles.groupedInput}
                  htmlName="auth-key"
                  type={visible ? 'text' : 'password'}
                  autoComplete="current-password"
                  hasAutoFocus
                  placeholder={t('auth.keyPlaceholder')}
                  aria-describedby="auth-feedback"
                  isDisabled={controlsDisabled}
                  value={candidate}
                  onChange={(value) => {
                    setCandidate(value)
                    handleInput()
                  }}
                  ref={inputRef}
                />
                <IconButton
                  xstyle={styles.reveal}
                  label={visible ? t('auth.conceal') : t('auth.reveal')}
                  icon={visible ? <EyeOff size={15} aria-hidden /> : <Eye size={15} aria-hidden />}
                  variant="ghost"
                  size="sm"
                  isDisabled={controlsDisabled}
                  onClick={() => setVisible((current) => !current)}
                />
              </InputGroup>

              {feedbackMessage !== '' && (
                <div id="auth-feedback">
                  <Banner status={feedbackTone} title={feedbackMessage} />
                </div>
              )}

              <Button
                type="submit"
                variant="primary"
                label={submitting ? t('auth.submitting') : t('auth.submit')}
                isLoading={submitting}
                isDisabled={controlsDisabled}
              />

              <p {...stylex.props(styles.sessionNote)}>
                {t('auth.sessionNotePrefix')}
                <code {...stylex.props(styles.code)}>localStorage</code>
                {t('auth.sessionNoteSuffix')}
              </p>
            </form>

            <div {...stylex.props(styles.help)}>
              <Collapsible
                trigger={t('auth.help.title')}
                isOpen={helpOpen}
                onOpenChange={setHelpOpen}
              >
                <div {...stylex.props(styles.helpSources)}>
                  <div>
                    <strong {...stylex.props(styles.helpSourceTitle)}>
                      {t('auth.help.accessKeyTitle')}
                    </strong>
                    <p {...stylex.props(styles.helpSourceBody)}>
                      {t('auth.help.accessKeyDescription')}
                    </p>
                  </div>
                  <div>
                    <strong {...stylex.props(styles.helpSourceTitle)}>
                      {t('auth.help.environmentTitle')}
                    </strong>
                    <p {...stylex.props(styles.helpSourceBody)}>
                      {t('auth.help.environmentDescription', { key: 'AUTH_KEY' })}
                    </p>
                  </div>
                  <div>
                    <strong {...stylex.props(styles.helpSourceTitle)}>
                      {t('auth.help.fileTitle')}
                    </strong>
                    <p {...stylex.props(styles.helpSourceBody)}>
                      {t('auth.help.fileDescription', {
                        path: '${DATA_DIR}/auth.key',
                        containerPath: '/app/data/auth.key',
                      })}
                    </p>
                  </div>
                  <div>
                    <strong {...stylex.props(styles.helpSourceTitle)}>
                      {t('auth.help.dockerTitle')}
                    </strong>
                    <p {...stylex.props(styles.helpSourceBody)}>
                      {t('auth.help.dockerDescription', {
                        command: 'docker exec -it gpt-load sh',
                      })}
                    </p>
                  </div>
                </div>
              </Collapsible>
            </div>

            <p {...stylex.props(styles.recovery)}>
              {t('auth.recoveryPrefix')}{' '}
              <button
                type="button"
                {...stylex.props(styles.recoveryAction)}
                disabled={controlsDisabled}
                onClick={() => void retryFromHelp()}
              >
                {t('auth.recoveryAction')}
              </button>{' '}
              {t('auth.recoverySuffix')}
            </p>

            <p {...stylex.props(styles.instance)}>
              {t('auth.instance', { origin: instanceOrigin })}
            </p>
          </div>
        </section>
      </div>
    </main>
  )
}
