import * as stylex from '@stylexjs/stylex'
import {
  AlertDialog,
  Button,
  CheckboxInput,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
  Selector,
  TextInput,
} from '@astryxdesign/core'
import { useEffect, useRef, useState } from 'react'

import {
  createProbeTemplate,
  runRealisticProbe,
  type RealisticProbeProtocol,
  type RealisticProbeResult,
} from '@shared/probe/realistic-probe'
import { useT } from '../../app/i18n'

const styles = stylex.create({
  body: { display: 'grid', gap: 'var(--space-3)', minWidth: 0 },
  wrapping: { minWidth: 0, overflowWrap: 'anywhere' },
  field: { display: 'grid', gap: 4, minWidth: 0 },
  editor: {
    width: '100%',
    boxSizing: 'border-box',
    minHeight: 220,
    fontFamily: 'monospace',
    resize: 'vertical',
  },
  preview: { whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', maxHeight: 240, overflow: 'auto' },
  actions: { display: 'flex', gap: 'var(--space-2)', flexWrap: 'wrap' },
})

const protocols: RealisticProbeProtocol[] = [
  'openai-completions',
  'openai-responses',
  'anthropic',
  'gemini',
]

export function isRealisticProbeProtocol(value: string): value is RealisticProbeProtocol {
  return protocols.some((protocol) => protocol === value)
}

export function RealisticProbeDialog({
  model,
  initialProtocol = 'openai-completions',
  onClose,
  onViewLog,
}: {
  model: string
  initialProtocol?: RealisticProbeProtocol
  onClose(): void
  onViewLog(id: string): void
}) {
  const t = useT()
  const [protocol, setProtocol] = useState<RealisticProbeProtocol>(initialProtocol)
  const [replacement, setReplacement] = useState<RealisticProbeProtocol | null>(null)
  const [geminiModel, setGeminiModel] = useState(model)
  const [geminiStream, setGeminiStream] = useState(false)
  const [body, setBody] = useState(() => createProbeTemplate(initialProtocol, model, false))
  const [accessKey, setAccessKey] = useState('')
  const [result, setResult] = useState<RealisticProbeResult | null>(null)
  const [pending, setPending] = useState(false)
  const generation = useRef(0)
  const controller = useRef<AbortController | null>(null)
  useEffect(
    () => () => {
      generation.current += 1
      controller.current?.abort()
    },
    [],
  )

  let parsed: Record<string, unknown> | null = null
  try {
    const value: unknown = JSON.parse(body)
    if (value !== null && typeof value === 'object' && !Array.isArray(value))
      parsed = value as Record<string, unknown>
  } catch {
    /* The editor can contain incomplete JSON while typing. */
  }
  const activeModel =
    protocol === 'gemini' ? geminiModel : typeof parsed?.model === 'string' ? parsed.model : ''
  const stream = protocol === 'gemini' ? geminiStream : parsed?.stream === true
  const endpoint =
    protocol === 'openai-completions'
      ? '/v1/chat/completions'
      : protocol === 'openai-responses'
        ? '/v1/responses'
        : protocol === 'anthropic'
          ? '/v1/messages'
          : `/v1beta/models/${encodeURIComponent(geminiModel)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`

  function close() {
    generation.current += 1
    controller.current?.abort()
    controller.current = null
    setAccessKey('')
    onClose()
  }
  function changeProtocol(next: RealisticProbeProtocol) {
    setResult(null)
    setProtocol(next)
    setBody(createProbeTemplate(next, activeModel || model, stream))
    setGeminiModel(activeModel || model)
    setGeminiStream(stream)
    setReplacement(null)
  }
  function changeStream(checked: boolean) {
    setResult(null)
    if (protocol === 'gemini') {
      setGeminiStream(checked)
      return
    }
    if (parsed) setBody(JSON.stringify({ ...parsed, stream: checked }, null, 2))
  }
  async function run() {
    if (pending || !accessKey || !parsed || !activeModel) return
    const current = ++generation.current
    const abort = new AbortController()
    controller.current = abort
    setPending(true)
    setResult(null)
    try {
      const value = await runRealisticProbe({
        protocol,
        body,
        model: activeModel,
        stream,
        accessKey,
        signal: abort.signal,
        onUpdate: (update) => {
          if (generation.current === current) setResult(update)
        },
      })
      if (generation.current === current) setResult(value)
    } catch {
      if (generation.current === current)
        setResult({
          outcome: 'failed',
          statusCode: null,
          requestId: null,
          firstOutputMs: null,
          durationMs: 0,
          output: '',
          terminal: null,
          error: t('monitor.realisticProbe.unexpected'),
          truncated: false,
        })
    } finally {
      if (generation.current === current) {
        setPending(false)
        controller.current = null
      }
    }
  }
  return (
    <>
      <Dialog
        isOpen
        onOpenChange={(open) => {
          if (!open) close()
        }}
        width={720}
        maxHeight="calc(100dvh - 36px)"
      >
        <Layout
          header={
            <DialogHeader
              title={t('monitor.realisticProbe.title')}
              onOpenChange={(open) => {
                if (!open) close()
              }}
              hasDivider
            />
          }
          content={
            <LayoutContent isScrollable>
              <div {...stylex.props(styles.body)}>
                <p>{t('monitor.realisticProbe.notice')}</p>
                <Selector
                  size="sm"
                  variant="input"
                  label={t('monitor.realisticProbe.protocol')}
                  options={protocols.map((value) => ({ value, label: value }))}
                  value={protocol}
                  isDisabled={pending}
                  onChange={(value) => {
                    if (isRealisticProbeProtocol(value) && value !== protocol) setReplacement(value)
                  }}
                />
                <div {...stylex.props(styles.wrapping)}>
                  {t('monitor.realisticProbe.endpoint')}: <code>{endpoint}</code>
                </div>
                {protocol === 'gemini' ? (
                  <TextInput
                    label={t('monitor.realisticProbe.model')}
                    value={geminiModel}
                    onChange={(value) => {
                      setGeminiModel(value)
                      setResult(null)
                    }}
                    isDisabled={pending}
                  />
                ) : (
                  <div {...stylex.props(styles.wrapping)}>
                    {t('monitor.realisticProbe.model')}: <code>{activeModel || '—'}</code>
                  </div>
                )}
                <CheckboxInput
                  value={stream}
                  isDisabled={pending || (protocol !== 'gemini' && !parsed)}
                  onChange={changeStream}
                  label={t('monitor.realisticProbe.stream')}
                />
                <label {...stylex.props(styles.field)}>
                  {t('monitor.realisticProbe.json')}
                  <textarea
                    {...stylex.props(styles.editor)}
                    value={body}
                    disabled={pending}
                    onChange={(event) => {
                      setBody(event.target.value)
                      setResult(null)
                    }}
                    spellCheck={false}
                  />
                </label>
                <TextInput
                  label={t('monitor.realisticProbe.key')}
                  type="password"
                  value={accessKey}
                  onChange={(value) => {
                    setAccessKey(value)
                    setResult(null)
                  }}
                  isDisabled={pending}
                  autoComplete="off"
                />
                <p>{t('monitor.realisticProbe.tools')}</p>
                {result && (
                  <section aria-live="polite" {...stylex.props(styles.wrapping)}>
                    <strong>{t(`monitor.realisticProbe.outcomes.${result.outcome}`)}</strong>
                    <div>
                      HTTP: {result.statusCode ?? '—'} · {t('monitor.realisticProbe.firstOutput')}:{' '}
                      {result.firstOutputMs === null ? '—' : `${result.firstOutputMs} ms`} ·{' '}
                      {t('monitor.realisticProbe.duration')}: {result.durationMs} ms
                    </div>
                    <div>
                      {t('monitor.realisticProbe.requestId')}:{' '}
                      {result.requestId ?? t('monitor.realisticProbe.noId')}
                    </div>
                    <div>
                      {t('monitor.realisticProbe.terminal')}: {result.terminal ?? '—'}
                    </div>
                    {result.error && <p>{result.error}</p>}
                    <div>{t('monitor.realisticProbe.preview')}</div>
                    <pre {...stylex.props(styles.preview)}>{result.output}</pre>
                    {result.truncated && <p>{t('monitor.realisticProbe.truncated')}</p>}
                    {result.requestId && (
                      <Button
                        variant="secondary"
                        size="sm"
                        label={t('monitor.realisticProbe.viewLog')}
                        isDisabled={pending}
                        onClick={() => onViewLog(result.requestId!)}
                      />
                    )}
                  </section>
                )}
              </div>
            </LayoutContent>
          }
          footer={
            <LayoutFooter hasDivider>
              <div {...stylex.props(styles.actions)}>
                <Button
                  variant="secondary"
                  size="sm"
                  label={t('monitor.realisticProbe.close')}
                  onClick={close}
                />
                {pending ? (
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('monitor.realisticProbe.stop')}
                    onClick={() => controller.current?.abort()}
                  />
                ) : (
                  <Button
                    size="sm"
                    label={t('monitor.realisticProbe.run')}
                    isDisabled={!accessKey || !parsed || !activeModel}
                    onClick={() => void run()}
                  />
                )}
              </div>
            </LayoutFooter>
          }
        />
      </Dialog>
      <AlertDialog
        isOpen={replacement !== null}
        onOpenChange={(open) => {
          if (!open) setReplacement(null)
        }}
        title={t('monitor.realisticProbe.protocol')}
        description={t('monitor.realisticProbe.replaceConfirm')}
        cancelLabel={t('common.cancel')}
        actionLabel={t('monitor.realisticProbe.replace')}
        onAction={() => {
          if (replacement) changeProtocol(replacement)
        }}
      />
    </>
  )
}
