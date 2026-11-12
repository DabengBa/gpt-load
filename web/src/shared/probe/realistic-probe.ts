export type RealisticProbeProtocol =
  'openai-completions' | 'openai-responses' | 'anthropic' | 'gemini'
export interface RealisticProbeResult {
  outcome: 'running' | 'completed' | 'incomplete' | 'failed' | 'canceled'
  statusCode: number | null
  requestId: string | null
  firstOutputMs: number | null
  durationMs: number
  output: string
  terminal: string | null
  error: string | null
  truncated: boolean
}
interface ProbeOptions {
  protocol: RealisticProbeProtocol
  body: string
  model: string
  stream: boolean
  accessKey: string
  signal: AbortSignal
  onUpdate?: (result: RealisticProbeResult) => void
}
const previewLimit = 128 * 1024
const eventLimit = 256 * 1024
const bodyLimit = 1024 * 1024

export function createProbeTemplate(
  protocol: RealisticProbeProtocol,
  model: string,
  stream: boolean,
): string {
  const prompt = 'Hello! Please respond briefly.'
  let body: object
  switch (protocol) {
    case 'openai-responses':
      body = {
        model,
        input: [{ role: 'user', content: [{ type: 'input_text', text: prompt }] }],
        max_output_tokens: 1024,
        store: false,
        stream,
      }
      break
    case 'openai-completions':
    case 'anthropic':
      body = { model, messages: [{ role: 'user', content: prompt }], max_tokens: 1024, stream }
      break
    case 'gemini':
      body = {
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { maxOutputTokens: 1024 },
      }
  }
  return JSON.stringify(body, null, 2)
}
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
function nonempty(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0
}
function validResponsesOutput(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.some(
      (item: unknown) =>
        record(item) &&
        ['reasoning', 'message', 'function_call', 'custom_tool_call'].includes(String(item.type)),
    ) &&
    value.every((item: unknown) => {
      if (!record(item)) return false
      if (item.type === 'reasoning') return nonempty(item.id) && Array.isArray(item.summary)
      if (item.type === 'message')
        return (
          Array.isArray(item.content) &&
          item.content.some(
            (part: unknown) =>
              record(part) &&
              ((part.type === 'output_text' && typeof part.text === 'string') ||
                (part.type === 'refusal' && typeof part.refusal === 'string')),
          )
        )
      if (item.type === 'function_call')
        return nonempty(item.call_id) && nonempty(item.name) && typeof item.arguments === 'string'
      if (item.type === 'custom_tool_call')
        return nonempty(item.call_id) && nonempty(item.name) && typeof item.input === 'string'
      return nonempty(item.type) && Object.keys(item).length > 1
    })
  )
}
function responsesHasOutput(value: unknown): boolean {
  return (
    record(value) &&
    Array.isArray(value.output) &&
    value.output.some(
      (item: unknown) =>
        record(item) &&
        (item.type === 'message' ||
          item.type === 'function_call' ||
          item.type === 'custom_tool_call'),
    )
  )
}
function validChatMessage(value: unknown): boolean {
  return (
    record(value) &&
    value.role === 'assistant' &&
    (typeof value.content === 'string' ||
      (record(value.function_call) &&
        nonempty(value.function_call.name) &&
        typeof value.function_call.arguments === 'string') ||
      (Array.isArray(value.tool_calls) &&
        value.tool_calls.some(
          (call: unknown) =>
            record(call) &&
            record(call.function) &&
            nonempty(call.function.name) &&
            typeof call.function.arguments === 'string',
        )))
  )
}
function validAnthropicContent(value: unknown): boolean {
  return (
    Array.isArray(value) &&
    value.some(
      (block: unknown) =>
        record(block) &&
        ((block.type === 'text' && typeof block.text === 'string') ||
          (block.type === 'tool_use' &&
            nonempty(block.id) &&
            nonempty(block.name) &&
            record(block.input))),
    )
  )
}
function validGeminiContent(value: unknown): boolean {
  return (
    record(value) &&
    Array.isArray(value.parts) &&
    value.parts.some(
      (part: unknown) =>
        record(part) &&
        (typeof part.text === 'string' ||
          (record(part.functionCall) && nonempty(part.functionCall.name))),
    )
  )
}
type Judgement = {
  outcome: 'completed' | 'incomplete' | 'failed'
  terminal: string | null
  error: string | null
}
type ToolCallParts = { name: boolean; arguments: boolean }
function updateToolParts(parts: ToolCallParts, value: unknown): boolean {
  if (!record(value)) return false
  if (value.name !== undefined && typeof value.name !== 'string') return false
  if (value.arguments !== undefined && typeof value.arguments !== 'string') return false
  parts.name ||= nonempty(value.name)
  parts.arguments ||= nonempty(value.arguments)
  return true
}
type CandidateState = {
  content: boolean
  finish: string | null
  toolCalls?: Map<number, ToolCallParts>
  functionCall?: ToolCallParts
  invalidTool?: boolean
}
const maxCandidates = 1024
function validChatToolState(state: CandidateState): boolean {
  if (state.invalidTool) return false
  if (
    state.toolCalls &&
    [...state.toolCalls.values()].some((tool) => !tool.name || !tool.arguments)
  )
    return false
  if (state.functionCall && (!state.functionCall.name || !state.functionCall.arguments))
    return false
  if (state.finish === 'tool_calls' && !state.toolCalls?.size) return false
  if (state.finish === 'function_call' && !state.functionCall) return false
  return true
}
function aggregateCandidates(
  states: Map<number, CandidateState>,
  expected: number,
  protocol: 'openai-completions' | 'gemini',
): Judgement | null {
  if (expected > maxCandidates)
    return { outcome: 'incomplete', terminal: null, error: 'Candidate count exceeds limit' }
  for (const state of states.values()) {
    if (
      state.finish &&
      (protocol === 'gemini'
        ? state.finish !== 'STOP'
        : !['stop', 'tool_calls', 'function_call'].includes(state.finish))
    )
      return {
        outcome: 'incomplete',
        terminal: state.finish,
        error: 'Generation stopped before completion',
      }
    if (protocol === 'openai-completions' && !validChatToolState(state))
      return {
        outcome: 'incomplete',
        terminal: state.finish,
        error: 'Incomplete tool call',
      }
  }
  if (states.size < expected) return null
  for (let i = 0; i < expected; i++) {
    const state = states.get(i)
    if (!state?.finish || !state.content) return null
  }
  for (const state of states.values()) if (!state.finish || !state.content) return null
  return { outcome: 'completed', terminal: states.get(0)?.finish ?? null, error: null }
}
function candidateIndex(value: unknown, fallback: number): number | null {
  if (value === undefined) return fallback < maxCandidates ? fallback : null
  return typeof value === 'number' &&
    Number.isSafeInteger(value) &&
    value >= 0 &&
    value < maxCandidates
    ? value
    : null
}
function judge(protocol: RealisticProbeProtocol, value: unknown, expected = 1): Judgement | null {
  if (!record(value)) return null
  if (value.error != null)
    return { outcome: 'failed', terminal: 'error', error: 'Upstream reported an error' }
  if (protocol === 'openai-responses') {
    if (value.object !== 'response' || !nonempty(value.status)) return null
    if (value.status === 'failed')
      return { outcome: 'failed', terminal: value.status, error: 'Response failed' }
    if (value.status === 'incomplete')
      return {
        outcome: 'incomplete',
        terminal:
          record(value.incomplete_details) && nonempty(value.incomplete_details.reason)
            ? value.incomplete_details.reason
            : value.status,
        error: 'Response incomplete',
      }
    if (value.status === 'completed' && validResponsesOutput(value.output))
      return { outcome: 'completed', terminal: value.status, error: null }
  } else if (protocol === 'openai-completions') {
    if (value.object !== 'chat.completion' || !Array.isArray(value.choices)) return null
    const states = new Map<number, CandidateState>()
    for (const [position, choice] of value.choices.entries()) {
      if (!record(choice)) return null
      const index = candidateIndex(choice.index, position)
      if (index === null || states.has(index)) return null
      const state: CandidateState = {
        content: validChatMessage(choice.message),
        finish: nonempty(choice.finish_reason) ? choice.finish_reason : null,
      }
      if (record(choice.message)) {
        if (choice.message.function_call != null) {
          state.functionCall = { name: false, arguments: false }
          if (!updateToolParts(state.functionCall, choice.message.function_call))
            state.invalidTool = true
        }
        if (choice.message.tool_calls != null) {
          state.toolCalls = new Map()
          if (!Array.isArray(choice.message.tool_calls) || choice.message.tool_calls.length === 0)
            state.invalidTool = true
          else
            for (const [toolIndex, call] of choice.message.tool_calls.entries()) {
              const parts = { name: false, arguments: false }
              if (!record(call) || !updateToolParts(parts, call.function)) state.invalidTool = true
              state.toolCalls.set(toolIndex, parts)
            }
        }
      }
      states.set(index, state)
    }
    return aggregateCandidates(states, expected, protocol)
  } else if (protocol === 'anthropic') {
    if (value.type !== 'message' || value.role !== 'assistant' || !nonempty(value.stop_reason))
      return null
    if (value.stop_reason === 'max_tokens' || value.stop_reason === 'refusal')
      return {
        outcome: 'incomplete',
        terminal: value.stop_reason,
        error: 'Generation stopped before completion',
      }
    if (
      (value.stop_reason === 'end_turn' ||
        value.stop_reason === 'stop_sequence' ||
        value.stop_reason === 'tool_use') &&
      validAnthropicContent(value.content)
    )
      return { outcome: 'completed', terminal: value.stop_reason, error: null }
  } else if (protocol === 'gemini') {
    if (record(value.promptFeedback) && nonempty(value.promptFeedback.blockReason))
      return {
        outcome: 'incomplete',
        terminal: value.promptFeedback.blockReason,
        error: 'Prompt blocked',
      }
    if (!Array.isArray(value.candidates)) return null
    const states = new Map<number, CandidateState>()
    for (const [position, candidate] of value.candidates.entries()) {
      if (!record(candidate)) return null
      const index = candidateIndex(candidate.index, position)
      if (index === null || states.has(index)) return null
      states.set(index, {
        content: validGeminiContent(candidate.content),
        finish: nonempty(candidate.finishReason) ? candidate.finishReason : null,
      })
    }
    return aggregateCandidates(states, expected, protocol)
  }
  return null
}

export async function runRealisticProbe(
  options: ProbeOptions,
  fetcher: typeof fetch = fetch,
): Promise<RealisticProbeResult> {
  const started = performance.now()
  let preview = ''
  let pendingPreview = ''
  let result: RealisticProbeResult = {
    outcome: 'running',
    statusCode: null,
    requestId: null,
    firstOutputMs: null,
    durationMs: 0,
    output: '',
    terminal: null,
    error: null,
    truncated: false,
  }
  // Include the JSON-escaped representation: the preview often contains raw JSON.
  const patterns = [...new Set([options.accessKey, JSON.stringify(options.accessKey).slice(1, -1)])]
    .filter(Boolean)
    .map((value) => {
      const failure = Array.from<number>({ length: value.length }).fill(0)
      for (let i = 1, matched = 0; i < value.length; i++) {
        while (matched && value[i] !== value[matched]) matched = failure[matched - 1]
        if (value[i] === value[matched]) matched++
        failure[i] = matched
      }
      return { value, failure }
    })
  const mask = (value: string) => {
    for (const pattern of patterns) value = value.split(pattern.value).join('[REDACTED]')
    return value
  }
  const savePreview = (safe: string) => {
    const remaining = previewLimit - preview.length
    if (safe.length > remaining) result.truncated = true
    if (remaining > 0) preview += safe.slice(0, remaining)
  }
  const flushPreview = () => {
    // A trailing prefix may be a key cut off by EOF, cancellation or the preview limit.
    if (pendingPreview) savePreview('[REDACTED]')
    pendingPreview = ''
  }
  const snapshot = () => {
    return {
      ...result,
      durationMs: Math.max(0, Math.round(performance.now() - started)),
      output: preview,
      truncated: result.truncated,
      terminal: result.terminal === null ? null : mask(result.terminal),
      error: result.error === null ? null : mask(result.error),
    }
  }
  const notify = (value: RealisticProbeResult) => {
    try {
      options.onUpdate?.(value)
    } catch {
      /* Observers cannot break the transport. */
    }
  }
  const update = () => {
    notify(snapshot())
  }
  const finish = (
    outcome: RealisticProbeResult['outcome'],
    terminal: string | null,
    error: string | null,
  ) => {
    flushPreview()
    result = { ...result, outcome, terminal, error }
    const final = snapshot()
    notify(final)
    return final
  }
  const append = (text: string) => {
    const combined = pendingPreview + text
    let start = 0
    let safe = ''
    while (start < combined.length) {
      let match = -1
      let length = 0
      for (const pattern of patterns) {
        const at = combined.indexOf(pattern.value, start)
        if (
          at >= 0 &&
          (match < 0 || at < match || (at === match && pattern.value.length > length))
        ) {
          match = at
          length = pattern.value.length
        }
      }
      if (match < 0) break
      safe += combined.slice(start, match) + '[REDACTED]'
      start = match + length
    }
    const tail = combined.slice(start)
    let suffix = 0
    for (const pattern of patterns) {
      let matched = 0
      for (const character of tail) {
        while (matched && character !== pattern.value[matched])
          matched = pattern.failure[matched - 1]
        if (character === pattern.value[matched]) matched++
      }
      suffix = Math.max(suffix, matched)
    }
    safe += tail.slice(0, tail.length - suffix)
    pendingPreview = suffix ? tail.slice(-suffix) : ''
    savePreview(safe)
    if (preview.length === previewLimit && pendingPreview) result.truncated = true
  }
  try {
    let body: unknown
    try {
      body = JSON.parse(options.body)
    } catch {
      return finish('failed', null, 'Invalid JSON request')
    }
    if (!record(body)) return finish('failed', null, 'Request must be a JSON object')
    if (!/^[\x21-\x7e]+$/.test(options.accessKey))
      return finish('failed', null, 'Invalid access key')
    const paths: Record<Exclude<RealisticProbeProtocol, 'gemini'>, string> = {
      'openai-completions': '/v1/chat/completions',
      'openai-responses': '/v1/responses',
      anthropic: '/v1/messages',
    }
    let path: string
    let stream: boolean
    if (options.protocol === 'gemini') {
      if (!options.model || /[\s/\u0000-\u001f\u007f-\u009f]/u.test(options.model))
        return finish('failed', null, 'Invalid Gemini model')
      stream = options.stream
      path = `/v1beta/models/${encodeURIComponent(options.model)}:${stream ? 'streamGenerateContent?alt=sse' : 'generateContent'}`
    } else {
      if (
        !nonempty(body.model) ||
        body.model.trim() !== body.model ||
        (body.stream !== undefined && typeof body.stream !== 'boolean')
      )
        return finish('failed', null, 'Invalid model or stream')
      stream = body.stream === true
      path = paths[options.protocol]
      if (!path) return finish('failed', null, 'Unsupported protocol')
    }
    const requestedCount =
      options.protocol === 'openai-completions'
        ? body.n
        : options.protocol === 'gemini' && record(body.generationConfig)
          ? body.generationConfig.candidateCount
          : undefined
    const expected =
      typeof requestedCount === 'number' &&
      Number.isSafeInteger(requestedCount) &&
      requestedCount > 0
        ? requestedCount
        : 1
    if (options.signal.aborted) return finish('canceled', null, 'Request canceled')
    const response = await fetcher(path, {
      method: 'POST',
      body: options.body,
      headers: { Authorization: `Bearer ${options.accessKey}`, 'Content-Type': 'application/json' },
      credentials: 'omit',
      redirect: 'error',
      signal: options.signal,
    })
    const requestId = response.headers.get('X-GPTLoad-Request-ID')
    result.statusCode = response.status
    result.requestId =
      requestId &&
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(requestId) &&
      !requestId.includes(options.accessKey)
        ? requestId
        : null
    update()
    if (!response.body)
      return response.ok
        ? finish('incomplete', null, 'Response body missing')
        : finish('failed', 'http_error', `HTTP ${response.status}`)
    const reader = response.body.getReader()
    const cancelReader = () => {
      void reader.cancel().catch(() => {})
    }
    options.signal.addEventListener('abort', cancelReader, { once: true })
    if (options.signal.aborted) cancelReader()
    if (!response.ok) {
      const errorDecoder = new TextDecoder()
      let bytesRead = 0
      const errorByteLimit = previewLimit * 4
      try {
        while (preview.length < previewLimit && bytesRead < errorByteLimit) {
          const part = await reader.read()
          if (options.signal.aborted) return finish('canceled', null, 'Request canceled')
          if (part.done) {
            append(errorDecoder.decode())
            return finish('failed', 'http_error', `HTTP ${response.status}`)
          }
          const remaining = errorByteLimit - bytesRead
          bytesRead += part.value.byteLength
          append(errorDecoder.decode(part.value.subarray(0, remaining), { stream: true }))
          update()
        }
        result.truncated = true
        return finish('failed', 'http_error', `HTTP ${response.status}`)
      } catch {
        return options.signal.aborted
          ? finish('canceled', null, 'Request canceled')
          : finish('failed', 'http_error', `HTTP ${response.status}`)
      } finally {
        options.signal.removeEventListener('abort', cancelReader)
        cancelReader()
        reader.releaseLock()
      }
    }
    const decoder = new TextDecoder('utf-8', { fatal: true })
    let buffer = ''
    let dataLines: string[] = []
    let eventName = ''
    let eventSize = 0
    let judgment: Judgement | null = null
    let outputSeen = false
    const candidates = new Map<number, CandidateState>()
    let anthropicFinish: string | null = null
    let validStreamContent = false
    const anthropicTools = new Map<number, string>()
    let anthropicToolInputSize = 0
    let invalidAnthropicTool = false
    const observeOutput = () => {
      outputSeen = true
      if (result.firstOutputMs === null)
        result.firstOutputMs = Math.max(0, Math.round(performance.now() - started))
    }
    const handleEvent = () => {
      if (dataLines.length === 0) {
        eventName = ''
        eventSize = 0
        return
      }
      const data = dataLines.join('\n')
      dataLines = []
      eventSize = 0
      append(data + '\n')
      if (data === '[DONE]') {
        if (options.protocol === 'openai-completions')
          judgment = aggregateCandidates(candidates, expected, 'openai-completions') ?? {
            outcome: 'incomplete',
            terminal: null,
            error: 'Missing or unfinished choice',
          }
        eventName = ''
        update()
        return
      }
      let parsed: unknown
      try {
        parsed = JSON.parse(data)
      } catch {
        judgment = { outcome: 'incomplete', terminal: null, error: 'Malformed stream event' }
        eventName = ''
        update()
        return
      }
      if (!record(parsed)) {
        judgment = { outcome: 'incomplete', terminal: null, error: 'Invalid stream event' }
        eventName = ''
        update()
        return
      }
      if (nonempty(eventName) && nonempty(parsed.type) && eventName !== parsed.type) {
        judgment = { outcome: 'incomplete', terminal: null, error: 'Conflicting stream event type' }
        eventName = ''
        update()
        return
      }
      const type = nonempty(parsed.type) ? parsed.type : eventName
      eventName = ''
      if (parsed.error != null || type === 'error' || type === 'response.failed') {
        judgment = {
          outcome: 'failed',
          terminal: type || 'error',
          error: 'Upstream reported an error',
        }
        update()
        return
      }
      if (options.protocol === 'openai-responses') {
        if (
          (type === 'response.output_text.delta' ||
            type === 'response.function_call_arguments.delta') &&
          nonempty(parsed.delta)
        )
          observeOutput()
        if (type === 'response.incomplete') {
          const incomplete = judge('openai-responses', parsed.response)
          judgment =
            incomplete?.outcome === 'incomplete'
              ? incomplete
              : {
                  outcome: 'incomplete',
                  terminal: type,
                  error: 'Response incomplete',
                }
        }
        if (type === 'response.completed') {
          const candidate = judge('openai-responses', parsed.response)
          judgment =
            candidate?.outcome === 'completed'
              ? { ...candidate, terminal: type }
              : { outcome: 'incomplete', terminal: type, error: 'Invalid terminal response' }
          if (candidate?.outcome === 'completed' && responsesHasOutput(parsed.response))
            observeOutput()
        }
      } else if (options.protocol === 'openai-completions') {
        if (Array.isArray(parsed.choices))
          for (const [position, choice] of parsed.choices.entries())
            if (record(choice)) {
              const index = candidateIndex(choice.index, position)
              if (index === null) {
                judgment = { outcome: 'incomplete', terminal: null, error: 'Invalid choice index' }
                break
              }
              const state = candidates.get(index) ?? { content: false, finish: null }
              if (record(choice.delta)) {
                if (nonempty(choice.delta.content)) {
                  state.content = true
                  observeOutput()
                }
                if (choice.delta.function_call != null) {
                  state.functionCall ??= { name: false, arguments: false }
                  if (!updateToolParts(state.functionCall, choice.delta.function_call))
                    state.invalidTool = true
                  if (state.functionCall.name || state.functionCall.arguments) {
                    state.content = true
                    observeOutput()
                  }
                }
                if (choice.delta.tool_calls != null) {
                  state.toolCalls ??= new Map()
                  if (!Array.isArray(choice.delta.tool_calls)) state.invalidTool = true
                  else
                    for (const [position, call] of choice.delta.tool_calls.entries()) {
                      if (!record(call)) {
                        state.invalidTool = true
                        continue
                      }
                      const toolIndex = candidateIndex(call.index, position)
                      if (toolIndex === null) {
                        state.invalidTool = true
                        continue
                      }
                      const parts = state.toolCalls.get(toolIndex) ?? {
                        name: false,
                        arguments: false,
                      }
                      if (call.function != null && !updateToolParts(parts, call.function))
                        state.invalidTool = true
                      state.toolCalls.set(toolIndex, parts)
                      if (parts.name || parts.arguments) {
                        state.content = true
                        observeOutput()
                      }
                    }
                }
              }
              if (nonempty(choice.finish_reason)) {
                if (state.finish && state.finish !== choice.finish_reason) {
                  judgment = {
                    outcome: 'incomplete',
                    terminal: state.finish,
                    error: 'Conflicting choice finish reasons',
                  }
                  break
                }
                state.finish = choice.finish_reason
              }
              candidates.set(index, state)
            }
      } else if (options.protocol === 'anthropic') {
        if (
          type === 'content_block_start' &&
          record(parsed.content_block) &&
          parsed.content_block.type === 'tool_use'
        ) {
          const index = candidateIndex(parsed.index, 0)
          if (
            index === null ||
            anthropicTools.has(index) ||
            !nonempty(parsed.content_block.id) ||
            !nonempty(parsed.content_block.name) ||
            !record(parsed.content_block.input)
          )
            invalidAnthropicTool = true
          else {
            anthropicTools.set(index, '')
            validStreamContent = true
            observeOutput()
          }
        }
        if (type === 'content_block_delta' && record(parsed.delta)) {
          if (parsed.delta.type === 'input_json_delta') {
            const index = candidateIndex(parsed.index, 0)
            if (
              index === null ||
              !anthropicTools.has(index) ||
              typeof parsed.delta.partial_json !== 'string'
            )
              invalidAnthropicTool = true
            else {
              anthropicToolInputSize += parsed.delta.partial_json.length
              if (anthropicToolInputSize > bodyLimit) invalidAnthropicTool = true
              else anthropicTools.set(index, anthropicTools.get(index)! + parsed.delta.partial_json)
              if (nonempty(parsed.delta.partial_json)) observeOutput()
            }
          } else if (nonempty(parsed.delta.text)) {
            validStreamContent = true
            observeOutput()
          }
        }
        if (type === 'message_delta' && record(parsed.delta) && nonempty(parsed.delta.stop_reason))
          anthropicFinish = parsed.delta.stop_reason
        if (type === 'message_stop' && anthropicFinish) {
          for (const input of anthropicTools.values()) {
            if (!input) continue
            try {
              if (!record(JSON.parse(input))) invalidAnthropicTool = true
            } catch {
              invalidAnthropicTool = true
            }
          }
          judgment = {
            outcome:
              anthropicFinish === 'end_turn' ||
              anthropicFinish === 'stop_sequence' ||
              anthropicFinish === 'tool_use'
                ? validStreamContent &&
                  !invalidAnthropicTool &&
                  (anthropicFinish !== 'tool_use' || anthropicTools.size > 0)
                  ? 'completed'
                  : 'incomplete'
                : 'incomplete',
            terminal: anthropicFinish,
            error: null,
          }
        }
      } else {
        if (Array.isArray(parsed.candidates))
          for (const [position, candidate] of parsed.candidates.entries())
            if (record(candidate)) {
              const index = candidateIndex(candidate.index, position)
              if (index === null) {
                judgment = {
                  outcome: 'incomplete',
                  terminal: null,
                  error: 'Invalid candidate index',
                }
                break
              }
              const state = candidates.get(index) ?? { content: false, finish: null }
              if (validGeminiContent(candidate.content)) {
                state.content = true
                observeOutput()
              }
              if (nonempty(candidate.finishReason)) {
                if (state.finish && state.finish !== candidate.finishReason) {
                  judgment = {
                    outcome: 'incomplete',
                    terminal: state.finish,
                    error: 'Conflicting candidate finish reasons',
                  }
                  break
                }
                state.finish = candidate.finishReason
              }
              candidates.set(index, state)
            }
        if (record(parsed.promptFeedback) && nonempty(parsed.promptFeedback.blockReason))
          judgment = {
            outcome: 'incomplete',
            terminal: parsed.promptFeedback.blockReason,
            error: 'Prompt blocked',
          }
      }
      update()
    }
    const consumeLine = (line: string) => {
      if (line === '') {
        handleEvent()
        return
      }
      if (line.startsWith(':')) return
      const colon = line.indexOf(':')
      const field = colon < 0 ? line : line.slice(0, colon)
      let value = colon < 0 ? '' : line.slice(colon + 1)
      if (value.startsWith(' ')) value = value.slice(1)
      eventSize += line.length
      if (eventSize > eventLimit) throw new Error('Event too large')
      if (field === 'data') dataLines.push(value)
      if (field === 'event') eventName = value
    }
    try {
      while (true) {
        if (options.signal.aborted) return finish('canceled', null, 'Request canceled')
        const part = await reader.read()
        if (options.signal.aborted) return finish('canceled', null, 'Request canceled')
        if (part.done) {
          buffer += decoder.decode()
          break
        }
        buffer += decoder.decode(part.value, { stream: true })
        if (!stream) {
          if (buffer.length > bodyLimit) {
            cancelReader()
            return finish('incomplete', null, 'Response body too large')
          }
          continue
        }
        let pos = 0
        for (let i = 0; i < buffer.length; i++) {
          if (buffer[i] !== '\r' && buffer[i] !== '\n') continue
          if (buffer[i] === '\r' && i === buffer.length - 1) break
          consumeLine(buffer.slice(pos, i))
          if (buffer[i] === '\r' && buffer[i + 1] === '\n') i++
          pos = i + 1
          if (judgment || options.signal.aborted) break
        }
        if (options.signal.aborted) return finish('canceled', null, 'Request canceled')
        buffer = buffer.slice(pos)
        if (buffer.length + eventSize > eventLimit && !judgment) throw new Error('Event too large')
        if (judgment) {
          cancelReader()
          break
        }
      }
      if (!stream) {
        append(buffer)
        let value: unknown
        try {
          value = JSON.parse(buffer)
        } catch {
          return finish('incomplete', null, 'Malformed JSON response')
        }
        const terminal = judge(options.protocol, value, expected)
        if (!terminal) return finish('incomplete', null, 'Invalid or nonterminal response')
        if (
          terminal.outcome === 'completed' &&
          (options.protocol !== 'openai-responses' || responsesHasOutput(value))
        )
          observeOutput()
        return finish(terminal.outcome, terminal.terminal, terminal.error)
      }
      if (!judgment && buffer.endsWith('\r')) {
        consumeLine(buffer.slice(0, -1))
        buffer = ''
      }
      const unfinishedEvent =
        dataLines.length > 0 || eventName !== '' || (buffer !== '' && !buffer.startsWith(':'))
      if (!judgment && options.protocol === 'gemini' && !unfinishedEvent)
        judgment = aggregateCandidates(candidates, expected, 'gemini')
      if (judgment) {
        const terminal: Judgement = judgment
        return finish(terminal.outcome, terminal.terminal, terminal.error)
      }
      return finish(
        'incomplete',
        null,
        outputSeen ? 'Stream ended after partial output' : 'Stream ended without terminal response',
      )
    } catch (error) {
      if (error instanceof Error && error.message === 'Event too large')
        return finish('incomplete', null, 'Stream event too large')
      throw error
    } finally {
      options.signal.removeEventListener('abort', cancelReader)
      cancelReader()
      reader.releaseLock()
    }
  } catch {
    return options.signal.aborted
      ? finish('canceled', null, 'Request canceled')
      : finish('incomplete', null, 'Request or response interrupted')
  }
}
