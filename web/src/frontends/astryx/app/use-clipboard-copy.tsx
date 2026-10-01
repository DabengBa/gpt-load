import { useRouter, useRouterState } from '@tanstack/react-router'
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'

import { copyText } from '@shared/lib/clipboard'
import { registerEphemeralStateCleaner } from '@shared/controllers/ephemeral-state'

import { CopyFallbackDialog } from '../components/CopyButton'

export type ClipboardCopyResult = 'success' | 'fallback' | 'cancelled'

export interface ClipboardCopy {
  copy(source: string | (() => string | Promise<string>)): Promise<ClipboardCopyResult>
  fallbackText: string | undefined
  pending: boolean
  reset(): void
  dialog: ReactNode
}

// React port of the classic useClipboardCopy composable: sequence-guarded
// copy with a manual fallback dialog, reset on route change, and registration
// with the ephemeral-state cleaner so revealed secrets never outlive the page.
export function useClipboardCopy(): ClipboardCopy {
  const router = useRouter()
  const [fallbackText, setFallbackText] = useState<string | undefined>(undefined)
  const [pending, setPending] = useState(false)
  const sequenceRef = useRef(0)
  const disposedRef = useRef(false)

  const reset = useCallback(() => {
    sequenceRef.current += 1
    setFallbackText(undefined)
    setPending(false)
  }, [])

  const copy = useCallback(
    async (source: string | (() => string | Promise<string>)): Promise<ClipboardCopyResult> => {
      if (disposedRef.current) return 'cancelled'
      reset()
      const operation = ++sequenceRef.current
      // The location is captured per operation and compared live: a route
      // change mid-copy invalidates the pending write even before the
      // component re-renders (classic watches route.fullPath the same way).
      const path = router.state.location.href
      const isCurrent = () =>
        !disposedRef.current &&
        sequenceRef.current === operation &&
        router.state.location.href === path
      setPending(true)
      try {
        const value = typeof source === 'function' ? await source() : source
        if (!isCurrent()) return 'cancelled'
        const copied = await copyText(value, undefined, isCurrent)
        if (!isCurrent()) return 'cancelled'
        if (copied) return 'success'
        setFallbackText(value)
        return 'fallback'
      } catch (error) {
        if (!isCurrent()) return 'cancelled'
        throw error
      } finally {
        if (isCurrent()) setPending(false)
      }
    },
    [reset, router],
  )

  // Route changes hide the fallback dialog and pending state — render-phase
  // adjustment mirroring the classic watch on route.fullPath.
  const fullPath = useRouterState({ select: (state) => state.location.href })
  const [lastFullPath, setLastFullPath] = useState(fullPath)
  if (lastFullPath !== fullPath) {
    setLastFullPath(fullPath)
    setFallbackText(undefined)
    setPending(false)
  }

  useEffect(() => {
    const unregister = registerEphemeralStateCleaner(reset)
    return () => {
      disposedRef.current = true
      unregister()
    }
  }, [reset])

  return {
    copy,
    fallbackText,
    pending,
    reset,
    dialog: <CopyFallbackDialog value={fallbackText} onClose={reset} />,
  }
}
