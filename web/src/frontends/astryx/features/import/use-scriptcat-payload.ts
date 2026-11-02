import { useEffect, useState } from 'react'

import { consumeScriptcatImportPayload } from '@shared/domain/import/scriptcat-payload'

export function useScriptcatPayload() {
  const [payload, setPayload] = useState(() => {
    try {
      return typeof window === 'undefined'
        ? null
        : consumeScriptcatImportPayload(window.sessionStorage)
    } catch {
      return null
    }
  })

  useEffect(() => {
    let pollCount = 0
    function receivePayload(): void {
      try {
        const next = consumeScriptcatImportPayload(window.sessionStorage)
        if (next !== null) {
          window.clearInterval(pollTimer)
          setPayload(next)
        }
      } catch {
        // Storage access can be blocked; manual import remains available.
      }
    }
    window.addEventListener('gpt-load:scriptcat-import', receivePayload)
    const pollTimer = window.setInterval(() => {
      receivePayload()
      pollCount += 1
      if (pollCount >= 20) window.clearInterval(pollTimer)
    }, 100)
    receivePayload()
    return () => {
      window.removeEventListener('gpt-load:scriptcat-import', receivePayload)
      window.clearInterval(pollTimer)
    }
  }, [])

  return payload
}
