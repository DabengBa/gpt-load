import * as stylex from '@stylexjs/stylex'
import { Toast } from '@astryxdesign/core'
import { useSyncExternalStore } from 'react'
import { createPortal } from 'react-dom'

import type { ToastTone } from '@shared/controllers/toast'

import { useAppServices } from './services'

const styles = stylex.create({
  host: {
    position: 'fixed',
    zIndex: 1100,
    bottom: 26,
    left: '50%',
    transform: 'translateX(-50%)',
    maxWidth: 'calc(100vw - 32px)',
  },
})

function toastType(tone: ToastTone): 'info' | 'error' {
  return tone === 'danger' ? 'error' : 'info'
}

// Single-toast host mirroring AppToastViewport.vue: the shared controller owns
// the message, tone, and auto-dismiss timing; Toast supplies the card chrome.
export function ToastHost() {
  const { toast } = useAppServices()
  const current = useSyncExternalStore(toast.subscribe, toast.getCurrent)
  if (current === null) return null
  return createPortal(
    <div {...stylex.props(styles.host)}>
      <Toast
        key={current.id}
        type={toastType(current.tone)}
        body={current.message}
        isAutoHide={false}
        autoHideDuration={0}
        onDismiss={() => toast.dismiss()}
      />
    </div>,
    document.body,
  )
}
