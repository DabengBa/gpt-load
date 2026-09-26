import * as stylex from '@stylexjs/stylex'
import { Dialog, DialogHeader } from '@astryxdesign/core/Dialog'
import { Layout, LayoutContent, LayoutFooter } from '@astryxdesign/core/Layout'
import { useEffect, useRef, type ReactNode } from 'react'

// Modal side panel — the Astryx counterpart of the classic AppDrawer.
//
// Spike (b) decision (ADR-0002): Astryx Dialog with `position={{ end: 0 }}`
// anchors the surface to the trailing edge; xstyle supplies the drawer chrome
// (full height, no radius, leading border, full-bleed <=520px). The native
// <dialog> provides focus trap, Escape, scrim dismissal, and invoker focus
// return — the entire a11y contract without a swizzle. BottomSheet was ruled
// out: its bottom-anchored gesture/snap model is a mobile-only interaction
// that does not match a desktop side panel, and suppressing handle/gestures
// is a heavier override than xstyle restyling.
const panelSlideIn = stylex.keyframes({
  from: { transform: 'translateX(100%)' },
  to: { transform: 'translateX(0)' },
})

const styles = stylex.create({
  panel: {
    // The dialog's own maxWidth keeps spacing-token gutters; a drawer goes
    // full-bleed under 520px like the classic one, so the clamp must lift.
    width: {
      default: 'min(92vw, 520px)',
      '@media (max-width: 520px)': '100vw',
    },
    maxWidth: {
      default: null,
      '@media (max-width: 520px)': '100vw',
    },
    height: '100dvh',
    margin: 0,
    borderRadius: 0,
    borderWidth: 0,
    borderInlineStartWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    boxShadow: 'var(--shadow-overlay, -16px 0 44px rgba(0, 0, 0, 0.12))',
    animationName: {
      default: panelSlideIn,
      '@media (prefers-reduced-motion: reduce)': 'none',
    },
    animationDuration: 'var(--duration-normal, 200ms)',
    animationTimingFunction: 'var(--easing-standard, ease-out)',
    animationFillMode: 'backwards',
  },
  layout: {
    height: '100%',
  },
  content: {
    paddingBlock: 14,
    paddingInline: 18,
  },
})

export interface DetailPanelProps {
  isOpen: boolean
  onOpenChange: (isOpen: boolean) => void
  /** DialogHeader title; also names the dialog via aria-labelledby. */
  title: string
  /** Optional subtitle rendered under the title (classic drawer description). */
  subtitle?: string
  /**
   * Mirrors the classic `dismissible`: false blocks Escape, scrim clicks, and
   * the close button. Defaults to true.
   */
  dismissible?: boolean
  footer?: ReactNode
  children: ReactNode
}

export function DetailPanel({
  isOpen,
  onOpenChange,
  title,
  subtitle,
  dismissible = true,
  footer,
  children,
}: DetailPanelProps) {
  const dialogRef = useRef<HTMLDialogElement | null>(null)

  // Hard focus containment. A native modal <dialog> inerts the page, but
  // Chromium wraps sequential focus through <body> at the tab-order edges —
  // silently, without focus events — while the classic reka-ui drawer keeps
  // activeElement strictly inside via sentinels. Parity therefore needs the
  // same mechanism reka uses: intercept Tab at the scope boundary.
  const containKeyDown = (event: React.KeyboardEvent<HTMLDialogElement>) => {
    if (event.key !== 'Tab') return
    const dialog = event.currentTarget
    if (!dialog.open) return
    const tabbables = dialog.querySelectorAll<HTMLElement>(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    )
    if (tabbables.length === 0) return
    const first = tabbables[0]
    const last = tabbables[tabbables.length - 1]
    const active = document.activeElement
    if (event.shiftKey ? active === first : active === last) {
      event.preventDefault()
      ;(event.shiftKey ? last : first).focus()
    }
  }

  // Covers focus escapes that do dispatch events (programmatic .focus() calls
  // land on inert-ineligible targets). Cleanup runs before Dialog's isOpen→
  // false effect, so the invoker focus restore on close is not intercepted.
  useEffect(() => {
    if (!isOpen) return
    const contain = (event: FocusEvent) => {
      const dialog = dialogRef.current
      const target = event.target
      if (!dialog?.open || !(target instanceof Node) || dialog.contains(target)) {
        return
      }
      // A focusable inside a nested top-layer overlay (a stacked Dialog or a
      // Layer popover portaled past this dialog's subtree) is a legitimate
      // target, not an escape — only recapture focus with no overlay ancestor.
      if (target instanceof Element && target.closest('dialog, [popover]')) {
        return
      }
      const anchor =
        dialog.querySelector<HTMLElement>('[data-autofocus]') ?? dialog
      anchor.focus()
    }
    document.addEventListener('focusin', contain)
    return () => document.removeEventListener('focusin', contain)
  }, [isOpen])

  return (
    <Dialog
      ref={dialogRef}
      isOpen={isOpen}
      onKeyDown={containKeyDown}
      // 'info' allows Escape + scrim; 'required' swallows both like the
      // classic non-dismissible contract (the close button also disappears).
      purpose={dismissible ? 'info' : 'required'}
      onOpenChange={dismissible ? onOpenChange : () => {}}
      position={{ end: 0, top: 0 }}
      width="min(92vw, 520px)"
      maxHeight="100dvh"
      padding={0}
      xstyle={styles.panel}
    >
      <Layout
        xstyle={styles.layout}
        header={
          <DialogHeader
            title={title}
            subtitle={subtitle}
            onOpenChange={dismissible ? onOpenChange : undefined}
            hasDivider
          />
        }
        // data-autofocus: Dialog focuses the first marked descendant after
        // showModal (component autofocus runs before the dialog opens and is
        // silently dropped). The scrollable content region is the right target
        // — same shape as reka-ui focusing the drawer content element.
        content={
          <LayoutContent
            isScrollable
            xstyle={styles.content}
            tabIndex={-1}
            data-autofocus
          >
            {children}
          </LayoutContent>
        }
        footer={
          footer !== undefined ? <LayoutFooter hasDivider>{footer}</LayoutFooter> : undefined
        }
      />
    </Dialog>
  )
}
