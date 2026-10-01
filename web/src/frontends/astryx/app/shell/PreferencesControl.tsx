import { IconButton } from '@astryxdesign/core/IconButton'
import { Popover } from '@astryxdesign/core/Popover'
import * as stylex from '@stylexjs/stylex'
import { useRouter } from '@tanstack/react-router'
import { LogOut, Menu, Monitor, Moon, Sun } from 'lucide-react'
import { useEffect, useId, useState, useSyncExternalStore, type ReactNode } from 'react'

import { supportedLocales, type AppLocale } from '@shared/preferences/locale'
import type { AppTheme } from '@shared/controllers/theme'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../i18n'
import { useAppServices } from '../services'
import { useThemePreference } from '../../theme/theme-preference'

const styles = stylex.create({
  panel: {
    display: 'grid',
    width: '100%',
    gap: '10px',
    minWidth: '216px',
  },
  group: {
    display: 'grid',
    gap: '6px',
    minWidth: 0,
  },
  label: {
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
    color: 'var(--color-text-secondary)',
    padding: '0 2px',
    fontSize: 'var(--text-supporting-size)',
    fontWeight: 400,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
  },
  segments: {
    display: 'grid',
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control, rgba(0,0,0,0.16))',
    borderRadius: 'var(--radius-md, 7px)',
  },
  segmentsThree: {
    gridTemplateColumns: 'repeat(3, minmax(0, 1fr))',
  },
  segment: {
    position: 'relative',
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    justifyContent: 'center',
    gap: '4px',
    borderLeftWidth: '1px',
    borderLeftStyle: 'solid',
    borderLeftColor: 'var(--color-border-control, rgba(0,0,0,0.16))',
    backgroundColor: 'var(--color-surface, #fff)',
    color: 'var(--color-text-secondary)',
    padding: '6px 4px',
    fontSize: 'var(--text-body-size)',
    cursor: 'pointer',
    textAlign: 'center',
  },
  segmentFirst: {
    borderLeftWidth: 0,
  },
  segmentChecked: {
    backgroundColor: 'var(--color-text-primary)',
    color: 'var(--color-surface, #fff)',
    fontWeight: 560,
  },
  segmentInput: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    opacity: 0,
  },
  divider: {
    height: '1px',
    margin: '0 -10px',
    backgroundColor: 'var(--color-border-subtle, rgba(0,0,0,0.08))',
  },
  action: {
    display: 'flex',
    width: '100%',
    alignItems: 'center',
    gap: 'var(--space-2, 8px)',
    borderWidth: 0,
    borderRadius: 'var(--radius-md, 7px)',
    backgroundColor: {
      default: 'transparent',
      ':hover': 'var(--color-surface-sunken, rgba(0,0,0,0.04))',
    },
    color: 'var(--color-text-primary)',
    padding: '7px 6px',
    fontSize: '12.5px',
    cursor: 'pointer',
  },
  mobileNav: {
    display: { default: 'none', '@media (max-width: 860px)': 'block' },
  },
})

interface SegmentOption<T extends string> {
  value: T
  labelKey: MessageId
  icon?: ReactNode
}

function Segments<T extends string>({
  name,
  label,
  options,
  value,
  onChange,
}: {
  name: string
  label: string
  options: readonly SegmentOption<T>[]
  value: T
  onChange(value: T): void
}) {
  const t = useT()
  return (
    <div {...stylex.props(styles.group)}>
      <span {...stylex.props(styles.label)}>{label}</span>
      <div {...stylex.props(styles.segments, styles.segmentsThree)} role="group" aria-label={label}>
        {options.map((option, index) => {
          const checked = option.value === value
          return (
            <label
              key={option.value}
              {...stylex.props(
                styles.segment,
                index === 0 && styles.segmentFirst,
                checked && styles.segmentChecked,
              )}
            >
              <input
                {...stylex.props(styles.segmentInput)}
                type="radio"
                name={name}
                value={option.value}
                checked={checked}
                onChange={() => onChange(option.value)}
              />
              {option.icon}
              <span>{t(option.labelKey)}</span>
            </label>
          )
        })}
      </div>
    </div>
  )
}

const themeOptions: readonly SegmentOption<AppTheme>[] = [
  { value: 'system', labelKey: 'shell.themeSystem', icon: <Monitor size={14} aria-hidden /> },
  { value: 'light', labelKey: 'shell.themeLight', icon: <Sun size={14} aria-hidden /> },
  { value: 'dark', labelKey: 'shell.themeDark', icon: <Moon size={14} aria-hidden /> },
]

const localeOptions: readonly SegmentOption<AppLocale>[] = [
  { value: 'zh-CN', labelKey: 'shell.localeZhShort' },
  { value: 'en-US', labelKey: 'shell.localeEnShort' },
  { value: 'ja-JP', labelKey: 'shell.localeJaShort' },
]

export function PreferencesPanel({
  showSignOut = false,
  mobileNav,
  onSignOut,
}: {
  showSignOut?: boolean
  mobileNav?: ReactNode
  onSignOut?(): void
}) {
  const t = useT()
  const services = useAppServices()
  const identity = useId()
  const [theme, setTheme] = useThemePreference()
  const locale = useSyncExternalStore(services.i18n.subscribe, services.i18n.getSnapshot).locale

  return (
    <div {...stylex.props(styles.panel)}>
      {mobileNav !== undefined && (
        <>
          <div {...stylex.props(styles.mobileNav)}>{mobileNav}</div>
          <div {...stylex.props(styles.mobileNav)} role="separator">
            <div {...stylex.props(styles.divider)} />
          </div>
        </>
      )}
      <Segments
        name={`${identity}-theme`}
        label={t('shell.theme')}
        options={themeOptions}
        value={theme}
        onChange={setTheme}
      />
      <Segments
        name={`${identity}-locale`}
        label={t('shell.language')}
        options={localeOptions}
        value={locale}
        onChange={(next) => {
          if (supportedLocales.includes(next)) void services.i18n.setLocale(next)
        }}
      />
      {showSignOut && <div {...stylex.props(styles.divider)} />}
      {showSignOut && (
        <button {...stylex.props(styles.action)} type="button" onClick={onSignOut}>
          <LogOut size={15} aria-hidden />
          {t('shell.signOut')}
        </button>
      )}
    </div>
  )
}

export function PreferencesControl({
  triggerLabel,
  showSignOut = false,
  mobileNav,
  onSignOut,
}: {
  triggerLabel?: string
  showSignOut?: boolean
  mobileNav?: ReactNode
  onSignOut?(): void
}) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const router = useRouter()
  useEffect(() => {
    if (!open || mobileNav === undefined) return
    return router.subscribe('onResolved', () => setOpen(false))
  }, [open, mobileNav, router])
  return (
    <Popover
      isOpen={open}
      onOpenChange={setOpen}
      placement="below"
      alignment="end"
      label={triggerLabel || t('shell.preferences')}
      content={
        <PreferencesPanel
          showSignOut={showSignOut}
          mobileNav={mobileNav}
          onSignOut={() => {
            setOpen(false)
            onSignOut?.()
          }}
        />
      }
    >
      <IconButton
        label={triggerLabel || t('shell.preferences')}
        icon={<Menu size={15} aria-hidden />}
        variant="secondary"
        className="preferences-trigger"
      />
    </Popover>
  )
}
