import * as stylex from '@stylexjs/stylex'
import { Popover, TextInput } from '@astryxdesign/core'
import { ChevronDown, Search } from 'lucide-react'
import { useId, useMemo, useState, type KeyboardEvent } from 'react'

import type { AccessProtocol } from '@shared/control/types'
import type { MessageId } from '@shared/i18n/message-ids'
import {
  clientGroup,
  gatewayClients,
  type GatewayClient,
  type GatewayClientGroup,
  type GatewayClientID,
} from '@shared/domain/home/gateway-clients'

import { useT } from '../../app/i18n'
import { ChannelIcon } from '../../components/ChannelIcon'

const styles = stylex.create({
  trigger: {
    display: 'inline-flex',
    minWidth: 190,
    minHeight: 'var(--control-sm)',
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: {
      default: 'var(--color-border-control)',
      ':hover:not(:disabled)': 'var(--color-text-faint)',
    },
    borderRadius: 'var(--radius-control)',
    backgroundColor: {
      default: 'var(--color-surface)',
      ':hover:not(:disabled)': 'var(--color-interactive-hover)',
    },
    color: 'var(--color-text)',
    paddingTop: 0,
    paddingBottom: 0,
    paddingInline: 10,
    fontFamily: 'inherit',
    fontSize: 'var(--text-button)',
    fontWeight: 560,
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    transitionProperty: 'border-color, background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
    opacity: { ':disabled': 0.55 },
  },
  triggerIcon: {
    display: 'inline-flex',
    width: 18,
    minWidth: 18,
    height: 18,
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 16,
  },
  triggerCurrent: {
    flex: '1',
    minWidth: 0,
    overflow: 'hidden',
    textAlign: 'left',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  triggerChevron: {
    flex: 'none',
    color: 'var(--color-text-faint)',
  },
  panel: {
    paddingTop: 8,
    paddingBottom: 8,
    paddingInline: 8,
  },
  list: {
    display: 'grid',
    gap: 1,
    marginTop: 'var(--space-3)',
    maxHeight: 320,
    overflowY: 'auto',
  },
  group: {
    margin: 0,
    color: 'var(--color-text-faint)',
    paddingTop: 9,
    paddingBottom: 3,
    paddingInline: 7,
    fontSize: 'var(--text-label-xs)',
    fontWeight: 700,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
  option: {
    display: 'grid',
    width: '100%',
    gridTemplateColumns: 'auto minmax(0, 1fr)',
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 0,
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'transparent',
    color: 'var(--color-text)',
    paddingTop: 7,
    paddingBottom: 7,
    paddingInline: 8,
    fontFamily: 'inherit',
    fontSize: 'var(--text-meta)',
    textAlign: 'left',
    cursor: 'pointer',
  },
  optionActive: {
    backgroundColor: 'var(--color-interactive-hover)',
  },
  optionSelected: {
    backgroundColor: 'var(--color-surface-sunken)',
    fontWeight: 600,
  },
  // No `disabled`: that would hide the row from keyboard and screen readers —
  // the dimmed look carries the "not for this key" signal instead.
  optionBlocked: {
    opacity: 0.5,
  },
  optionIcon: {
    display: 'inline-flex',
    width: 20,
    minWidth: 20,
    height: 20,
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: 17,
  },
  optionName: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  empty: {
    margin: 0,
    color: 'var(--color-text-faint)',
    paddingTop: 10,
    paddingBottom: 10,
    paddingInline: 8,
    fontSize: 'var(--text-meta)',
  },
})

const groupOrder: readonly GatewayClientGroup[] = ['commandLine', 'desktop', 'web']

export function ClientPicker({
  value,
  protocols,
  disabled = false,
  onChange,
}: {
  value: GatewayClientID
  protocols: readonly AccessProtocol[]
  disabled?: boolean
  onChange(id: GatewayClientID): void
}) {
  const t = useT()
  const identity = useId()
  const listID = `${identity}-clients`
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)

  const selected =
    gatewayClients.find((entry) => entry.id === value) ?? gatewayClients[0]!

  const label = (entry: GatewayClient): string =>
    t(`home.ledger.connection.clients.${entry.id}` as MessageId)

  // cc-switch has no fixed requiredProtocol (it depends on the target app), so
  // only declared requiredProtocol entries can be flagged unsupported.
  const unsupported = (entry: GatewayClient): boolean =>
    Boolean(entry.requiredProtocol && !protocols.includes(entry.requiredProtocol))

  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase()
    if (needle === '') return [...gatewayClients]
    return gatewayClients.filter(
      (entry) =>
        label(entry).toLocaleLowerCase().includes(needle) ||
        entry.id.includes(needle) ||
        entry.searchTerms.some((term) => term.includes(needle)),
    )
    // eslint-disable-next-line react-hooks/exhaustive-deps -- label reads `t`
  }, [query])

  const sections = useMemo(() => {
    const available = matches.filter((entry) => !unsupported(entry))
    const blocked = matches.filter((entry) => unsupported(entry))
    const result: Array<{ key: string; title: string; entries: GatewayClient[] }> = []
    for (const group of groupOrder) {
      const entries = available.filter((entry) => clientGroup(entry.kind) === group)
      if (entries.length > 0) {
        result.push({
          key: group,
          title: t(`home.ledger.connection.groups.${group}` as MessageId),
          entries,
        })
      }
    }
    if (blocked.length > 0) {
      result.push({
        key: 'unsupported',
        title: t('home.ledger.connection.groups.unsupported'),
        entries: blocked,
      })
    }
    return result
    // eslint-disable-next-line react-hooks/exhaustive-deps -- unsupported reads protocols
  }, [matches, protocols])

  const flat = useMemo(() => sections.flatMap((section) => section.entries), [sections])

  // Route/query changes mid-edit reset the cursor to the first row — same as
  // the classic watch on the flattened list.
  const [lastFlat, setLastFlat] = useState(flat)
  if (lastFlat !== flat) {
    setLastFlat(flat)
    setActiveIndex(0)
  }

  function onOpenChange(next: boolean): void {
    setOpen(next)
    if (!next) {
      setQuery('')
      return
    }
    setActiveIndex(Math.max(flat.findIndex((entry) => entry.id === value), 0))
  }

  function choose(entry: GatewayClient): void {
    if (disabled) return
    setOpen(false)
    if (entry.id !== value) onChange(entry.id)
  }

  function onSearchKeyDown(event: KeyboardEvent<HTMLInputElement>): void {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setActiveIndex((index) => Math.min(index + 1, flat.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActiveIndex((index) => Math.max(index - 1, 0))
    } else if (event.key === 'Enter') {
      const entry = flat[activeIndex]
      if (entry) {
        event.preventDefault()
        choose(entry)
      }
    }
  }

  return (
    <Popover
      isOpen={open}
      onOpenChange={onOpenChange}
      placement="below"
      alignment="start"
      width={300}
      hasAutoFocus
      label={t('home.ledger.connection.selectClient')}
      content={
        <div {...stylex.props(styles.panel)}>
          <TextInput
            label={t('home.ledger.connection.searchClients')}
            isLabelHidden
            placeholder={t('home.ledger.connection.searchClientsPlaceholder')}
            value={query}
            onChange={setQuery}
            onKeyDown={onSearchKeyDown}
            hasClear
            startIcon={<Search size={14} aria-hidden="true" />}
            size="sm"
          />
          <div id={listID} {...stylex.props(styles.list)}>
            {sections.map((section) => (
              <div key={section.key} role="group" aria-label={section.title}>
                <p {...stylex.props(styles.group)}>{section.title}</p>
                {section.entries.map((entry) => {
                  const index = flat.findIndex((candidate) => candidate.id === entry.id)
                  return (
                    <button
                      key={entry.id}
                      {...stylex.props(
                        styles.option,
                        flat[activeIndex]?.id === entry.id && styles.optionActive,
                        entry.id === value && styles.optionSelected,
                        section.key === 'unsupported' && styles.optionBlocked,
                      )}
                      type="button"
                      aria-current={entry.id === value ? 'true' : undefined}
                      onClick={() => choose(entry)}
                      onMouseEnter={() => setActiveIndex(index)}
                    >
                      <span {...stylex.props(styles.optionIcon)}>
                        <ChannelIcon icon={entry.icon} mark={entry.mark} />
                      </span>
                      <span {...stylex.props(styles.optionName)}>{label(entry)}</span>
                    </button>
                  )
                })}
              </div>
            ))}
            {sections.length === 0 && (
              <p {...stylex.props(styles.empty)}>
                {t('home.ledger.connection.noClientMatches')}
              </p>
            )}
          </div>
        </div>
      }
    >
      {(triggerProps) => (
        <button
          {...triggerProps}
          {...stylex.props(styles.trigger)}
          type="button"
          disabled={disabled}
          aria-controls={listID}
          aria-label={t('home.ledger.connection.selectClient')}
        >
          <span {...stylex.props(styles.triggerIcon)}>
            <ChannelIcon icon={selected.icon} mark={selected.mark} />
          </span>
          <span {...stylex.props(styles.triggerCurrent)}>{label(selected)}</span>
          <ChevronDown {...stylex.props(styles.triggerChevron)} size={14} aria-hidden="true" />
        </button>
      )}
    </Popover>
  )
}
