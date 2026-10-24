import * as stylex from '@stylexjs/stylex'
import { Button, Popover, TextInput } from '@astryxdesign/core'
import { ChevronDown, Search } from 'lucide-react'
import { useEffect, useId, useState, type KeyboardEvent } from 'react'

import type { ChannelDto } from '@shared/control/resources/channels'

import { useT } from '@app/app/i18n'
import { ChannelIcon } from './ChannelIcon'
import { InlineNotice } from './InlineNotice'

const MAX_FEATURED_CHANNELS = 4
const FEATURED_CHANNEL_IDS = ['openai', 'anthropic', 'gemini', 'openai_compatible']

interface ChannelMatch {
  channel: ChannelDto
  rank: number
  reason: string
}

const styles = stylex.create({
  root: {
    minWidth: 0,
    borderBottomWidth: '1px',
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingTop: '22px',
    paddingBottom: 'var(--space-6)',
  },
  rootCompact: {
    borderBottomWidth: 0,
    paddingTop: 0,
    paddingBottom: 0,
  },
  header: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
    marginBottom: 'var(--space-3)',
  },
  heading: {
    margin: 0,
    fontSize: 'var(--title-section)',
    fontWeight: 650,
    letterSpacing: '-0.01em',
  },
  current: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  currentName: {
    color: 'var(--color-text-muted)',
    fontWeight: 560,
  },
  row: {
    position: 'relative',
  },
  selector: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  divider: {
    width: '1px',
    alignSelf: 'stretch',
    backgroundColor: 'var(--color-border-subtle)',
  },
  chips: {
    display: 'flex',
    minWidth: 0,
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
  },
  chip: {
    display: 'inline-flex',
    minHeight: {
      default: 'var(--control-sm)',
      '@media (max-width: 680px)': 'var(--touch-target)',
    },
    alignItems: 'center',
    gap: 6,
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: {
      default: 'var(--color-border-control)',
      ':hover:not(:disabled)': 'var(--color-action)',
    },
    borderRadius: 'var(--radius-control)',
    backgroundColor: {
      default: 'var(--color-surface)',
      ':hover:not(:disabled)': 'var(--color-action-soft)',
    },
    color: 'var(--color-text)',
    paddingBlock: 0,
    paddingInline: '10px',
    fontSize: 'var(--text-sm)',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.5 },
  },
  chipSelected: {
    borderColor: 'var(--color-action)',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    fontWeight: 560,
  },
  mobileSegment: {
    minHeight: { default: null, '@media (max-width: 680px)': 'var(--touch-target)' },
  },
  caret: {
    transitionProperty: 'transform',
    transitionDuration: 'var(--duration-fast)',
  },
  caretOpen: {
    transform: 'rotate(180deg)',
  },
  panel: {
    display: 'grid',
    gap: 'var(--space-2)',
    padding: 'var(--space-2)',
    minWidth: 0,
  },
  compatibleHelp: {
    margin: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  options: {
    display: 'grid',
    maxHeight: '260px',
    overflowY: 'auto',
  },
  option: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    borderWidth: 0,
    borderRadius: 'var(--radius-control)',
    backgroundColor: { default: 'transparent', ':hover': 'var(--color-surface-sunken)' },
    paddingBlock: '6px',
    paddingInline: '8px',
    textAlign: 'start',
    cursor: 'pointer',
    color: 'var(--color-text)',
    fontSize: 'var(--text-sm)',
  },
  optionActive: {
    backgroundColor: 'var(--color-surface-sunken)',
  },
  optionName: {
    minWidth: 0,
    flexGrow: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  optionReason: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    maxWidth: '40%',
  },
})

function highlightSegments(text: string, rawQuery: string): { text: string; matched: boolean }[] {
  const trimmed = rawQuery.trim()
  if (!trimmed) return [{ text, matched: false }]
  const at = text.toLocaleLowerCase().indexOf(trimmed.toLocaleLowerCase())
  if (at < 0) return [{ text, matched: false }]
  const segments: { text: string; matched: boolean }[] = []
  if (at > 0) segments.push({ text: text.slice(0, at), matched: false })
  segments.push({ text: text.slice(at, at + trimmed.length), matched: true })
  if (at + trimmed.length < text.length) {
    segments.push({ text: text.slice(at + trimmed.length), matched: false })
  }
  return segments
}

export interface ChannelPresetPickerProps {
  value: string | null
  channels: readonly ChannelDto[]
  selectedChannel: ChannelDto | null
  loading: boolean
  error: boolean
  disabled?: boolean
  hideHeader?: boolean
  compact?: boolean
  onSelect(channel: ChannelDto): void
  onRetry(): void
}

export function ChannelPresetPicker({
  value,
  channels,
  selectedChannel,
  loading,
  error,
  disabled = false,
  hideHeader = false,
  compact = false,
  onSelect,
  onRetry,
}: ChannelPresetPickerProps) {
  const t = useT()
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [activeIndex, setActiveIndex] = useState(0)
  const identity = useId()
  const optionId = (channelID: string) => `${identity}-channel-${channelID}`
  const channelListId = `${identity}-channel-list`
  const initialLoading = loading && channels.length === 0
  const loadFailed = error && channels.length === 0

  const byID = new Map(channels.map((channel) => [channel.channel_id, channel]))
  const preferredIDs = new Set(FEATURED_CHANNEL_IDS)
  const featuredChannels = [
    ...FEATURED_CHANNEL_IDS.map((id) => byID.get(id)).filter(
      (channel): channel is ChannelDto => channel !== undefined,
    ),
    ...channels.filter((channel) => !preferredIDs.has(channel.channel_id)),
  ].slice(0, MAX_FEATURED_CHANNELS)
  const activeChannels = channels
  const featuredChannelIDs = new Set(featuredChannels.map((channel) => channel.channel_id))
  const extraChannel =
    selectedChannel && !featuredChannelIDs.has(selectedChannel.channel_id) ? selectedChannel : null
  const otherChannels = activeChannels.filter(
    (channel) => !featuredChannelIDs.has(channel.channel_id),
  )

  function matchChannel(channel: ChannelDto, normalizedQuery: string): ChannelMatch | null {
    const name = channel.name.toLocaleLowerCase()
    if (name.startsWith(normalizedQuery)) return { channel, rank: 100, reason: '' }
    if (name.includes(normalizedQuery)) return { channel, rank: 80, reason: '' }
    const id = channel.channel_id
    if (id.startsWith(normalizedQuery)) return { channel, rank: 70, reason: id }
    if (id.includes(normalizedQuery)) return { channel, rank: 60, reason: id }
    for (const term of channel.search_terms) {
      const lower = term.toLocaleLowerCase()
      if (lower.startsWith(normalizedQuery)) return { channel, rank: 50, reason: term }
      if (lower.includes(normalizedQuery)) return { channel, rank: 30, reason: term }
    }
    // The server-side search this replaced also matched descriptions, which is
    // the only way "microsoft" reaches Azure OpenAI. Rank it last so it never
    // outranks a name or alias hit.
    if (channel.description.toLocaleLowerCase().includes(normalizedQuery)) {
      return { channel, rank: 10, reason: channel.description }
    }
    return null
  }

  const normalizedQuery = query.trim().toLocaleLowerCase()
  const searchMatches: ChannelMatch[] = (() => {
    if (!normalizedQuery) {
      return otherChannels.map((channel) => ({ channel, rank: 0, reason: '' }))
    }
    const matches = activeChannels
      .map((channel, index) => ({ index, match: matchChannel(channel, normalizedQuery) }))
      .filter((row): row is { index: number; match: ChannelMatch } => row.match !== null)
      .sort((a, b) => b.match.rank - a.match.rank || a.index - b.index)
      .map((row) => row.match)
    return matches
  })()
  const compatibleAlternative =
    normalizedQuery && searchMatches.length === 0
      ? activeChannels.find((channel) => channel.channel_id === 'openai_compatible')
      : undefined
  const rankedMatches: ChannelMatch[] = compatibleAlternative
    ? [{ channel: compatibleAlternative, rank: 0, reason: '' }]
    : searchMatches
  const compatibleHelpId = `${channelListId}-compatible-help`

  // Classic watch(rankedMatches) resets the highlight; fingerprint on the
  // ordered channel ids so identical lists don't retrigger.
  const matchFingerprint = rankedMatches.map((match) => match.channel.channel_id).join(',')
  const [lastFingerprint, setLastFingerprint] = useState(matchFingerprint)
  if (lastFingerprint !== matchFingerprint) {
    setLastFingerprint(matchFingerprint)
    setActiveIndex(0)
  }

  const activeOptionId = rankedMatches[activeIndex]
    ? optionId(rankedMatches[activeIndex].channel.channel_id)
    : undefined

  // Focus stays in the search input, so the highlighted option has to be
  // scrolled into view explicitly — otherwise Enter can select a row below
  // the fold.
  useEffect(() => {
    if (!activeOptionId) return
    document.getElementById(activeOptionId)?.scrollIntoView({ block: 'nearest' })
  }, [activeOptionId])

  function onPopoverOpenChange(open: boolean): void {
    setPopoverOpen(open)
    if (open) {
      setQuery('')
      setActiveIndex(0)
    }
  }

  function channelSelected(channel: ChannelDto): boolean {
    return value === channel.channel_id
  }

  function choose(channel: ChannelDto): void {
    if (disabled) return
    setPopoverOpen(false)
    onSelect(channel)
  }

  function onSearchKeydown(event: KeyboardEvent<HTMLInputElement>): void {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setActiveIndex(Math.min(activeIndex + 1, rankedMatches.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setActiveIndex(Math.max(activeIndex - 1, 0))
    } else if (event.key === 'Enter') {
      const match = rankedMatches[activeIndex]
      if (match) {
        event.preventDefault()
        choose(match.channel)
      }
    }
  }

  return (
    <section
      {...stylex.props(styles.root, compact && styles.rootCompact)}
      aria-labelledby={hideHeader ? undefined : 'channel-picker-heading'}
      aria-label={hideHeader ? t('import.presets.title') : undefined}
    >
      {!hideHeader && (
        <div {...stylex.props(styles.header)}>
          <h2 id="channel-picker-heading" {...stylex.props(styles.heading)}>
            {t('import.presets.title')}
          </h2>
          {selectedChannel && (
            <span {...stylex.props(styles.current)}>
              <span>{t('import.presets.current')}</span>
              <ChannelIcon icon={selectedChannel.icon} mark={selectedChannel.mark} />
              <span {...stylex.props(styles.currentName)}>{selectedChannel.name}</span>
            </span>
          )}
        </div>
      )}

      <div {...stylex.props(styles.row)}>
        {loading && channels.length > 0 && (
          <span role="status" aria-label={t('import.presets.loading')} />
        )}

        {initialLoading ? (
          <InlineNotice tone="neutral">{t('import.presets.loading')}</InlineNotice>
        ) : loadFailed ? (
          <InlineNotice
            tone="danger"
            action={
              <Button variant="ghost" size="sm" label={t('common.retry')} onClick={onRetry} />
            }
          >
            {t('import.presets.loadFailed')}
          </InlineNotice>
        ) : (
          <div {...stylex.props(styles.selector)}>
            <div {...stylex.props(styles.chips)}>
              {featuredChannels.map((channel) => (
                <button
                  key={channel.channel_id}
                  type="button"
                  {...stylex.props(styles.chip, channelSelected(channel) && styles.chipSelected)}
                  disabled={disabled}
                  aria-pressed={channelSelected(channel)}
                  onClick={() => choose(channel)}
                >
                  <ChannelIcon icon={channel.icon} mark={channel.mark} />
                  <span>{channel.name}</span>
                </button>
              ))}

              {
                <>
                  <span {...stylex.props(styles.divider)} aria-hidden="true" />
                  <Popover
                    isOpen={popoverOpen}
                    onOpenChange={onPopoverOpenChange}
                    placement="below"
                    alignment="center"
                    width={320}
                    hasAutoFocus
                    label={t('import.presets.more')}
                    content={
                      <div {...stylex.props(styles.panel)}>
                        <TextInput
                          label={t('import.presets.search')}
                          isLabelHidden
                          placeholder={t('import.presets.search')}
                          value={query}
                          onChange={setQuery}
                          onKeyDown={onSearchKeydown}
                          hasClear
                          startIcon={<Search size={14} aria-hidden="true" />}
                          size="sm"
                          role="combobox"
                          aria-autocomplete="list"
                          aria-expanded={popoverOpen}
                          aria-controls={rankedMatches.length ? channelListId : undefined}
                          aria-activedescendant={activeOptionId}
                        />
                        {searchMatches.length === 0 && (
                          <InlineNotice tone="neutral" appearance="hint">
                            {t('import.presets.noMatches')}
                          </InlineNotice>
                        )}
                        {compatibleAlternative && (
                          <p id={compatibleHelpId} {...stylex.props(styles.compatibleHelp)}>
                            {t('import.presets.compatibleHelp')}
                          </p>
                        )}
                        {rankedMatches.length > 0 && (
                          <div
                            id={channelListId}
                            {...stylex.props(styles.options)}
                            role="listbox"
                            aria-label={t('import.presets.more')}
                          >
                            {rankedMatches.map((match, index) => (
                              <button
                                key={match.channel.channel_id}
                                id={optionId(match.channel.channel_id)}
                                type="button"
                                role="option"
                                aria-selected={channelSelected(match.channel)}
                                aria-describedby={
                                  compatibleAlternative ? compatibleHelpId : undefined
                                }
                                {...stylex.props(
                                  styles.option,
                                  index === activeIndex && styles.optionActive,
                                )}
                                onClick={() => choose(match.channel)}
                                onMouseEnter={() => setActiveIndex(index)}
                              >
                                <ChannelIcon icon={match.channel.icon} mark={match.channel.mark} />
                                <span {...stylex.props(styles.optionName)}>
                                  {compatibleAlternative
                                    ? t('import.presets.useCompatible')
                                    : highlightSegments(match.channel.name, query).map(
                                        (segment, segIndex) =>
                                          segment.matched ? (
                                            <mark key={segIndex}>{segment.text}</mark>
                                          ) : (
                                            segment.text
                                          ),
                                      )}
                                </span>
                                {match.reason && (
                                  <span {...stylex.props(styles.optionReason)}>
                                    {highlightSegments(match.reason, query).map(
                                      (segment, segIndex) =>
                                        segment.matched ? (
                                          <mark key={segIndex}>{segment.text}</mark>
                                        ) : (
                                          segment.text
                                        ),
                                    )}
                                  </span>
                                )}
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    }
                  >
                    {(triggerProps) => (
                      <button
                        {...triggerProps}
                        type="button"
                        {...stylex.props(styles.chip, extraChannel !== null && styles.chipSelected)}
                        disabled={disabled}
                        aria-pressed={extraChannel !== null}
                      >
                        {extraChannel && (
                          <ChannelIcon icon={extraChannel.icon} mark={extraChannel.mark} />
                        )}
                        <span>{extraChannel?.name ?? t('import.presets.more')}</span>
                        <ChevronDown
                          size={13}
                          aria-hidden="true"
                          {...stylex.props(styles.caret, popoverOpen && styles.caretOpen)}
                        />
                      </button>
                    )}
                  </Popover>
                </>
              }
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
