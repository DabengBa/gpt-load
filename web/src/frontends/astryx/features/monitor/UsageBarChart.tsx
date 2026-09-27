import * as stylex from '@stylexjs/stylex'
import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type FocusEvent as ReactFocusEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
  type PointerEvent as ReactPointerEvent,
} from 'react'

import {
  buildUsageBarGeometry,
  completeUsageBarSeries,
  isUsageBarSeriesUsable,
  type UsageBarDatum,
} from '@shared/domain/monitor/usage-bar-chart'
import { formatISOInstant, formatInteger, formatLocalTimeRange } from '@shared/lib/format'

import { useMediaQuery } from '../../app/use-media-query'

const COMPACT = '@media (max-width: 620px)'
const MOTION_OK = '@media (prefers-reduced-motion: no-preference)'

const chartHeight = 158

// Classic Vue <Transition> fades: the keyed data layer fades in on mount and
// on every series change, the tooltip/guide on appear. Keyed remounting plus
// an enter animation reproduces the enter half; the classic leave fade has no
// React equivalent without a transition library, so exits are instant.
const fadeIn = stylex.keyframes({
  from: { opacity: 0 },
  to: { opacity: 1 },
})

const styles = stylex.create({
  chart: {
    position: 'relative',
    display: 'grid',
    minWidth: 0,
    margin: 0,
    cursor: 'crosshair',
  },
  visuallyHidden: {
    position: 'absolute',
    width: 1,
    height: 1,
    overflow: 'hidden',
    clipPath: 'inset(50%)',
    whiteSpace: 'nowrap',
  },
  frame: {
    position: 'relative',
    height: '100%',
  },
  plotStack: {
    position: 'relative',
    aspectRatio: {
      default: 'var(--chart-aspect-ratio)',
      [COMPACT]: '2 / 1',
    },
    minHeight: { [COMPACT]: 120 },
  },
  dataLayer: {
    height: '100%',
    // `<Transition name="usage-bar-chart__data" appear>` — fade on mount and
    // on every keyed remount (seriesKey changes).
    animationName: { [MOTION_OK]: fadeIn },
    animationDuration: 'var(--duration-data)',
    animationTimingFunction: 'var(--easing-data)',
  },
  graphic: {
    display: 'block',
    width: '100%',
    height: '100%',
  },
  gridLine: {
    stroke: 'var(--color-border-subtle)',
    strokeWidth: 1,
    strokeDasharray: 'var(--chart-grid-dash)',
  },
  primaryBar: {
    fill: 'var(--color-action)',
    opacity: 0.78,
    transitionProperty: { [MOTION_OK]: 'fill' },
    transitionDuration: { [MOTION_OK]: 'var(--duration-data)' },
    transitionTimingFunction: { [MOTION_OK]: 'var(--easing-data)' },
  },
  secondaryBar: {
    fill: 'var(--color-success)',
    opacity: 0.78,
    transitionProperty: { [MOTION_OK]: 'fill' },
    transitionDuration: { [MOTION_OK]: 'var(--duration-data)' },
    transitionTimingFunction: { [MOTION_OK]: 'var(--easing-data)' },
  },
  guide: {
    stroke: 'var(--color-border-strong)',
  },
  // `<Transition name="usage-bar-chart__active">` — fade on appear.
  activeMarkers: {
    animationName: { [MOTION_OK]: fadeIn },
    animationDuration: 'var(--duration-fast)',
    animationTimingFunction: 'var(--easing-standard)',
  },
  tooltip: {
    position: 'absolute',
    zIndex: 2,
    minWidth: 134,
    transform: 'translate(-50%, -100%)',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-strong)',
    borderRadius: 8,
    backgroundColor: 'var(--color-surface-raised)',
    boxShadow: 'var(--shadow-chart-tooltip)',
    paddingBlock: 8,
    paddingInline: 11,
    pointerEvents: 'none',
    whiteSpace: 'nowrap',
    // Same active enter fade as the guide marker.
    animationName: { [MOTION_OK]: fadeIn },
    animationDuration: 'var(--duration-fast)',
    animationTimingFunction: 'var(--easing-standard)',
  },
  tooltipStart: {
    transform: 'translate(0, -100%)',
  },
  tooltipEnd: {
    transform: 'translate(-100%, -100%)',
  },
  tooltipTime: {
    display: 'block',
    marginBottom: 5,
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 11,
  },
  tooltipRow: {
    display: 'flex',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: 14,
    fontSize: 12,
  },
  // Classic `.row + .row` — every tooltip row after the first gets the gap;
  // the sibling combinator is not in the stylex allowlist, so all rows after
  // the primary one apply this explicitly.
  tooltipRowAdjacent: {
    marginTop: 2,
  },
  tooltipKey: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 6,
    color: 'var(--color-text-muted)',
  },
  tooltipKeyDetail: {
    paddingLeft: 13,
  },
  tooltipSwatch: {
    display: 'block',
    width: 7,
    height: 7,
    flex: '0 0 auto',
    borderRadius: 2,
  },
  tooltipSwatchPrimary: {
    backgroundColor: 'var(--color-action)',
  },
  tooltipSwatchSecondary: {
    backgroundColor: 'var(--color-success)',
  },
  tooltipValue: {
    color: 'var(--color-text)',
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    fontWeight: 600,
  },
})

type TooltipAlignment = 'start' | 'center' | 'end'

/**
 * Usage bucket bar chart — classic UsageBarChart.vue. Hover, click, touch, and
 * full keyboard selection (ArrowLeft/Right, Home/End, Escape) with an
 * aria-live point announcement.
 */
export function UsageBarChart({
  series,
  title,
  description,
  emptyLabel,
  primaryLabel,
  secondaryLabel,
  primaryZeroDisplay,
  secondaryZeroDisplay,
  detailZeroDisplay,
  rangeStart,
  rangeEnd,
  locale = 'en-US',
  grouped = false,
}: {
  series: readonly UsageBarDatum[]
  title: string
  description: string
  emptyLabel: string
  primaryLabel: string
  secondaryLabel?: string
  primaryZeroDisplay?: string
  secondaryZeroDisplay?: string
  detailZeroDisplay?: string
  rangeStart: number
  rangeEnd: number
  locale?: string
  grouped?: boolean
}) {
  const compactChart = useMediaQuery('(max-width: 620px)')
  const width = compactChart ? chartHeight * 2 : 1000
  const descriptionID = `usage-bar-chart-description-${useId()}`
  const chartElement = useRef<HTMLElement | null>(null)
  const [activePointIndex, setActivePointIndex] = useState<number | null>(null)
  const [selectionAnnouncement, setSelectionAnnouncement] = useState('')
  const announcementGeneration = useRef(0)

  const chartSeries: UsageBarDatum[] = (() => {
    if (!isUsageBarSeriesUsable(series, rangeStart, rangeEnd)) return []
    const completed = completeUsageBarSeries(series, rangeStart, rangeEnd)
    const detailTemplate = series.find((datum) => datum.details?.length)?.details
    if (!detailTemplate) return completed
    return completed.map((datum) =>
      datum.details
        ? datum
        : {
            ...datum,
            details: detailTemplate.map((detail) => ({
              label: detail.label,
              display: detailZeroDisplay ?? formatInteger(0, locale),
            })),
          },
    )
  })()

  const seriesKey = `${locale}:${primaryLabel}:${secondaryLabel ?? ''}:${grouped}:${primaryZeroDisplay ?? ''}:${secondaryZeroDisplay ?? ''}:${detailZeroDisplay ?? ''}:${rangeStart}:${rangeEnd}:${chartSeries
    .map(
      (datum) =>
        `${datum.bucket_start_ms}:${datum.bucket_end_ms}:${datum.primary_value}:${datum.secondary_value}:${datum.primary_display ?? ''}:${datum.secondary_display ?? ''}:${datum.details?.map((detail) => `${detail.label}:${detail.display}`).join(',') ?? ''}`,
    )
    .join('|')}`

  const geometry = buildUsageBarGeometry(
    chartSeries,
    width,
    chartHeight,
    rangeStart,
    rangeEnd,
    grouped,
  )

  const activePoint = (() => {
    if (activePointIndex === null) return undefined
    const datum = chartSeries[activePointIndex]
    const point = geometry.points[activePointIndex]
    return datum && point ? { datum, point } : undefined
  })()

  let tooltipStyle: CSSProperties | undefined
  let tooltipAlignment: TooltipAlignment = 'center'
  const activePointPosition = activePoint?.point
  if (activePointPosition) {
    const left =
      activePointPosition.x < width * 0.16
        ? '2px'
        : activePointPosition.x > width * 0.84
          ? 'calc(100% - 2px)'
          : `${(activePointPosition.x / width) * 100}%`
    tooltipStyle = {
      left,
      top: `max(82px, calc(${(activePointPosition.y / chartHeight) * 100}% - 12px))`,
    }
    if (activePointPosition.x < width * 0.16) tooltipAlignment = 'start'
    if (activePointPosition.x > width * 0.84) tooltipAlignment = 'end'
  }

  // watch(seriesKey): a changed series invalidates the active point and any
  // queued announcement. State reset uses the render-adjustment pattern; the
  // generation ref bumps in an effect because refs must not be written during
  // render.
  const [prevSeriesKey, setPrevSeriesKey] = useState(seriesKey)
  if (seriesKey !== prevSeriesKey) {
    setPrevSeriesKey(seriesKey)
    setActivePointIndex(null)
    setSelectionAnnouncement('')
  }
  useEffect(() => {
    announcementGeneration.current += 1
  }, [seriesKey])

  // document pointerdown (capture): picking outside the chart closes the
  // current selection. Written directly against the ref + stable setters so
  // the listener never goes stale.
  useEffect(() => {
    const onExternalPointerDown = (event: PointerEvent): void => {
      const target = event.target
      if (target instanceof Node && chartElement.current?.contains(target)) return
      announcementGeneration.current += 1
      setActivePointIndex(null)
      setSelectionAnnouncement('')
    }
    document.addEventListener('pointerdown', onExternalPointerDown, true)
    return () => document.removeEventListener('pointerdown', onExternalPointerDown, true)
  }, [])

  function formatBucketTime(datum: UsageBarDatum): string {
    return formatLocalTimeRange(datum.bucket_start_ms, datum.bucket_end_ms, locale)
  }

  function primaryTooltipValue(datum: UsageBarDatum): string {
    return (
      datum.primary_display ??
      (datum.primary_value === 0 ? primaryZeroDisplay : undefined) ??
      formatInteger(datum.primary_value, locale)
    )
  }

  function secondaryTooltipValue(datum: UsageBarDatum): string {
    return (
      datum.secondary_display ??
      (datum.secondary_value === 0 ? secondaryZeroDisplay : undefined) ??
      formatInteger(datum.secondary_value, locale)
    )
  }

  function pointAnnouncement(index: number): string {
    const datum = chartSeries[index]
    if (!datum) return ''
    const primary = `${formatBucketTime(datum)} · ${primaryLabel} ${primaryTooltipValue(datum)}`
    const values =
      grouped && secondaryLabel
        ? `${primary} · ${secondaryLabel} ${secondaryTooltipValue(datum)}`
        : primary
    return (datum.details ?? []).reduce(
      (announcement, detail) => `${announcement} · ${detail.label} ${detail.display}`,
      values,
    )
  }

  function clearAnnouncement(): void {
    announcementGeneration.current += 1
    setSelectionAnnouncement('')
  }

  function announcePoint(index: number): void {
    const message = pointAnnouncement(index)
    const generation = ++announcementGeneration.current
    setSelectionAnnouncement('')
    // Classic defers through nextTick so the live region commits the empty
    // string before the message lands; queueMicrotask gives the same
    // clear-then-set ordering for identical repeated announcements.
    queueMicrotask(() => {
      if (generation === announcementGeneration.current) {
        setSelectionAnnouncement(message)
      }
    })
  }

  function selectPoint(index: number | null, announce = false): void {
    setActivePointIndex(index)
    if (index === null) clearAnnouncement()
    else if (announce) announcePoint(index)
  }

  function nearestPointIndex(event: { clientX: number }): number | null {
    const element = chartElement.current
    const points = geometry.points
    if (!element || points.length === 0) return null
    const bounds = element.getBoundingClientRect()
    if (bounds.width <= 0) return null
    const x = Math.max(0, Math.min(width, ((event.clientX - bounds.left) / bounds.width) * width))
    let nearest: number | null = null
    let nearestDistance = Infinity
    points.forEach((point, index) => {
      const distance = Math.abs(point.x - x)
      if (distance < nearestDistance) {
        nearest = index
        nearestDistance = distance
      }
    })
    return nearest
  }

  function selectNearest(event: { clientX: number }, announce = false): void {
    const index = nearestPointIndex(event)
    selectPoint(index, announce && index !== null)
  }

  function onPointerMove(event: ReactPointerEvent<HTMLElement>): void {
    if (event.pointerType !== 'touch') selectNearest(event)
  }

  function onCommitSelection(event: ReactMouseEvent<HTMLElement>): void {
    selectNearest(event, true)
  }

  function onPointerLeave(event: ReactPointerEvent<HTMLElement>): void {
    if (event.pointerType !== 'touch') selectPoint(null)
  }

  function onKeydown(event: ReactKeyboardEvent<HTMLElement>): void {
    const lastIndex = chartSeries.length - 1
    if (lastIndex < 0) return
    if (event.key === 'Escape') {
      selectPoint(null)
      return
    }
    let next: number | null = null
    if (event.key === 'ArrowLeft') next = Math.max(0, (activePointIndex ?? 0) - 1)
    if (event.key === 'ArrowRight') next = Math.min(lastIndex, (activePointIndex ?? -1) + 1)
    if (event.key === 'Home') next = 0
    if (event.key === 'End') next = lastIndex
    if (next !== null) {
      event.preventDefault()
      selectPoint(next, true)
    }
  }

  function closeOnFocusLeave(event: ReactFocusEvent<HTMLElement>): void {
    const next = event.relatedTarget
    if (next instanceof Node && chartElement.current?.contains(next)) return
    selectPoint(null)
  }

  return (
    <figure
      ref={chartElement}
      {...stylex.props(styles.chart)}
      tabIndex={0}
      role="group"
      aria-label={title}
      aria-describedby={descriptionID}
      onClick={onCommitSelection}
      onBlur={closeOnFocusLeave}
      onKeyDown={onKeydown}
      onPointerLeave={onPointerLeave}
      onPointerMove={onPointerMove}
    >
      <span id={descriptionID} {...stylex.props(styles.visuallyHidden)}>
        {chartSeries.length === 0 ? emptyLabel : description}
      </span>
      <span {...stylex.props(styles.visuallyHidden)} aria-live="polite" aria-atomic="true">
        {selectionAnnouncement}
      </span>
      <div {...stylex.props(styles.plotStack)}>
        <div key={seriesKey} {...stylex.props(styles.dataLayer)}>
          <div {...stylex.props(styles.frame)}>
            <svg
              {...stylex.props(styles.graphic)}
              viewBox={`0 0 ${width} ${chartHeight}`}
              aria-hidden="true"
            >
              <g>
                {[
                  geometry.plotTop,
                  (geometry.plotTop + geometry.baseline) / 2,
                  geometry.baseline,
                ].map((lineY) => (
                  <line
                    key={lineY}
                    {...stylex.props(styles.gridLine)}
                    x1={0}
                    y1={lineY}
                    x2={width}
                    y2={lineY}
                  />
                ))}
              </g>
              <g>
                {geometry.primaryBars
                  .filter((bar) => bar.value > 0)
                  .map((bar, index) => (
                    <rect
                      key={`${bar.x}:${index}`}
                      {...stylex.props(styles.primaryBar)}
                      x={bar.x}
                      y={bar.y}
                      width={bar.width}
                      height={bar.height}
                      rx={1.5}
                    />
                  ))}
              </g>
              {grouped ? (
                <g>
                  {geometry.secondaryBars
                    .filter((bar) => bar.value > 0)
                    .map((bar, index) => (
                      <rect
                        key={`${bar.x}:${index}`}
                        {...stylex.props(styles.secondaryBar)}
                        x={bar.x}
                        y={bar.y}
                        width={bar.width}
                        height={bar.height}
                        rx={1.5}
                      />
                    ))}
                </g>
              ) : null}
              {activePoint ? (
                <g {...stylex.props(styles.activeMarkers)}>
                  <line
                    {...stylex.props(styles.guide)}
                    x1={activePoint.point.x}
                    y1={geometry.plotTop}
                    x2={activePoint.point.x}
                    y2={geometry.baseline}
                  />
                </g>
              ) : null}
            </svg>
            {activePoint ? (
              <div
                {...stylex.props(
                  styles.tooltip,
                  tooltipAlignment === 'start' && styles.tooltipStart,
                  tooltipAlignment === 'end' && styles.tooltipEnd,
                )}
                style={tooltipStyle}
                role="tooltip"
              >
                <time
                  {...stylex.props(styles.tooltipTime)}
                  dateTime={formatISOInstant(activePoint.datum.bucket_start_ms)}
                >
                  {formatBucketTime(activePoint.datum)}
                </time>
                <span {...stylex.props(styles.tooltipRow)}>
                  <span {...stylex.props(styles.tooltipKey)}>
                    <i
                      {...stylex.props(styles.tooltipSwatch, styles.tooltipSwatchPrimary)}
                      aria-hidden="true"
                    />
                    {primaryLabel}
                  </span>
                  <strong {...stylex.props(styles.tooltipValue)}>
                    {primaryTooltipValue(activePoint.datum)}
                  </strong>
                </span>
                {grouped && secondaryLabel ? (
                  <span {...stylex.props(styles.tooltipRow, styles.tooltipRowAdjacent)}>
                    <span {...stylex.props(styles.tooltipKey)}>
                      <i
                        {...stylex.props(styles.tooltipSwatch, styles.tooltipSwatchSecondary)}
                        aria-hidden="true"
                      />
                      {secondaryLabel}
                    </span>
                    <strong {...stylex.props(styles.tooltipValue)}>
                      {secondaryTooltipValue(activePoint.datum)}
                    </strong>
                  </span>
                ) : null}
                {(activePoint.datum.details ?? []).map((detail) => (
                  <span
                    key={detail.label}
                    {...stylex.props(styles.tooltipRow, styles.tooltipRowAdjacent)}
                  >
                    <span {...stylex.props(styles.tooltipKey, styles.tooltipKeyDetail)}>
                      {detail.label}
                    </span>
                    <strong {...stylex.props(styles.tooltipValue)}>{detail.display}</strong>
                  </span>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </figure>
  )
}
