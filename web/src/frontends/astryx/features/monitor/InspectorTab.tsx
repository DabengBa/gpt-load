import * as stylex from '@stylexjs/stylex'
import { Badge, Banner, Button, EmptyState, Skeleton, Tooltip } from '@astryxdesign/core'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useRouterState } from '@tanstack/react-router'
import { ArrowRight, ChevronRight, Route as RouteIcon } from 'lucide-react'
import { useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { useIntl } from 'react-intl'

import { controlQueryKeys } from '@shared/control/query-keys'
import { enabledDataProtocols } from '@shared/control/protocols'
import { listAccessKeyOptions } from '@shared/control/resources/access-keys'
import { listChannels } from '@shared/control/resources/channels'
import { groupOptionsQueryOptions } from '@shared/control/resources/groups'
import {
  inspectRoute,
  type RouteInspectCredentialDto,
  type RouteInspectGroupDto,
  type RouteInspectReasonCode,
  type RouteInspectRequest,
  type RouteInspectResponseDto,
} from '@shared/control/resources/route-inspection'
import type { AccessProtocol } from '@shared/control/types'
import { RequestCancelledError } from '@shared/http/errors'
import { isValidMonitorText, normalizeMonitorText } from '@shared/domain/monitor/filter-validation'
import { formatInteger, formatISOInstant, formatLocalInstant } from '@shared/lib/format'
import { pagePath } from '@shared/routing/page-routes'
import type { SharedRouteQuery } from '@shared/routing/route-query'
import type { MessageId } from '@shared/i18n/message-ids'
import { inspectorMonitorQuery, parseInspectorMonitorState } from '@shared/routing/monitor-route'

import { useAppServices } from '../../app/services'
import { useStableLoading } from '../../app/collection-loading'
import { useT } from '../../app/i18n'
import { InspectorForm, type InspectorFormOption } from './InspectorForm'
import { MonitorSectionHeading } from './MonitorSectionHeading'

type InspectorField = 'protocol' | 'externalModel' | 'accessKey'
type InspectorErrors = Partial<Record<InspectorField, MessageId>>
type StatusTone = 'success' | 'warning' | 'danger' | 'neutral'

const badgeVariants = {
  neutral: 'neutral',
  success: 'success',
  warning: 'warning',
  danger: 'error',
} as const

const knownReasons = new Set<RouteInspectReasonCode>([
  'access_key_disabled',
  'access_key_expired',
  'protocol_filtered',
  'model_filtered',
  'model_required_by_filter',
  'operation_unsupported',
  'native_route_required',
  'no_route_target',
  'group_disabled',
  'group_filtered',
  'no_available_group',
  'no_credentials',
  'credential_blacklisted',
  'credential_cooldown',
  'credential_auth_unavailable',
  'credential_not_allowed',
  'no_available_credential',
  'entry_disabled',
  'entry_blacklisted',
  'entry_cooldown',
  'entry_weight_zero',
  'tier_demoted',
])

interface InspectorDraft {
  protocol: AccessProtocol | ''
  model: string
  accessKeyID: string
}

function readProtocol(raw: unknown): AccessProtocol | '' {
  return typeof raw === 'string' && enabledDataProtocols.some((protocol) => protocol === raw)
    ? (raw as AccessProtocol)
    : ''
}

function readText(raw: unknown): string {
  return normalizeMonitorText(raw) ?? ''
}

function readPositiveID(raw: unknown): string {
  if (typeof raw !== 'string' || !/^\d+$/.test(raw)) return ''
  const value = Number(raw)
  return Number.isSafeInteger(value) && value > 0 ? String(value) : ''
}

function sameInspectionRequest(
  left: RouteInspectRequest | undefined,
  right: RouteInspectRequest,
): boolean {
  return (
    left?.protocol === right.protocol &&
    left.external_model === right.external_model &&
    left.access_key_id === right.access_key_id
  )
}

function validateInspectorDraft(
  draft: InspectorDraft,
  configuredModels: readonly string[],
  accessKeyIDs: ReadonlySet<number> | undefined,
): { request?: RouteInspectRequest; errors: InspectorErrors } {
  const errors: InspectorErrors = {}
  if (!enabledDataProtocols.some((protocol) => protocol === draft.protocol)) {
    errors.protocol = 'monitor.inspector.errors.protocol'
  }
  if (
    draft.model === '' ||
    !isValidMonitorText(draft.model) ||
    !configuredModels.includes(draft.model)
  ) {
    errors.externalModel = 'monitor.inspector.errors.model'
  }
  const accessKeyID = Number(draft.accessKeyID)
  if (
    !/^\d+$/.test(draft.accessKeyID) ||
    !Number.isSafeInteger(accessKeyID) ||
    accessKeyID <= 0 ||
    accessKeyIDs === undefined ||
    !accessKeyIDs.has(accessKeyID)
  ) {
    errors.accessKey = 'monitor.inspector.errors.accessKey'
  }
  if (Object.keys(errors).length > 0) return { errors }
  return {
    errors,
    request: {
      protocol: draft.protocol as AccessProtocol,
      external_model: draft.model,
      access_key_id: accessKeyID,
    },
  }
}

function groupDetailHref(id: number): string {
  return `${pagePath('groups')}/${id}`
}

const styles = stylex.create({
  tab: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(272px, 300px) minmax(0, 1fr)',
      '@media (max-width: 1120px)': 'minmax(0, 1fr)',
    },
    alignItems: 'start',
    gap: 'var(--space-6)',
  },
  formSticky: {
    position: { default: 'sticky', '@media (max-width: 1120px)': 'static' },
    top: 'calc(var(--topbar-height, 56px) + var(--space-4))',
  },
  stack: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-5)',
  },
  section: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-3)',
  },
  empty: {
    minHeight: '330px',
    borderWidth: '1px',
    borderStyle: 'dashed',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
  },
  skeleton: {
    display: 'grid',
    gap: 'var(--space-2)',
    minHeight: '330px',
  },
  refreshing: {
    minHeight: 'var(--space-4)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  summary: {
    minWidth: 0,
    overflow: 'hidden',
    borderWidth: '1px',
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-card)',
    backgroundColor: 'var(--color-surface)',
  },
  summarySuccess: {
    borderLeftWidth: '3px',
    borderLeftColor: 'var(--color-success)',
  },
  summaryDanger: {
    borderLeftWidth: '3px',
    borderLeftColor: 'var(--color-danger)',
  },
  summaryHeader: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    gap: { default: 'var(--space-5)', '@media (max-width: 860px)': 'var(--space-3)' },
    flexDirection: { default: 'row', '@media (max-width: 860px)': 'column' },
    paddingBlock: { default: '18px', '@media (max-width: 560px)': '16px' },
    paddingInline: { default: '20px', '@media (max-width: 560px)': '14px' },
  },
  summaryContent: {
    minWidth: 0,
  },
  summaryTitleRow: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-3)',
  },
  summaryTitle: {
    margin: 0,
    fontSize: '1.18rem',
    fontWeight: 650,
    letterSpacing: '-0.015em',
    lineHeight: 'var(--line-compact)',
  },
  summaryReason: {
    marginTop: 'var(--space-2)',
    marginBottom: 0,
    marginInline: 0,
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
  },
  summaryReasonCode: {
    marginLeft: 'var(--space-2)',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  summaryMeta: {
    display: 'grid',
    flexShrink: 0,
    justifyItems: { default: 'end', '@media (max-width: 860px)': 'start' },
    gap: 'var(--space-1)',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    fontVariantNumeric: 'tabular-nums',
    whiteSpace: 'nowrap',
  },
  facts: {
    display: 'grid',
    gridTemplateColumns: {
      default: 'repeat(3, minmax(0, 1fr))',
      '@media (max-width: 860px)': 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 560px)': 'minmax(0, 1fr)',
    },
    margin: 0,
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  fact: {
    minWidth: 0,
    minHeight: '74px',
    paddingBlock: '12px',
    paddingInline: '16px',
  },
  factTerm: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  factValue: {
    marginTop: '7px',
    marginBottom: 0,
    marginInline: 0,
    overflow: 'hidden',
    color: 'var(--color-text)',
    fontSize: 'var(--text-meta)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  factMono: {
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
  },
  factNumber: {
    fontFamily: 'var(--font-mono)',
    fontVariantNumeric: 'tabular-nums',
    fontSize: '1rem',
  },
  ledger: {
    minWidth: 0,
    borderBottomWidth: { default: '1px', '@media (max-width: 860px)': 0 },
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-control)',
    display: { default: 'block', '@media (max-width: 860px)': 'grid' },
    gap: '10px',
  },
  ledgerHeader: {
    display: { default: 'grid', '@media (max-width: 860px)': 'none' },
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(170px, 1.35fr) 128px 96px 112px minmax(132px, 0.95fr) 34px',
      '@media (max-width: 1180px)':
        'minmax(160px, 1.3fr) 120px 88px 104px minmax(120px, 0.9fr) 30px',
    },
    alignItems: 'center',
    columnGap: { default: 'var(--space-4)', '@media (max-width: 1180px)': 'var(--space-3)' },
    minHeight: '38px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
    fontWeight: 500,
    letterSpacing: '0.04em',
  },
  candidate: {
    minWidth: 0,
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    borderBottomWidth: { '@media (max-width: 860px)': '1px' },
    borderBottomStyle: { '@media (max-width: 860px)': 'solid' },
    borderBottomColor: { '@media (max-width: 860px)': 'var(--color-border-subtle)' },
    borderInlineStartWidth: { '@media (max-width: 860px)': '1px' },
    borderInlineStartStyle: { '@media (max-width: 860px)': 'solid' },
    borderInlineStartColor: { '@media (max-width: 860px)': 'var(--color-border-subtle)' },
    borderInlineEndWidth: { '@media (max-width: 860px)': '1px' },
    borderInlineEndStyle: { '@media (max-width: 860px)': 'solid' },
    borderInlineEndColor: { '@media (max-width: 860px)': 'var(--color-border-subtle)' },
    borderRadius: { '@media (max-width: 860px)': 'var(--radius-control)' },
    backgroundColor: { '@media (max-width: 860px)': 'var(--color-surface)' },
    overflow: { '@media (max-width: 860px)': 'hidden' },
  },
  candidateSummary: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'minmax(170px, 1.35fr) 128px 96px 112px minmax(132px, 0.95fr) 34px',
      '@media (max-width: 1180px)':
        'minmax(160px, 1.3fr) 120px 88px 104px minmax(120px, 0.9fr) 30px',
      '@media (max-width: 860px)': 'minmax(0, 1.5fr) minmax(112px, 0.7fr) 28px',
      '@media (max-width: 560px)': 'minmax(0, 1fr) 28px',
    },
    alignItems: 'center',
    columnGap: { default: 'var(--space-4)', '@media (max-width: 1180px)': 'var(--space-3)' },
    gap: { '@media (max-width: 860px)': '12px 16px' },
    minHeight: '80px',
    paddingBlock: { default: '12px', '@media (max-width: 860px)': '15px' },
    paddingInline: { '@media (max-width: 860px)': '15px' },
    cursor: 'pointer',
    listStyle: 'none',
    transitionProperty: 'background-color',
    transitionDuration: 'var(--duration-fast)',
    transitionTimingFunction: 'var(--easing-standard)',
  },
  candidateIdentity: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    gridColumn: { '@media (max-width: 860px)': '1' },
  },
  candidateIdentityText: {
    overflow: 'hidden',
    fontWeight: 620,
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  candidateIdentitySub: {
    overflow: 'hidden',
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-label-xs)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  candidateStatus: {
    display: 'grid',
    minWidth: 0,
    justifyItems: 'start',
    gap: 'var(--space-1)',
    gridColumn: { '@media (max-width: 860px)': '2', '@media (max-width: 560px)': '1' },
  },
  candidateStatusSmall: {
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  candidateMeasure: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    alignSelf: { '@media (max-width: 860px)': 'end' },
    gridColumn: { '@media (max-width: 560px)': '1' },
  },
  candidateMeasureCredentials: {
    gridColumn: { '@media (max-width: 860px)': '1' },
  },
  candidateMeasureWeight: {
    gridColumn: { '@media (max-width: 860px)': '2', '@media (max-width: 560px)': '1' },
  },
  candidateMeasureValue: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-meta)',
    fontWeight: 500,
    fontVariantNumeric: 'tabular-nums',
  },
  candidateShare: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    alignSelf: { '@media (max-width: 860px)': 'end' },
    gridColumn: {
      '@media (max-width: 860px)': '1 / span 2',
      '@media (max-width: 560px)': '1',
    },
  },
  candidateDisclosure: {
    display: 'grid',
    color: 'var(--color-text-faint)',
    placeItems: 'center',
    gridColumn: {
      '@media (max-width: 860px)': '3',
      '@media (max-width: 560px)': '2',
    },
    gridRow: {
      '@media (max-width: 860px)': '1 / span 2',
      '@media (max-width: 560px)': '1 / span 5',
    },
  },
  cellLabel: {
    display: { default: 'none', '@media (max-width: 860px)': 'block' },
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-sans)',
    fontSize: 'var(--text-label-xs)',
  },
  credentialDetails: {
    display: 'grid',
    gap: 'var(--space-3)',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    backgroundColor: 'color-mix(in srgb, var(--color-surface-sunken) 62%, var(--color-surface))',
    paddingTop: '14px',
    paddingBottom: '16px',
    paddingInline: { default: '16px', '@media (max-width: 560px)': '13px' },
  },
  credentialDetailsHeader: {
    display: 'flex',
    minWidth: 0,
    alignItems: { default: 'center', '@media (max-width: 560px)': 'flex-start' },
    justifyContent: 'space-between',
    flexDirection: { default: 'row', '@media (max-width: 560px)': 'column' },
    gap: { default: 'var(--space-4)', '@media (max-width: 560px)': 'var(--space-2)' },
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-sm)',
  },
  credentialDetailsHeaderInner: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 'var(--space-2)',
  },
  credentialDetailsTitle: {
    color: 'var(--color-text)',
    fontWeight: 620,
  },
  credentialDetailsEmpty: {
    margin: 0,
    color: 'var(--color-text-muted)',
    paddingBlock: 'var(--space-3)',
    fontSize: 'var(--text-sm)',
  },
  credentialDetailsFooter: {
    display: 'flex',
    minWidth: 0,
    alignItems: 'center',
    justifyContent: 'flex-end',
  },
  ledgerScroll: {
    overflowX: 'auto',
    minWidth: 0,
  },
  ledgerGrid: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: 'var(--ledger-record-list-grid, minmax(0, 1fr))',
    columnGap: 'var(--ledger-record-list-column-gap, 16px)',
    alignItems: 'center',
  },
  ledgerHeadCell: {
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  credentialRecord: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: 'var(--ledger-record-list-grid, minmax(0, 1fr))',
    columnGap: 'var(--ledger-record-list-column-gap, 16px)',
    alignItems: 'center',
    minHeight: 'var(--ledger-record-list-record-min-height, 58px)',
    paddingBlock: 'var(--ledger-record-list-record-padding-block, 9px)',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  credentialRecordFirst: {
    borderTopColor: 'var(--color-border-control)',
  },
  credentialStatus: {
    display: 'flex',
    minWidth: 0,
    alignItems: { default: 'center', '@media (max-width: 860px)': 'flex-start' },
    flexWrap: 'wrap',
    gap: 'var(--space-1)',
    gridColumn: { '@media (max-width: 860px)': '1 / -1' },
  },
  credentialCode: {
    color: 'var(--color-text-faint)',
    fontSize: '10px',
  },
  credentialMono: {
    fontFamily: 'var(--font-mono)',
    fontSize: 'var(--text-sm)',
    fontVariantNumeric: 'tabular-nums',
  },
  exclusionRecord: {
    display: 'grid',
    minWidth: 0,
    gridTemplateColumns: {
      default: 'var(--ledger-record-list-grid, minmax(0, 1fr))',
      '@media (max-width: 860px)': 'repeat(2, minmax(0, 1fr))',
      '@media (max-width: 560px)': 'minmax(0, 1fr)',
    },
    columnGap: 'var(--ledger-record-list-column-gap, 16px)',
    alignItems: 'center',
    minHeight: '72px',
    paddingBlock: '11px',
    borderTopWidth: '1px',
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  exclusionIdentity: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
    gridColumn: { '@media (max-width: 860px)': '1 / -1', '@media (max-width: 560px)': '1' },
  },
  exclusionReason: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-sm)',
    gridColumn: { '@media (max-width: 860px)': '1 / -1', '@media (max-width: 560px)': '1' },
  },
  shareMeter: {
    width: '100%',
    height: '5px',
    overflow: 'hidden',
    borderRadius: '999px',
    backgroundColor: 'var(--color-border-subtle)',
  },
  shareMeterFill: {
    display: 'block',
    height: '100%',
    borderRadius: 'inherit',
    backgroundColor: 'var(--color-action)',
  },
})

function InspectorLedger({
  label,
  scrollHint,
  rowCount,
  grid,
  header,
  children,
}: {
  label: string
  scrollHint: string
  rowCount: number
  grid: string
  header: ReactNode
  children: ReactNode
}) {
  const listRef = useRef<HTMLDivElement | null>(null)
  const [overflowing, setOverflowing] = useState(false)
  useEffect(() => {
    const element = listRef.current
    if (element === null) return
    const update = (): void => {
      setOverflowing(element.scrollWidth > element.clientWidth + 1)
    }
    const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(update) : undefined
    observer?.observe(element)
    const fallbackFrame = observer === undefined ? requestAnimationFrame(update) : undefined
    window.addEventListener('resize', update)
    return () => {
      observer?.disconnect()
      if (fallbackFrame !== undefined) cancelAnimationFrame(fallbackFrame)
      window.removeEventListener('resize', update)
    }
  }, [rowCount])
  const accessibleLabel = overflowing && scrollHint !== '' ? `${label} · ${scrollHint}` : label
  return (
    <div
      ref={listRef}
      {...stylex.props(styles.ledgerScroll)}
      role="table"
      data-testid="ledger-record-list"
      aria-label={accessibleLabel}
      aria-rowcount={rowCount}
      tabIndex={overflowing ? 0 : undefined}
      style={{ '--ledger-record-list-grid': grid } as CSSProperties}
    >
      {header}
      {children}
    </div>
  )
}

export function InspectorTab() {
  const t = useT()
  const intl = useIntl()
  const locale = intl.locale
  const navigate = useNavigate()
  const { apiClient } = useAppServices()
  const monitorPath = pagePath('monitor')

  const rawSearch = useRouterState({
    select: (state) => state.location.search as SharedRouteQuery,
  })
  const routeState = useMemo(() => parseInspectorMonitorState(rawSearch), [rawSearch])

  const fieldSignature = `${routeState.protocol ?? ''}\0${routeState.externalModel ?? ''}\0${
    routeState.accessKeyID ?? ''
  }`
  const watchSignature = `${fieldSignature}\0${routeState.run ? 1 : 0}`

  const [draft, setDraft] = useState<InspectorDraft>(() => ({
    protocol: readProtocol(routeState.protocol),
    model: readText(routeState.externalModel),
    accessKeyID: readPositiveID(routeState.accessKeyID),
  }))
  const [fieldErrors, setFieldErrors] = useState<InspectorErrors>({})
  const [pending, setPending] = useState(false)
  const [failed, setFailed] = useState(false)
  const [resultStale, setResultStale] = useState(false)
  const [submitted, setSubmitted] = useState<RouteInspectRequest | undefined>()
  const [observation, setObservation] = useState<RouteInspectResponseDto | undefined>()
  const ownerRef = useRef(0)
  const controllerRef = useRef<AbortController | undefined>(undefined)
  const summaryRef = useRef<HTMLHeadingElement | null>(null)

  // Classic watch 1 (on protocol/model/accessKeyID/run): reset local state when
  // the route fields change, or when `run` drops with unchanged fields. Carried
  // as a render adjustment; the abort/owner bookkeeping happens in the
  // follow-up effect (refs are not writable during render).
  const [routeSync, setRouteSync] = useState({ fields: fieldSignature, watch: watchSignature })
  if (routeSync.watch !== watchSignature) {
    if (routeSync.fields !== fieldSignature || !routeState.run) {
      setDraft({
        protocol: readProtocol(routeState.protocol),
        model: readText(routeState.externalModel),
        accessKeyID: readPositiveID(routeState.accessKeyID),
      })
      setFieldErrors({})
      setPending(false)
      setFailed(false)
      setResultStale(false)
      setSubmitted(undefined)
      setObservation(undefined)
    }
    setRouteSync({ fields: fieldSignature, watch: watchSignature })
  }

  const accessKeyOptionsQuery = useQuery({
    queryKey: controlQueryKeys.accessKeys.options(),
    queryFn: ({ signal }) => listAccessKeyOptions(apiClient, signal),
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
  const groupOptionsQuery = useQuery(groupOptionsQueryOptions(apiClient, true))
  const channelsQuery = useQuery({
    queryKey: controlQueryKeys.channels.list(''),
    queryFn: ({ signal }) => listChannels(apiClient, '', signal),
    staleTime: Number.POSITIVE_INFINITY,
    refetchOnMount: false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })

  const protocolOptions = useMemo<InspectorFormOption[]>(
    () => [
      { value: '', label: t('monitor.inspector.form.selectProtocol') },
      ...enabledDataProtocols.map((value) => ({ value, label: value })),
    ],
    [t],
  )
  const configuredModels = useMemo(
    () =>
      [...new Set((groupOptionsQuery.data ?? []).flatMap((group) => group.models))].sort(
        (left, right) => left.localeCompare(right),
      ),
    [groupOptionsQuery.data],
  )
  const missingModelOption =
    groupOptionsQuery.isSuccess && draft.model !== '' && !configuredModels.includes(draft.model)
  const modelOptions = useMemo<InspectorFormOption[]>(
    () => [
      { value: '', label: t('monitor.inspector.form.selectModel') },
      ...(missingModelOption
        ? [
            {
              value: draft.model,
              label: t('monitor.inspector.form.missingModelOption', { model: draft.model }),
            },
          ]
        : []),
      ...configuredModels.map((model) => ({ value: model, label: model })),
    ],
    [t, missingModelOption, draft.model, configuredModels],
  )
  const missingAccessKeyOption =
    accessKeyOptionsQuery.isSuccess &&
    draft.accessKeyID !== '' &&
    !accessKeyOptionsQuery.data?.some((accessKey) => String(accessKey.id) === draft.accessKeyID)
  const accessKeyOptions = useMemo<InspectorFormOption[]>(() => {
    const options = (accessKeyOptionsQuery.data ?? []).map((accessKey) => ({
      value: String(accessKey.id),
      label: t('monitor.inspector.form.accessKeyOption', {
        name: accessKey.name,
        status: t(`monitor.inspector.accessKeyStatus.${accessKey.status}` as MessageId),
      }),
    }))
    return [
      { value: '', label: t('monitor.inspector.form.selectAccessKey') },
      ...(missingAccessKeyOption
        ? [
            {
              value: draft.accessKeyID,
              label: t('monitor.inspector.form.missingAccessKeyOption'),
            },
          ]
        : []),
      ...options,
    ]
  }, [t, missingAccessKeyOption, draft.accessKeyID, accessKeyOptionsQuery.data])
  const optionsPending = accessKeyOptionsQuery.isPending || groupOptionsQuery.isPending
  const optionsFailed = accessKeyOptionsQuery.isError || groupOptionsQuery.isError
  const accessKeyIDs = useMemo(
    () =>
      accessKeyOptionsQuery.data === undefined
        ? undefined
        : new Set(accessKeyOptionsQuery.data.map((accessKey) => accessKey.id)),
    [accessKeyOptionsQuery.data],
  )
  const channelsByID = useMemo<Record<string, string>>(
    () =>
      Object.fromEntries(
        (channelsQuery.data?.items ?? []).map((channel) => [channel.channel_id, channel.name]),
      ),
    [channelsQuery.data],
  )

  // Mirror of the mutable inspection state so the auto-run effect can guard
  // without listing it in deps (classic watch reads live refs; adding these to
  // deps would re-trigger the run after every state change).
  const liveRef = useRef({ pending, submitted, observation, draft })
  useEffect(() => {
    liveRef.current = { pending, submitted, observation, draft }
  }, [pending, submitted, observation, draft])

  // Classic watch 2 (immediate on run/fields + options data): auto-run the
  // inspection when the route requests it. The abort/owner side of the
  // route-change reset lives here too (render adjustments can't touch refs).
  const cleanupSigRef = useRef({ fields: fieldSignature, run: routeState.run })
  useEffect(() => {
    const prev = cleanupSigRef.current
    const fieldsChanged = prev.fields !== fieldSignature
    cleanupSigRef.current = { fields: fieldSignature, run: routeState.run }
    if (fieldsChanged || !routeState.run) {
      ownerRef.current += 1
      controllerRef.current?.abort()
      controllerRef.current = undefined
    }
    if (!routeState.run) return
    const { request, errors } = validateInspectorDraft(
      liveRef.current.draft,
      configuredModels,
      accessKeyIDs,
    )
    if (request === undefined) {
      // Deep-linked invalid input: surface the same field errors classic shows.
      if (Object.keys(errors).length > 0) setFieldErrors(errors)
      return
    }
    const live = liveRef.current
    if (live.pending && sameInspectionRequest(live.submitted, request)) return
    if (live.observation !== undefined && sameInspectionRequest(live.submitted, request)) return
    void runInspection(request)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mirrors the classic watch deps
  }, [fieldSignature, routeState.run, configuredModels, accessKeyIDs])

  const resultLoadingActive = pending && observation === undefined
  const resultLoading = useStableLoading(resultLoadingActive)
  const resultRefreshing = pending && observation !== undefined

  // Move focus to the summary after a completed non-routable inspection.
  useEffect(() => {
    if (observation !== undefined && !observation.routable) {
      summaryRef.current?.focus()
    }
  }, [observation])

  useEffect(
    () => () => {
      ownerRef.current += 1
      controllerRef.current?.abort()
      controllerRef.current = undefined
    },
    [],
  )

  async function runInspection(request: RouteInspectRequest): Promise<void> {
    controllerRef.current?.abort()
    const currentOwner = ++ownerRef.current
    const controller = new AbortController()
    controllerRef.current = controller
    setPending(true)
    setFailed(false)
    setSubmitted(request)

    try {
      const result = await inspectRoute(apiClient, request, controller.signal)
      if (currentOwner === ownerRef.current && !controller.signal.aborted) {
        setObservation(result)
        setResultStale(false)
      }
    } catch (error: unknown) {
      if (
        currentOwner === ownerRef.current &&
        !controller.signal.aborted &&
        !(error instanceof RequestCancelledError)
      ) {
        setFailed(true)
        setResultStale(liveRef.current.observation !== undefined)
      }
    } finally {
      if (currentOwner === ownerRef.current) {
        controllerRef.current = undefined
        setPending(false)
      }
    }
  }

  function inspect(): void {
    const { request, errors } = validateInspectorDraft(draft, configuredModels, accessKeyIDs)
    setFieldErrors(errors)
    if (request === undefined) return
    const current = routeState
    if (
      current.run &&
      current.protocol === request.protocol &&
      current.externalModel === request.external_model &&
      current.accessKeyID === String(request.access_key_id)
    ) {
      void runInspection(request)
      return
    }
    void navigate({
      to: monitorPath,
      search: inspectorMonitorQuery({
        protocol: request.protocol,
        externalModel: request.external_model,
        accessKeyID: String(request.access_key_id),
        run: true,
        expandedGroupIDs: [],
      }),
      resetScroll: false,
    })
  }

  function retryOptions(): void {
    void Promise.all([accessKeyOptionsQuery.refetch(), groupOptionsQuery.refetch()])
  }

  function isKnownReason(reason: string | null): boolean {
    return reason !== null && knownReasons.has(reason as RouteInspectReasonCode)
  }

  function reasonLabel(reason: string | null): string {
    if (reason === null) return t('monitor.inspector.reasons.none')
    return isKnownReason(reason) ? t(`monitor.inspector.reasons.${reason}` as MessageId) : ''
  }

  function modelLabel(value: string | null): string {
    return value ?? t('monitor.inspector.result.modelNotSpecified')
  }

  function formattedInteger(value: number): string {
    return formatInteger(value, locale)
  }

  function channelName(channelID: string): string {
    return channelsByID[channelID]?.trim() || channelID
  }

  function includedGroupIdentity(group: RouteInspectGroupDto): string {
    return `${channelName(group.channel_id)} · ${modelLabel(group.upstream_model)}`
  }

  function excludedGroupIdentity(group: RouteInspectGroupDto): string {
    return `${channelName(group.channel_id)} · ${t(
      `monitor.inspector.routeModes.${group.route_mode}` as MessageId,
    )} · ${modelLabel(group.upstream_model)}`
  }

  const inputChanged =
    submitted !== undefined &&
    (draft.protocol !== submitted.protocol ||
      draft.model !== submitted.external_model ||
      draft.accessKeyID !== String(submitted.access_key_id))

  const includedGroups = useMemo(
    () => (observation?.groups ?? []).filter((group) => group.included),
    [observation],
  )
  const excludedGroups = useMemo(
    () => (observation?.groups ?? []).filter((group) => !group.included),
    [observation],
  )
  const weightedMix = observation?.route_strategy === 'weighted_mix'
  const activeRouteMode = useMemo<'native' | 'converted' | null>(() => {
    if (includedGroups.some((group) => group.routable && group.route_mode === 'native')) {
      return 'native'
    }
    if (includedGroups.some((group) => group.routable && group.route_mode === 'converted')) {
      return 'converted'
    }
    return null
  }, [includedGroups])
  const orderedIncludedGroups = useMemo(
    () =>
      [...includedGroups].sort((left, right) => {
        const routeModeOrder = routeModePriority(left) - routeModePriority(right)
        if (routeModeOrder !== 0) return routeModeOrder
        if (left.routable !== right.routable) return left.routable ? -1 : 1
        const entryWeightOrder = right.entry_weight - left.entry_weight
        if (entryWeightOrder !== 0) return entryWeightOrder
        const priorityOrder = left.priority - right.priority
        return priorityOrder !== 0 ? priorityOrder : left.group_id - right.group_id
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps -- uses weightedMix via helpers
    [includedGroups, weightedMix],
  )

  function routeModePriority(group: RouteInspectGroupDto): number {
    return weightedMix || group.route_mode === 'native' ? 0 : 1
  }

  function isActiveCandidate(group: RouteInspectGroupDto): boolean {
    return group.routable && (weightedMix || group.route_mode === activeRouteMode)
  }

  const activeGroups = includedGroups.filter(isActiveCandidate)
  const availableCredentialCount = activeGroups.reduce(
    (total, group) => total + groupAvailableCredentialCount(group),
    0,
  )

  function routePriorityTone(group: RouteInspectGroupDto): StatusTone {
    if (!isActiveCandidate(group)) return 'neutral'
    return group.route_mode === 'native' ? 'success' : 'warning'
  }

  function routePriorityLabel(group: RouteInspectGroupDto): string {
    return weightedMix
      ? t(`monitor.inspector.routeModes.${group.route_mode}` as MessageId)
      : t(`monitor.inspector.groups.priority.${group.route_mode}` as MessageId)
  }

  function groupStatusLabel(group: RouteInspectGroupDto): string {
    if (!group.routable) return t('monitor.inspector.result.notRoutable')
    return isActiveCandidate(group)
      ? t('monitor.inspector.groups.weightedCandidate')
      : t('monitor.inspector.groups.standbyCandidate')
  }

  function credentialTone(credential: RouteInspectCredentialDto): StatusTone {
    if (credential.available) return 'success'
    if (credential.reason_code === 'credential_cooldown') return 'warning'
    if (credential.reason_code === 'credential_blacklisted') return 'danger'
    return 'neutral'
  }

  function credentialStatusLabel(credential: RouteInspectCredentialDto): string {
    return credential.available
      ? t('monitor.inspector.credentials.available')
      : reasonLabel(credential.reason_code)
  }

  function groupAvailableCredentialCount(group: RouteInspectGroupDto): number {
    return group.credentials.filter((credential) => credential.available).length
  }

  function groupShare(group: RouteInspectGroupDto): number {
    // 占比一律以后端归一化结果为准(设计 §8.2);0 是有效值,不做前端回退。
    return Math.round(group.effective_share * 1_000) / 10
  }

  function groupShareLabel(group: RouteInspectGroupDto): string {
    if (!isActiveCandidate(group)) return t('monitor.inspector.groups.standbyShare')
    return new Intl.NumberFormat(locale, { style: 'percent', maximumFractionDigits: 1 }).format(
      group.effective_share,
    )
  }

  function candidateCredentialSummary(group: RouteInspectGroupDto): string {
    const available = groupAvailableCredentialCount(group)
    return t('monitor.inspector.credentials.summary', {
      available: formattedInteger(available),
      unavailable: formattedInteger(group.credentials.length - available),
    })
  }

  function orderedCredentials(group: RouteInspectGroupDto): RouteInspectCredentialDto[] {
    return [...group.credentials].sort((left, right) => {
      if (left.available !== right.available) return left.available ? -1 : 1
      const reasonOrder = (left.reason_code ?? '').localeCompare(right.reason_code ?? '')
      return reasonOrder !== 0 ? reasonOrder : left.credential_id - right.credential_id
    })
  }

  function groupExpanded(groupID: number): boolean {
    return routeState.expandedGroupIDs.includes(groupID)
  }

  function setGroupExpanded(groupID: number, expanded: boolean): void {
    const current = new Set(routeState.expandedGroupIDs)
    if (expanded === current.has(groupID)) return
    if (expanded) current.add(groupID)
    else current.delete(groupID)
    void navigate({
      to: monitorPath,
      search: inspectorMonitorQuery({ ...routeState, expandedGroupIDs: [...current] }),
      resetScroll: false,
    })
  }

  function accessKeyStatusTone(status: 'active' | 'disabled'): 'success' | 'neutral' {
    return status === 'active' ? 'success' : 'neutral'
  }

  const scrollHint = t('monitor.scrollHint')

  return (
    <div {...stylex.props(styles.tab)}>
      <div {...stylex.props(styles.formSticky)}>
        <InspectorForm
          protocol={draft.protocol}
          model={draft.model}
          accessKeyId={draft.accessKeyID}
          protocolOptions={protocolOptions}
          modelOptions={modelOptions}
          accessKeyOptions={accessKeyOptions}
          errors={fieldErrors}
          optionsPending={optionsPending}
          optionsFailed={optionsFailed}
          missingAccessKey={missingAccessKeyOption === true}
          submitPending={pending}
          onProtocolChange={(value) =>
            setDraft((prev) => ({ ...prev, protocol: value as AccessProtocol | '' }))
          }
          onModelChange={(value) => setDraft((prev) => ({ ...prev, model: value }))}
          onAccessKeyIdChange={(value) => setDraft((prev) => ({ ...prev, accessKeyID: value }))}
          onSubmit={inspect}
          onRetryOptions={retryOptions}
        />
      </div>

      <div {...stylex.props(styles.stack)}>
        {resultRefreshing && (
          <div {...stylex.props(styles.refreshing)} role="status">
            {t('monitor.inspector.request.loading')}
          </div>
        )}
        {resultLoadingActive || resultLoading ? (
          <div
            {...stylex.props(styles.skeleton)}
            aria-label={t('monitor.inspector.request.loading')}
          >
            <Skeleton height={74} radius={2} />
            <Skeleton height={180} radius={2} />
          </div>
        ) : failed && observation === undefined ? (
          <EmptyState
            title={t('monitor.inspector.request.failed')}
            icon={<RouteIcon size={20} aria-hidden />}
            actions={
              <Button variant="secondary" size="sm" label={t('common.retry')} onClick={inspect} />
            }
          />
        ) : observation === undefined ? (
          <EmptyState
            {...stylex.props(styles.empty)}
            title={t('monitor.inspector.empty.title')}
            description={t('monitor.inspector.empty.description')}
            icon={<RouteIcon size={22} strokeWidth={1.7} aria-hidden />}
          />
        ) : (
          <>
            {pending && <Banner status="info" title={t('monitor.inspector.request.loading')} />}
            {inputChanged && !pending && (
              <Banner status="warning" title={t('monitor.inspector.result.inputChanged')} />
            )}
            {resultStale && (
              <Banner
                status="warning"
                title={t('monitor.inspector.result.stale')}
                endContent={
                  <Button
                    variant="secondary"
                    size="sm"
                    label={t('common.retry')}
                    onClick={inspect}
                  />
                }
              />
            )}

            <section
              {...stylex.props(
                styles.summary,
                observation.routable ? styles.summarySuccess : styles.summaryDanger,
              )}
              aria-labelledby="route-summary-title"
            >
              <header {...stylex.props(styles.summaryHeader)}>
                <div {...stylex.props(styles.summaryContent)}>
                  <div {...stylex.props(styles.summaryTitleRow)}>
                    <h2
                      id="route-summary-title"
                      ref={summaryRef}
                      tabIndex={-1}
                      {...stylex.props(styles.summaryTitle)}
                    >
                      {observation.routable
                        ? t('monitor.inspector.result.routableTitle')
                        : t('monitor.inspector.result.notRoutableTitle')}
                    </h2>
                    <Badge
                      variant={observation.routable ? 'success' : 'error'}
                      label={
                        observation.routable
                          ? t('monitor.inspector.result.routable')
                          : t('monitor.inspector.result.notRoutable')
                      }
                    />
                  </div>
                  {isKnownReason(observation.reason_code) && (
                    <p {...stylex.props(styles.summaryReason)}>
                      {t('monitor.inspector.result.reasonLine', {
                        reason: reasonLabel(observation.reason_code),
                      })}
                      <code {...stylex.props(styles.summaryReasonCode)}>
                        {observation.reason_code}
                      </code>
                    </p>
                  )}
                </div>
                <div {...stylex.props(styles.summaryMeta)}>
                  <span>
                    {t('monitor.inspector.result.routeStrategy', {
                      strategy: t(
                        `settings.runtime.routeStrategies.${observation.route_strategy}` as MessageId,
                      ),
                    })}
                  </span>
                  <time dateTime={formatISOInstant(observation.observed_at_ms)}>
                    {t('monitor.inspector.result.observedAt', {
                      time: formatLocalInstant(observation.observed_at_ms, locale),
                    })}
                  </time>
                </div>
              </header>

              <dl {...stylex.props(styles.facts)}>
                <div {...stylex.props(styles.fact)}>
                  <dt {...stylex.props(styles.factTerm)}>
                    {t('monitor.inspector.result.accessKey')}
                  </dt>
                  <dd {...stylex.props(styles.factValue)}>
                    <Tooltip content={observation.access_key.name}>
                      <span>{observation.access_key.name}</span>
                    </Tooltip>
                  </dd>
                </div>
                <div {...stylex.props(styles.fact)}>
                  <dt {...stylex.props(styles.factTerm)}>
                    {t('monitor.inspector.result.accessKeyStatus')}
                  </dt>
                  <dd {...stylex.props(styles.factValue)}>
                    <Badge
                      variant={badgeVariants[accessKeyStatusTone(observation.access_key.status)]}
                      label={t(
                        `monitor.inspector.accessKeyStatus.${observation.access_key.status}` as MessageId,
                      )}
                    />
                  </dd>
                </div>
                <div {...stylex.props(styles.fact)}>
                  <dt {...stylex.props(styles.factTerm)}>
                    {t('monitor.inspector.result.protocol')}
                  </dt>
                  <dd {...stylex.props(styles.factValue, styles.factMono)}>
                    <Tooltip content={`${observation.protocol} · ${observation.operation}`}>
                      <span>{`${observation.protocol} · ${observation.operation}`}</span>
                    </Tooltip>
                  </dd>
                </div>
                <div {...stylex.props(styles.fact)}>
                  <dt {...stylex.props(styles.factTerm)}>
                    {t('monitor.inspector.result.externalModel')}
                  </dt>
                  <dd {...stylex.props(styles.factValue, styles.factMono)}>
                    <Tooltip content={modelLabel(observation.external_model)}>
                      <span>{modelLabel(observation.external_model)}</span>
                    </Tooltip>
                  </dd>
                </div>
                <div {...stylex.props(styles.fact)}>
                  <dt {...stylex.props(styles.factTerm)}>
                    {t('monitor.inspector.result.candidateGroups')}
                  </dt>
                  <dd {...stylex.props(styles.factValue, styles.factNumber)}>
                    {formattedInteger(includedGroups.length)}
                  </dd>
                </div>
                <div {...stylex.props(styles.fact)}>
                  <dt {...stylex.props(styles.factTerm)}>
                    {t('monitor.inspector.result.availableCredentials')}
                  </dt>
                  <dd {...stylex.props(styles.factValue, styles.factNumber)}>
                    {formattedInteger(availableCredentialCount)}
                  </dd>
                </div>
              </dl>
            </section>

            <section {...stylex.props(styles.section)} aria-labelledby="route-candidates-title">
              <MonitorSectionHeading
                id="route-candidates-title"
                title={t('monitor.inspector.groups.title')}
                description={t(
                  `monitor.inspector.groups.description.${observation.route_strategy}` as MessageId,
                )}
                meta={t('monitor.inspector.groups.count', {
                  count: formattedInteger(includedGroups.length),
                })}
              />

              {includedGroups.length === 0 ? (
                <Banner status="info" title={t('monitor.inspector.groups.completeEmpty')} />
              ) : (
                <div
                  {...stylex.props(styles.ledger)}
                  role="table"
                  aria-label={t('monitor.inspector.groups.tableLabel')}
                  aria-rowcount={orderedIncludedGroups.length + 1}
                >
                  <div {...stylex.props(styles.ledgerHeader)} role="row" aria-rowindex={1}>
                    <span role="columnheader">{t('monitor.inspector.groups.columns.group')}</span>
                    <span role="columnheader">{t('monitor.inspector.groups.columns.status')}</span>
                    <span role="columnheader">
                      {t('monitor.inspector.groups.columns.credentials')}
                    </span>
                    <span role="columnheader">{t('monitor.inspector.groups.columns.weight')}</span>
                    <span role="columnheader">{t('monitor.inspector.groups.columns.share')}</span>
                    <span role="columnheader">{t('monitor.inspector.groups.columns.actions')}</span>
                  </div>

                  {orderedIncludedGroups.map((group, index) => (
                    <details
                      key={group.group_id}
                      {...stylex.props(styles.candidate)}
                      role="row"
                      aria-rowindex={index + 2}
                      open={groupExpanded(group.group_id)}
                      onToggle={(event) =>
                        setGroupExpanded(group.group_id, event.currentTarget.open)
                      }
                    >
                      <summary {...stylex.props(styles.candidateSummary)}>
                        <div {...stylex.props(styles.candidateIdentity)} role="cell">
                          <Tooltip content={group.group_name}>
                            <strong {...stylex.props(styles.candidateIdentityText)}>
                              {group.group_name}
                            </strong>
                          </Tooltip>
                          <Tooltip content={includedGroupIdentity(group)}>
                            <small {...stylex.props(styles.candidateIdentitySub)}>
                              {includedGroupIdentity(group)}
                            </small>
                          </Tooltip>
                        </div>
                        <div {...stylex.props(styles.candidateStatus)} role="cell">
                          <span {...stylex.props(styles.cellLabel)}>
                            {t('monitor.inspector.groups.columns.status')}
                          </span>
                          <Badge
                            variant={badgeVariants[routePriorityTone(group)]}
                            label={routePriorityLabel(group)}
                          />
                          <small {...stylex.props(styles.candidateStatusSmall)}>
                            {groupStatusLabel(group)}
                          </small>
                          <small {...stylex.props(styles.candidateStatusSmall)}>
                            <code>P{group.priority}</code>
                          </small>
                          {group.entry_cooldown_until_ms !== null && (
                            <small {...stylex.props(styles.candidateStatusSmall)}>
                              {t('monitor.inspector.groups.entryCooldown')}{' '}
                              {formatLocalInstant(group.entry_cooldown_until_ms, locale)}
                            </small>
                          )}
                        </div>
                        <div
                          {...stylex.props(
                            styles.candidateMeasure,
                            styles.candidateMeasureCredentials,
                          )}
                          role="cell"
                        >
                          <span {...stylex.props(styles.cellLabel)}>
                            {t('monitor.inspector.groups.columns.credentials')}
                          </span>
                          <strong {...stylex.props(styles.candidateMeasureValue)}>
                            {formattedInteger(groupAvailableCredentialCount(group))} /{' '}
                            {formattedInteger(group.credentials.length)}
                          </strong>
                          <small {...stylex.props(styles.candidateStatusSmall)}>
                            {t('monitor.inspector.groups.availableTotal')}
                          </small>
                        </div>
                        <div
                          {...stylex.props(styles.candidateMeasure, styles.candidateMeasureWeight)}
                          role="cell"
                        >
                          <span {...stylex.props(styles.cellLabel)}>
                            {t('monitor.inspector.groups.columns.weight')}
                          </span>
                          <strong {...stylex.props(styles.candidateMeasureValue)}>
                            {formattedInteger(group.entry_weight)}
                          </strong>
                        </div>
                        <div {...stylex.props(styles.candidateShare)} role="cell">
                          <span {...stylex.props(styles.cellLabel)}>
                            {t('monitor.inspector.groups.columns.share')}
                          </span>
                          <div
                            {...stylex.props(styles.shareMeter)}
                            role="progressbar"
                            aria-label={t('monitor.inspector.groups.shareLabel', {
                              name: group.group_name,
                              share: groupShareLabel(group),
                            })}
                            aria-valuemin={0}
                            aria-valuemax={100}
                            aria-valuenow={groupShare(group)}
                          >
                            <i
                              {...stylex.props(styles.shareMeterFill)}
                              style={{ width: `${groupShare(group)}%` }}
                            />
                          </div>
                          <small {...stylex.props(styles.candidateStatusSmall)}>
                            {groupShareLabel(group)}
                          </small>
                        </div>
                        <span {...stylex.props(styles.candidateDisclosure)} role="cell">
                          <ChevronRight size={17} aria-hidden="true" />
                        </span>
                      </summary>

                      <div {...stylex.props(styles.credentialDetails)}>
                        <header {...stylex.props(styles.credentialDetailsHeader)}>
                          <div {...stylex.props(styles.credentialDetailsHeaderInner)}>
                            <strong {...stylex.props(styles.credentialDetailsTitle)}>
                              {t('monitor.inspector.credentials.title')}
                            </strong>
                            <span>{candidateCredentialSummary(group)}</span>
                          </div>
                        </header>

                        {group.credentials.length === 0 ? (
                          <p {...stylex.props(styles.credentialDetailsEmpty)}>
                            {t('monitor.inspector.credentials.noneReturned')}
                          </p>
                        ) : (
                          <InspectorLedger
                            label={t('monitor.inspector.credentials.tableLabel', {
                              name: group.group_name,
                            })}
                            scrollHint={scrollHint}
                            rowCount={group.credentials.length + 1}
                            grid="88px minmax(170px, 1.4fr) minmax(148px, 1fr)"
                            header={
                              <div
                                {...stylex.props(styles.ledgerGrid)}
                                role="row"
                                aria-rowindex={1}
                                data-testid="ledger-record-list__header"
                              >
                                <span role="columnheader" {...stylex.props(styles.ledgerHeadCell)}>
                                  {t('monitor.inspector.credentials.columns.credential')}
                                </span>
                                <span role="columnheader" {...stylex.props(styles.ledgerHeadCell)}>
                                  {t('monitor.inspector.credentials.columns.status')}
                                </span>
                                <span role="columnheader" {...stylex.props(styles.ledgerHeadCell)}>
                                  {t('monitor.inspector.credentials.columns.cooldown')}
                                </span>
                              </div>
                            }
                          >
                            {orderedCredentials(group).map((credential, credentialIndex) => (
                              <article
                                key={credential.credential_id}
                                {...stylex.props(
                                  styles.credentialRecord,
                                  credentialIndex === 0 && styles.credentialRecordFirst,
                                )}
                                role="row"
                                aria-rowindex={credentialIndex + 2}
                              >
                                <div role="cell" {...stylex.props(styles.credentialMono)}>
                                  <span {...stylex.props(styles.cellLabel)}>
                                    {t('monitor.inspector.credentials.columns.credential')}
                                  </span>
                                  <code>{credential.credential_id}</code>
                                </div>
                                <div role="cell" {...stylex.props(styles.credentialStatus)}>
                                  <span {...stylex.props(styles.cellLabel)}>
                                    {t('monitor.inspector.credentials.columns.status')}
                                  </span>
                                  <Badge
                                    variant={badgeVariants[credentialTone(credential)]}
                                    label={credentialStatusLabel(credential)}
                                  />
                                  {credential.reason_code !== null && (
                                    <code {...stylex.props(styles.credentialCode)}>
                                      {credential.reason_code}
                                    </code>
                                  )}
                                </div>
                                <div role="cell" {...stylex.props(styles.credentialMono)}>
                                  <span {...stylex.props(styles.cellLabel)}>
                                    {t('monitor.inspector.credentials.columns.cooldown')}
                                  </span>
                                  {credential.cooldown_until_ms !== null ? (
                                    formatLocalInstant(credential.cooldown_until_ms, locale)
                                  ) : (
                                    <span>{t('monitor.inspector.credentials.none')}</span>
                                  )}
                                </div>
                              </article>
                            ))}
                          </InspectorLedger>
                        )}

                        <footer {...stylex.props(styles.credentialDetailsFooter)}>
                          <Button
                            href={groupDetailHref(group.group_id)}
                            variant="secondary"
                            size="sm"
                            label={t('monitor.inspector.groups.viewGroup')}
                            icon={<ArrowRight size={15} aria-hidden="true" />}
                          />
                        </footer>
                      </div>
                    </details>
                  ))}
                </div>
              )}
            </section>

            <section {...stylex.props(styles.section)} aria-labelledby="route-exclusions-title">
              <MonitorSectionHeading
                id="route-exclusions-title"
                title={t('monitor.inspector.excluded.title')}
                description={t('monitor.inspector.excluded.description')}
                meta={t('monitor.inspector.groups.count', {
                  count: formattedInteger(excludedGroups.length),
                })}
              />

              {excludedGroups.length === 0 ? (
                <Banner status="info" title={t('monitor.inspector.excluded.empty')} />
              ) : (
                <InspectorLedger
                  label={t('monitor.inspector.excluded.tableLabel')}
                  scrollHint={scrollHint}
                  rowCount={excludedGroups.length + 1}
                  grid="minmax(190px, 1.2fr) 120px minmax(220px, 1.45fr) minmax(160px, 0.9fr)"
                  header={
                    <div
                      {...stylex.props(styles.ledgerGrid)}
                      role="row"
                      aria-rowindex={1}
                      data-testid="ledger-record-list__header"
                    >
                      <span role="columnheader" {...stylex.props(styles.ledgerHeadCell)}>
                        {t('monitor.inspector.groups.columns.group')}
                      </span>
                      <span role="columnheader" {...stylex.props(styles.ledgerHeadCell)}>
                        {t('monitor.inspector.groups.columns.status')}
                      </span>
                      <span role="columnheader" {...stylex.props(styles.ledgerHeadCell)}>
                        {t('monitor.inspector.result.reason')}
                      </span>
                      <span role="columnheader" {...stylex.props(styles.ledgerHeadCell)}>
                        {t('monitor.inspector.excluded.reasonCode')}
                      </span>
                    </div>
                  }
                >
                  {excludedGroups.map((group, index) => (
                    <article
                      key={group.group_id}
                      {...stylex.props(styles.exclusionRecord)}
                      role="row"
                      aria-rowindex={index + 2}
                    >
                      <div role="cell" {...stylex.props(styles.exclusionIdentity)}>
                        <span {...stylex.props(styles.cellLabel)}>
                          {t('monitor.inspector.groups.columns.group')}
                        </span>
                        <Tooltip content={group.group_name}>
                          <strong {...stylex.props(styles.candidateIdentityText)}>
                            {group.group_name}
                          </strong>
                        </Tooltip>
                        <Tooltip content={excludedGroupIdentity(group)}>
                          <small {...stylex.props(styles.candidateIdentitySub)}>
                            {excludedGroupIdentity(group)}
                          </small>
                        </Tooltip>
                      </div>
                      <div role="cell">
                        <span {...stylex.props(styles.cellLabel)}>
                          {t('monitor.inspector.groups.columns.status')}
                        </span>
                        <Badge variant="neutral" label={t('monitor.inspector.groups.excluded')} />
                      </div>
                      {isKnownReason(group.reason_code) && (
                        <div role="cell" {...stylex.props(styles.exclusionReason)}>
                          <span {...stylex.props(styles.cellLabel)}>
                            {t('monitor.inspector.result.reason')}
                          </span>
                          {reasonLabel(group.reason_code)}
                        </div>
                      )}
                      {isKnownReason(group.reason_code) && (
                        <div role="cell" {...stylex.props(styles.credentialMono)}>
                          <span {...stylex.props(styles.cellLabel)}>
                            {t('monitor.inspector.excluded.reasonCode')}
                          </span>
                          <code {...stylex.props(styles.candidateStatusSmall)}>
                            {group.reason_code ?? '—'}
                          </code>
                        </div>
                      )}
                    </article>
                  ))}
                </InspectorLedger>
              )}
            </section>
          </>
        )}
      </div>
    </div>
  )
}
