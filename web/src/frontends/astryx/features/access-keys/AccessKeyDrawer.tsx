import * as stylex from '@stylexjs/stylex'
import { Button } from '@astryxdesign/core'
import { RotateCcw, Save, Trash2 } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'

import type { ChannelDto } from '@shared/control/resources/channels'
import type {
  AccessKeyDto,
  AccessKeyFiltersDto,
  AccessProtocol,
  GroupOptionDto,
} from '@shared/control/types'
import {
  areAccessKeyCostLimitRulesValid,
  isAccessKeyDraftDirty,
  isAccessKeyDraftValid,
} from '@shared/domain/access-keys/access-key-patch'
import {
  accessKeyProtocolOptions,
  buildAccessKeyModelOptions,
  buildAccessKeyProtocolCandidates,
} from '@shared/domain/access-keys/access-key-options'
import {
  materializeAccessKeyFilters,
  validateAccessKeyScope,
  type AccessKeyScopeDimension,
  type AccessKeyScopeMode,
  type GroupCatalogState,
} from '@shared/domain/access-keys/access-key-scope'
import {
  isAccessKeyDrawerCreateOperationActive,
  isAccessKeyDrawerUnsavedDirty,
  type AccessKeyDrawerSnapshot,
} from '@shared/controllers/access-key-drawer'
import type { PendingAccessKeyCreateOperation } from '@shared/domain/access-keys/access-key-create-operation'
import type { PendingAccessKeyEditOperation } from '@shared/domain/access-keys/access-key-edit-operation'
import type { PendingAccessKeyRotateOperation } from '@shared/domain/access-keys/access-key-rotate-operation'
import { isValidPriceMultiplier } from '@shared/lib/price-multiplier'
import type { MessageId } from '@shared/i18n/message-ids'

import { useT } from '../../app/i18n'
import { useAccessKeyDrawerController } from '../../app/use-access-key-drawer'
import { useUnsavedChanges } from '../../app/use-unsaved-changes'
import { DetailPanel } from '../../components/DetailPanel'
import { AccessKeyCostLimitEditor } from './AccessKeyCostLimitEditor'
import { AccessKeyDeleteDialog } from './AccessKeyDeleteDialog'
import { AccessKeyFormFields, type AccessKeyFormFieldsHandle } from './AccessKeyFormFields'
import { AccessKeyOperationFeedback } from './AccessKeyOperationFeedback'
import { AccessKeyPolicyFields } from './AccessKeyPolicyFields'
import { AccessKeyRotateDialog } from './AccessKeyRotateDialog'
import { AccessKeyScopeEditor, type AccessKeyScopeOption } from './AccessKeyScopeEditor'

const styles = stylex.create({
  form: {
    display: 'block',
    fontSize: 'var(--text-body)',
  },
  section: {
    marginTop: 22,
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: 20,
  },
  sectionTitle: {
    marginTop: 0,
    marginBottom: 4,
    fontSize: 'var(--text-meta)',
  },
  sectionDescription: {
    marginTop: 0,
    marginBottom: 12,
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  scopeLogic: {
    display: 'grid',
    gridTemplateColumns: {
      default: '1fr auto 1fr auto 1fr',
      '@media (max-width: 480px)': '1fr',
    },
    alignItems: 'center',
    gap: 6,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    padding: 9,
    textAlign: 'center',
  },
  scopeLogicTerm: {
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-surface)',
    color: 'var(--color-text-muted)',
    padding: 6,
    fontSize: 'var(--text-label-xs)',
  },
  scopeLogicJoin: {
    color: 'var(--color-text-faint)',
    fontFamily: 'var(--font-mono)',
    fontSize: 9,
  },
  scopeEditors: {
    marginTop: 12,
  },
  scopeWarning: {
    display: 'flex',
    alignItems: 'flex-start',
    gap: 10,
    marginTop: 12,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
    paddingBlock: 10,
    paddingInline: 12,
    fontSize: 'var(--text-sm)',
  },
  scopeWarningGlyph: {
    display: 'grid',
    width: 17,
    height: 17,
    flexShrink: 0,
    placeItems: 'center',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'currentColor',
    borderRadius: '50%',
    fontFamily: 'var(--font-serif)',
    fontSize: 'var(--text-label-xs)',
    fontWeight: 700,
  },
  scopeWarningText: {
    margin: 0,
  },
  footer: {
    display: 'flex',
    alignItems: 'center',
    gap: 'var(--space-2)',
    width: '100%',
  },
  management: {
    display: 'flex',
    flexShrink: 0,
    gap: 'var(--space-2)',
  },
  saveBlocker: {
    minWidth: 0,
    flexGrow: 1,
    overflow: 'hidden',
    margin: 0,
    color: 'var(--color-danger)',
    fontSize: 'var(--text-label-xs)',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
})

function drawerDerived(snapshot: AccessKeyDrawerSnapshot) {
  const createOperationActive = isAccessKeyDrawerCreateOperationActive(snapshot)
  return {
    editing: snapshot.base !== null,
    createOperationActive,
    formLocked:
      snapshot.pending ||
      snapshot.rotationPending ||
      createOperationActive ||
      snapshot.editReconciliation !== null,
    closeBlocked: snapshot.pending || snapshot.rotationPending,
    dirty: isAccessKeyDraftDirty(snapshot.draft, snapshot.base),
    unsavedDirty: isAccessKeyDrawerUnsavedDirty(snapshot),
  }
}

export function AccessKeyDrawer({
  open,
  accessKey,
  groups,
  channels,
  total,
  groupCatalogState,
  createOperation,
  editOperation,
  rotateOperation,
  onCreateOperation,
  onEditOperation,
  onRotateOperation,
  onOpenChange,
  onSaved,
  onRotated,
  onDeleted,
}: {
  open: boolean
  accessKey: AccessKeyDto | null
  groups: readonly GroupOptionDto[]
  channels: readonly ChannelDto[]
  total: number
  groupCatalogState: GroupCatalogState
  createOperation: PendingAccessKeyCreateOperation | null
  editOperation: PendingAccessKeyEditOperation | null
  rotateOperation: PendingAccessKeyRotateOperation | null
  onCreateOperation(operation: PendingAccessKeyCreateOperation | null): void
  onEditOperation(operation: PendingAccessKeyEditOperation | null): void
  onRotateOperation(operation: PendingAccessKeyRotateOperation | null): void
  onOpenChange(open: boolean): Promise<void>
  onSaved(kind: 'created' | 'updated', name: string): void
  onRotated(name: string): void
  onDeleted(name: string): void
}) {
  const t = useT()
  const formFieldsRef = useRef<AccessKeyFormFieldsHandle | null>(null)
  const openRef = useRef(open)
  // Synced in an effect (refs must not be written during render). Declared
  // before the open/close effect below so it always runs first in the commit
  // where `open` flips — the controller's isOpen() getter is only read from
  // async callbacks, never during render.
  useEffect(() => {
    openRef.current = open
  }, [open])

  const groupCatalog = useMemo(
    () => ({ state: groupCatalogState, ids: groups.map(({ id }) => id) }),
    [groupCatalogState, groups],
  )

  const { controller, snapshot } = useAccessKeyDrawerController({
    isOpen: () => openRef.current,
    onSaved: (kind, name) => void onSaved(kind, name),
    onDeleted: (name) => void onDeleted(name),
    onCreateOperation,
    onEditOperation,
  })

  const [rotateOpen, setRotateOpen] = useState(false)
  const [deleteOpen, setDeleteOpen] = useState(false)

  const { draft, base } = snapshot
  const derived = drawerDerived(snapshot)
  const { createOperationActive, formLocked, closeBlocked, dirty } = derived

  // Draft field mutation helper — every edit path replaces the draft object so
  // the snapshot diff stays trivially detectable.
  const patchDraft = (patch: Partial<typeof draft>) =>
    controller.setDraft({ ...draft, ...patch })
  const patchFilters = (patch: Partial<AccessKeyFiltersDto>) =>
    controller.setDraft({ ...draft, filters: { ...draft.filters, ...patch } })
  const patchScopeModes = (dimension: AccessKeyScopeDimension, mode: AccessKeyScopeMode) =>
    controller.setDraft({ ...draft, scopeModes: { ...draft.scopeModes, [dimension]: mode } })

  const protocolOptions = useMemo(() => accessKeyProtocolOptions(), [])
  const selectedGroupIDs = useMemo(
    () => (draft.scopeModes.groups === 'restricted' ? draft.filters.groups : []),
    [draft.scopeModes.groups, draft.filters.groups],
  )
  const supportedProtocolOptions = useMemo(
    () => buildAccessKeyProtocolCandidates([...groups], [...channels], selectedGroupIDs),
    [groups, channels, selectedGroupIDs],
  )
  const catalogModelOptions = useMemo(
    () => buildAccessKeyModelOptions([...groups], [], selectedGroupIDs),
    [groups, selectedGroupIDs],
  )
  const modelOptions = useMemo<AccessKeyScopeOption[]>(() => {
    const catalog = new Set(catalogModelOptions)
    return buildAccessKeyModelOptions([...groups], draft.filters.models, selectedGroupIDs).map(
      (model) => ({
        value: model,
        label: model,
        description: catalog.has(model)
          ? undefined
          : t('accessKeys.drawer.modelCustomUnavailable'),
      }),
    )
  }, [catalogModelOptions, groups, draft.filters.models, selectedGroupIDs, t])

  const groupProtocolMismatch =
    groupCatalogState !== 'loading' &&
    groupCatalogState !== 'error' &&
    draft.scopeModes.groups === 'restricted' &&
    draft.scopeModes.protocols === 'restricted' &&
    !draft.filters.protocols.some((protocol) => supportedProtocolOptions.includes(protocol))
  const modelMismatch =
    draft.scopeModes.models === 'restricted' &&
    draft.filters.models.some((model) => !catalogModelOptions.includes(model))
  const valid =
    isAccessKeyDraftValid(draft, base, groupCatalog) && !groupProtocolMismatch

  const scopeValid = validateAccessKeyScope({
    base: base?.filters ?? null,
    filters: draft.filters,
    modes: draft.scopeModes,
    groupCatalog,
  })
  const scopeFeedbackKey = ((): MessageId | '' => {
    if (groupProtocolMismatch) return 'accessKeys.drawer.groupProtocolMismatch'
    if (scopeValid) return ''
    const effective = materializeAccessKeyFilters(draft.filters, draft.scopeModes)
    if (
      (['groups', 'protocols', 'models'] as const).some(
        (dimension) =>
          draft.scopeModes[dimension] === 'restricted' && effective[dimension].length === 0,
      )
    ) {
      return 'accessKeys.drawer.scopeIncomplete'
    }
    if (groupCatalogState === 'loading' || groupCatalogState === 'error') {
      return 'accessKeys.drawer.groupScopeUnavailable'
    }
    if (groupCatalogState === 'stale') return 'accessKeys.drawer.staleGroupScopeInvalid'
    return 'accessKeys.drawer.scopeIncomplete'
  })()

  const scopeSaveBlockerKey = ((): MessageId | '' => {
    switch (scopeFeedbackKey) {
      case 'accessKeys.drawer.groupProtocolMismatch':
        return 'accessKeys.drawer.saveBlockedGroupProtocol'
      case 'accessKeys.drawer.groupScopeUnavailable':
        return 'accessKeys.drawer.saveBlockedGroupUnavailable'
      case 'accessKeys.drawer.staleGroupScopeInvalid':
        return 'accessKeys.drawer.saveBlockedStaleScope'
      case 'accessKeys.drawer.scopeIncomplete':
        return 'accessKeys.drawer.saveBlockedScope'
      default:
        return ''
    }
  })()

  const saveBlockerKey = ((): MessageId | '' => {
    if (snapshot.pending) return 'accessKeys.drawer.saveBlockedPending'
    if (snapshot.editReconciliation || createOperationActive) return ''
    if (draft.name.trim().length === 0) return 'accessKeys.drawer.saveBlockedName'
    if (!Number.isSafeInteger(draft.rpm_limit) || draft.rpm_limit < 0) {
      return 'accessKeys.drawer.saveBlockedRPM'
    }
    if (!isValidPriceMultiplier(draft.price_multiplier)) {
      return 'common.priceMultiplier.invalid'
    }
    if (!areAccessKeyCostLimitRulesValid(draft.costLimitRules)) {
      return 'accessKeys.drawer.saveBlockedCostLimits'
    }
    if (scopeSaveBlockerKey) return scopeSaveBlockerKey
    if (!valid) return 'accessKeys.drawer.saveBlockedInvalid'
    if (!dirty) return 'accessKeys.drawer.saveBlockedNoChanges'
    return ''
  })()

  const mutationFeedbackKey: MessageId | '' = (() => {
    if (snapshot.mutationState === 'idle') return ''
    if (snapshot.editReconciliation) {
      return snapshot.mutationState === 'reconciling'
        ? 'accessKeys.drawer.editReconciling'
        : 'accessKeys.drawer.editIndeterminate'
    }
    return snapshot.mutationState === 'reconciling'
      ? 'accessKeys.drawer.saveReconciling'
      : 'accessKeys.drawer.saveIndeterminate'
  })()

  const rotateActionDisabled =
    snapshot.pending ||
    snapshot.rotationPending ||
    createOperationActive ||
    snapshot.editReconciliation !== null ||
    dirty

  const groupOptions = useMemo<AccessKeyScopeOption[]>(() => {
    const baseGroupIDs = new Set(base?.filters.groups ?? [])
    const options: AccessKeyScopeOption[] = groups.map((group) => ({
      value: group.id,
      label: group.name,
      disabled: groupCatalogState === 'stale' && !baseGroupIDs.has(group.id),
    }))
    const known = new Set(groups.map(({ id }) => id))
    for (const id of draft.filters.groups) {
      if (!known.has(id)) {
        options.push({
          value: id,
          label: t('accessKeys.drawer.unknownGroup'),
          disabled: false,
        })
      }
    }
    return options
  }, [groups, base, draft.filters.groups, groupCatalogState, t])

  const unsaved = useUnsavedChanges({
    dirty: derived.unsavedDirty,
    blocked: closeBlocked,
  })

  // Classic watch(open/accessKey, immediate): open → resetForOpen; close →
  // clearLocalState. Runs in an effect so the controller stays consistent when
  // the view remounts the drawer (v-if gate).
  useEffect(() => {
    if (open) {
      controller.resetForOpen({ accessKey, createOperation, editOperation })
      // Two Vue nextTicks collapse into one post-commit paint — the name input
      // is present by the time effects run.
      requestAnimationFrame(() => formFieldsRef.current?.focusName())
    } else {
      controller.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reset keys on open+target only
  }, [open, accessKey, controller])

  const setOpen = async (value: boolean) => {
    if (!value && !(await unsaved.confirmDiscard())) return
    if (!value) controller.close()
    await onOpenChange(value)
  }

  const canChangeScopeValue = (
    dimension: AccessKeyScopeDimension,
    value: number | string,
    adding: boolean,
  ): boolean => {
    if (formLocked || draft.scopeModes[dimension] !== 'restricted') return false
    if (dimension !== 'groups') {
      if (groupCatalogState === 'loading' || groupCatalogState === 'error') return false
      return true
    }
    if (groupCatalogState === 'ready') return true
    if (groupCatalogState !== 'stale') return false
    if (!adding) return true
    return base?.filters.groups.includes(value as number) ?? false
  }

  const setScopeMode = (dimension: AccessKeyScopeDimension, nextMode: AccessKeyScopeMode) => {
    const catalogBlocksChange =
      dimension === 'groups'
        ? groupCatalogState !== 'ready'
        : groupCatalogState === 'loading' || groupCatalogState === 'error'
    if (formLocked || catalogBlocksChange || (nextMode !== 'all' && nextMode !== 'restricted')) {
      return
    }
    patchScopeModes(dimension, nextMode)
  }

  const setGroups = (groupIDs: number[]) => {
    const current = new Set(draft.filters.groups)
    const next = new Set(groupIDs)
    for (const groupID of next) {
      if (!current.has(groupID) && !canChangeScopeValue('groups', groupID, true)) return
    }
    for (const groupID of current) {
      if (!next.has(groupID) && !canChangeScopeValue('groups', groupID, false)) return
    }
    patchFilters({ groups: [...next] })
  }

  const setProtocols = (protocols: AccessProtocol[]) => {
    const current = new Set(draft.filters.protocols)
    const next = new Set(protocols)
    for (const protocol of next) {
      if (!current.has(protocol) && !canChangeScopeValue('protocols', protocol, true)) return
    }
    for (const protocol of current) {
      if (!next.has(protocol) && !canChangeScopeValue('protocols', protocol, false)) return
    }
    patchFilters({ protocols: [...next] })
  }

  const setModels = (models: string[]) => {
    const current = new Set(draft.filters.models)
    const next = new Set(models.map((model) => model.trim()).filter(Boolean))
    for (const model of next) {
      if (!current.has(model) && !canChangeScopeValue('models', model, true)) return
    }
    for (const model of current) {
      if (!next.has(model) && !canChangeScopeValue('models', model, false)) return
    }
    patchFilters({ models: [...next] })
  }

  const addModel = () => {
    const model = snapshot.modelInput.trim()
    if (!model || draft.filters.models.includes(model)) return
    if (!canChangeScopeValue('models', model, true)) return
    patchFilters({ models: [...draft.filters.models, model] })
    controller.setModelInput('')
  }

  const handleRotated = (next: AccessKeyDto) => {
    controller.applyRotated(next)
    onRotated(next.name)
  }

  const handleDeleted = (name: string) => {
    controller.applyDeleted(name)
  }

  const saveLabel = snapshot.editReconciliation
    ? t('accessKeys.drawer.checkResult')
    : createOperationActive
      ? t('accessKeys.drawer.checkResult')
      : derived.editing
        ? t('accessKeys.drawer.saveChanges')
        : t('accessKeys.drawer.createKey')

  return (
    <DetailPanel
      isOpen={open}
      onOpenChange={(value) => void setOpen(value)}
      title={t(derived.editing ? 'accessKeys.drawer.editTitle' : 'accessKeys.drawer.createTitle')}
      subtitle={t(
        derived.editing
          ? 'accessKeys.drawer.editDescription'
          : 'accessKeys.drawer.createDescription',
      )}
      dismissible={!closeBlocked}
      footer={
        <div {...stylex.props(styles.footer)}>
          {derived.editing && base && (
            <div {...stylex.props(styles.management)}>
              <Button
                variant="secondary"
                size="sm"
                isDisabled={rotateActionDisabled}
                icon={<RotateCcw size={15} />}
                label={
                  rotateOperation
                    ? t('accessKeys.rotate.checkResult')
                    : t('accessKeys.rotate.open')
                }
                onClick={() => setRotateOpen(true)}
              />
              <AccessKeyRotateDialog
                accessKey={base}
                open={rotateOpen}
                onOpenChange={setRotateOpen}
                operation={rotateOperation}
                onPendingChange={(value) => controller.setRotationPending(value)}
                onOperationChange={onRotateOperation}
                onRotated={handleRotated}
              />
              <Button
                variant="destructive"
                size="sm"
                isDisabled={formLocked}
                icon={<Trash2 size={15} />}
                label={t('accessKeys.delete.open')}
                onClick={() => setDeleteOpen(true)}
              />
              <AccessKeyDeleteDialog
                accessKey={base}
                total={total}
                open={deleteOpen}
                onOpenChange={setDeleteOpen}
                onDeleted={handleDeleted}
              />
            </div>
          )}
          <p
            id="access-key-save-blocker"
            {...stylex.props(styles.saveBlocker)}
            role="status"
            aria-live="polite"
            title={saveBlockerKey ? t(saveBlockerKey) : undefined}
          >
            {saveBlockerKey ? t(saveBlockerKey) : ''}
          </p>
          <Button
            variant="secondary"
            size="sm"
            label={t('common.cancel')}
            isDisabled={closeBlocked}
            onClick={() => void setOpen(false)}
          />
          <Button
            size="sm"
            icon={<Save size={15} aria-hidden />}
            label={saveLabel}
            isLoading={snapshot.pending}
            isDisabled={
              !snapshot.editReconciliation && !createOperationActive && (!valid || !dirty)
            }
            onClick={() => void controller.save(valid)}
          />
        </div>
      }
    >
      <form
        id="access-key-drawer-form"
        {...stylex.props(styles.form)}
        onSubmit={(event) => {
          event.preventDefault()
          void controller.save(valid)
        }}
      >
        <AccessKeyOperationFeedback
          failed={snapshot.failed}
          editNotApplied={snapshot.editNotApplied}
          mutationFeedbackKey={mutationFeedbackKey}
        />

        <section>
          <h3 {...stylex.props(styles.sectionTitle)}>
            {t('accessKeys.drawer.basicInformation')}
          </h3>
          <p {...stylex.props(styles.sectionDescription)}>
            {t('accessKeys.drawer.basicInformationDescription')}
          </p>
          <AccessKeyFormFields
            ref={formFieldsRef}
            name={draft.name}
            status={draft.status}
            rpmLimit={draft.rpm_limit}
            priceMultiplier={draft.price_multiplier}
            disabled={formLocked}
            onNameChange={(value) => patchDraft({ name: value })}
            onStatusChange={(value) => patchDraft({ status: value })}
            onRpmLimitChange={(value) => patchDraft({ rpm_limit: value })}
            onPriceMultiplierChange={(value) => patchDraft({ price_multiplier: value })}
          />
        </section>

        <section {...stylex.props(styles.section)}>
          <AccessKeyCostLimitEditor
            value={draft.costLimitRules}
            runtimeStatus={base?.cost_limit_status ?? null}
            disabled={formLocked}
            onChange={(costLimitRules) => patchDraft({ costLimitRules })}
          />
        </section>

        <section {...stylex.props(styles.section)}>
          <h3 {...stylex.props(styles.sectionTitle)}>{t('accessKeys.drawer.accessPolicy')}</h3>
          <p {...stylex.props(styles.sectionDescription)}>
            {t('accessKeys.drawer.accessPolicyDescription')}
          </p>
          <AccessKeyPolicyFields
            expirationMode={draft.expirationMode}
            expiresAt={draft.expires_at_ms}
            baseExpiresAt={base?.expires_at_ms}
            sourceMode={draft.sourceMode}
            allowedCidrs={draft.filters.allowed_cidrs}
            disabled={formLocked}
            onExpirationModeChange={(value) => patchDraft({ expirationMode: value })}
            onExpiresAtChange={(value) => patchDraft({ expires_at_ms: value })}
            onSourceModeChange={(value) => patchDraft({ sourceMode: value })}
            onAllowedCidrsChange={(value) => patchFilters({ allowed_cidrs: value })}
          />
        </section>

        <section {...stylex.props(styles.section)}>
          <h3 {...stylex.props(styles.sectionTitle)}>
            {t('accessKeys.drawer.permissionScope')}
          </h3>
          <p {...stylex.props(styles.sectionDescription)}>
            {t('accessKeys.drawer.permissionScopeDescription')}
          </p>
          <div
            {...stylex.props(styles.scopeLogic)}
            aria-label={t('accessKeys.drawer.scopeLogic')}
          >
            <span {...stylex.props(styles.scopeLogicTerm)}>
              {t('accessKeys.drawer.scopeLogicGroups')}
            </span>
            <b {...stylex.props(styles.scopeLogicJoin)}>AND</b>
            <span {...stylex.props(styles.scopeLogicTerm)}>
              {t('accessKeys.drawer.scopeLogicProtocols')}
            </span>
            <b {...stylex.props(styles.scopeLogicJoin)}>AND</b>
            <span {...stylex.props(styles.scopeLogicTerm)}>
              {t('accessKeys.drawer.scopeLogicModels')}
            </span>
          </div>
          <div {...stylex.props(styles.scopeEditors)}>
            <AccessKeyScopeEditor
              modelInput={snapshot.modelInput}
              modes={draft.scopeModes}
              filters={draft.filters}
              groupOptions={groupOptions}
              groupCatalogState={groupCatalogState}
              protocolOptions={protocolOptions}
              modelOptions={modelOptions}
              disabled={formLocked}
              modelMismatch={modelMismatch}
              onSetScopeMode={setScopeMode}
              onGroupsChange={setGroups}
              onProtocolsChange={setProtocols}
              onModelsChange={setModels}
              onModelInputChange={(value) => controller.setModelInput(value)}
              onAddModel={addModel}
            />
          </div>
          <div {...stylex.props(styles.scopeWarning)}>
            <span {...stylex.props(styles.scopeWarningGlyph)} aria-hidden="true">
              !
            </span>
            <p {...stylex.props(styles.scopeWarningText)}>
              {t('accessKeys.drawer.scopeExpansionWarning')}
            </p>
          </div>
        </section>
      </form>
      {unsaved.dialog}
    </DetailPanel>
  )
}
