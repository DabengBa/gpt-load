import * as stylex from '@stylexjs/stylex'
import { Button, SegmentedControl, SegmentedControlItem, TextInput } from '@astryxdesign/core'
import { Plus, Search, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

import type { AccessKeyFiltersDto, AccessProtocol } from '@shared/control/types'
import type {
  AccessKeyScopeDimension,
  AccessKeyScopeMode,
  AccessKeyScopeModes,
  GroupCatalogState,
} from '@shared/domain/access-keys/access-key-scope'

import { useT } from '../../app/i18n'
import { plainTextInputAttrs } from '../../components/input-attrs'

export type AccessKeyScopeValue = string | number

/** Same shape as classic `SearchableMultiSelectOption`. */
export interface AccessKeyScopeOption {
  value: AccessKeyScopeValue
  label: string
  description?: string
  disabled?: boolean
}

const styles = stylex.create({
  scopeEditor: {
    minWidth: 0,
    margin: 0,
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    padding: 0,
  },
  // `.scope-editor + .scope-editor { margin-top: 10px }` — StyleX has no
  // sibling selectors, so the spacing rides on the 2nd/3rd fieldset.
  scopeEditorSibling: {
    marginTop: '10px',
  },
  srOnly: {
    position: 'absolute',
    width: '1px',
    height: '1px',
    margin: '-1px',
    padding: 0,
    overflow: 'hidden',
    clip: 'rect(0 0 0 0)',
    whiteSpace: 'nowrap',
    borderWidth: 0,
  },
  scopeHead: {
    display: 'flex',
    minHeight: '48px',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '12px',
    borderBottomWidth: 1,
    borderBottomStyle: 'solid',
    borderBottomColor: 'var(--color-border-subtle)',
    paddingBlock: '8px',
    paddingInline: '10px',
  },
  scopeHeadTitle: {
    display: 'block',
    fontSize: 'var(--text-sm)',
  },
  scopeHeadDescription: {
    display: 'block',
    marginTop: '1px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  scopeBody: {
    padding: '10px',
  },
  permissionNote: {
    display: 'flex',
    alignItems: 'center',
    gap: '7px',
    color: 'var(--color-text-faint)',
    fontSize: 'var(--text-label-xs)',
  },
  permissionNoteDot: {
    width: '6px',
    height: '6px',
    flex: 'none',
    borderRadius: '50%',
    backgroundColor: 'var(--color-action)',
  },
  modelRisk: {
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-warning)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-warning-bg)',
    color: 'var(--color-warning)',
    marginTop: '8px',
    marginBottom: 0,
    marginInline: 0,
    paddingBlock: 'var(--space-2)',
    paddingInline: 'var(--space-3)',
    fontSize: 'var(--text-label-xs)',
  },
  modelEntry: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: 'var(--space-2)',
    marginTop: '8px',
  },
  modelEntryInput: {
    minWidth: 0,
    height: 'var(--control-compact)',
    flex: '1 1 220px',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-control)',
    borderRadius: 'var(--radius-control)',
    backgroundColor: 'var(--color-surface-sunken)',
    color: 'var(--color-text)',
    paddingBlock: 0,
    paddingInline: '10px',
    fontFamily: 'inherit',
    fontSize: 'var(--text-sm)',
    cursor: { ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
  },
})

const selectStyles = stylex.create({
  root: {
    display: 'grid',
    gap: '8px',
  },
  chips: {
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '6px',
  },
  chip: {
    display: 'inline-flex',
    minHeight: '28px',
    alignItems: 'center',
    gap: 'var(--space-1)',
    borderRadius: 'var(--radius-tag)',
    backgroundColor: 'var(--color-action-soft)',
    color: 'var(--color-action)',
    paddingInlineStart: '9px',
    fontSize: 'var(--text-label-xs)',
  },
  chipContent: {
    display: 'grid',
    gap: 'var(--space-1)',
  },
  chipDescription: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  chipRemove: {
    display: 'inline-flex',
    width: '28px',
    height: '28px',
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-text-muted)',
    padding: 0,
    fontFamily: 'inherit',
    fontSize: 'inherit',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
  },
  picker: {
    display: 'grid',
    gap: '8px',
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
    paddingTop: '9px',
  },
  head: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 'var(--space-2)',
  },
  count: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  clear: {
    borderWidth: 0,
    backgroundColor: 'transparent',
    color: 'var(--color-action)',
    padding: 0,
    fontFamily: 'inherit',
    fontSize: 'inherit',
    textDecorationLine: 'underline',
    cursor: { default: 'pointer', ':disabled': 'not-allowed' },
    opacity: { ':disabled': 0.55 },
  },
  options: {
    display: 'grid',
    maxHeight: '252px',
    overflowY: 'auto',
    borderWidth: 1,
    borderStyle: 'solid',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
  },
  option: {
    display: 'flex',
    alignItems: 'center',
    minHeight: 'var(--touch-target)',
    gap: 'var(--space-2)',
    paddingBlock: 0,
    paddingInline: 'var(--space-3)',
  },
  optionDivider: {
    borderTopWidth: 1,
    borderTopStyle: 'solid',
    borderTopColor: 'var(--color-border-subtle)',
  },
  optionDisabled: {
    cursor: 'not-allowed',
    opacity: 0.55,
  },
  optionInput: {
    width: '15px',
    height: '15px',
    flex: 'none',
    margin: 0,
    cursor: { ':disabled': 'not-allowed' },
  },
  optionContent: {
    display: 'grid',
    minWidth: 0,
    gap: 'var(--space-1)',
  },
  optionDescription: {
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
  feedback: {
    display: 'flex',
    minHeight: '38px',
    alignItems: 'center',
    margin: 0,
    borderWidth: 1,
    borderStyle: 'dashed',
    borderColor: 'var(--color-border-subtle)',
    borderRadius: 'var(--radius-control)',
    paddingBlock: 0,
    paddingInline: 'var(--space-3)',
    color: 'var(--color-text-muted)',
    fontSize: 'var(--text-label-xs)',
  },
})

/**
 * Always-open searchable checkbox list — the React port of classic
 * `SearchableMultiSelect` restricted to the `always-open` mode this editor is
 * the only consumer of. Selected values render as removable chips; the picker
 * (count, clear, optional search, option rows) stays permanently expanded.
 */
function ScopeMultiSelect({
  id,
  label,
  options,
  value,
  onChange,
  disabled = false,
  loading = false,
  searchable = true,
  autoFocusSearch = false,
  searchLabel,
  searchPlaceholder = '',
  emptyLabel,
  loadingLabel,
}: {
  id: string
  label: string
  options: AccessKeyScopeOption[]
  value: AccessKeyScopeValue[]
  onChange: (value: AccessKeyScopeValue[]) => void
  disabled?: boolean
  loading?: boolean
  searchable?: boolean
  autoFocusSearch?: boolean
  searchLabel: string
  searchPlaceholder?: string
  emptyLabel: string
  loadingLabel: string
}) {
  const t = useT()
  const searchRef = useRef<HTMLInputElement>(null)
  const autofocusedRef = useRef(false)
  const [query, setQuery] = useState('')

  const selected = new Set(value)
  const normalizedQuery = query.trim().toLocaleLowerCase()
  const filteredOptions =
    !searchable || normalizedQuery === ''
      ? options
      : options.filter((option) =>
          `${option.label} ${option.description ?? ''}`
            .toLocaleLowerCase()
            .includes(normalizedQuery),
        )
  const selectedOptions = value.map((entry) => {
    const option = options.find((candidate) => candidate.value === entry)
    return {
      value: entry,
      label: option?.label ?? String(entry),
      description: option?.description,
    }
  })
  const selectedCountText = t('accessKeys.drawer.selectedCount', { count: value.length })
  const inert = disabled || loading

  // Classic `auto-focus-search`: focus the search input once the picker is
  // visible and interactive (mount for this always-open variant).
  useEffect(() => {
    if (autofocusedRef.current) return
    if (!autoFocusSearch || !searchable || inert) return
    autofocusedRef.current = true
    searchRef.current?.focus()
  }, [autoFocusSearch, searchable, inert])

  function toggle(entry: AccessKeyScopeValue, checked: boolean): void {
    if (inert) return
    onChange(
      checked ? [...new Set([...value, entry])] : value.filter((current) => current !== entry),
    )
  }

  function remove(entry: AccessKeyScopeValue): void {
    if (inert) return
    onChange(value.filter((current) => current !== entry))
  }

  function clear(): void {
    if (inert) return
    onChange([])
  }

  return (
    <section {...stylex.props(selectStyles.root)} aria-labelledby={`${id}-label`}>
      <span id={`${id}-label`} {...stylex.props(styles.srOnly)}>
        {label}
      </span>
      <div {...stylex.props(selectStyles.chips)} aria-label={selectedCountText}>
        {selectedOptions.map((option) => (
          <span key={String(option.value)} {...stylex.props(selectStyles.chip)}>
            <span {...stylex.props(selectStyles.chipContent)}>
              <span>{option.label}</span>
              {option.description ? (
                <small {...stylex.props(selectStyles.chipDescription)}>{option.description}</small>
              ) : null}
            </span>
            <button
              type="button"
              {...stylex.props(selectStyles.chipRemove)}
              aria-label={t('accessKeys.drawer.removeSelection', { label: option.label })}
              disabled={inert}
              onClick={() => remove(option.value)}
            >
              <X size={15} aria-hidden="true" />
            </button>
          </span>
        ))}
      </div>

      <div id={`${id}-picker`} {...stylex.props(selectStyles.picker)}>
        <div {...stylex.props(selectStyles.head)}>
          <span {...stylex.props(selectStyles.count)} aria-live="polite">
            {selectedCountText}
          </span>
          {value.length > 0 && (
            <button
              type="button"
              {...stylex.props(selectStyles.clear)}
              disabled={inert}
              onClick={clear}
            >
              {t('accessKeys.drawer.clearSelected')}
            </button>
          )}
        </div>

        {searchable && (
          <TextInput
            id={`${id}-search`}
            ref={searchRef}
            label={searchLabel}
            isLabelHidden
            placeholder={searchPlaceholder}
            value={query}
            isDisabled={inert}
            hasClear
            size="sm"
            startIcon={<Search size={15} aria-hidden="true" />}
            onChange={setQuery}
            {...plainTextInputAttrs}
          />
        )}

        {loading ? (
          <p {...stylex.props(selectStyles.feedback)} role="status">
            {loadingLabel}
          </p>
        ) : filteredOptions.length === 0 ? (
          <p {...stylex.props(selectStyles.feedback)}>{emptyLabel}</p>
        ) : (
          <div {...stylex.props(selectStyles.options)} role="group" aria-label={label}>
            {filteredOptions.map((option, index) => {
              const optionDisabled = inert || Boolean(option.disabled)
              return (
                <label
                  key={String(option.value)}
                  {...stylex.props(
                    selectStyles.option,
                    index > 0 && selectStyles.optionDivider,
                    optionDisabled && selectStyles.optionDisabled,
                  )}
                >
                  <input
                    type="checkbox"
                    {...stylex.props(selectStyles.optionInput)}
                    checked={selected.has(option.value)}
                    disabled={optionDisabled}
                    onChange={(event) => toggle(option.value, event.target.checked)}
                  />
                  <span {...stylex.props(selectStyles.optionContent)}>
                    <span>{option.label}</span>
                    {option.description ? (
                      <small {...stylex.props(selectStyles.optionDescription)}>
                        {option.description}
                      </small>
                    ) : null}
                  </span>
                </label>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}

/**
 * Port of classic `AccessKeyScopeEditor.vue` — three fieldsets (Groups,
 * Protocols, Models) each with an All/Specified `SegmentedControl`; a
 * restricted dimension expands to an always-open multi-select, and models
 * additionally allow free-form entry.
 *
 * Emit mapping: `setScopeMode` → `onSetScopeMode`, `update:groups` →
 * `onGroupsChange`, `update:protocols` → `onProtocolsChange`,
 * `update:models` → `onModelsChange`, `update:modelInput` →
 * `onModelInputChange`, `addModel` → `onAddModel`.
 */
export function AccessKeyScopeEditor({
  modes,
  filters,
  groupOptions,
  groupCatalogState,
  protocolOptions,
  modelOptions,
  modelInput,
  disabled,
  modelMismatch,
  onSetScopeMode,
  onGroupsChange,
  onProtocolsChange,
  onModelsChange,
  onModelInputChange,
  onAddModel,
}: {
  modes: AccessKeyScopeModes
  filters: AccessKeyFiltersDto
  groupOptions: AccessKeyScopeOption[]
  groupCatalogState: GroupCatalogState
  protocolOptions: readonly AccessProtocol[]
  modelOptions: AccessKeyScopeOption[]
  modelInput: string
  disabled: boolean
  modelMismatch: boolean
  onSetScopeMode: (dimension: AccessKeyScopeDimension, mode: AccessKeyScopeMode) => void
  onGroupsChange: (groupIDs: number[]) => void
  onProtocolsChange: (protocols: AccessProtocol[]) => void
  onModelsChange: (models: string[]) => void
  onModelInputChange: (value: string) => void
  onAddModel: () => void
}) {
  const t = useT()
  const protocolMultiSelectOptions: AccessKeyScopeOption[] = protocolOptions.map((protocol) => ({
    value: protocol,
    label: protocol,
  }))

  const catalogUnavailable = (): boolean =>
    groupCatalogState === 'loading' || groupCatalogState === 'error'

  const optionDisabled = (dimension: 'protocols' | 'models'): boolean =>
    disabled || modes[dimension] !== 'restricted' || catalogUnavailable()

  const modeDisabled = (dimension: AccessKeyScopeDimension): boolean => {
    if (disabled) return true
    return dimension === 'groups' ? groupCatalogState !== 'ready' : catalogUnavailable()
  }

  const scopeModeControl = (dimension: AccessKeyScopeDimension, label: string) => (
    <SegmentedControl
      value={modes[dimension]}
      label={label}
      size="sm"
      isDisabled={modeDisabled(dimension)}
      onChange={(value) => {
        if (value === 'all' || value === 'restricted') onSetScopeMode(dimension, value)
      }}
    >
      <SegmentedControlItem value="all" label={t('accessKeys.drawer.scopeAll')} />
      <SegmentedControlItem value="restricted" label={t('accessKeys.drawer.scopeRestricted')} />
    </SegmentedControl>
  )

  const updateGroups = (values: AccessKeyScopeValue[]): void => {
    onGroupsChange(values.filter((entry): entry is number => typeof entry === 'number'))
  }

  const updateModels = (values: AccessKeyScopeValue[]): void => {
    onModelsChange(values.filter((entry): entry is string => typeof entry === 'string'))
  }

  const updateProtocols = (values: AccessKeyScopeValue[]): void => {
    const allowed = new Set<AccessProtocol>(protocolOptions)
    onProtocolsChange(
      values.filter(
        (entry): entry is AccessProtocol =>
          typeof entry === 'string' && allowed.has(entry as AccessProtocol),
      ),
    )
  }

  return (
    <>
      <fieldset {...stylex.props(styles.scopeEditor)}>
        <legend {...stylex.props(styles.srOnly)}>{t('accessKeys.drawer.groups')}</legend>
        <div {...stylex.props(styles.scopeHead)}>
          <div>
            <strong {...stylex.props(styles.scopeHeadTitle)}>
              {t('accessKeys.drawer.groups')}
            </strong>
            <small {...stylex.props(styles.scopeHeadDescription)}>
              {t('accessKeys.drawer.groupsDescription')}
            </small>
          </div>
          {scopeModeControl('groups', t('accessKeys.drawer.groups'))}
        </div>
        <div {...stylex.props(styles.scopeBody)}>
          {modes.groups === 'all' ? (
            <div {...stylex.props(styles.permissionNote)}>
              <i aria-hidden="true" {...stylex.props(styles.permissionNoteDot)} />
              {t('accessKeys.drawer.allGroupsAllowed')}
            </div>
          ) : (
            <ScopeMultiSelect
              id="access-key-groups"
              label={t('accessKeys.drawer.groupSelectorLabel')}
              searchLabel={t('accessKeys.drawer.searchGroups')}
              searchPlaceholder={t('accessKeys.drawer.searchGroupsPlaceholder')}
              emptyLabel={t('accessKeys.drawer.noGroupOptions')}
              loadingLabel={t('accessKeys.drawer.groupOptionsLoading')}
              options={groupOptions}
              value={filters.groups}
              disabled={disabled || catalogUnavailable()}
              loading={groupCatalogState === 'loading'}
              autoFocusSearch
              onChange={updateGroups}
            />
          )}
        </div>
      </fieldset>

      <fieldset {...stylex.props(styles.scopeEditor, styles.scopeEditorSibling)}>
        <legend {...stylex.props(styles.srOnly)}>{t('accessKeys.drawer.protocols')}</legend>
        <div {...stylex.props(styles.scopeHead)}>
          <div>
            <strong {...stylex.props(styles.scopeHeadTitle)}>
              {t('accessKeys.drawer.protocols')}
            </strong>
            <small {...stylex.props(styles.scopeHeadDescription)}>
              {t('accessKeys.drawer.protocolsDescription')}
            </small>
          </div>
          {scopeModeControl('protocols', t('accessKeys.drawer.protocols'))}
        </div>
        <div {...stylex.props(styles.scopeBody)}>
          {modes.protocols === 'all' ? (
            <div {...stylex.props(styles.permissionNote)}>
              <i aria-hidden="true" {...stylex.props(styles.permissionNoteDot)} />
              {t('accessKeys.drawer.allProtocolsAllowed')}
            </div>
          ) : (
            <ScopeMultiSelect
              id="access-key-protocols"
              label={t('accessKeys.drawer.protocolSelectorLabel')}
              searchLabel={t('accessKeys.drawer.protocols')}
              emptyLabel={t('accessKeys.drawer.noProtocolOptions')}
              loadingLabel={t('accessKeys.drawer.groupOptionsLoading')}
              options={protocolMultiSelectOptions}
              value={filters.protocols}
              disabled={optionDisabled('protocols')}
              searchable={false}
              onChange={updateProtocols}
            />
          )}
        </div>
      </fieldset>

      <fieldset {...stylex.props(styles.scopeEditor, styles.scopeEditorSibling)}>
        <legend {...stylex.props(styles.srOnly)}>{t('accessKeys.drawer.models')}</legend>
        <div {...stylex.props(styles.scopeHead)}>
          <div>
            <strong {...stylex.props(styles.scopeHeadTitle)}>
              {t('accessKeys.drawer.models')}
            </strong>
            <small {...stylex.props(styles.scopeHeadDescription)}>
              {t('accessKeys.drawer.modelsScopeDescription')}
            </small>
          </div>
          {scopeModeControl('models', t('accessKeys.drawer.models'))}
        </div>
        <div {...stylex.props(styles.scopeBody)}>
          {modes.models === 'all' ? (
            <div {...stylex.props(styles.permissionNote)}>
              <i aria-hidden="true" {...stylex.props(styles.permissionNoteDot)} />
              {t('accessKeys.drawer.allModelsAllowed')}
            </div>
          ) : (
            <>
              <ScopeMultiSelect
                id="access-key-models"
                label={t('accessKeys.drawer.modelSelectorLabel')}
                searchLabel={t('accessKeys.drawer.searchModels')}
                searchPlaceholder={t('accessKeys.drawer.searchModelsPlaceholder')}
                emptyLabel={t('accessKeys.drawer.noModelOptions')}
                loadingLabel={t('accessKeys.drawer.groupOptionsLoading')}
                options={modelOptions}
                value={filters.models}
                disabled={optionDisabled('models')}
                loading={groupCatalogState === 'loading'}
                autoFocusSearch
                onChange={updateModels}
              />
              <div {...stylex.props(styles.modelEntry)}>
                <input
                  {...stylex.props(styles.modelEntryInput)}
                  value={modelInput}
                  type="text"
                  autoComplete="off"
                  placeholder={t('accessKeys.drawer.modelPlaceholder')}
                  disabled={optionDisabled('models')}
                  onChange={(event) => onModelInputChange(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter') {
                      event.preventDefault()
                      onAddModel()
                    }
                  }}
                />
                <Button
                  variant="secondary"
                  size="sm"
                  isDisabled={optionDisabled('models') || modelInput.trim() === ''}
                  icon={<Plus size={14} aria-hidden="true" />}
                  label={t('accessKeys.drawer.addModel')}
                  onClick={onAddModel}
                />
              </div>
              {modelMismatch && (
                <p {...stylex.props(styles.modelRisk)} role="status">
                  {t('accessKeys.drawer.modelRouteRisk')}
                </p>
              )}
            </>
          )}
        </div>
      </fieldset>
    </>
  )
}
