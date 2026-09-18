<script setup lang="ts">
import { Check, ChevronDown } from '@lucide/vue'
import { computed, nextTick, ref, useAttrs } from 'vue'
import {
  ComboboxAnchor,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxItemIndicator,
  ComboboxPortal,
  ComboboxRoot,
} from 'reka-ui'

import OverflowTooltip from './OverflowTooltip.vue'

interface SelectOption {
  value: string
  label: string
}

// Reka Combobox reserves an empty string for clearing the selection. Keep the
// AppSelect contract for "any" options while using a safe internal value for
// the primitive component.
const emptyOptionValue = '__app_select_empty__'

defineOptions({ inheritAttrs: false })

const props = withDefaults(
  defineProps<{
    modelValue?: string
    label: string
    options: SelectOption[]
    searchPlaceholder: string
    emptyLabel: string
    disabled?: boolean
    variant?: 'default' | 'embedded'
    size?: 'sm' | 'md' | 'compact'
  }>(),
  {
    modelValue: undefined,
    disabled: false,
    variant: 'default',
    size: 'md',
  },
)
const emit = defineEmits<{ 'update:modelValue': [value: string] }>()
const attrs = useAttrs()
const open = ref(false)
const query = ref('')
const triggerElement = ref<HTMLElement>()
const normalizedOptions = computed(() =>
  props.options.map((option) => ({
    ...option,
    raw: option.value,
    value: option.value === '' ? emptyOptionValue : option.value,
  })),
)
const filteredOptions = computed(() => {
  const keyword = query.value.trim().toLocaleLowerCase()
  if (!keyword) return normalizedOptions.value
  return normalizedOptions.value.filter((option) =>
    option.label.toLocaleLowerCase().includes(keyword),
  )
})
const normalizedModelValue = computed(() =>
  props.modelValue === '' ? emptyOptionValue : props.modelValue,
)
const selectedLabel = computed(
  () => props.options.find((option) => option.value === props.modelValue)?.label ?? '',
)

function select(value: unknown): void {
  if (typeof value !== 'string') return
  emit('update:modelValue', value === emptyOptionValue ? '' : value)
}

// The popup input carries the filter, so the trigger stays a real button:
// reka's own ComboboxTrigger is intentionally skipped because it is hard-wired
// tabindex=-1 for the input-in-anchor pattern.
function onOpenChange(value: boolean): void {
  open.value = value
  if (value) {
    query.value = ''
    return
  }
  // Match the focus-return contract reka gives its own trigger: only reclaim
  // focus when the close left nothing focused.
  void nextTick(() => {
    if (!document.activeElement || document.activeElement === document.body) {
      triggerElement.value?.focus()
    }
  })
}
</script>

<template>
  <ComboboxRoot
    :model-value="normalizedModelValue"
    :open="open"
    :disabled="props.disabled"
    :ignore-filter="true"
    @update:model-value="select"
    @update:open="onOpenChange"
  >
    <ComboboxAnchor as-child>
      <button
        ref="triggerElement"
        v-bind="attrs"
        type="button"
        class="app-select__trigger"
        :class="[`app-select__trigger--${props.variant}`, `app-select__trigger--${props.size}`]"
        :aria-label="label"
        aria-haspopup="listbox"
        :aria-expanded="open"
        :disabled="props.disabled"
        @click="onOpenChange(!open)"
      >
        <OverflowTooltip
          as="span"
          class="app-select__value"
          :content="selectedLabel"
          :focusable="false"
        >
          {{ selectedLabel }}
        </OverflowTooltip>
        <ChevronDown class="app-select__chevron" :size="16" aria-hidden="true" />
      </button>
    </ComboboxAnchor>
    <ComboboxPortal>
      <ComboboxContent
        class="app-select__content searchable-select__content"
        position="popper"
        :side-offset="6"
      >
        <ComboboxInput
          class="searchable-select__search"
          :model-value="query"
          :placeholder="searchPlaceholder"
          :aria-label="searchPlaceholder"
          :display-value="() => ''"
          @update:model-value="query = String($event)"
          @keydown.enter.prevent
        />
        <ComboboxEmpty class="searchable-select__empty">{{ emptyLabel }}</ComboboxEmpty>
        <ComboboxItem
          v-for="(option, index) in filteredOptions"
          :key="`${option.value}:${index}`"
          class="app-select__item"
          :value="option.value"
          :text-value="option.label"
          :data-value="option.raw"
        >
          <ComboboxItemIndicator class="app-select__indicator">
            <Check :size="15" aria-hidden="true" />
          </ComboboxItemIndicator>
          <span>{{ option.label }}</span>
        </ComboboxItem>
      </ComboboxContent>
    </ComboboxPortal>
  </ComboboxRoot>
</template>

<style>
.app-select__content.searchable-select__content {
  display: flex;
  min-width: var(--reka-combobox-trigger-width);
  max-width: var(--reka-combobox-content-available-width);
  max-height: min(320px, var(--reka-combobox-content-available-height));
  flex-direction: column;
  padding: 0;
}
.searchable-select__search {
  position: sticky;
  top: 0;
  z-index: 1;
  display: block;
  min-height: 36px;
  flex: none;
  border: 0;
  border-bottom: 1px solid var(--color-border-subtle);
  border-radius: var(--radius-control) var(--radius-control) 0 0;
  background: var(--color-surface);
  color: var(--color-text);
  padding: 8px 10px;
  font: inherit;
  font-size: var(--text-sm);
  outline: none;
}
.searchable-select__search::placeholder {
  color: var(--color-text-faint);
}
.searchable-select__empty {
  padding: 10px 12px;
  color: var(--color-text-faint);
  font-size: var(--text-sm);
}
.searchable-select__content .app-select__item {
  margin: 0 var(--space-1);
}
.searchable-select__content .app-select__item:first-of-type {
  margin-top: var(--space-1);
}
.searchable-select__content .app-select__item:last-of-type {
  margin-bottom: var(--space-1);
}
</style>
