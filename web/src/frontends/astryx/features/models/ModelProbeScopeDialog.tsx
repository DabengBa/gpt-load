import {
  Button,
  Dialog,
  DialogHeader,
  Layout,
  LayoutContent,
  LayoutFooter,
} from '@astryxdesign/core'

import { useT } from '../../app/i18n'

// A batch that mixes enabled and disabled groups must ask before spending an
// upstream call on a group that is not serving traffic. Dismissing the dialog
// aborts the probe; the two explicit choices keep the scope unambiguous.
export function ModelProbeScopeDialog({
  open,
  total,
  disabledCount,
  onOpenChange,
  onProbeAll,
  onProbeEnabled,
}: {
  open: boolean
  total: number
  disabledCount: number
  onOpenChange(open: boolean): void
  onProbeAll(): void
  onProbeEnabled(): void
}) {
  const t = useT()
  const enabledCount = total - disabledCount

  return (
    <Dialog isOpen={open} onOpenChange={onOpenChange} width={440}>
      <Layout
        header={
          <DialogHeader
            title={t('monitor.modelProbe.disabledScope.title')}
            subtitle={t('monitor.modelProbe.disabledScope.description', {
              total,
              disabled: disabledCount,
            })}
            onOpenChange={onOpenChange}
            hasDivider
          />
        }
        content={<LayoutContent />}
        footer={
          <LayoutFooter hasDivider>
            <Button
              variant="secondary"
              size="sm"
              label={t('monitor.modelProbe.disabledScope.cancel')}
              onClick={() => onOpenChange(false)}
            />
            <Button
              variant="secondary"
              size="sm"
              isDisabled={enabledCount === 0}
              label={t('monitor.modelProbe.disabledScope.enabledOnly', {
                count: enabledCount,
              })}
              onClick={onProbeEnabled}
            />
            <Button
              variant="primary"
              size="sm"
              label={t('monitor.modelProbe.disabledScope.includeDisabled', {
                count: total,
              })}
              onClick={onProbeAll}
            />
          </LayoutFooter>
        }
      />
    </Dialog>
  )
}
