import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { QueryClient, QueryObserver } from '@tanstack/query-core'

import { applyInvalidationPlan, mutationInvalidationPlans } from '@shared/control/invalidation'
import { controlQueryKeys as k } from '@shared/control/query-keys'

const source = (path: string) =>
  readFile(fileURLToPath(new URL(`../src/frontends/astryx/${path}`, import.meta.url)), 'utf8')

async function flush(): Promise<void> {
  for (let index = 0; index < 4; index++) await new Promise((resolve) => setImmediate(resolve))
}

test('credential invalidation refetches the group settings read and keeps it readable meanwhile', async () => {
  const client = new QueryClient()
  const key = k.groups.settings(7)
  const saved = { site_access: { snapshot: { status: 'ok', group_ratio: 0.5 } } }
  const refreshed = { site_access: { snapshot: { status: 'pending', group_ratio: null } } }
  let calls = 0
  let settle: (() => void) | undefined
  try {
    client.setQueryData(key, saved)
    const observer = new QueryObserver(client, {
      queryKey: key,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      queryFn: () => {
        calls++
        return new Promise((resolve) => {
          settle = () => resolve(refreshed)
        })
      },
    })
    const unsubscribe = observer.subscribe(() => {})

    // The credential plan must invalidate the current group settings read, otherwise
    // the detail page keeps the old key's ratio until the 60 minute sync.
    const invalidated = applyInvalidationPlan(
      client,
      mutationInvalidationPlans.group.importCredentials(7),
    )
    await flush()
    assert.equal(calls, 1)
    // Cached data stays readable while the refetch is in flight, so the settings
    // tab keeps its draft source and never falls back to an empty/loading state.
    assert.deepEqual(observer.getCurrentResult().data, saved)

    settle?.()
    await invalidated
    assert.deepEqual(client.getQueryData(key), refreshed)
    unsubscribe()
  } finally {
    client.clear()
  }
})

test('updating a credential invalidates that group settings read', async () => {
  const editor = await source('features/groups/credentials/GroupApiKeyEditor.tsx')

  assert.match(editor, /controlQueryKeys/)
  assert.match(
    editor,
    /invalidateQueries\(\{\s*queryKey:\s*controlQueryKeys\.groups\.settings\(groupId\)/s,
  )
  assert.match(editor, /exact:\s*true/)
  assert.match(editor, /invalidateGroupSettingsDependents/)
  // The settings refetch must be part of the reconcile step, after the credential
  // cache write and before the pending flag is cleared, so the snapshot follows the key.
  const cacheWrite = editor.indexOf('cacheCredentialItem(queryClient, groupId, result)')
  const settingsRefresh = editor.indexOf('controlQueryKeys.groups.settings(groupId)')
  const pendingReset = editor.lastIndexOf('setPending(false)')
  assert.ok(
    cacheWrite >= 0 && settingsRefresh > cacheWrite,
    'settings refresh must follow the cache write',
  )
  assert.ok(
    pendingReset > settingsRefresh,
    'settings refresh must run while the update is still pending',
  )
})

test('an unsaved settings draft survives the background refetch', async () => {
  const tab = await source('features/groups/settings/GroupSettingsTab.tsx')

  // Hydration is skipped while the form is dirty or a mutation is in flight, so an
  // invalidated settings query cannot overwrite unsaved edits with the new snapshot.
  assert.match(
    tab,
    /if \(latest === savedRef\.current \|\| isDirty \|\| isPending \|\| isDeleted\) return/,
  )
  assert.match(
    tab,
    /!latest \|\|\s+latest === savedRef\.current \|\|\s+stateRef\.current\.dirty \|\|\s+stateRef\.current\.mutationPending \|\|\s+stateRef\.current\.deleted/s,
  )
  assert.match(tab, /wasDirtyRef\.current/)
})
