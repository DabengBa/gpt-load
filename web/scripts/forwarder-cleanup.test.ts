import test from 'node:test'
import assert from 'node:assert/strict'
import { access, readFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const root = new URL('../', import.meta.url)
const file = (path: string) => fileURLToPath(new URL(path, root))

async function exists(path: string): Promise<boolean> {
  try {
    await access(file(path))
    return true
  } catch {
    return false
  }
}

test('unused announcement forwarders stay removed', async () => {
  assert.equal(await exists('src/shared/control/client.ts'), false)
  assert.equal(await exists('src/shared/control/resources/index.ts'), false)

  // The announcement resource is imported directly; a re-export barrel would be
  // dead weight and lint cannot catch an unused module.
  const view = await readFile(
    file('src/frontends/astryx/features/announcements/AnnouncementsView.tsx'),
    'utf8',
  )
  assert.match(view, /from '@shared\/control\/resources\/announcements'/)
  assert.doesNotMatch(view, /@shared\/control\/client|@shared\/control\/resources'/)
})

for (const [name, path, removed] of [
  [
    'site access section',
    'src/frontends/astryx/features/groups/settings/GroupSettingsSiteAccessSection.tsx',
    'input',
  ],
  [
    'import connection section',
    'src/frontends/astryx/features/import/ImportConnectionSection.tsx',
    'siteToken',
  ],
] as const) {
  test(`${name} keeps only styles the component applies`, async () => {
    const source = await readFile(file(path), 'utf8')
    assert.doesNotMatch(
      source,
      new RegExp(`^\\s{2}${removed}: \\{`, 'm'),
      `${removed} style left over`,
    )
    assert.doesNotMatch(source, new RegExp(`styles\\.${removed}\\b`))
  })
}
