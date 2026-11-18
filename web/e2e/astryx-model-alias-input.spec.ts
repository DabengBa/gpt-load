import { expect, test } from '@playwright/test'
import { mockGroupFingerprint } from './fixtures/group-fingerprint-mock'
import { mockGroupModels } from './fixtures/group-models-mock'

test('existing model ID can be edited and saved without losing its configuration', async ({
  page,
}) => {
  const { puts } = await mockGroupModels(page, {
    models: [{ id: 'model-old', alias: 'public', alias_enabled: true, weight: 3, priority: 2 }],
  })
  await page.goto('/groups/1?tab=models')
  const row = page.getByTestId('model-alias-editor__record').first()
  const id = row.getByLabel('Model ID', { exact: true })
  const save = page.getByRole('button', { name: 'Save settings', exact: true })
  await expect(id).toHaveValue('model-old')
  await id.fill('')
  await expect(save).toBeDisabled()
  await id.fill('model-new')
  await expect(save).toBeEnabled()
  await save.click()
  await expect.poll(() => puts.length).toBe(1)
  expect(puts[0]).toMatchObject({
    models: [{ id: 'model-new', alias: 'public', entry_id: 'e000000000000' }],
  })
  await expect(id).toHaveValue('model-new')
  await page.reload()
  await expect(id).toHaveValue('model-new')
  await expect(row.getByLabel('Client alias for model-new')).toHaveValue('public')
  await expect(row.getByTestId('group-models__test-alias')).toHaveText('tst000')
  await expect(row.getByLabel('Weight', { exact: true })).toHaveValue('3')
  await expect(row.getByLabel('Priority', { exact: true })).toHaveValue('2')
})

test('model alias is an always available text field, with no checkbox', async ({ page }) => {
  await mockGroupFingerprint(page)
  await page.goto('/groups/1?tab=models')
  const row = page.getByTestId('model-alias-editor__record').first()
  const alias = row.getByLabel('Client alias for gpt-5.4')
  await expect(alias).toBeVisible()
  await expect(alias).toHaveValue('')
  await expect(row.getByLabel('Enable alias for gpt-5.4')).toHaveCount(0)
  await expect(row.getByLabel('Enable model gpt-5.4')).toBeChecked()
  await alias.fill('my-alias')
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeEnabled()
  await alias.fill('')
  await expect(page.getByRole('button', { name: 'Save settings', exact: true })).toBeDisabled()
})

test('existing aliases survive saves, discard restores them, and clearing disables them', async ({
  page,
}) => {
  const { puts } = await mockGroupModels(page, {
    models: [{ id: 'model-a', alias: 'historic', alias_enabled: true }, { id: 'model-b' }],
  })
  await page.goto('/groups/1?tab=models')
  const alias = page.getByLabel('Client alias for model-a')
  const otherAlias = page.getByLabel('Client alias for model-b')
  const save = page.getByRole('button', { name: 'Save settings', exact: true })
  await expect(alias).toHaveValue('historic')
  await otherAlias.fill('  public-name  ')
  await save.click()
  await expect.poll(() => puts.length).toBe(1)
  expect(puts[0]).toMatchObject({
    models: [
      { id: 'model-a', alias: 'historic', alias_enabled: true },
      { id: 'model-b', alias: 'public-name', alias_enabled: true },
    ],
  })
  await expect(otherAlias).toHaveValue('public-name')
  await alias.fill('')
  await page.getByRole('button', { name: 'Discard', exact: true }).click()
  await expect(alias).toHaveValue('historic')
  await alias.fill('   ')
  await save.click()
  await expect.poll(() => puts.length).toBe(2)
  expect(puts[1]).toMatchObject({
    models: [
      { id: 'model-a', alias: '', alias_enabled: false },
      { id: 'model-b', alias: 'public-name', alias_enabled: true },
    ],
  })
  await expect(alias).toHaveValue('')
  await page.reload()
  await expect(alias).toHaveValue('')
  await expect(otherAlias).toHaveValue('public-name')
})

test('a newly added model accepts an alias without a toggle', async ({ page }) => {
  const { puts } = await mockGroupModels(page, { models: [] })
  await page.goto('/groups/1?tab=models')
  await page.getByRole('button', { name: 'Add a model', exact: true }).click()
  const testAlias = page
    .getByTestId('model-alias-editor__record')
    .first()
    .getByTestId('group-models__test-alias')
  await expect(testAlias).toHaveText('Assigned after save')
  await page.getByLabel('Model ID', { exact: true }).fill('new-model')
  await page.getByLabel('Client alias for new-model').fill('new-public')
  await page.getByRole('button', { name: 'Save settings', exact: true }).click()
  await expect.poll(() => puts.length).toBe(1)
  expect(puts[0]).toMatchObject({
    models: [{ id: 'new-model', alias: 'new-public', alias_enabled: true }],
  })
  await expect(page.getByLabel('Client alias for new-model')).toHaveValue('new-public')
  await expect(testAlias).toHaveText('tst000')
})
