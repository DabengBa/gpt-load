const { test, expect } = require('playwright/test');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

for (const file of ['dir-1.html', 'dir-2.html', 'dir-3.html']) {
  test(`${file}: editing any configuration invalidates the result immediately`, async ({ page }) => {
    await page.goto(pathToFileURL(path.join(__dirname, file)).href);
    await page.getByLabel('模型名称').fill('example-model');
    for (const edit of [
      () => page.getByLabel('Base URL（演示）').fill('https://changed.invalid/v1'),
      () => page.getByLabel('模型名称').fill('changed-model'),
      () => page.getByLabel('连接类型').selectOption('azure'),
    ]) {
      await page.getByRole('button', { name: '保存演示配置' }).click();
      await page.getByRole('button', { name: '运行模拟测试' }).click();
      await expect(page.getByRole('status')).toContainText('模拟通过');
      await edit();
      await expect(page.locator('#save-note')).toContainText('未保存', { timeout: 1000 });
      await expect(page.getByRole('status')).toContainText('已过期', { timeout: 1000 });
      await expect(page.getByRole('status')).not.toContainText('模拟通过');
      await page.getByRole('button', { name: '运行模拟测试' }).click();
      await expect(page.getByRole('status')).toContainText('重新保存');
    }
  });

  test(`${file}: long model results wrap and current probe scope is accurate`, async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 });
    await page.goto(pathToFileURL(path.join(__dirname, file)).href);
    await expect(page.locator('.scope')).toContainText('现有');
    await expect(page.locator('.scope')).toContainText('OpenAI Compatible');
    await page.getByLabel('模型名称').fill('longmodel'.repeat(60));
    await page.getByRole('button', { name: '保存演示配置' }).click();
    await page.getByRole('button', { name: '运行模拟测试' }).click();
    const metrics = await page.locator('#result').evaluate(el => ({ scroll: el.scrollWidth, client: el.clientWidth }));
    expect(metrics.scroll).toBeLessThanOrEqual(metrics.client);
    const paragraph = await page.locator('#result p').evaluate(el => ({ scroll: el.scrollWidth, client: el.clientWidth }));
    expect(paragraph.scroll).toBeLessThanOrEqual(paragraph.client);
  });
}

test('direction B: saved configuration provides keyboard-accessible route to test', async ({ page }) => {
  await page.goto(pathToFileURL(path.join(__dirname, 'dir-2.html')).href);
  await page.getByLabel('模型名称').fill('example-model');
  await page.getByRole('button', { name: '保存演示配置' }).click();
  await page.getByRole('button', { name: '前往模拟测试' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByLabel('模拟结果')).toBeFocused();
});

for (const file of ['dir-1.html', 'dir-2.html', 'dir-3.html']) {
  test(`${file}: offline simulation and cautious result`, async ({ page }) => {
    const outbound = [];
    page.on('request', req => { if (/^https?:/.test(req.url())) outbound.push(req.url()); });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(pathToFileURL(path.join(__dirname, file)).href);
    await expect(page.getByLabel('演示 Key')).toHaveAttribute('readonly', '');
    await page.getByRole('button', { name: '运行模拟测试' }).click();
    await expect(page.getByRole('status')).toContainText('请先填写模型');
    await page.getByRole('button', { name: '保存演示配置' }).click();
    await expect(page.getByRole('status')).toContainText('请填写演示 Base URL 和模型名称');
    await page.getByLabel('连接类型').selectOption('azure');
    await expect(page.getByText('Azure 专用配置入口')).toBeVisible();
    await page.getByLabel('连接类型').selectOption('bedrock');
    await expect(page.getByText('Bedrock 专用配置入口')).toBeVisible();
    await page.getByLabel('连接类型').selectOption('vertex');
    await expect(page.getByText('Vertex 专用配置入口')).toBeVisible();
    await page.getByLabel('连接类型').selectOption('multi');
    await expect(page.getByText('其他 / 多协议入口')).toBeVisible();
    await page.getByLabel('连接类型').selectOption('compatible');
    await page.getByLabel('模型名称').fill('example-model');
    await page.getByRole('button', { name: '保存演示配置' }).click();
    await page.getByLabel('模拟结果').selectOption('failed');
    await page.getByRole('button', { name: '运行模拟测试' }).click();
    await expect(page.getByRole('status')).toContainText('不等于协议不兼容');
    await page.getByLabel('模拟结果').selectOption('passed');
    await page.getByRole('button', { name: '运行模拟测试' }).click();
    await expect(page.getByRole('status')).toContainText('不证明');
    await page.getByLabel('模拟结果').selectOption('unknown');
    await page.getByRole('button', { name: '运行模拟测试' }).click();
    await expect(page.getByRole('status')).toContainText('无法判断');
    await page.getByLabel('模型名称').fill('changed-model');
    await page.getByRole('button', { name: '运行模拟测试' }).click();
    await expect(page.getByRole('status')).toContainText('重新保存');
    await page.reload();
    await expect(page.getByLabel('模型名称')).toHaveValue('');
    await expect(page.getByRole('status')).toContainText('尚未测试');
    expect(outbound).toEqual([]);
    expect(errors).toEqual([]);
  });
}
