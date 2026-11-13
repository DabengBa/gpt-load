import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';
import { createRequire } from 'node:module';
import { mkdir } from 'node:fs/promises';

const require = createRequire(new URL('../../web/package.json', import.meta.url));
const { chromium } = require('@playwright/test');
const url = new URL('./management-demo.html', import.meta.url).href;
// Browser integration lane: file URL, synthetic in-memory data, no server.
// Each test owns its page; browser closes after the suite (~10 seconds).
// Screenshots are replaceable artifacts in ignored web/test-results.
let browser;
before(async () => { browser = await chromium.launch({ headless: true }); });
after(async () => { await browser?.close(); });
async function withPage(run, viewport = { width: 1440, height: 900 }) {
  const page = await browser.newPage({ viewport, reducedMotion: 'reduce' });
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => { if (!request.url().startsWith('file:')) requests.push(request.url()); });
  try {
    await page.goto(url);
    await run(page);
    assert.deepEqual(errors, [], 'No page errors');
    assert.deepEqual(requests, [], 'No network or external resource requests');
  } finally { await page.close(); }
}

test('search filters groups by name and channel, handles no results and reset', async () => {
  await withPage(async page => {
    const rows = page.locator('[data-group]');
    assert.equal(await rows.count(), 5);
    const search = page.getByRole('searchbox', { name: '搜索分组' });
    await search.fill('  OPENAI  ');
    assert.equal(await rows.filter({ visible: true }).count(), 1);
    assert.match(await page.locator('#result-count').innerText(), /1/);
    await search.fill('不存在的分组');
    assert.equal(await rows.filter({ visible: true }).count(), 0);
    assert.equal(await page.getByText('没有找到匹配的分组').isVisible(), true);
    await page.getByRole('button', { name: '清除搜索' }).click();
    assert.equal(await search.inputValue(), '');
    assert.equal(await rows.filter({ visible: true }).count(), 5);
    assert.equal(await search.evaluate(el => document.activeElement === el), true);
  });
});

test('search field boundary contrasts with its surface', async () => {
  await withPage(async page => {
    const ratio = await page.getByRole('searchbox', { name: '搜索分组' }).evaluate(el => {
      const luminance = color => {
        const channels = color.match(/\d+/g).slice(0, 3).map(Number).map(value => {
          value /= 255;
          return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
      };
      const style = getComputedStyle(el);
      const a = luminance(style.borderTopColor), b = luminance(style.backgroundColor);
      return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    });
    assert.ok(ratio >= 3, `Search boundary contrast ${ratio.toFixed(2)} must be at least 3:1`);
  });
});

test('details show selected group, close with Escape and restore focus', async () => {
  await withPage(async page => {
    const trigger = page.getByRole('button', { name: '查看 OpenAI 主分组详情' });
    await trigger.click();
    const dialog = page.getByRole('dialog', { name: 'OpenAI 主分组' });
    assert.equal(await dialog.isVisible(), true);
    assert.match(await dialog.innerText(), /OpenAI/);
    assert.equal(await dialog.evaluate(el => el.contains(document.activeElement)), true);
    await page.keyboard.press('Escape');
    assert.equal(await dialog.isVisible(), false);
    assert.equal(await trigger.evaluate(el => document.activeElement === el), true);
    await page.getByRole('button', { name: '查看 Anthropic 对话详情' }).click();
    assert.equal(await page.getByRole('dialog', { name: 'Anthropic 对话' }).isVisible(), true);
    await page.getByRole('button', { name: '关闭详情' }).click();
  });
});

test('navigation represents administrator routes without pretending they work', async () => {
  await withPage(async page => {
    const nav = page.getByRole('navigation', { name: '主导航' });
    for (const name of ['首页', '分组', '公告', '调度', '监控', '日志', '设置']) {
      assert.equal(await nav.getByText(name, { exact: true }).count(), 1);
    }
    assert.equal(await nav.locator('[aria-current=page]').innerText(), '分组');
    assert.match(await page.locator('body').innerText(), /模拟数据/);
    assert.match(await page.locator('body').innerText(), /管理员/);
    assert.equal(await nav.locator('a[href]').count(), 0, 'No dead or network navigation');
  });
});

test('responsive layout and mobile navigation at 320–1440px; capture screenshots', async () => {
  await withPage(async page => {
    const output = new URL('../../web/test-results/management-demo/', import.meta.url);
    await mkdir(output, { recursive: true });
    for (const width of [1440, 1100, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `No page overflow at ${width}px`);
      assert.equal(await page.getByRole('button', { name: '查看 OpenAI 主分组详情' }).isVisible(), true);
      if ([1440, 390, 320].includes(width)) await page.screenshot({ path: new URL(`${width}.png`, output).pathname, fullPage: true });
    }
    const toggle = page.getByRole('button', { name: '打开导航' });
    await toggle.click();
    assert.equal(await toggle.getAttribute('aria-expanded'), 'true');
    assert.equal(await page.getByRole('navigation', { name: '主导航' }).isVisible(), true);
    await page.keyboard.press('Escape');
    assert.equal(await toggle.getAttribute('aria-expanded'), 'false');
    assert.equal(await toggle.evaluate(el => document.activeElement === el), true);
  });
});
