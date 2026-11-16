// Run with NODE_PATH pointing to an installed Playwright package; no server/network.
const { chromium } = require('playwright');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const browser = await chromium.launch({ headless: true });
  let problems = 0;
  for (const file of ['thumbs.html', 'dir-1.html', 'dir-2.html', 'dir-3.html']) {
    for (const [width, height] of [[1440, 900], [375, 812]]) {
      const page = await browser.newPage({ viewport: { width, height } });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('request', r => { if (/^https?:/.test(r.url())) errors.push(`network: ${r.url()}`); });
      await page.goto(pathToFileURL(path.join(__dirname, file)).href);
      await page.screenshot({ path: path.join(__dirname, `${file.slice(0, -5)}-${width}.png`), fullPage: true });
      if (width === 1440) await page.screenshot({ path: path.join(__dirname, `${file.slice(0, -5)}-first.png`) });
      const size = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, viewport: document.documentElement.clientWidth }));
      if (size.scroll > size.viewport) errors.push(`horizontal overflow ${size.scroll} > ${size.viewport}`);
      console.log(`${file} ${width}x${height}: ${errors.length ? errors.join('; ') : 'OK'}`);
      problems += errors.length;
      await page.close();
    }
  }
  await browser.close();
  if (problems) process.exitCode = 1;
})().catch(e => { console.error(e); process.exitCode = 1; });
