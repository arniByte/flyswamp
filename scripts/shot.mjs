// Headless screenshot of a page served by `vite` (used to check visuals without a display).
// usage: node scripts/shot.mjs <url> <out.png> [waitMs] [width] [height]
import { chromium } from '../web/node_modules/playwright/index.mjs';

const [url, out, wait = '4000', w = '1280', h = '800'] = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || undefined, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +w, height: +h } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`${m.type()}: ${m.text()}`); });
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
try {
  await page.goto(url, { waitUntil: 'load' });
  await page.waitForTimeout(+wait);
  await page.screenshot({ path: out, timeout: 180000 });
  const info = await page.evaluate(() => window.__shotInfo ?? null);
  console.log(JSON.stringify({ out, errors: errors.slice(0, 20), info }));
} catch (e) {
  console.log(JSON.stringify({ out, failed: e.message.split('\n')[0], errors: errors.slice(0, 20) }));
} finally {
  await browser.close();
}
