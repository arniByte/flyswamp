// The whole CNS inside the running game (headless Chromium): loads, runs in the Web Worker, and under the P2
// sugar stimulus (right labellar sugar GRNs, 150 Hz) drives MN9_L as in the gate. Also measures how fast the
// worker keeps up with the game clock at ×1 and ×8.
// usage: (cd web && npx vite preview --port 4173) & node validation/web_live_check.mjs [url] [out.json]
import fs from 'node:fs';
import { chromium } from '../web/node_modules/playwright/index.mjs';

const url = process.argv[2] ?? 'http://localhost:4173/', out = process.argv[3];
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/opt/pw-browsers/chromium', args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto(`${url}?mode=brain&lite&seed=3`, { waitUntil: 'load' });
const t0 = Date.now();
await page.waitForFunction(() => window.__flyswamp?.cns?.status === 'работает', null, { timeout: 180000, polling: 200 });
const loadS = (Date.now() - t0) / 1000;

// Counts spikes of MN9_L and all neurons over a window of simulated time, straight from the worker's ticks.
const measure = (speed, drive, seconds) => page.evaluate(async ([speed, drive, seconds]) => {
  const c = window.__flyswamp.cns;
  c._post({ type: 'speed', value: speed });
  c._post({ type: 'drive', entries: drive ? c.header.sets.sugar_R.map((i) => [i, 150]) : [] });
  await new Promise((r) => setTimeout(r, 1000)); // settle: onset transient and the speed change
  let mn9 = 0, all = 0;
  const sim0 = c.simMs, wall0 = performance.now();
  const on = (e) => { if (e.data.type !== 'tick') return; all += e.data.spikes.length; for (const i of e.data.spikes) if (i === c.mn9) mn9++; };
  c.worker.addEventListener('message', on);
  await new Promise((r) => setTimeout(r, seconds * 1000));
  c.worker.removeEventListener('message', on);
  const simS = (c.simMs - sim0) / 1000, wallS = (performance.now() - wall0) / 1000;
  return { speed, drive, sim_s: simS, wall_s: wallS, factor: simS / wallS, mn9_l_hz: mn9 / simS, spikes_per_sim_s: all / simS };
}, [speed, drive, seconds]);

const mode = await page.evaluate(() => window.__flyswamp.cns.mode);
const runs = [await measure(1, false, 3), await measure(1, true, 5), await measure(8, true, 5)];
const res = { url, mode, load_s: loadS, runs, page_errors: errors };
if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));
console.log(JSON.stringify(res, null, 1));
await browser.close();
process.exit(errors.length || mode !== 'worker' ? 1 : 0);
