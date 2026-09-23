// Headless swamp runs: does the fly learn to avoid sundews while still finding fruit?
// usage: node scripts/experiment.mjs [minutes=20] [seeds=4] [--arms]
import fs from 'node:fs';
import { World, DT } from '../web/src/sim/world.js';

const args = process.argv.slice(2);
const minutes = +(args.find((a) => /^\d+$/.test(a)) ?? 20);
const seeds = +(args.filter((a) => /^\d+$/.test(a))[1] ?? 4);
const armsRace = args.includes('--arms');
const circuit = JSON.parse(fs.readFileSync(new URL('../web/public/data/circuit.json', import.meta.url)));

const CONDITIONS = {
  intact: {},
  'no dopamine': { dopamine: false },
  'no APL': { apl: false },
  'shuffled PN→KC': { shufflePnKc: true },
};
const WINDOWS = [[0, 2], [2, 5], [5, 10], [10, 15], [15, 20], [20, 30]].filter(([, b]) => b <= minutes);

function run(brain, seed) {
  const w = new World(circuit, { seed, brain, armsRace });
  const steps = Math.round((minutes * 60) / DT);
  for (let i = 0; i < steps; i++) w.step(DT);
  return w;
}

function rates(w) {
  const tl = [{ t: 0, eaten: 0, trapped: 0, strikes: 0 }, ...w.timeline];
  const at = (t) => tl.reduce((best, x) => (Math.abs(x.t - t) < Math.abs(best.t - t) ? x : best));
  return WINDOWS.map(([m0, m1]) => {
    const a = at(m0 * 60), b = at(m1 * 60);
    return { trapped: (b.trapped - a.trapped) / (m1 - m0), eaten: (b.eaten - a.eaten) / (m1 - m0) };
  });
}

const rows = [];
for (const [name, brain] of Object.entries(CONDITIONS)) {
  const acc = [];
  const mem = [];
  const t0 = Date.now();
  for (let s = 1; s <= seeds; s++) {
    const w = run(brain, s);
    acc.push(rates(w));
    mem.push(w.memory);
  }
  const nW = acc[0].length;
  const mean = (f) => Array.from({ length: nW }, (_, i) => acc.reduce((s, r) => s + f(r[i]), 0) / acc.length);
  const trap = mean((r) => r.trapped), eat = mean((r) => r.eaten);
  const avg = (k) => mem.reduce((s, m) => s + m[k].drive, 0) / mem.length;
  rows.push({ name, trap, eat, fruit: avg('fruit'), sundew: avg('sundew'), frog: avg('frog'), sec: (Date.now() - t0) / 1000 });
}

const f2 = (x) => x.toFixed(2);
console.log(`# ${minutes} min × ${seeds} seeds per condition${armsRace ? ', arms race on' : ''}; rates per minute in windows ${WINDOWS.map(([a, b]) => `${a}–${b}`).join(', ')} min\n`);
console.log('| condition | sundew traps/min | fruit meals/min | final drive: fruit / sundew / frog |');
console.log('|---|---|---|---|');
for (const r of rows) {
  console.log(`| ${r.name} | ${r.trap.map(f2).join(' → ')} | ${r.eat.map(f2).join(' → ')} | ${f2(r.fruit)} / ${f2(r.sundew)} / ${f2(r.frog)} |`);
}
console.error(rows.map((r) => `${r.name}: ${r.sec.toFixed(1)}s`).join(', '));
