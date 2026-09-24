// P2 calibration of w_syn and spike-frequency adaptation on MaleCNS, as planned in docs/ROADMAP.md.
// Every parameter set runs p2_bench.mjs in a child process; the criteria are fixed relative to the FlyWire
// reference run (results/p2_bench_flywire630.json) and written down before the grid ran.
//   grid [b,b,...]          calibration seeds: MN9 benchmarks on 1-5, offset on 1-20; a set that fails
//                           sugar -> MN9 is not run further. Optional list restricts b (mV).
//   holdout <w> <b> <tau>   the chosen set once on held-out seeds: MN9 benchmarks on 101-110, offset on 101-150
//   control <w> <b> <tau>   the chosen set on three shuffled graphs (--shuffle 1..3), MN9 benchmarks on 101-105
// usage: node validation/p2_grid.mjs grid|holdout|control ... [--workers 3] [--graph NAME] [--w 0.2,0.21]
//        --graph picks .cache/graphs/NAME (default malecns_min1); results for other graphs get the name as suffix.
//        --w replaces the w_syn values of the grid.
import fs from 'node:fs';
import { spawn } from 'node:child_process';

const here = new URL('.', import.meta.url).pathname;
const args = process.argv.slice(2);
const opt = (name, fallback) => { const k = args.indexOf(name); return k >= 0 ? args.splice(k, 2)[1] : fallback; };
const WORKERS = Number(opt('--workers', 3));
const GRAPH_NAME = opt('--graph', 'malecns_min1');
const W = opt('--w', '0.18,0.2,0.22,0.25,0.275').split(',').map(Number), B = [0, 0.5, 1, 2, 4], TAU = [50, 200, 1000];
const GRAPH = `${here}../.cache/graphs/${GRAPH_NAME}`;
const SUFFIX = GRAPH_NAME === 'malecns_min1' ? '' : `_${GRAPH_NAME}`;
const OUT = `${here}results/p2_grid${SUFFIX}`;
const [mode, ...rest] = args;

const ref = JSON.parse(fs.readFileSync(`${here}results/p2_bench_flywire630.json`, 'utf8'));
const R = {
  sugar: ref.benchmarks.sugar.mn9_best.hz[0], responding: ref.benchmarks.sugar.responding[0],
  suppression: ref.bitter_suppression, water: ref.benchmarks.water.mn9_best.hz[0],
};
const CRITERIA = {
  sugar: (b) => within(b.sugar.mn9_best.hz[0], R.sugar) && b.sugar.responding[0] <= 10 * R.responding,
  bitter_veto: (b, res) => res.bitter_suppression != null && res.bitter_suppression >= 0.5 * R.suppression,
  bitter_alone: (b) => b.bitter.mn9_best.hz[0] <= 5,
  water: (b) => within(b.water.mn9_best.hz[0], R.water),
  offset: (b) => b.offset.quiet_seeds === b.offset.n_seeds,
};
function within(x, r) { return x >= 0.5 * r && x <= 2 * r; }

const label = ({ w, b, tau }) => `w${w}_b${b}${b ? `_t${tau}` : ''}`;
const params = ({ w, b, tau }) => ({ wSyn: w, ...(b ? { adaptB: b, tauAdapt: tau } : {}) });

// one p2_bench run in a child process; resolves to its parsed result
function bench(set, bench, seeds, name, extra = []) {
  const out = `${OUT}/${label(set)}_${name}.json`;
  if (fs.existsSync(out)) return Promise.resolve(JSON.parse(fs.readFileSync(out, 'utf8')));
  return new Promise((resolve, reject) => {
    const p = spawn(process.execPath, [`${here}p2_bench.mjs`, GRAPH, 'malecns', seeds, out, ...extra], {
      env: { ...process.env, LIF_PARAMS: JSON.stringify(params(set)), BENCH: bench }, stdio: ['ignore', 'pipe', 'inherit'],
    });
    p.on('exit', (code) => (code === 0 ? resolve(JSON.parse(fs.readFileSync(out, 'utf8'))) : reject(new Error(`${label(set)} ${name}: exit ${code}`))));
  });
}

async function evaluate(set, mn9Seeds, offsetSeeds, tag) {
  const sugar = await bench(set, 'sugar', mn9Seeds, `${tag}sugar`);
  const b = { sugar: sugar.benchmarks.sugar };
  const pass = { sugar: CRITERIA.sugar(b) };
  if (pass.sugar) {
    const rest = await bench(set, 'sugar,sugar_bitter,bitter,water', mn9Seeds, `${tag}mn9`);
    Object.assign(b, rest.benchmarks);
    const off = await bench(set, 'offset', offsetSeeds, `${tag}offset`);
    b.offset = off.benchmarks.offset;
    for (const k of ['bitter_veto', 'bitter_alone', 'water', 'offset']) pass[k] = CRITERIA[k](b, rest);
    b.bitter_suppression = rest.bitter_suppression;
  }
  const row = {
    ...set, pass, passes_all: Object.keys(CRITERIA).every((k) => pass[k]),
    sugar_hz: b.sugar.mn9_best.hz[0], responding: b.sugar.responding[0], suppression: b.bitter_suppression ?? null,
    bitter_hz: b.bitter?.mn9_best.hz[0] ?? null, water_hz: b.water?.mn9_best.hz[0] ?? null,
    quiet: b.offset ? `${b.offset.quiet_seeds}/${b.offset.n_seeds}` : null,
  };
  console.log(`${label(set).padEnd(16)} sugar ${row.sugar_hz.toFixed(1)} Hz (${row.responding.toFixed(0)} n)` +
    (pass.sugar ? `, veto ${(100 * row.suppression).toFixed(0)} %, bitter ${row.bitter_hz.toFixed(1)}, water ${row.water_hz.toFixed(1)}, quiet ${row.quiet}` : '') +
    ` → ${row.passes_all ? 'PASS' : `fail ${Object.keys(pass).filter((k) => !pass[k]).join(',')}`}`);
  return row;
}

async function pool(tasks) {
  const results = [];
  let next = 0;
  await Promise.all(Array.from({ length: WORKERS }, async () => {
    while (next < tasks.length) { const k = next++; results[k] = await tasks[k](); }
  }));
  return results;
}

fs.mkdirSync(OUT, { recursive: true });
if (mode === 'grid') {
  const bs = rest[0] ? rest[0].split(',').map(Number) : B;
  const sets = W.flatMap((w) => bs.flatMap((b) => (b ? TAU.map((tau) => ({ w, b, tau })) : [{ w, b: 0, tau: null }])));
  const rows = await pool(sets.map((s) => () => evaluate(s, '1-5', '1-20', '')));
  const file = `${here}results/p2_grid_summary${SUFFIX}.json`;
  const old = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')).rows : [];
  const all = [...old.filter((o) => !rows.some((r) => label(r) === label(o))), ...rows];
  // selection rule from the plan: smallest b among passing sets, then MN9 closest to the reference
  const passing = all.filter((r) => r.passes_all).sort((a, c) => a.b - c.b || Math.abs(Math.log(a.sugar_hz / R.sugar)) - Math.abs(Math.log(c.sugar_hz / R.sugar)));
  fs.writeFileSync(file, JSON.stringify({ graph: GRAPH_NAME, reference: R, criteria: 'docs/ROADMAP.md, P2 calibration plan', rows: all, chosen: passing[0] ?? null }, null, 1));
  console.log(passing[0] ? `chosen: ${label(passing[0])}` : 'no set passes');
} else if (mode === 'holdout') {
  const [w, b, tau] = rest.map(Number);
  const row = await evaluate({ w, b, tau: b ? tau : null }, '101-110', '101-150', 'holdout_');
  fs.writeFileSync(`${here}results/p2_holdout${SUFFIX}.json`, JSON.stringify({ reference: R, row }, null, 1));
} else if (mode === 'control') {
  const [w, b, tau] = rest.map(Number), set = { w, b, tau: b ? tau : null };
  const rows = await pool([1, 2, 3].map((k) => async () => {
    const r = await bench(set, 'sugar,sugar_bitter,bitter,water', '101-105', `shuffle${k}`, ['--shuffle', String(k)]);
    const b2 = r.benchmarks;
    return { shuffle: k, sugar_hz: b2.sugar.mn9_best.hz[0], water_hz: b2.water.mn9_best.hz[0], sugar_passes: CRITERIA.sugar(b2), water_passes: CRITERIA.water(b2) };
  }));
  const fails = rows.every((r) => !r.sugar_passes && !r.water_passes);
  fs.writeFileSync(`${here}results/p2_control${SUFFIX}.json`, JSON.stringify({ reference: R, set, rows, control_fails_as_required: fails }, null, 1));
  console.log(rows.map((r) => `shuffle ${r.shuffle}: sugar ${r.sugar_hz.toFixed(1)}, water ${r.water_hz.toFixed(1)} Hz`).join('\n'));
  console.log(`control ${fails ? 'fails sugar and water as required' : 'PASSES a wiring benchmark — gate fails'}`);
} else {
  console.error('usage: node validation/p2_grid.mjs grid [b,...] | holdout <w> <b> <tau> | control <w> <b> <tau> [--workers N]');
  process.exit(1);
}
