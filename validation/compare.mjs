// Statistical comparison of two per-trial spike count files (shiu_reference.py, shipped_results.py,
// run_lif.mjs). Equivalence here means: the two simulators are indistinguishable given their own
// trial-to-trial noise, not that spike trains match (the random streams differ).
// usage: node validation/compare.mjs <a.json> <b.json> [report.json]
import fs from 'node:fs';
import { mulberry32 } from '../web/src/sim/rng.js';

const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
const variance = (x) => { const m = mean(x); return x.reduce((a, b) => a + (b - m) ** 2, 0) / (x.length - 1); };

function pearson(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxy / Math.sqrt(sxx * syy);
}

function welch(a, b) {
  const se = Math.sqrt(variance(a) / a.length + variance(b) / b.length);
  const d = mean(a) - mean(b);
  return se > 0 ? d / se : d === 0 ? 0 : Infinity * Math.sign(d);
}

// two-sided normal tail, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
function normalTail(z) {
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.3275911 * x);
  const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
  return y * Math.exp(-x * x);
}

function criticalZ(alpha) {
  let lo = 0, hi = 40;
  for (let k = 0; k < 100; k++) { const m = (lo + hi) / 2; if (normalTail(m) > alpha) lo = m; else hi = m; }
  return hi;
}

export function compare(A, B, splits = 50) {
  const t = A.meta.t_run_s;
  const ids = [...new Set([...Object.keys(A.counts), ...Object.keys(B.counts)])];
  const zeros = (n) => new Array(n).fill(0);
  const ca = ids.map((id) => A.counts[id] ?? zeros(A.meta.n_trials));
  const cb = ids.map((id) => B.counts[id] ?? zeros(B.meta.n_trials));
  const ra = ca.map((c) => mean(c) / t), rb = cb.map((c) => mean(c) / t);

  // split-half: correlation across simulators vs within each, same number of trials per side
  const rand = mulberry32(12345);
  const half = (n) => {
    const idx = [...Array(n).keys()];
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
    return [idx.slice(0, n >> 1), idx.slice(n >> 1, 2 * (n >> 1))];
  };
  const sub = (counts, idx) => counts.map((c) => idx.reduce((s, k) => s + c[k], 0) / idx.length);
  let within = 0, cross = 0;
  for (let s = 0; s < splits; s++) {
    const [a1, a2] = half(A.meta.n_trials), [b1, b2] = half(B.meta.n_trials);
    within += (pearson(sub(ca, a1), sub(ca, a2)) + pearson(sub(cb, b1), sub(cb, b2))) / 2;
    cross += (pearson(sub(ca, a1), sub(cb, b2)) + pearson(sub(ca, a2), sub(cb, b1))) / 2;
  }
  within /= splits; cross /= splits;

  const z = ca.map((c, i) => welch(c, cb[i]));
  const zCrit = criticalZ(0.05 / ids.length);
  const outliers = ids.map((id, i) => ({ id, a: ra[i], b: rb[i], z: z[i] }))
    .filter((o) => Math.abs(o.z) > zCrit).sort((p, q) => Math.abs(q.z) - Math.abs(p.z));

  const mn9 = A.meta.mn9, k = ids.indexOf(mn9);
  const sdRate = (c) => Math.sqrt(variance(c)) / t;
  const res = {
    a: A.meta.source, b: B.meta.source, neurons: ids.length,
    active_1hz: { a: ra.filter((r) => r > 1).length, b: rb.filter((r) => r > 1).length },
    pearson_means: pearson(ra, rb), split_half: { within, cross },
    mn9: k < 0 ? null : { a: [ra[k], sdRate(ca[k])], b: [rb[k], sdRate(cb[k])], z: z[k] },
    bonferroni_z: zCrit, outliers: outliers.length, top_outliers: outliers.slice(0, 10),
  };
  res.gates = {
    mn9_z_below_3: res.mn9 !== null && Math.abs(res.mn9.z) < 3,
    cross_ge_within_minus_0_02: cross >= within - 0.02,
    outliers_le_1pct: outliers.length <= 0.01 * ids.length,
  };
  res.pass = Object.values(res.gates).every(Boolean);
  return res;
}

function main() {
  const [pa, pb, out] = process.argv.slice(2);
  const res = compare(JSON.parse(fs.readFileSync(pa, 'utf8')), JSON.parse(fs.readFileSync(pb, 'utf8')));
  if (out) fs.writeFileSync(out, JSON.stringify(res, null, 1));
  const f = (x) => x.toFixed(3);
  console.log(`A: ${res.a}\nB: ${res.b}`);
  console.log(`neurons ${res.neurons} (>1 Hz: A ${res.active_1hz.a}, B ${res.active_1hz.b})`);
  console.log(`pearson of mean rates ${f(res.pearson_means)}; split-half within ${f(res.split_half.within)}, cross ${f(res.split_half.cross)}`);
  if (res.mn9) console.log(`MN9 A ${res.mn9.a[0].toFixed(1)} ± ${res.mn9.a[1].toFixed(1)} Hz, B ${res.mn9.b[0].toFixed(1)} ± ${res.mn9.b[1].toFixed(1)} Hz, z ${res.mn9.z.toFixed(2)}`);
  console.log(`neurons differing at Bonferroni |z| > ${res.bonferroni_z.toFixed(2)}: ${res.outliers}`);
  for (const o of res.top_outliers.slice(0, 5)) console.log(`  ${o.id}: ${o.a.toFixed(1)} vs ${o.b.toFixed(1)} Hz (z ${o.z.toFixed(1)})`);
  console.log(`gates ${JSON.stringify(res.gates)} → ${res.pass ? 'PASS' : 'FAIL'}`);
}

if (import.meta.url === `file://${process.argv[1]}`) main();
