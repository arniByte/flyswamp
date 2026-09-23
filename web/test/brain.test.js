import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { MushroomBody } from '../src/sim/brain.js';
import { makeOdorLibrary } from '../src/sim/odors.js';

const circuit = JSON.parse(fs.readFileSync(new URL('../public/data/circuit.json', import.meta.url)));
const lib = makeOdorLibrary(circuit.glomeruli);
const scale = (v, k) => v.map((x) => x * k);
const zero = new Float32Array(circuit.glomeruli.length);
const refs = [lib.fruit, lib.sundew, lib.frog, scale(lib.fruit, 0.3), scale(lib.sundew, 0.3), scale(lib.frog, 0.3)];

function brain(opts) {
  const mb = new MushroomBody(circuit, opts);
  mb.calibrate(refs);
  return mb;
}

function conditioning(mb) {
  const dt = 0.02;
  const trial = (odor, key) => {
    for (let t = 0; t < 1.5; t += dt) mb.step(scale(lib[odor], 0.6), dt, { [key]: t > 1.0 ? 1 : 0 });
    for (let t = 0; t < 2; t += dt) mb.step(zero, dt);
  };
  for (let i = 0; i < 3; i++) trial('sundew', 'punish');
  for (let i = 0; i < 3; i++) trial('fruit', 'reward');
  const v = (n) => mb.evaluate(scale(lib[n], 0.6)).valence;
  return { fruit: v('fruit'), sundew: v('sundew'), frog: v('frog') };
}

test('edge indices stay inside their groups', () => {
  const g = Object.fromEntries(circuit.groups.map((x) => [x.name, x.count]));
  const nDan = g.PAM + g.PPL1;
  const spec = {
    glom_pn: [circuit.glomeruli.length, g.PN], pn_kc: [g.PN, g.KC], kc_mbon: [g.KC, g.MBON],
    kc_apl: [g.KC, g.APL], apl_kc: [g.APL, g.KC], dan_mbon: [nDan, g.MBON], mbon_dan: [g.MBON, nDan],
  };
  for (const [k, [nPre, nPost]] of Object.entries(spec)) {
    const e = circuit.edges[k];
    assert.equal(e.length % 3, 0, k);
    for (let i = 0; i < e.length; i += 3) {
      assert.ok(e[i] >= 0 && e[i] < nPre && e[i + 1] >= 0 && e[i + 1] < nPost && e[i + 2] > 0, `${k} edge ${i / 3}`);
    }
  }
  assert.equal(circuit.neurons.bodyId.length, circuit.groups.reduce((s, x) => s + x.count, 0));
});

test('KC code is sparse and APL makes it concentration tolerant', () => {
  const mb = brain();
  const s = mb.evaluate(lib.fruit).sparsity;
  assert.ok(s > 0.03 && s < 0.2, `sparsity ${s}`);
  const ratio = (m) => m.evaluate(scale(lib.fruit, 2)).active / m.evaluate(scale(lib.fruit, 0.3)).active;
  assert.ok(ratio(mb) < ratio(brain({ apl: false })), 'APL should flatten the concentration response');
});

test('naive valence is zero for every odor', () => {
  const mb = brain();
  for (const n of ['fruit', 'sundew', 'frog', 'swamp']) assert.equal(mb.evaluate(lib[n]).valence, 0, n);
});

test('differential conditioning separates fruit from its sundew mimic', () => {
  const v = conditioning(brain());
  assert.ok(v.fruit > 0, `fruit ${v.fruit}`);
  assert.ok(v.sundew < -0.2, `sundew ${v.sundew}`);
  assert.ok(Math.abs(v.frog) < 0.05, `frog untouched ${v.frog}`);
});

test('blocking dopamine abolishes learning', () => {
  const v = conditioning(brain({ dopamine: false }));
  for (const [k, x] of Object.entries(v)) assert.ok(Math.abs(x) < 1e-3, `${k} ${x}`);
});
