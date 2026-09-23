import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { World, DT } from '../src/sim/world.js';

const circuit = JSON.parse(fs.readFileSync(new URL('../public/data/circuit.json', import.meta.url)));
const run = (w, seconds) => { for (let i = 0; i < seconds / DT; i++) w.step(DT); return w; };

test('plumes carry odor downwind, not upwind', () => {
  const w = new World(circuit, { layout: 'empty' });
  w.wind.angle = 0; // blowing towards +x
  const e = w.add('fruit', 0, 0);
  assert.ok(w.concentration(e, 120, 0) > 5 * w.concentration(e, -120, 0));
});

test('a minute in the swamp stays finite and inside the arena', () => {
  const w = run(new World(circuit, { seed: 2 }), 60);
  const f = w.fly;
  for (const v of [f.x, f.y, f.z, f.heading, f.energy, w.mb.valence]) assert.ok(Number.isFinite(v));
  assert.ok(Math.hypot(f.x, f.z) <= 300.001);
});

test('with dopamine the fly ends up avoiding sundews more than without', () => {
  const intact = run(new World(circuit, { seed: 5 }), 240);
  const blocked = run(new World(circuit, { seed: 5, brain: { dopamine: false } }), 240);
  assert.ok(intact.memory.sundew.drive < blocked.memory.sundew.drive - 0.2,
    `sundew drive intact ${intact.memory.sundew.drive.toFixed(2)} vs blocked ${blocked.memory.sundew.drive.toFixed(2)}`);
});
