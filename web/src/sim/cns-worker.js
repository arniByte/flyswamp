// Web Worker around CNSRunner: decodes the whole-CNS graph, then keeps simulated time level with the
// wall clock and posts the spikes of every tick. If the brain falls behind, the world gets `factor` < 1.
import { CNSRunner } from './cns-runner.js';
import { concat, decodeCNS, gunzip } from './cnsgraph.js';

const TICK_MS = 30, MAX_LAG_MS = 250;
let runner = null, wall0 = 0, sim0 = 0, paused = false, speed = 1, lastWall = 0, lastSim = 0;

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const t0 = performance.now();
      const graph = decodeCNS(m.header, await gunzip(concat(m.chunks)));
      runner = new CNSRunner(graph, m.header.engine, m.seed ?? 1);
      self.postMessage({ type: 'ready', decodeMs: performance.now() - t0, n: graph.n, nnz: graph.nnz });
      wall0 = lastWall = performance.now();
      sim0 = lastSim = 0;
      setTimeout(tick, 0);
    } else if (m.type === 'drive') runner?.setDrive(m.entries);
    else if (m.type === 'pause') { paused = m.value; resync(); }
    else if (m.type === 'speed') { speed = m.value; resync(); }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err?.message ?? err) });
  }
};

function resync() {
  if (!runner) return;
  wall0 = performance.now();
  sim0 = runner.simMs;
}

function tick() {
  const now = performance.now();
  if (!paused) {
    let target = sim0 + (now - wall0) * speed;
    // too far behind: accept the lag instead of trying to catch up, and report the slow-down
    if (target - runner.simMs > MAX_LAG_MS) { sim0 = runner.simMs; wall0 = now; target = runner.simMs + TICK_MS * speed; }
    const spikes = runner.advance(target, TICK_MS * 0.8);
    const wallDt = now - lastWall, simDt = runner.simMs - lastSim;
    lastWall = now; lastSim = runner.simMs;
    self.postMessage({ type: 'tick', spikes, simMs: runner.simMs, factor: wallDt > 0 ? simDt / wallDt : 1 }, [spikes.buffer]);
  }
  setTimeout(tick, Math.max(0, TICK_MS - (performance.now() - now)));
}
