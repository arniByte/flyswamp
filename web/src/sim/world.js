// The swamp: odor plumes, fruit, sundews, frogs and one fly steered by its mushroom body.
// Pure logic (no DOM, no three.js) so it runs headless in Node for experiments.
//
// Frame: x/z ground plane in millimetres, y up. Heading ψ: forward = (cos ψ, -sin ψ), left = (-sin ψ, -cos ψ);
// turning left increases ψ (same as a three.js rotation.y of ψ on a model facing +x).
import { MushroomBody } from './brain.js';
import { makeOdorLibrary, innateValence, jitter, cosine, SUNDEW_SIGNATURE } from './odors.js';
import { mulberry32, gaussian } from './rng.js';

export const DT = 0.02; // s, one brain/world tick
const ARENA = 300;      // mm radius
const KINDS = {
  fruit: { radius: 13, strength: 1.0 },
  sundew: { radius: 11, strength: 0.9 },
  frog: { radius: 26, strength: 0.8 },
};
export const W_INNATE = 0.7;
export const W_MB = 1.4;

export class World {
  constructor(circuit, { seed = 1, brain = {}, armsRace = false, layout = 'default' } = {}) {
    this.circuit = circuit;
    this.rand = mulberry32(seed);
    this.lib = makeOdorLibrary(circuit.glomeruli);
    this.gIndex = Object.fromEntries(circuit.glomeruli.map((g, i) => [g, i]));
    this.nG = circuit.glomeruli.length;
    this.mb = new MushroomBody(circuit, { seed, ...brain });
    const s = (v, k) => v.map((x) => x * k);
    this.mb.calibrate([this.lib.fruit, this.lib.sundew, this.lib.frog, s(this.lib.fruit, 0.3), s(this.lib.sundew, 0.3), s(this.lib.frog, 0.3)]);

    this.armsRace = armsRace;
    this.t = 0;
    this.nextId = 1;
    this.entities = [];
    this.wind = { angle: 0.4, speed: 1 };
    this.fly = {
      x: 0, y: 30, z: 0, heading: 0, speed: 60, turn: 0, wander: 0,
      state: 'fly', stateTime: 0, target: null, energy: 1, pitch: 0, roll: 0, perch: null,
    };
    this.teach = { reward: 0, punish: 0 };
    this.motor = { turn: 0, walk: 0, escape: 0, struggle: 0 };
    this.stats = { eaten: 0, trapped: 0, strikes: 0, escapes: 0, caught: 0 };
    this.events = [];
    this.timeline = [];
    this.sense = { u: new Float32Array(this.nG), intensity: 0, drive: 0, innate: 0, mb: 0, gradient: 0 };
    this._u = new Float32Array(this.nG);
    this._tmp = new Float32Array(this.nG);
    this.memory = { fruit: 0, sundew: 0, frog: 0 };
    this._probeT = 0;
    this._sampleT = 0;
    this._evolveT = 0;
    this._respawn = [];

    if (layout === 'default') this.defaultLayout();
    this.probeMemory();
  }

  // ---------- entities ----------

  defaultLayout() {
    const place = (kind, n, rMin, rMax) => {
      for (let i = 0; i < n; i++) {
        const p = this.freeSpot(rMin, rMax, kind === 'frog' ? 90 : 55);
        this.add(kind, p.x, p.z);
      }
    };
    place('fruit', 6, 60, 260);
    place('sundew', 5, 60, 260);
    place('frog', 2, 120, 250);
    const p = this.freeSpot(0, 80, 40);
    Object.assign(this.fly, { x: p.x, z: p.z });
  }

  freeSpot(rMin, rMax, clearance) {
    for (let tries = 0; tries < 200; tries++) {
      const r = rMin + (rMax - rMin) * Math.sqrt(this.rand());
      const a = this.rand() * Math.PI * 2;
      const x = r * Math.cos(a), z = r * Math.sin(a);
      if (this.entities.every((e) => Math.hypot(e.x - x, e.z - z) > clearance + e.radius)) return { x, z };
    }
    return { x: (this.rand() - 0.5) * rMax, z: (this.rand() - 0.5) * rMax };
  }

  add(kind, x, z) {
    const k = KINDS[kind];
    const base = kind === 'sundew' && this.sundewRecipe ? this.sundewRecipe : this.lib[kind];
    const e = {
      id: this.nextId++, kind, x, z, radius: k.radius, strength: k.strength,
      odor: jitter(base, this.rand, kind === 'sundew' ? 0.06 : 0.12),
      phase: this.rand() * 10, bites: kind === 'fruit' ? 5 : 0, heading: this.rand() * Math.PI * 2,
      strike: 0, cooldown: 0, hop: 20 + 20 * this.rand(), trapped: 0,
    };
    this.entities.push(e);
    return e;
  }

  remove(id) {
    this.entities = this.entities.filter((e) => e.id !== id);
    if (this.fly.perch === id) this.takeoff();
  }

  // ---------- odor field ----------

  concentration(e, x, z) {
    const w = this.wind;
    const wx = Math.cos(w.angle), wz = Math.sin(w.angle);
    const dx = x - e.x, dz = z - e.z;
    const d2 = dx * dx + dz * dz;
    const along = dx * wx + dz * wz;
    const perp2 = Math.max(0, d2 - along * along);
    const r0 = e.radius + 15;
    const near = Math.exp(-d2 / (2 * r0 * r0));
    let plume = 0;
    if (along > 0) {
      const sig = 14 + 0.32 * along;
      const flicker = 0.72 + 0.28 * Math.sin(0.045 * along - 2.6 * this.t + e.phase);
      plume = Math.exp(-perp2 / (2 * sig * sig)) / (1 + along / 200) * flicker;
    }
    return e.strength * Math.max(near, plume);
  }

  // Glomerular input at a point: sum of plumes plus a faint swamp background.
  odorAt(x, z, out = this._u) {
    const amb = 0.05;
    for (let g = 0; g < this.nG; g++) out[g] = amb * this.lib.swamp[g];
    let total = 0;
    for (const e of this.entities) {
      const c = this.concentration(e, x, z);
      if (c < 1e-3) continue;
      total += c;
      for (let g = 0; g < this.nG; g++) out[g] += c * e.odor[g];
    }
    return total;
  }

  intensityAt(x, z) {
    let s = 0;
    for (const e of this.entities) s += this.concentration(e, x, z);
    return s;
  }

  // ---------- main tick ----------

  step(dt = DT) {
    this.t += dt;
    const f = this.fly;
    f.stateTime += dt;
    const fwdX = Math.cos(f.heading), fwdZ = -Math.sin(f.heading);
    const leftX = -Math.sin(f.heading), leftZ = -Math.cos(f.heading);

    // sense: odor at the head, total intensity at both antennae (spacing exaggerated for a 2D plume)
    const hx = f.x + 2 * fwdX, hz = f.z + 2 * fwdZ;
    const I = this.odorAt(hx, hz);
    const IL = this.intensityAt(hx + 5 * leftX, hz + 5 * leftZ);
    const IR = this.intensityAt(hx - 5 * leftX, hz - 5 * leftZ);

    const reward = this.teach.reward > 0 ? 1 : 0;
    const punish = this.teach.punish > 0 ? 1 : 0;
    this.teach.reward = Math.max(0, this.teach.reward - dt);
    this.teach.punish = Math.max(0, this.teach.punish - dt);
    this.mb.step(this._u, dt, { reward, punish });

    const fade = Math.min(1, I / 0.08);
    const innate = innateValence(this._u, this.gIndex) * fade;
    // Design choice: hunger lowers the weight of aversive memories, so a starving fly takes risks.
    const hunger = 1 - f.energy;
    const vmb = this.mb.valence * fade * (this.mb.valence < 0 ? 1 - 0.6 * hunger : 1);
    const drive = Math.max(-1, Math.min(1, W_INNATE * innate + W_MB * vmb));
    const grad = (IL - IR) / (IL + IR + 1e-4);
    Object.assign(this.sense, { intensity: I, drive, innate, mb: vmb, gradient: grad });
    this.sense.u.set(this._u);

    if (f.state === 'fly' || f.state === 'escape') this.flyStep(dt, I, drive, grad);
    else if (f.state === 'land') this.landStep(dt);
    else if (f.state === 'feed') this.feedStep(dt);
    else if (f.state === 'trapped') this.trappedStep(dt);

    this.frogsStep(dt);
    f.energy = Math.max(0.02, f.energy - dt * (f.state === 'fly' ? 0.008 : 0.003));
    this.motor.escape *= Math.exp(-dt / 0.15);

    for (const r of this._respawn) r.t -= dt;
    while (this._respawn.length && this._respawn[0].t <= 0) {
      const r = this._respawn.shift();
      const p = this.freeSpot(50, 270, 55);
      this.add(r.kind, p.x, p.z);
    }

    if ((this._probeT -= dt) <= 0) { this._probeT = 0.5; this.probeMemory(); }
    if ((this._sampleT -= dt) <= 0) { this._sampleT = 5; this.sample(); }
    if (this.armsRace && (this._evolveT -= dt) <= 0) { this._evolveT = 12; this.evolveSundews(); }
  }

  flyStep(dt, I, drive, grad) {
    const f = this.fly;
    const escaping = f.state === 'escape';
    const fwdX = Math.cos(f.heading), fwdZ = -Math.sin(f.heading);
    // wander: Ornstein-Uhlenbeck turn noise (search/casting when there is no odor)
    const casting = I < 0.03 ? 1.6 : 0.6;
    f.wander += (-f.wander * 1.5 * dt) + casting * 2.2 * Math.sqrt(dt) * gaussian(this.rand);
    let turn = f.wander;
    if (I > 0.02 && !escaping) {
      // approach: turn toward the stronger antenna and upwind; avoid: the reverse
      const up = Math.atan2(Math.sin(this.wind.angle), -Math.cos(this.wind.angle)); // heading that faces the wind
      const rel = angleDiff(up, f.heading);
      turn += drive * (7 * grad + 1.1 * Math.sin(rel));
    }
    // arena wall: soft steering near the edge, hard clamp at it
    const r = Math.hypot(f.x, f.z);
    const toCenter = Math.atan2(f.z, -f.x);
    if (r > ARENA - 60) turn += 4 * Math.sin(angleDiff(toCenter, f.heading)) * (r - ARENA + 60) / 60;
    f.turn += (Math.max(-5, Math.min(5, turn)) - f.turn) * Math.min(1, dt / 0.12);
    f.heading += f.turn * dt;

    const target = escaping ? 170 : 55 + (I > 0.03 ? 45 * Math.max(0, drive) + 70 * Math.max(0, -drive) : 0);
    f.speed += (target - f.speed) * Math.min(1, dt / 0.3);
    f.x += fwdX * f.speed * dt;
    f.z += fwdZ * f.speed * dt;
    const r2 = Math.hypot(f.x, f.z);
    if (r2 > ARENA) {
      f.x *= ARENA / r2;
      f.z *= ARENA / r2;
      f.heading = toCenter + 0.5 * gaussian(this.rand);
    }
    const alt = escaping ? 55 : 22 + 6 * Math.sin(this.t * 0.9 + 1.3) - (I > 0.1 && drive > 0 ? 8 : 0);
    f.y += (alt - f.y) * Math.min(1, dt / 0.5);
    f.pitch = escaping ? 0.35 : -0.12 - 0.002 * (f.speed - 55); // nose down to fly forward, up while escaping
    f.roll = -0.35 * f.turn / 5;
    this.motor.turn = Math.max(-1, Math.min(1, f.turn / 3));
    this.motor.walk = 0;
    this.motor.struggle = 0;
    if (escaping && f.stateTime > 0.6) this.setState('fly');

    // land when the fly wants what is right below
    if (!escaping && f.stateTime > 1.0) {
      const hunger = 1 - f.energy;
      const want = 0.22 - 0.18 * hunger;
      for (const e of this.entities) {
        if (e.kind === 'frog') continue;
        const d = Math.hypot(e.x - f.x, e.z - f.z);
        if (d < e.radius + 6 && drive > want && !(e.kind === 'fruit' && e.bites <= 0)) {
          f.target = e.id;
          this.setState('land');
          break;
        }
      }
    }
  }

  landStep(dt) {
    const f = this.fly;
    const e = this.entities.find((x) => x.id === f.target);
    if (!e) return this.takeoff();
    const top = e.kind === 'fruit' ? 2 * e.radius * 0.8 : 4.5; // matches the prop heights in render/props.js
    const k = Math.min(1, dt / 0.25);
    f.x += (e.x - f.x) * k;
    f.z += (e.z - f.z) * k;
    f.y += (top - f.y) * k;
    f.speed *= 0.9;
    f.pitch *= 0.9;
    f.roll *= 0.9;
    if (f.stateTime > 0.7) {
      f.perch = e.id;
      if (e.kind === 'fruit') {
        this.setState('feed');
        this.teach.reward = 2.0;
        this.stats.eaten++;
        e.bites--;
        this.log('feed', 'Муха ест гнилой фрукт → сахар активирует PAM (награда)');
      } else {
        this.setState('trapped');
        this.teach.punish = 1.5;
        this.stats.trapped++;
        e.trapped = 1;
        f.energy = Math.max(0.02, f.energy - 0.08);
        this.log('trap', 'Росянка! Липкие железы держат муху → PPL1 (наказание)');
      }
    }
  }

  feedStep(dt) {
    const f = this.fly;
    f.energy = Math.min(1, f.energy + 0.12 * dt);
    this.motor.walk = 0.3;
    this.motor.turn = 0.2 * Math.sin(this.t * 3);
    f.heading += 0.25 * Math.sin(this.t * 0.7) * dt;
    if (f.stateTime > 2.5) {
      const e = this.entities.find((x) => x.id === f.perch);
      if (e && e.bites <= 0) {
        this.entities = this.entities.filter((x) => x !== e);
        this._respawn.push({ kind: 'fruit', t: 6 });
        this.log('info', 'Фрукт съеден. Скоро где-то созреет новый.');
      }
      this.takeoff();
    }
  }

  trappedStep(dt) {
    const f = this.fly;
    this.motor.struggle = 0.8 + 0.2 * Math.sin(this.t * 17);
    this.motor.turn = Math.sin(this.t * 11);
    if (f.stateTime > 3.0) {
      const e = this.entities.find((x) => x.id === f.perch);
      if (e) e.trapped = 0;
      this.log('info', 'Муха вырвалась из слизи росянки.');
      this.takeoff(true);
    }
  }

  takeoff(hard = false) {
    const f = this.fly;
    f.perch = null;
    f.target = null;
    f.heading += Math.PI * (0.6 + 0.8 * this.rand());
    f.speed = hard ? 120 : 70;
    this.setState(hard ? 'escape' : 'fly');
  }

  setState(s) {
    this.fly.state = s;
    this.fly.stateTime = 0;
  }

  frogsStep(dt) {
    const f = this.fly;
    for (const e of this.entities) {
      if (e.kind !== 'frog') continue;
      e.cooldown = Math.max(0, e.cooldown - dt);
      e.strike = Math.max(0, e.strike - dt);
      const dx = f.x - e.x, dz = f.z - e.z;
      const d = Math.hypot(dx, dz);
      const toFly = Math.atan2(-dz, dx);
      if (d < 140) e.heading += angleDiff(toFly, e.heading) * Math.min(1, dt * 2.5);
      if ((e.hop -= dt) <= 0 && e.strike <= 0) {
        e.hop = 18 + 25 * this.rand();
        const p = this.freeSpot(100, 260, 80);
        e.x += (p.x - e.x) * 0.5;
        e.z += (p.z - e.z) * 0.5;
        e.jump = 0.6;
      }
      e.jump = Math.max(0, (e.jump ?? 0) - dt);
      const catchable = f.state === 'fly' || f.state === 'feed';
      if (catchable && e.cooldown <= 0 && d < 60 && f.y < 38 && Math.abs(angleDiff(toFly, e.heading)) < 0.5) {
        e.cooldown = 12;
        if (this.rand() < 0.5) continue; // the frog hesitates
        e.strike = 0.35;
        e.strikeAt = { x: f.x, y: f.y, z: f.z };
        this.stats.strikes++;
        this.teach.punish = 0.8;
        this.motor.escape = 1;
        const pEscape = f.state === 'feed' ? 0.55 : 0.8;
        if (this.rand() < pEscape) {
          this.stats.escapes++;
          this.log('frog', 'Лягушка бьёт языком! Гигантское волокно DNp01 → побег');
          f.heading = toFly + (this.rand() < 0.5 ? 1 : -1) * 0.4;
          f.perch = null;
          this.setState('escape');
        } else {
          this.stats.caught++;
          this.teach.punish = 1.6;
          f.energy = Math.max(0.02, f.energy - 0.3);
          this.log('frog', 'Поймана языком… и выплюнута. Сильное наказание PPL1');
          f.perch = null;
          this.setState('escape');
        }
      }
    }
  }

  // ---------- learning read-outs ----------

  attraction(u) {
    const vmb = this.mb.evaluate(u).valence;
    return { mb: vmb, innate: innateValence(u, this.gIndex), drive: W_INNATE * innateValence(u, this.gIndex) + W_MB * vmb };
  }

  probeMemory() {
    const probe = (vec) => {
      for (let g = 0; g < this.nG; g++) this._tmp[g] = 0.6 * vec[g] + 0.05 * this.lib.swamp[g];
      return this.attraction(this._tmp);
    };
    const sundews = this.entities.filter((e) => e.kind === 'sundew');
    const sundew = sundews.length ? meanVec(sundews.map((e) => e.odor)) : this.lib.sundew;
    this.memory = { fruit: probe(this.lib.fruit), sundew: probe(sundew), frog: probe(this.lib.frog) };
    this.memory.mimicry = cosine(sundew, this.lib.fruit);
  }

  sample() {
    this.timeline.push({
      t: this.t, ...this.stats, energy: this.fly.energy,
      vFruit: this.memory.fruit.drive, vSundew: this.memory.sundew.drive, vFrog: this.memory.frog.drive,
      mimicry: this.memory.mimicry,
    });
  }

  // Arms race: each sundew tries small scent mutations and keeps the one the fly finds most attractive.
  evolveSundews() {
    const test = new Float32Array(this.nG);
    const score = (odor) => {
      for (let g = 0; g < this.nG; g++) test[g] = 0.6 * odor[g] + 0.05 * this.lib.swamp[g];
      return this.attraction(test).drive;
    };
    const sig = Object.entries(SUNDEW_SIGNATURE).map(([g, min]) => [this.gIndex[g], min]);
    let changed = 0;
    for (const e of this.entities) {
      if (e.kind !== 'sundew') continue;
      let best = e.odor, bestScore = score(e.odor);
      const norm = Math.hypot(...e.odor);
      for (let k = 0; k < 8; k++) {
        const m = Float32Array.from(best);
        for (let j = 0; j < 3; j++) {
          const g = Math.floor(this.rand() * this.nG);
          m[g] = Math.max(0, Math.min(1.2, m[g] + 0.12 * gaussian(this.rand)));
        }
        for (const [g, min] of sig) m[g] = Math.max(m[g], min);
        const s = norm / (Math.hypot(...m) || 1);
        for (let g = 0; g < this.nG; g++) m[g] *= s;
        const sc = score(m);
        if (sc > bestScore + 0.01) { best = m; bestScore = sc; }
      }
      if (best !== e.odor) { e.odor = best; changed++; }
    }
    if (changed) {
      const sundews = this.entities.filter((e) => e.kind === 'sundew');
      this.sundewRecipe = meanVec(sundews.map((e) => e.odor));
      this.log('evolve', `Росянки подстроили запах под память мухи (${changed} шт.)`);
    }
  }

  // ---------- user actions ----------

  giveSugar() { this.teach.reward = 1.0; this.log('feed', 'Ты дал сахар → PAM. Муха запомнит то, что сейчас нюхает'); }
  giveShock() { this.teach.punish = 1.0; this.log('trap', 'Ты дал удар → PPL1. Запах вокруг станет неприятным'); }

  log(type, text) {
    this.events.push({ t: this.t, type, text });
    if (this.events.length > 60) this.events.shift();
  }
}

function angleDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function meanVec(list) {
  const out = new Float32Array(list[0].length);
  for (const v of list) for (let i = 0; i < v.length; i++) out[i] += v[i] / list.length;
  return out;
}
