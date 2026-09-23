// Odors are stylized activation patterns over the real antennal lobe glomeruli.
// Anchors from the literature (the rest of each pattern is invented, see README):
//   DM1 (Or42b) and VA2 (Or92a) mediate innate attraction to vinegar   — Semmelhack & Wang 2009
//   DP1m (Ir64a) responds to acids                                     — Ai et al. 2010
//   DL5 (Or7a) responds to green-leaf volatiles such as E2-hexenal
//   DA2 (Or56a) is a labeled line for geosmin, an aversive microbe cue — Stensmyr et al. 2012
//   V (Gr21a/Gr63a) senses CO2, innately avoided                        — Suh et al. 2004
import { mulberry32 } from './rng.js';

export const RECIPES = {
  fruit: { DM1: 1.0, VA2: 0.8, DM2: 0.9, DM4: 0.7, DM3: 0.5, VM2: 0.5, DP1m: 0.4, VM7d: 0.3, DL1: 0.2 },
  sundew: { DM1: 0.8, VA2: 0.7, DM4: 0.3, DM2: 0.2, DC1: 0.8, DL5: 0.7, VC3: 0.5, VM3: 0.4, DL2d: 0.3 },
  frog: { DA2: 0.9, V: 1.0, DL4: 0.3, VM5d: 0.3, VA1d: 0.2 },
  swamp: { DA2: 0.5, VM5d: 0.3, V: 0.2, DL3: 0.2 },
};

// Glomeruli the sundew cannot silence: its own leaf chemistry (used by the arms race).
export const SUNDEW_SIGNATURE = { DC1: 0.35, DL5: 0.3, VC3: 0.2 };

export const ODOR_COLORS = { fruit: 0xffb347, sundew: 0xff4f8b, frog: 0x7bd88f, swamp: 0x6f7f6a };

// Every real odor weakly drives many receptors; add a seeded broad floor per family.
function broadFloor(glomeruli, seed, level) {
  const rand = mulberry32(seed);
  return glomeruli.map(() => (rand() < 0.35 ? rand() * level : 0));
}

export function recipeToVector(recipe, glomeruli, floor = null) {
  const v = new Float32Array(glomeruli.length);
  glomeruli.forEach((g, i) => {
    v[i] = (recipe[g] ?? 0) + (floor ? floor[i] : 0);
  });
  return v;
}

export function makeOdorLibrary(glomeruli) {
  const seeds = { fruit: 11, sundew: 23, frog: 37, swamp: 51 };
  const lib = {};
  for (const [name, recipe] of Object.entries(RECIPES)) {
    lib[name] = recipeToVector(recipe, glomeruli, broadFloor(glomeruli, seeds[name], 0.12));
  }
  return lib;
}

// Per-source variation so no two fruits smell exactly alike.
export function jitter(vec, rand, amount = 0.12) {
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) out[i] = Math.max(0, vec[i] * (1 + amount * (rand() * 2 - 1)));
  return out;
}

export function cosine(a, b) {
  let ab = 0, aa = 0, bb = 0;
  for (let i = 0; i < a.length; i++) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return ab / Math.sqrt(aa * bb + 1e-12);
}

// Innate, hard-wired valence of an odor mixture: a stand-in for lateral horn labeled lines.
export function innateValence(u, gIndex) {
  const attract = 0.5 * ((u[gIndex.DM1] ?? 0) + (u[gIndex.VA2] ?? 0));
  const aversive = Math.max(u[gIndex.DA2] ?? 0, u[gIndex.V] ?? 0);
  return Math.tanh(1.2 * attract - 1.6 * aversive);
}
