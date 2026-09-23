// Flatten circuit state into one activity value (0..1) per displayed neuron, in circuit.json order.
export function makeActivityMapper(circuit) {
  const g = Object.fromEntries(circuit.groups.map((x) => [x.name, x]));
  const N = circuit.neurons.bodyId.length;
  const out = new Float32Array(N);
  const ornGlom = Int32Array.from(circuit.orn_glom);
  const nPam = circuit.dan.n_pam;
  const dnType = [];
  for (let i = 0; i < g.DN.count; i++) dnType.push(circuit.types[circuit.neurons.type[g.DN.offset + i]]);
  const dnSide = circuit.neurons.side.slice(g.DN.offset, g.DN.offset + g.DN.count);

  // motor: { turn: -1..1 (left positive), walk: 0..1, escape: 0..1, struggle: 0..1 }
  return function collect(mb, motor = {}) {
    for (let i = 0; i < g.ORN.count; i++) out[g.ORN.offset + i] = Math.min(1, 1.2 * mb.u[ornGlom[i]]);
    out.set(mb.pn, g.PN.offset);
    out.set(mb.kc, g.KC.offset);
    for (let i = 0; i < g.MBON.count; i++) out[g.MBON.offset + i] = Math.min(1, 0.6 * mb.mbon[i]);
    for (let i = 0; i < g.PAM.count; i++) out[g.PAM.offset + i] = Math.min(1, mb.dan[i]);
    for (let i = 0; i < g.PPL1.count; i++) out[g.PPL1.offset + i] = Math.min(1, mb.dan[nPam + i]);
    for (let i = 0; i < g.APL.count; i++) out[g.APL.offset + i] = Math.min(1, 6 * mb.apl[i]);
    const { turn = 0, walk = 0, escape = 0, struggle = 0 } = motor;
    for (let i = 0; i < g.DN.count; i++) {
      const t = dnType[i];
      const left = dnSide[i] === 0;
      let v = 0.03;
      if (t === 'DNa01' || t === 'DNa02') v += Math.max(0, left ? turn : -turn);   // steering DNs, ipsilateral turns
      if (t === 'DNp09') v += 0.6 * walk;                                         // forward walking
      if (t === 'DNp01') v += escape;                                             // giant fiber
      if (t === 'MDN') v += struggle;                                             // backward walking
      out[g.DN.offset + i] = Math.min(1, v);
    }
    return out;
  };
}
