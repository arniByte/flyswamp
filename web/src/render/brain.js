// X-ray brain: real MaleCNS geometry (shells, neuropils, glomeruli, MB compartments),
// skeletons of every simulated neuron and a cloud of all 141k somata, lit by live activity.
import * as THREE from 'three';

export const GROUP_STYLE = {
  ORN: { color: 0xffa640, label: 'ORN — обонятельные рецепторы' },
  PN: { color: 0xd8ff4a, label: 'PN — проекционные нейроны' },
  KC: { color: 0x45d6ff, label: 'KC — клетки Кеньона' },
  MBON: { color: 0x7dff9b, label: 'MBON — выход грибовидного тела' },
  PAM: { color: 0xffc933, label: 'PAM — дофамин «награда»' },
  PPL1: { color: 0xc26bff, label: 'PPL1 — дофамин «наказание»' },
  APL: { color: 0x5a7dff, label: 'APL — обратное торможение' },
  DN: { color: 0xeaf6ff, label: 'DN — нисходящие моторные' },
};
const MBON_AVOID = new THREE.Color(0xff4f7d);
const MAX_RATE = { ORN: 60, PN: 80, KC: 25, MBON: 50, PAM: 30, PPL1: 30, APL: 40, DN: 60 }; // Hz, for spike sampling only
const TEX_W = 128;

const REGION_COLORS = [0x5d7bff, 0x3f5a8f, 0x6a9cc8, 0x9fc4ff, 0x777777, 0x555566];

function fresnelMaterial(color, strength, power = 2.0) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uStrength: { value: strength }, uGlow: { value: 0 }, uGlowColor: { value: new THREE.Color(color) }, uPower: { value: power } },
    vertexShader: /* glsl */ `
      varying vec3 vN; varying vec3 vV;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor; uniform vec3 uGlowColor; uniform float uStrength; uniform float uGlow; uniform float uPower;
      varying vec3 vN; varying vec3 vV;
      void main() {
        float nl = length(vN);
        float f = pow(1.0 - (nl > 1e-6 ? abs(dot(vN / nl, normalize(vV))) : 0.0), uPower);
        vec3 c = uColor * (0.08 + f) * uStrength + uGlowColor * uGlow * (0.35 + 0.8 * f);
        gl_FragColor = vec4(c, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  });
}

const NEURON_VERT = /* glsl */ `
  uniform sampler2D uAct; uniform sampler2D uColor; uniform float uGroupOn[8]; uniform float uGroupGain[8]; uniform float uRestShown[8]; uniform float uPointSize; uniform float uBase; uniform float uPx;
  attribute float nid; attribute float dist;
  varying vec3 vColor; varying float vAlpha;
  void main() {
    int id = int(nid + 0.5);
    ivec2 t = ivec2(id % ${TEX_W}, id / ${TEX_W});
    vec4 a = texelFetch(uAct, t, 0);     // r: rate, g: spike glow, b: seconds since spike
    vec4 c = texelFetch(uColor, t, 0);   // rgb: colour, a: group index
    int gi = int(c.a + 0.5);
    float on = uGroupOn[gi] * uGroupGain[gi];
    // Show only a fixed sample of resting neurons in dense groups; active ones always light up.
    float shown = step(fract(sin(float(id) * 12.9898) * 43758.5453), uRestShown[gi]);
    on *= max(shown, smoothstep(0.02, 0.12, a.r + a.g));
    float dp = dist - a.b / 0.16; // pow() of a negative base is NaN in GLSL, which bloom would smear everywhere
    float pulse = a.b < 0.25 ? exp(-dp * dp / 0.004) : 0.0;
    float lum = uBase + 0.9 * a.r + 0.7 * a.g + 1.1 * pulse;
    vColor = mix(c.rgb, vec3(1.0), clamp(0.4 * a.g + 0.5 * pulse, 0.0, 0.6)) * (0.45 + 0.55 * min(lum, 1.0));
    vAlpha = clamp(lum, 0.0, 1.0) * on;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uPointSize * uPx * (0.6 + 1.4 * a.r + a.g) * on / max(-mv.z, 1e-3);
  }`;
const NEURON_FRAG = /* glsl */ `
  varying vec3 vColor; varying float vAlpha;
  void main() { if (vAlpha < 0.004) discard; gl_FragColor = vec4(vColor, vAlpha);
#include <tonemapping_fragment>
#include <colorspace_fragment> }`;
const SOMA_FRAG = /* glsl */ `
  varying vec3 vColor; varying float vAlpha;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r = dot(p, p);
    if (r > 1.0 || vAlpha < 0.004) discard;
    gl_FragColor = vec4(vColor, vAlpha * (1.0 - r * r));
#include <tonemapping_fragment>
#include <colorspace_fragment>
  }`;

export class BrainView {
  constructor({ circuit, anatomy, cns, neurons, somas }) {
    this.circuit = circuit;
    this.group = new THREE.Group();
    this.group.name = 'cns';
    this.groups = circuit.groups;
    this.N = circuit.neurons.bodyId.length;
    this.groupIndex = new Uint8Array(this.N);
    this.groups.forEach((g, gi) => this.groupIndex.fill(gi, g.offset, g.offset + g.count));
    this.maxRate = Float32Array.from(this.groupIndex, (gi) => MAX_RATE[this.groups[gi].name] ?? 30);
    this.time = 0;
    this.px = { value: 800 }; // pixels per unit size at unit depth; set from the camera via setViewport()

    this._buildTextures();
    this._buildMeshes(cns);
    this._buildSkeletons(anatomy, neurons);
    this._buildSomaCloud(anatomy, somas);
    this._buildCompartmentMap();
  }

  _buildTextures() {
    const H = Math.ceil(this.N / TEX_W);
    this.act = new Float32Array(TEX_W * H * 4);
    for (let i = 0; i < this.N; i++) this.act[4 * i + 2] = 10; // no recent spike
    this.actTex = new THREE.DataTexture(this.act, TEX_W, H, THREE.RGBAFormat, THREE.FloatType);
    this.actTex.needsUpdate = true;
    const col = new Float32Array(TEX_W * H * 4);
    const val = this.circuit.mbon.valence;
    const mbonOff = this.groups.find((g) => g.name === 'MBON').offset;
    const c = new THREE.Color();
    for (let i = 0; i < this.N; i++) {
      const gi = this.groupIndex[i];
      const name = this.groups[gi].name;
      c.set(GROUP_STYLE[name].color);
      if (name === 'MBON' && val[i - mbonOff] < 0) c.copy(MBON_AVOID);
      col.set([c.r, c.g, c.b, gi], 4 * i);
    }
    this.colorTex = new THREE.DataTexture(col, TEX_W, H, THREE.RGBAFormat, THREE.FloatType);
    this.colorTex.needsUpdate = true;
    this.groupOn = new Array(8).fill(1);
    // Thousands of overlapping KC axons would saturate additive blending, so they are dimmer per line.
    const gain = { ORN: 0.8, PN: 0.7, KC: 0.55, MBON: 1.0, PAM: 0.9, PPL1: 1.0, APL: 0.5, DN: 1.0 };
    this.groupGain = this.groups.map((g) => gain[g.name] ?? 1).concat(new Array(8 - this.groups.length).fill(1));
    const rest = { KC: 0.12 };
    this.restShown = this.groups.map((g) => rest[g.name] ?? 1).concat(new Array(8 - this.groups.length).fill(1));
  }

  _neuronMaterial(pointSize, base, frag, blending = THREE.NormalBlending) {
    return new THREE.ShaderMaterial({
      uniforms: { uAct: { value: this.actTex }, uColor: { value: this.colorTex }, uGroupOn: { value: this.groupOn }, uGroupGain: { value: this.groupGain }, uRestShown: { value: this.restShown }, uPointSize: { value: pointSize }, uBase: { value: base }, uPx: this.px },
      vertexShader: NEURON_VERT, fragmentShader: frag, transparent: true, depthWrite: false, blending,
    });
  }

  _buildMeshes(cns) {
    this.meshes = {};
    const style = {
      shell: [0x5fb4ff, 0.32, 2.2], neuropil: [0x4f78d8, 0.18, 2.0], glomerulus: [0xffb45a, 0.22, 1.6], compartment: [0x86a8ff, 0.2, 1.8],
    };
    const highlight = { 'AL(L)': 0.3, 'AL(R)': 0.3, 'CA(L)': 0.35, 'CA(R)': 0.35, 'PED(L)': 0.3, 'PED(R)': 0.3, 'LH(L)': 0.3, 'LH(R)': 0.3 };
    cns.traverse((o) => {
      if (!o.isMesh) return;
      const kind = o.material.name;
      o.geometry.computeVertexNormals();
      const [color, strength, power] = style[kind] ?? style.neuropil;
      o.material = fresnelMaterial(color, highlight[o.name] ?? strength, power);
      o.userData.kind = kind;
      o.renderOrder = 1;
      this.meshes[o.name] = o;
    });
    // Glomerulus meshes are named like "DM1(L)"; map them to glomerulus indices.
    this.glomMeshes = [];
    this.circuit.glomeruli.forEach((g, gi) => {
      for (const side of ['L', 'R']) {
        const m = this.meshes[`${g}(${side})`] ?? (g.startsWith('VM6') ? this.meshes[`VM6(${side})`] : null);
        if (m) this.glomMeshes.push([gi, m]);
      }
    });
    this.cns = cns;
    this.group.add(cns);
  }

  _buildSkeletons(anatomy, buf) {
    const L = anatomy.neurons;
    const n = L.vertices;
    const pos = new Int16Array(buf, L.positions.offset, n * 3);
    const nid = new Uint16Array(buf, L.nid.offset, n);
    const dist = new Uint8Array(buf, L.dist.offset, n);
    const parent = new Int32Array(buf, L.parent.offset, n);
    let m = 0;
    for (let i = 0; i < n; i++) if (parent[i] >= 0) m++;
    const index = new Uint32Array(2 * m);
    for (let i = 0, k = 0; i < n; i++) if (parent[i] >= 0) { index[k++] = parent[i]; index[k++] = i; }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('nid', new THREE.BufferAttribute(nid, 1));
    geo.setAttribute('dist', new THREE.BufferAttribute(dist, 1, true));
    geo.setIndex(new THREE.BufferAttribute(index, 1));
    this.skeletons = new THREE.LineSegments(geo, this._neuronMaterial(0, 0.12, NEURON_FRAG));
    this.skeletons.scale.setScalar(1 / anatomy.scale);
    this.skeletons.frustumCulled = false;
    this.skeletons.renderOrder = 3;
    this.group.add(this.skeletons);

    const soma = new Int16Array(buf, L.soma.offset, this.N * 3);
    const sg = new THREE.BufferGeometry();
    sg.setAttribute('position', new THREE.BufferAttribute(soma, 3));
    sg.setAttribute('nid', new THREE.BufferAttribute(Float32Array.from({ length: this.N }, (_, i) => i), 1));
    sg.setAttribute('dist', new THREE.BufferAttribute(new Float32Array(this.N), 1));
    this.somaPoints = new THREE.Points(sg, this._neuronMaterial(0.0035, 0.3, SOMA_FRAG));
    this.somaPoints.scale.setScalar(1 / anatomy.scale);
    this.somaPoints.frustumCulled = false;
    this.somaPoints.renderOrder = 4;
    this.group.add(this.somaPoints);
    this.somaPositions = soma;
    this.quantScale = anatomy.scale;
  }

  _buildSomaCloud(anatomy, buf) {
    const n = anatomy.somas.count;
    const pos = new Int16Array(buf, 0, n * 3);
    const reg = new Uint8Array(buf, n * 6, n);
    const col = new Float32Array(n * 3);
    const c = new THREE.Color();
    for (let i = 0; i < n; i++) {
      c.set(REGION_COLORS[reg[i]] ?? 0x555555);
      col.set([c.r, c.g, c.b], 3 * i);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    this.cloudMaterial = new THREE.ShaderMaterial({
      uniforms: { uSize: { value: 0.003 }, uStrength: { value: 0.07 }, uPx: this.px },
      vertexShader: /* glsl */ `
        uniform float uSize; uniform float uPx; attribute vec3 color; varying vec3 vC;
        void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vC = color; gl_Position = projectionMatrix * mv; gl_PointSize = max(1.0, uSize * uPx / max(-mv.z, 1e-3)); }`,
      fragmentShader: /* glsl */ `
        uniform float uStrength; varying vec3 vC;
        void main() { vec2 p = gl_PointCoord * 2.0 - 1.0; float r = dot(p, p); if (r > 1.0) discard; gl_FragColor = vec4(vC * uStrength * (1.0 - r), 1.0);
#include <tonemapping_fragment>
#include <colorspace_fragment> }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.cloud = new THREE.Points(geo, this.cloudMaterial);
    this.cloud.scale.setScalar(1 / anatomy.scale);
    this.cloud.frustumCulled = false;
    this.cloud.renderOrder = 2;
    this.group.add(this.cloud);
  }

  // DAN instance names carry their compartments, e.g. PAM01(y5)_L or PPL105(a'2a2)_R.
  _buildCompartmentMap() {
    const inst = this.circuit.neurons.instance;
    const lobe = { y: 'g', a: 'a', "a'": "a'", B: 'b', "B'": "b'", b: 'b', "b'": "b'" };
    this.compartmentDans = {};
    for (const gname of ['PAM', 'PPL1']) {
      const g = this.groups.find((x) => x.name === gname);
      for (let i = g.offset; i < g.offset + g.count; i++) {
        const m = inst[i].match(/\(([^)]*)\)_([LR])$/);
        if (!m) continue;
        const primary = m[1].split('<')[0];
        for (const [, l, k] of primary.matchAll(/(y|a'|a|B'|B|b'|b)(\d)/g)) {
          const key = `${lobe[l]}${k}(${m[2]})`;
          if (this.meshes[key]) (this.compartmentDans[key] ??= []).push(i);
        }
      }
    }
  }

  setViewport(heightPx, fovDeg) {
    this.px.value = heightPx / (2 * Math.tan((fovDeg * Math.PI) / 360));
  }

  setGroupVisible(name, on) {
    const gi = this.groups.findIndex((g) => g.name === name);
    if (gi >= 0) this.groupOn[gi] = on ? 1 : 0;
  }

  setContext({ cloud = true, shells = true, neuropils = true } = {}) {
    this.cloud.visible = cloud;
    for (const m of Object.values(this.meshes)) {
      if (m.userData.kind === 'shell') m.visible = shells;
      if (m.userData.kind === 'neuropil') m.visible = neuropils;
    }
  }

  // activity: Float32Array(N) in 0..1; glom: glomerular input per glomerulus (0..1)
  update(dt, activity, glom, rand = Math.random) {
    this.time += dt;
    const a = this.act;
    const decay = Math.exp(-dt / 0.09);
    const smooth = Math.min(1, dt / 0.06);
    for (let i = 0; i < this.N; i++) {
      const r = activity[i];
      a[4 * i] += (r - a[4 * i]) * smooth;
      a[4 * i + 1] *= decay;
      a[4 * i + 2] += dt;
      if (rand() < 1 - Math.exp(-r * this.maxRate[i] * dt)) {
        a[4 * i + 1] = 1;
        a[4 * i + 2] = 0;
      }
    }
    this.actTex.needsUpdate = true;
    for (const [gi, m] of this.glomMeshes) {
      const u = m.material.uniforms.uGlow;
      u.value += (Math.min(1.5, 1.6 * glom[gi]) - u.value) * smooth;
    }
    for (const [key, dans] of Object.entries(this.compartmentDans)) {
      let s = 0;
      for (const i of dans) s += Math.max(0, activity[i] - 0.1);
      const u = this.meshes[key].material.uniforms;
      const target = Math.min(1.5, 2.5 * s / dans.length);
      u.uGlow.value += (target - u.uGlow.value) * smooth;
      const isPam = this.groups[this.groupIndex[dans[0]]].name === 'PAM';
      u.uGlowColor.value.set(isPam ? GROUP_STYLE.PAM.color : GROUP_STYLE.PPL1.color);
    }
  }

  // Soma position of neuron i in the fly frame (mm).
  somaOf(i, target = new THREE.Vector3()) {
    const s = this.somaPositions, q = this.quantScale;
    return target.set(s[3 * i] / q, s[3 * i + 1] / q, s[3 * i + 2] / q);
  }
}
