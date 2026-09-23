// Visuals for world entities: rotting fruit, sundews (Drosera) and frogs. Sizes in millimetres.
import * as THREE from 'three';
import { mulberry32 } from '../sim/rng.js';

const UP = new THREE.Vector3(0, 1, 0);

function noise3(x, y, z) {
  return Math.sin(x * 1.7 + Math.sin(y * 2.3)) * 0.5 + Math.sin(y * 2.9 + Math.sin(z * 1.3)) * 0.3 + Math.sin(z * 3.7 + x) * 0.2;
}

function padUnder(radius, color = 0x3f6d2a) {
  const geo = new THREE.CircleGeometry(radius, 40, 0.15, Math.PI * 2 - 0.3).rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, roughness: 0.55, side: THREE.DoubleSide }));
  mesh.position.y = 0.5;
  mesh.receiveShadow = true;
  return mesh;
}

// ---------- fruit ----------

function fruitGeometry(r, seed) {
  const rand = mulberry32(seed);
  const geo = new THREE.SphereGeometry(r, 48, 32);
  const pos = geo.attributes.position;
  const col = new Float32Array(pos.count * 3);
  const o = [rand() * 10, rand() * 10, rand() * 10];
  const skin = new THREE.Color(0x5a1f3a), bruise = new THREE.Color(0x2a140f), mould = new THREE.Color(0x9fa889), flesh = new THREE.Color(0xc2733a);
  const c = new THREE.Color();
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i).normalize();
    const n = noise3(v.x * 3 + o[0], v.y * 3 + o[1], v.z * 3 + o[2]);
    const sag = v.y > 0 ? 1 - 0.12 * v.y * v.y : 1 - 0.25 * v.y * v.y; // squashed, heavier at the bottom
    pos.setXYZ(i, v.x * r * (1 + 0.06 * n), v.y * r * sag * (1 + 0.04 * n), v.z * r * (1 + 0.06 * n));
    c.copy(skin).lerp(bruise, THREE.MathUtils.clamp(0.5 + n, 0, 1) * 0.8);
    const spot = noise3(v.x * 9 + o[1], v.y * 9, v.z * 9 + o[2]);
    if (spot > 0.55) c.lerp(mould, (spot - 0.55) * 2.2);
    if (spot < -0.75) c.lerp(flesh, 0.7); // split skin
    col.set([c.r, c.g, c.b], 3 * i);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.computeVertexNormals();
  return geo;
}

class FruitView {
  constructor(e) {
    this.group = new THREE.Group();
    this.group.add(padUnder(e.radius * 2.2));
    this.body = new THREE.Mesh(
      fruitGeometry(e.radius, e.id * 13),
      new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.35, clearcoat: 0.7, clearcoatRoughness: 0.3, sheen: 0.6, sheenColor: 0x88607a }),
    );
    this.body.castShadow = true;
    this.body.position.y = e.radius * 0.8;
    this.group.add(this.body);
    this.r = e.radius;
  }

  update(e, dt, t) {
    const k = 0.55 + 0.45 * Math.max(0, e.bites) / 5; // shrinks as it is eaten
    this.body.scale.setScalar(k);
    this.body.position.y = this.r * 0.8 * k;
  }

  topY(e) {
    return this.body.position.y + this.r * this.body.scale.y * 0.8;
  }
}

// ---------- sundew ----------

class SundewView {
  constructor(e) {
    const rand = mulberry32(e.id * 31 + 7);
    this.group = new THREE.Group();
    const moss = new THREE.Mesh(
      new THREE.SphereGeometry(e.radius * 1.9, 24, 12, 0, Math.PI * 2, 0, Math.PI / 2).scale(1, 0.1, 1),
      new THREE.MeshStandardMaterial({ color: 0x2f4a1c, roughness: 1 }),
    );
    moss.receiveShadow = true;
    this.group.add(moss);
    this.base = 2.5;

    const nLeaves = 9;
    const perLeaf = 26;
    const leafGeo = new THREE.SphereGeometry(1, 16, 8).scale(3.2, 0.35, 2.4);
    const stalkGeo = new THREE.CylinderGeometry(0.35, 0.5, 1, 6).rotateZ(Math.PI / 2).translate(0.5, 0, 0);
    const leafMat = new THREE.MeshPhysicalMaterial({ color: 0x8a2a1c, roughness: 0.5, clearcoat: 0.4, sheen: 0.5, sheenColor: 0xff7a5a });
    const stalkMat = new THREE.MeshStandardMaterial({ color: 0x6d7a30, roughness: 0.6 });
    this.tentacles = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.08, 0.12, 1, 5).translate(0, 0.5, 0), new THREE.MeshStandardMaterial({ color: 0xc23a2c, roughness: 0.5 }), nLeaves * perLeaf);
    this.drops = new THREE.InstancedMesh(
      new THREE.SphereGeometry(0.28, 10, 8),
      new THREE.MeshPhysicalMaterial({ color: 0xffc2d2, roughness: 0.02, clearcoat: 1, transparent: true, opacity: 0.85, emissive: 0xff3d6e, emissiveIntensity: 0.6 }),
      nLeaves * perLeaf,
    );
    this.tentacleBase = [];
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3();
    let k = 0;
    for (let i = 0; i < nLeaves; i++) {
      const a = (i / nLeaves) * Math.PI * 2 + rand() * 0.3;
      const reach = e.radius * (0.55 + 0.25 * rand());
      const lift = 0.25 + 0.2 * rand();
      const leaf = new THREE.Group();
      leaf.rotation.set(0, a, lift);
      leaf.position.y = this.base;
      const stalk = new THREE.Mesh(stalkGeo, stalkMat);
      stalk.scale.set(reach - 3, 1, 1);
      const blade = new THREE.Mesh(leafGeo, leafMat);
      blade.position.x = reach;
      blade.castShadow = true;
      leaf.add(stalk, blade);
      this.group.add(leaf);
      leaf.updateMatrix();
      for (let j = 0; j < perLeaf; j++) {
        // tentacles fan out from the leaf blade, longer at the rim
        const u = rand() * Math.PI * 2, rr = Math.sqrt(rand());
        const local = new THREE.Vector3(reach + 3.0 * rr * Math.cos(u), 0.3, 2.2 * rr * Math.sin(u));
        const dir = new THREE.Vector3(0.6 * rr * Math.cos(u), 1, 0.6 * rr * Math.sin(u)).normalize();
        const len = 1.2 + 2.2 * rr;
        this.tentacleBase.push({ local, dir, len, leaf: leaf.matrix.clone() });
        p.copy(local).applyMatrix4(leaf.matrix);
        q.setFromUnitVectors(UP, dir.clone().transformDirection(leaf.matrix));
        s.set(1, len, 1);
        this.tentacles.setMatrixAt(k, m.compose(p, q, s));
        k++;
      }
    }
    this.tentacles.castShadow = true;
    this.group.add(this.tentacles, this.drops);
    this.curl = 0;
    this._updateTentacles(0);
  }

  _updateTentacles(curl) {
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(), tip = new THREE.Vector3();
    this.tentacleBase.forEach((b, i) => {
      const dir = b.dir.clone().lerp(new THREE.Vector3(-1, 0.4, 0), curl * 0.7).normalize(); // bend towards the leaf centre
      p.copy(b.local).applyMatrix4(b.leaf);
      const wdir = dir.transformDirection(b.leaf);
      q.setFromUnitVectors(UP, wdir);
      s.set(1, b.len, 1);
      this.tentacles.setMatrixAt(i, m.compose(p, q, s));
      tip.copy(p).addScaledVector(wdir, b.len);
      this.drops.setMatrixAt(i, m.compose(tip, q.identity(), s.set(1, 1, 1)));
    });
    this.tentacles.instanceMatrix.needsUpdate = true;
    this.drops.instanceMatrix.needsUpdate = true;
  }

  update(e, dt, t) {
    const target = e.trapped ? 1 : 0;
    const prev = this.curl;
    this.curl += (target - this.curl) * Math.min(1, dt / 0.8);
    if (Math.abs(prev - this.curl) > 1e-3) this._updateTentacles(this.curl);
    this.drops.material.emissiveIntensity = 0.45 + 0.25 * Math.sin(t * 2 + e.phase);
  }

  topY() {
    return this.base + 2; // standing on the leaf blades, among the tentacles
  }
}

// ---------- frog ----------

function frogGeometry() {
  const geo = new THREE.SphereGeometry(1, 48, 32);
  const pos = geo.attributes.position;
  const col = new Float32Array(pos.count * 3);
  const back = new THREE.Color(0x3f7a2c), belly = new THREE.Color(0xd9d09a), spot = new THREE.Color(0x1f3a14);
  const c = new THREE.Color(), v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    // pear-shaped body: wide hips at -x, head at +x; flat belly
    const x = v.x, y = v.y, z = v.z;
    const width = 1 - 0.18 * x;
    pos.setXYZ(i, x * 1.35, y * (y < 0 ? 0.55 : 0.8) * width, z * 0.95 * width);
    c.copy(back).lerp(belly, THREE.MathUtils.smoothstep(-y, -0.1, 0.4));
    if (y > 0 && noise3(x * 5, y * 5, z * 5) > 0.6) c.lerp(spot, 0.8);
    col.set([c.r, c.g, c.b], 3 * i);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
  geo.computeVertexNormals();
  return geo;
}

class FrogView {
  constructor(e) {
    this.group = new THREE.Group();
    this.group.add(padUnder(e.radius * 2.4, 0x355f25));
    const R = e.radius;
    this.frog = new THREE.Group();
    this.frog.position.y = 1;
    const skin = new THREE.MeshPhysicalMaterial({ vertexColors: true, roughness: 0.35, clearcoat: 0.9, clearcoatRoughness: 0.2 });
    const body = new THREE.Mesh(frogGeometry(), skin);
    body.scale.setScalar(R);
    body.position.y = R * 0.45;
    body.castShadow = true;
    this.frog.add(body);

    const eyeMat = new THREE.MeshPhysicalMaterial({ color: 0xd4a52a, roughness: 0.1, clearcoat: 1 });
    const pupilMat = new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.2 });
    for (const s of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(R * 0.2, 20, 16), eyeMat);
      eye.position.set(R * 0.75, R * 0.95, s * R * 0.42);
      const pupil = new THREE.Mesh(new THREE.SphereGeometry(R * 0.2, 16, 12).scale(0.35, 0.18, 0.6), pupilMat);
      pupil.position.set(R * 0.14, 0.02 * R, s * R * 0.04);
      eye.add(pupil);
      this.frog.add(eye);
      // folded hind leg and front leg
      const hind = new THREE.Mesh(new THREE.CapsuleGeometry(R * 0.22, R * 0.9, 6, 12), skin);
      hind.position.set(-R * 0.75, R * 0.28, s * R * 0.75);
      hind.rotation.set(0, s * 0.5, Math.PI / 2.4);
      const fore = new THREE.Mesh(new THREE.CapsuleGeometry(R * 0.1, R * 0.5, 4, 8), skin);
      fore.position.set(R * 0.7, R * 0.2, s * R * 0.55);
      fore.rotation.set(s * 0.3, 0, -0.3);
      hind.castShadow = fore.castShadow = true;
      this.frog.add(hind, fore);
    }
    this.throat = new THREE.Mesh(new THREE.SphereGeometry(R * 0.32, 20, 14), new THREE.MeshPhysicalMaterial({ color: 0xe8dca0, roughness: 0.3, clearcoat: 0.8 }));
    this.throat.position.set(R * 0.95, R * 0.25, 0);
    this.frog.add(this.throat);
    this.mouth = new THREE.Vector3(R * 1.3, R * 0.55, 0);

    this.tongueMat = new THREE.MeshPhysicalMaterial({ color: 0xe86a7a, roughness: 0.3, clearcoat: 1 });
    this.tongue = null;
    this.group.add(this.frog);
    this.R = R;
  }

  update(e, dt, t, fly) {
    this.frog.rotation.y = e.heading;
    const croak = Math.max(0, Math.sin(t * 1.3 + e.phase));
    this.throat.scale.setScalar(1 + 0.7 * croak ** 8);
    this.frog.position.y = 1 + (e.jump > 0 ? 40 * Math.sin((1 - e.jump / 0.6) * Math.PI) : 0);
    // tongue: out and back within the 0.35 s strike
    if (this.tongue) {
      this.group.remove(this.tongue);
      this.tongue.geometry.dispose();
      this.tongue = null;
    }
    if (e.strike > 0 && e.strikeAt) {
      const phase = 1 - e.strike / 0.35;
      const reach = Math.sin(Math.min(1, phase) * Math.PI);
      const start = this.mouth.clone().applyAxisAngle(UP, e.heading).add(this.frog.position);
      const target = new THREE.Vector3(e.strikeAt.x - e.x, e.strikeAt.y, e.strikeAt.z - e.z);
      const end = start.clone().lerp(target, reach);
      const mid = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, 6 * reach, 0));
      const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
      this.tongue = new THREE.Mesh(new THREE.TubeGeometry(curve, 16, 1.4, 8), this.tongueMat);
      this.group.add(this.tongue);
    }
  }

  topY() {
    return this.R;
  }
}

const VIEWS = { fruit: FruitView, sundew: SundewView, frog: FrogView };

export class Props {
  constructor(scene) {
    this.group = new THREE.Group();
    scene.add(this.group);
    this.views = new Map();
  }

  sync(world, dt) {
    const alive = new Set();
    for (const e of world.entities) {
      alive.add(e.id);
      let v = this.views.get(e.id);
      if (!v) {
        v = new VIEWS[e.kind](e);
        v.group.userData.entityId = e.id;
        this.group.add(v.group);
        this.views.set(e.id, v);
      }
      v.group.position.set(e.x, 0, e.z);
      v.update(e, dt, world.t, world.fly);
    }
    for (const [id, v] of this.views) {
      if (alive.has(id)) continue;
      this.group.remove(v.group);
      v.group.traverse((o) => { o.geometry?.dispose(); });
      this.views.delete(id);
    }
  }

  // Height of the surface the fly stands on when perched on entity id.
  perchHeight(world, id) {
    const e = world.entities.find((x) => x.id === id);
    const v = this.views.get(id);
    return e && v ? v.topY(e) : 0;
  }
}
