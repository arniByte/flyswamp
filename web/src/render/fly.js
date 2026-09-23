// Anatomical fly (flybody mesh) with a joint rig, procedural motion and an x-ray material swap.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const MATERIALS = {
  cuticle: () => new THREE.MeshPhysicalMaterial({ color: 0xa8662f, roughness: 0.45, clearcoat: 0.6, clearcoatRoughness: 0.35, sheen: 0.4, sheenColor: 0xffd29a }),
  pale: () => new THREE.MeshPhysicalMaterial({ color: 0xd9b27a, roughness: 0.5, clearcoat: 0.3 }),
  black: () => new THREE.MeshPhysicalMaterial({ color: 0x15100c, roughness: 0.35, clearcoat: 0.8 }),
  eye: () => new THREE.MeshPhysicalMaterial({ color: 0x9a0a05, roughness: 0.25, clearcoat: 1, clearcoatRoughness: 0.15, sheen: 1, sheenColor: 0xff6040, iridescence: 0.25 }),
  ocelli: () => new THREE.MeshPhysicalMaterial({ color: 0x2a1408, roughness: 0.2, clearcoat: 1 }),
  vein: () => new THREE.MeshPhysicalMaterial({ color: 0x3a2412, roughness: 0.5, transparent: true, opacity: 0.85 }),
  membrane: () => new THREE.MeshPhysicalMaterial({
    color: 0xdfe8ff, roughness: 0.15, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false,
    iridescence: 1, iridescenceIOR: 1.6, iridescenceThicknessRange: [180, 620],
  }),
};

// Fresnel rim shader for x-ray: the cuticle turns into faint glass so the CNS inside is visible.
function xrayMaterial(color) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uStrength: { value: 1 } },
    vertexShader: /* glsl */ `
      varying vec3 vN; varying vec3 vV;
      void main() {
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
        vN = normalize(normalMatrix * normal); vV = normalize(-mv.xyz);
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor; uniform float uStrength; varying vec3 vN; varying vec3 vV;
      void main() {
        float nl = length(vN);
        float f = pow(1.0 - (nl > 1e-6 ? abs(dot(vN / nl, normalize(vV))) : 0.0), 2.2);
        gl_FragColor = vec4(uColor * (0.015 + 0.55 * f) * uStrength, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  });
}

export async function loadFly(url, rigUrl) {
  const [gltf, rig] = await Promise.all([new GLTFLoader().loadAsync(url), fetch(rigUrl).then((r) => r.json())]);
  return new Fly(gltf.scene, rig);
}

const LEGS = ['T1', 'T2', 'T3'].flatMap((seg) => ['left', 'right'].map((side) => ({ seg, side })));
// Tripod gait: T1/T3 of one side move with T2 of the other.
const TRIPOD = { T1left: 0, T3left: 0, T2right: 0, T1right: Math.PI, T3right: Math.PI, T2left: Math.PI };
const WING_REST = { yaw: 1.5, roll: 0.7, pitch: -1 }; // folded over the abdomen (MJCF spring references)
const STROKE = { center: -0.15, amp: 1.15 };

// Translucent stroke-plane "blur" shown while the wings beat; denser near stroke reversal,
// where a real wing spends most of its time.
function strokeFan(baseAngle) {
  const geo = new THREE.RingGeometry(0.25, 2.55, 40, 1, baseAngle + STROKE.center - STROKE.amp, 2 * STROKE.amp);
  const mat = new THREE.ShaderMaterial({
    uniforms: { uOpacity: { value: 0 }, uA0: { value: baseAngle + STROKE.center }, uAmp: { value: STROKE.amp } },
    vertexShader: /* glsl */ `varying vec2 vP; void main() { vP = position.xy; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform float uOpacity; uniform float uA0; uniform float uAmp; varying vec2 vP;
      void main() {
        float a = atan(vP.y, vP.x) - uA0;
        a = mod(a + 3.14159265, 6.2831853) - 3.14159265;
        float x = clamp(abs(a) / uAmp, 0.0, 0.985);
        float dwell = 0.35 / sqrt(1.0 - x * x);
        float r = length(vP) / 2.55;
        float alpha = uOpacity * min(dwell, 1.4) * smoothstep(0.1, 0.35, r) * (1.0 - smoothstep(0.85, 1.0, r));
        gl_FragColor = vec4(vec3(0.78, 0.84, 0.9), alpha);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
  });
  const m = new THREE.Mesh(geo, mat);
  m.renderOrder = 3;
  return m;
}

export class Fly {
  constructor(scene, rig) {
    // MuJoCo frame (x forward, y left, z up) → three.js (y up). Forward stays +x, left becomes -z.
    this.root = new THREE.Group();
    this.model = scene;
    this.model.rotation.x = -Math.PI / 2;
    this.root.add(this.model);

    this.bodies = {};
    this.rest = {};
    this.meshes = [];
    this.model.traverse((o) => {
      if (o.isMesh) {
        o.userData.kind = o.material.name;
        o.userData.normal = MATERIALS[o.material.name]?.() ?? o.material;
        o.userData.xray = xrayMaterial(o.material.name === 'eye' ? 0xb8402a : o.material.name === 'membrane' ? 0x4a8fc0 : 0x3f8fd8);
        o.material = o.userData.normal;
        o.castShadow = o.userData.kind !== 'membrane';
        o.renderOrder = o.userData.kind === 'membrane' ? 2 : 0;
        this.meshes.push(o);
      } else if (o.name) {
        this.bodies[o.name] = o;
        this.rest[o.name] = o.quaternion.clone();
      }
    });

    this.joints = {};
    this.bodyJoints = {};
    for (const j of rig.joints) {
      this.joints[j.name] = { ...j, axis: new THREE.Vector3(...j.axis), q: 0 };
      (this.bodyJoints[j.body] ??= []).push(j.name);
    }
    this._q = new THREE.Quaternion();
    this.xray = 0;
    this.gait = 0;
    this._buildFans();
  }

  _buildFans() {
    this.fans = [];
    for (const side of ['left', 'right']) {
      const wing = this.bodies[`wing_${side}`];
      const blade = this.meshes.find((m) => m.name === `wing_${side}_brown`);
      if (!wing || !blade) continue;
      // direction of the wing blade in its hinge frame gives the stroke sector
      blade.geometry.computeBoundingBox();
      const c = blade.geometry.boundingBox.getCenter(new THREE.Vector3()).applyQuaternion(blade.quaternion).add(blade.position);
      const fan = strokeFan(Math.atan2(c.y, c.x));
      fan.position.copy(wing.position);
      fan.quaternion.copy(this.rest[`wing_${side}`]);
      wing.parent.add(fan);
      this.fans.push(fan);
    }
  }

  // Procedural motion for the behavioural state coming from the world simulation.
  animate(dt, t, state, motor) {
    const flying = state === 'fly' || state === 'escape' || state === 'land';
    const fanTarget = flying && !this.xray ? 0.55 : 0;
    for (const f of this.fans) {
      const u = f.material.uniforms.uOpacity;
      u.value += (fanTarget - u.value) * Math.min(1, dt / 0.08);
      f.visible = u.value > 0.01;
    }
    for (const side of ['left', 'right']) {
      if (flying) {
        // ~200 Hz wing beat sampled at frame rate: a fresh random phase each frame reads as a blurred beat
        const ph = Math.random() * Math.PI * 2;
        this.set(`wing_yaw_${side}`, STROKE.center + STROKE.amp * Math.sin(ph));
        this.set(`wing_roll_${side}`, 0.15 * Math.cos(ph));
        this.set(`wing_pitch_${side}`, 0.9 * Math.cos(ph));
        this.set(`haltere_${side}`, 0.9 * Math.sin(ph + Math.PI));
      } else if (state === 'trapped' && Math.sin(t * 5) > 0.2) {
        const ph = Math.random() * Math.PI * 2;
        this.set(`wing_yaw_${side}`, 0.6 + 0.6 * Math.sin(ph));
        this.set(`wing_roll_${side}`, 0.5);
        this.set(`wing_pitch_${side}`, 0.2);
      } else {
        this.set(`wing_yaw_${side}`, WING_REST.yaw);
        this.set(`wing_roll_${side}`, WING_REST.roll);
        this.set(`wing_pitch_${side}`, WING_REST.pitch);
      }
    }

    const walk = state === 'feed' ? 0.25 : 0;
    const struggle = state === 'trapped' ? 1 : 0;
    this.gait += dt * (walk ? 6 : 0);
    for (const { seg, side } of LEGS) {
      const k = `${seg}_${side}`;
      if (flying) {
        this.set(`coxa_${k}`, 0.5);
        this.set(`femur_${k}`, 1.4);
        this.set(`tibia_${k}`, -1.1);
        this.set(`tarsus_${k}`, 0.3);
      } else if (struggle) {
        const r = Math.sin(t * (14 + 3 * LEGS.findIndex((l) => l.seg === seg && l.side === side)) + seg.length);
        this.set(`coxa_${k}`, 0.3 + 0.45 * r);
        this.set(`femur_${k}`, 0.5 + 0.5 * r);
        this.set(`tibia_${k}`, -0.2 + 0.6 * Math.cos(t * 13));
        this.set(`tarsus_${k}`, 0.2);
      } else {
        const ph = this.gait + TRIPOD[`${seg}${side}`];
        const swing = walk * Math.sin(ph);
        this.set(`coxa_${k}`, swing * 0.8);
        this.set(`femur_${k}`, Math.max(0, walk * 0.6 * Math.cos(ph)));
        this.set(`tibia_${k}`, 0);
        this.set(`tarsus_${k}`, 0);
      }
    }

    const feeding = state === 'feed';
    const pe = feeding ? 0.5 + 0.5 * Math.sin(t * 5) : 0; // proboscis extension, pumping while feeding
    this.set('rostrum', -1.0 * pe);
    this.set('haustellum', -1.2 * pe);
    this.set('head_abduct', 0.15 * (motor.turn ?? 0));
    this.set('head', feeding ? -0.2 : 0);
    this.set('abdomen', 0.05 * Math.sin(t * 2.2));
    for (const side of ['left', 'right']) this.set(`antenna_${side}`, 0.08 * Math.sin(t * 3.1 + (side === 'left' ? 0 : 1.7)));
    this.applyPose();
  }

  set(name, angle) {
    const j = this.joints[name];
    if (!j) return;
    j.q = j.range ? Math.min(j.range[1], Math.max(j.range[0], angle)) : angle;
  }

  // MuJoCo composes a body's hinge joints in definition order after its rest rotation.
  applyPose() {
    for (const [body, names] of Object.entries(this.bodyJoints)) {
      const node = this.bodies[body];
      if (!node) continue;
      node.quaternion.copy(this.rest[body]);
      for (const n of names) {
        const j = this.joints[n];
        if (j.q) node.quaternion.multiply(this._q.setFromAxisAngle(j.axis, j.q));
      }
    }
  }

  setXray(on, strength = 1) {
    this.xray = on ? 1 : 0;
    for (const m of this.meshes) {
      m.material = on ? m.userData.xray : m.userData.normal;
      if (on) m.material.uniforms.uStrength.value = strength;
    }
  }

  worldOf(bodyName, target = new THREE.Vector3()) {
    return this.bodies[bodyName].getWorldPosition(target);
  }
}
