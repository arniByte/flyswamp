// Odor made visible: particles released by each source drift downwind and fade.
// Purely illustrative; the fly smells the analytic plume in world.js, not these particles.
import * as THREE from 'three';
import { ODOR_COLORS } from '../sim/odors.js';

const MAX = 2400;
const LIFE = 7; // s

export class Plumes {
  constructor(scene) {
    this.pos = new Float32Array(MAX * 3);
    this.col = new Float32Array(MAX * 3);
    this.age = new Float32Array(MAX).fill(LIFE);
    this.vel = new Float32Array(MAX * 3);
    this.seed = new Float32Array(MAX);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(this.pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(this.col, 3));
    geo.setAttribute('age', new THREE.BufferAttribute(this.age, 1));
    this.material = new THREE.ShaderMaterial({
      uniforms: { uPx: { value: 600 }, uLife: { value: LIFE }, uOpacity: { value: 1 } },
      vertexShader: /* glsl */ `
        uniform float uPx; uniform float uLife; attribute vec3 color; attribute float age; varying vec3 vC; varying float vA;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          float k = age / uLife;
          vC = color; vA = smoothstep(0.0, 0.08, k) * (1.0 - k);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = min(48.0, (1.5 + 7.0 * k) * uPx / max(-mv.z, 1.0)); // puff diameter in mm
        }`,
      fragmentShader: /* glsl */ `
        uniform float uOpacity; varying vec3 vC; varying float vA;
        void main() { vec2 p = gl_PointCoord * 2.0 - 1.0; float r = dot(p, p); if (r > 1.0) discard;
          gl_FragColor = vec4(vC * vA * 0.12 * uOpacity * (1.0 - r), 1.0);
#include <tonemapping_fragment>
#include <colorspace_fragment> }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geo, this.material);
    this.points.frustumCulled = false;
    scene.add(this.points);
    this.next = 0;
    this.carry = new Map();
    this.enabled = true;
  }

  update(world, dt) {
    if (!this.enabled) return;
    const wx = Math.cos(world.wind.angle), wz = Math.sin(world.wind.angle);
    const speed = 38;
    const c = new THREE.Color();
    for (const e of world.entities) {
      const rate = 26 * e.strength; // particles per second
      let acc = (this.carry.get(e.id) ?? 0) + rate * dt;
      c.set(ODOR_COLORS[e.kind]);
      while (acc >= 1) {
        acc -= 1;
        const i = this.next;
        this.next = (this.next + 1) % MAX;
        const a = Math.random() * Math.PI * 2, r = e.radius * Math.random();
        this.pos.set([e.x + r * Math.cos(a), 2 + 12 * Math.random(), e.z + r * Math.sin(a)], 3 * i);
        this.vel.set([speed * wx, 2 + 3 * Math.random(), speed * wz], 3 * i);
        this.col.set([c.r, c.g, c.b], 3 * i);
        this.age[i] = 0;
        this.seed[i] = Math.random() * 100;
      }
      this.carry.set(e.id, acc);
    }
    const t = world.t;
    for (let i = 0; i < MAX; i++) {
      if (this.age[i] >= LIFE) continue;
      this.age[i] += dt;
      const s = this.seed[i];
      // meander: slow lateral wobble grows with age, as filaments spread
      const wob = 14 * Math.sin(t * 0.8 + s) * (this.age[i] / LIFE);
      this.pos[3 * i] += (this.vel[3 * i] - wob * wz * 0.2) * dt;
      this.pos[3 * i + 1] += this.vel[3 * i + 1] * dt * 0.3;
      this.pos[3 * i + 2] += (this.vel[3 * i + 2] + wob * wx * 0.2) * dt;
    }
    const g = this.points.geometry.attributes;
    g.position.needsUpdate = g.color.needsUpdate = g.age.needsUpdate = true;
  }

  setPixelScale(px) {
    this.material.uniforms.uPx.value = px;
  }
}
