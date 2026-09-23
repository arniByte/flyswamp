// Camera modes: follow the fly, x-ray close-up, brain only, overview map. Orbit controls stay live in all of them.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export const MODES = {
  follow: { dist: 30, min: 6, max: 420, near: 0.4, far: 6000, dir: [-0.8, 0.45, 0.55] },
  xray: { dist: 5, min: 2.5, max: 40, near: 0.08, far: 6000, dir: [0.45, 0.75, 0.5] },
  brain: { dist: 1.45, min: 0.6, max: 8, near: 0.02, far: 400, dir: [0.9, 0.35, 0.3] },
  map: { dist: 720, min: 120, max: 1600, near: 2, far: 8000, dir: [0.0, 0.8, 0.6] },
};

export class CameraRig {
  constructor(camera, dom) {
    this.camera = camera;
    this.controls = new OrbitControls(camera, dom);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.zoomSpeed = 0.8;
    this.mode = 'follow';
    this.lastFocus = new THREE.Vector3();
    this.blend = 0;
    this._v = new THREE.Vector3();
  }

  setMode(mode, focus) {
    const m = MODES[mode];
    this.mode = mode;
    this.blend = 1; // seconds of transition left
    this.controls.minDistance = m.min * 0.5;
    this.controls.maxDistance = m.max;
    this.camera.near = m.near;
    this.camera.far = m.far;
    this.camera.updateProjectionMatrix();
    // keep the current viewing direction unless it is far from the mode's preferred one
    const cur = this._v.subVectors(this.camera.position, this.controls.target).normalize();
    const pref = new THREE.Vector3(...m.dir).normalize();
    this.dir = cur.dot(pref) > 0.2 && mode !== 'map' ? cur.clone() : pref;
    this.startTarget = this.controls.target.clone();
    this.startDist = this.camera.position.distanceTo(this.controls.target);
    this.lastFocus.copy(focus);
  }

  update(dt, focus) {
    const m = MODES[this.mode];
    const c = this.controls;
    if (this.blend > 0) {
      this.blend = Math.max(0, this.blend - dt);
      const k = 1 - this.blend;
      const e = k * k * (3 - 2 * k);
      c.target.lerpVectors(this.startTarget, focus, e);
      const dist = THREE.MathUtils.lerp(this.startDist, m.dist, e);
      this.camera.position.copy(c.target).addScaledVector(this.dir, dist);
      c.minDistance = Math.min(m.min, dist);
    } else {
      // carry the camera along with the focus so orbiting stays relative to the fly
      this._v.subVectors(focus, this.lastFocus);
      this.camera.position.add(this._v);
      c.target.copy(focus);
      c.minDistance = m.min;
    }
    this.lastFocus.copy(focus);
    c.update();
  }
}
