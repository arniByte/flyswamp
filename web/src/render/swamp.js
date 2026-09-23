// Swamp at dusk, insect scale (millimetres): sky dome, reflective water, lily pads, reeds, fireflies.
// Blue-hour palette: dark enough for the fireflies and the x-ray to read, light enough to see the swamp.
import * as THREE from 'three';
import { mulberry32 } from '../sim/rng.js';

const SKY = { zenith: 0x1c2e4c, horizon: 0x6e8f98, glow: 0xa6c6bc };
export const FOG = 0x34504e;

function skyDome() {
  const mat = new THREE.ShaderMaterial({
    uniforms: { uZenith: { value: new THREE.Color(SKY.zenith) }, uHorizon: { value: new THREE.Color(SKY.horizon) }, uGlow: { value: new THREE.Color(SKY.glow) }, uMoon: { value: new THREE.Vector3(-0.45, 0.35, -0.82).normalize() } },
    vertexShader: /* glsl */ `varying vec3 vDir; void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: /* glsl */ `
      uniform vec3 uZenith; uniform vec3 uHorizon; uniform vec3 uGlow; uniform vec3 uMoon; varying vec3 vDir;
      float hash(vec3 p) { return fract(sin(dot(p, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }
      void main() {
        float h = clamp(vDir.y, -0.2, 1.0);
        vec3 c = mix(uHorizon, uZenith, smoothstep(0.0, 0.55, h));
        float m = max(dot(normalize(vDir), uMoon), 0.0);
        c += uGlow * pow(m, 18.0) * 0.8 + vec3(0.9, 0.95, 1.0) * smoothstep(0.9985, 0.9992, m);
        vec3 q = floor(vDir * 260.0);
        float star = step(0.9965, hash(q)) * smoothstep(0.08, 0.4, h);
        c += vec3(star * 0.7);
        gl_FragColor = vec4(c, 1.0);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`,
    side: THREE.BackSide, depthWrite: false, fog: false,
  });
  const m = new THREE.Mesh(new THREE.SphereGeometry(3000, 48, 24), mat);
  m.renderOrder = -1;
  return m;
}

// Tileable ripple normal map drawn on a canvas (sum of random sine waves → height → normals).
function rippleNormalMap(size = 256, seed = 7) {
  const rand = mulberry32(seed);
  const waves = Array.from({ length: 14 }, () => ({ kx: Math.round((rand() - 0.5) * 12), ky: Math.round((rand() - 0.5) * 12), p: rand() * 6.28, a: 0.3 + rand() }));
  const h = new Float32Array(size * size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let v = 0;
    for (const w of waves) v += w.a * Math.sin((2 * Math.PI * (w.kx * x + w.ky * y)) / size + w.p);
    h[y * size + x] = v;
  }
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(size, size);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    const dx = h[y * size + ((x + 1) % size)] - h[y * size + ((x - 1 + size) % size)];
    const dy = h[((y + 1) % size) * size + x] - h[((y - 1 + size) % size) * size + x];
    const n = new THREE.Vector3(-dx * 0.18, -dy * 0.18, 1).normalize();
    const i = 4 * (y * size + x);
    img.data[i] = (n.x * 0.5 + 0.5) * 255;
    img.data[i + 1] = (n.y * 0.5 + 0.5) * 255;
    img.data[i + 2] = (n.z * 0.5 + 0.5) * 255;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  return tex;
}

function padTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(128, 128, 10, 128, 128, 128);
  grd.addColorStop(0, '#5f8f3e');
  grd.addColorStop(0.7, '#3f6d2a');
  grd.addColorStop(1, '#2b4d1f');
  g.fillStyle = grd;
  g.fillRect(0, 0, 256, 256);
  g.strokeStyle = 'rgba(160,200,110,0.35)';
  g.lineWidth = 2;
  for (let i = 0; i < 26; i++) {
    const a = (i / 26) * Math.PI * 2;
    g.beginPath();
    g.moveTo(128, 128);
    g.quadraticCurveTo(128 + 60 * Math.cos(a + 0.1), 128 + 60 * Math.sin(a + 0.1), 128 + 126 * Math.cos(a), 128 + 126 * Math.sin(a));
    g.stroke();
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// A lily pad: disc with the characteristic notch, rim slightly curled up.
function padGeometry() {
  const geo = new THREE.CircleGeometry(1, 48, 0.12, Math.PI * 2 - 0.24);
  const pos = geo.attributes.position;
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i);
    const r = Math.hypot(x, y);
    pos.setZ(i, 0.06 * r * r * r);
  }
  geo.rotateX(-Math.PI / 2);
  geo.computeVertexNormals();
  return geo;
}

export class Swamp {
  constructor(scene, renderer, { radius = 320, seed = 3 } = {}) {
    this.scene = scene;
    this.group = new THREE.Group();
    scene.add(this.group);
    const rand = mulberry32(seed);
    scene.fog = new THREE.FogExp2(FOG, 0.0016);
    scene.background = new THREE.Color(FOG);

    this.sky = skyDome();
    this.group.add(this.sky);

    // Environment map from the sky, so water and wet surfaces reflect it.
    const envScene = new THREE.Scene();
    envScene.add(skyDome());
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(envScene, 0.02).texture;
    scene.environmentIntensity = 0.9;

    const hemi = new THREE.HemisphereLight(0x9fb8c8, 0x3a3020, 1.6);
    const moon = new THREE.DirectionalLight(0xdfe8ff, 2.2);
    moon.position.set(-450, 380, -800);
    moon.castShadow = true;
    moon.shadow.mapSize.set(2048, 2048);
    const sc = moon.shadow.camera;
    sc.left = sc.bottom = -90;
    sc.right = sc.top = 90;
    sc.near = 10;
    sc.far = 2400;
    moon.shadow.bias = -0.0004;
    moon.shadow.normalBias = 0.3;
    this.moon = moon;
    this.group.add(hemi, moon, moon.target);
    const warm = new THREE.PointLight(0xffa860, 900, 420, 1.6); // lantern-like warm fill from one side
    warm.position.set(260, 90, 180);
    this.group.add(warm);

    // Water
    this.normalMap = rippleNormalMap();
    this.normalMap.repeat.set(40, 40);
    const water = new THREE.Mesh(
      new THREE.CircleGeometry(2600, 96).rotateX(-Math.PI / 2),
      new THREE.MeshPhysicalMaterial({
        color: 0x2e4a40, roughness: 0.16, metalness: 0.0, normalMap: this.normalMap, normalScale: new THREE.Vector2(0.35, 0.35),
        clearcoat: 1, clearcoatRoughness: 0.08, envMapIntensity: 1.2,
      }),
    );
    water.receiveShadow = true;
    water.name = 'water';
    this.water = water;
    this.group.add(water);

    // Lily pads (instanced), kept off the spawn spots handled by props
    const padTex = padTexture();
    const padMat = new THREE.MeshStandardMaterial({ map: padTex, roughness: 0.55, metalness: 0, side: THREE.DoubleSide });
    const nPads = 70;
    this.pads = new THREE.InstancedMesh(padGeometry(), padMat, nPads);
    this.pads.receiveShadow = true;
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
    for (let i = 0; i < nPads; i++) {
      const r = 40 + (radius + 60) * Math.sqrt(rand());
      const a = rand() * Math.PI * 2;
      const size = 18 + 40 * rand() * rand();
      p.set(r * Math.cos(a), 0.4 + rand() * 0.3, r * Math.sin(a));
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), rand() * Math.PI * 2);
      s.set(size, size, size);
      this.pads.setMatrixAt(i, m.compose(p, q, s));
      this.pads.setColorAt(i, new THREE.Color().setHSL(0.24 + 0.05 * rand(), 0.45, 0.35 + 0.2 * rand()));
    }
    this.group.add(this.pads);

    // Reeds and cattails ringing the pond, swaying in the wind
    this.time = { value: 0 };
    const reedGeo = new THREE.CylinderGeometry(0.6, 2.2, 1, 5, 6).translate(0, 0.5, 0);
    const reedMat = new THREE.MeshStandardMaterial({ color: 0x3b5a2a, roughness: 0.8 });
    // bend(h) is the sideways offset in mm at local height h: quadratic along a reed, constant for its head
    const sway = (bend) => (shader) => {
      shader.uniforms.uTime = this.time;
      shader.vertexShader = 'uniform float uTime;\n' + shader.vertexShader.replace('#include <begin_vertex>', `
        #include <begin_vertex>
        float h = position.y;
        vec4 wp = instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
        float b = ${bend};
        transformed.x += b * sin(uTime * 0.9 + wp.x * 0.02 + wp.z * 0.013);
        transformed.z += 0.7 * b * cos(uTime * 0.7 + wp.x * 0.017);`);
    };
    reedMat.onBeforeCompile = sway('14.0 * h * h');
    const nReeds = 420;
    this.reeds = new THREE.InstancedMesh(reedGeo, reedMat, nReeds);
    const headGeo = new THREE.CapsuleGeometry(3.2, 22, 4, 8).translate(0, 1, 0);
    const headMat = new THREE.MeshStandardMaterial({ color: 0x4a2c17, roughness: 0.9 });
    headMat.onBeforeCompile = sway('14.0 * 0.74');
    const heads = [];
    for (let i = 0; i < nReeds; i++) {
      const r = radius + 20 + 160 * Math.sqrt(rand());
      const a = rand() * Math.PI * 2;
      const hgt = 120 + 260 * rand();
      p.set(r * Math.cos(a), 0, r * Math.sin(a));
      q.setFromEuler(new THREE.Euler((rand() - 0.5) * 0.15, rand() * 6, (rand() - 0.5) * 0.15));
      s.set(1 + rand(), hgt, 1 + rand());
      this.reeds.setMatrixAt(i, m.compose(p, q, s));
      if (rand() < 0.3) heads.push({ p: p.clone(), q: q.clone(), h: hgt });
    }
    this.group.add(this.reeds);
    // Cattail heads sit at the top of their reed; the sway shader uses the same instance origin.
    this.cattails = new THREE.InstancedMesh(headGeo, headMat, heads.length);
    heads.forEach((c, i) => {
      const top = new THREE.Vector3(0, c.h * 0.86, 0).applyQuaternion(c.q).add(c.p);
      this.cattails.setMatrixAt(i, m.compose(top, c.q, new THREE.Vector3(1, 1, 1)));
    });
    this.group.add(this.cattails);

    // Half-sunk logs
    const barkMat = new THREE.MeshStandardMaterial({ color: 0x3a2a1c, roughness: 0.95 });
    for (let i = 0; i < 3; i++) {
      const log = new THREE.Mesh(new THREE.CylinderGeometry(14, 16, 220 + 120 * rand(), 12, 1), barkMat);
      const a = rand() * Math.PI * 2, r = 200 + 80 * rand();
      log.position.set(r * Math.cos(a), 3, r * Math.sin(a));
      log.rotation.set(Math.PI / 2, 0, rand() * Math.PI);
      log.castShadow = log.receiveShadow = true;
      this.group.add(log);
    }

    // Fireflies
    const nFly = 90;
    const fpos = new Float32Array(nFly * 3);
    this.fireflySeed = new Float32Array(nFly * 4);
    for (let i = 0; i < nFly; i++) {
      const r = 60 + 420 * Math.sqrt(rand()), a = rand() * Math.PI * 2;
      this.fireflySeed.set([r * Math.cos(a), 20 + 160 * rand(), r * Math.sin(a), rand() * 100], 4 * i);
    }
    const fgeo = new THREE.BufferGeometry();
    fgeo.setAttribute('position', new THREE.BufferAttribute(fpos, 3));
    fgeo.setAttribute('seed', new THREE.BufferAttribute(Float32Array.from({ length: nFly }, (_, i) => this.fireflySeed[4 * i + 3]), 1));
    const fmat = new THREE.ShaderMaterial({
      uniforms: { uTime: this.time, uPx: { value: 600 } },
      vertexShader: /* glsl */ `
        uniform float uTime; uniform float uPx; attribute float seed; varying float vB;
        void main() {
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          vB = pow(max(0.0, sin(uTime * (0.7 + fract(seed) * 0.8) + seed)), 6.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = (2.0 + 5.0 * vB) * uPx / max(-mv.z, 1.0);
        }`,
      fragmentShader: /* glsl */ `
        varying float vB;
        void main() { vec2 p = gl_PointCoord * 2.0 - 1.0; float r = dot(p, p); if (r > 1.0) discard;
          gl_FragColor = vec4(vec3(0.85, 1.0, 0.45) * (0.15 + vB) * (1.0 - r), 1.0);
#include <tonemapping_fragment>
#include <colorspace_fragment> }`,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.fireflies = new THREE.Points(fgeo, fmat);
    this.fireflies.frustumCulled = false;
    this.group.add(this.fireflies);
  }

  // follow: point the shadow frustum at what the camera looks at
  update(dt, focus) {
    this.time.value += dt;
    this.normalMap.offset.x += dt * 0.004;
    this.normalMap.offset.y += dt * 0.0025;
    const pos = this.fireflies.geometry.attributes.position;
    const t = this.time.value;
    for (let i = 0; i < pos.count; i++) {
      const [x, y, z, sd] = this.fireflySeed.subarray(4 * i, 4 * i + 4);
      pos.setXYZ(i, x + 25 * Math.sin(t * 0.15 + sd), y + 10 * Math.sin(t * 0.3 + sd * 2), z + 25 * Math.cos(t * 0.12 + sd));
    }
    pos.needsUpdate = true;
    if (focus) {
      this.moon.target.position.copy(focus);
      this.moon.position.set(focus.x - 450, focus.y + 380, focus.z - 800);
    }
  }

  setPixelScale(px) {
    this.fireflies.material.uniforms.uPx.value = px;
  }
}
