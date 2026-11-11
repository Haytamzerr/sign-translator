/**
 * The 3D figure, drawn with three.js.
 *
 * A smooth, faceless studio mannequin: a sculpted torso, a neck, an
 * egg-shaped head, and limbs made of tapered segments joined by spheres of
 * matching size, so the surface reads as one continuous body. Fingers taper
 * towards the tips and the palm is a solid slab with thickness.
 *
 * Every segment and every joint of a frame is drawn in one go (instancing),
 * so a frame costs only a handful of draw calls. Lighting comes from a soft
 * studio environment plus one key light that casts shadows (a hand in front
 * of the chest darkens it), which is what makes the shapes read as solid.
 *
 * Each frame: solveFigure() gives joint positions, update() moves the solids,
 * render() draws the main view and, for letters, a close-up of the hand.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { cross, normalize, type Arm, type Figure, type Vec3 } from '../core/learn/figure';

export interface MannequinPalette {
  body: string;
  right: string;
  left: string;
  /** Background of the hand close-up. */
  inset: string;
}

/** Thickness of each part, in shoulder widths. */
const R = {
  shoulder: 0.15,
  upperArm: 0.128,
  elbow: 0.104,
  forearm: 0.1,
  wrist: 0.066,
  neck: 0.13,
};

/** Radius of each hand joint (wrist, thumb 1-4, index 5-8, ...), in shoulder widths. */
const HAND_R = [
  0.05,
  0.042, 0.038, 0.033, 0.027,
  0.036, 0.03, 0.026, 0.022,
  0.037, 0.031, 0.027, 0.022,
  0.035, 0.029, 0.025, 0.021,
  0.032, 0.026, 0.022, 0.019,
];
/** Bones drawn as tapered segments (the palm's edges are covered by the slab). */
const FINGER_BONES: [number, number][] = [
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
  [0, 1], [0, 5], [0, 17], [5, 9], [9, 13], [13, 17],
];
/** Outline of the palm slab. */
const PALM_RING = [0, 1, 2, 5, 9, 13, 17];
const PALM_THICKNESS = 0.05;

const MAX_SEGMENTS = 96;
const MAX_JOINTS = 96;

/** Outline of the torso, turned around its vertical axis: (radius, height). */
const TORSO_PROFILE: [number, number][] = [
  [0.001, 0.15], [0.13, 0.145], [0.27, 0.11], [0.4, 0.05], [0.5, -0.04], [0.56, -0.16], [0.58, -0.32], [0.565, -0.55],
  [0.52, -0.82], [0.47, -1.06], [0.45, -1.26], [0.46, -1.46], [0.47, -1.66], [0.42, -1.86], [0.24, -1.98], [0.001, -2],
];
/** Outline of the head, chin at the bottom: (radius, height), scaled by the head's radii. */
const HEAD_PROFILE: [number, number][] = [
  [0.001, -1], [0.38, -0.96], [0.62, -0.82], [0.8, -0.58], [0.93, -0.28], [0.99, 0.02], [0.98, 0.32],
  [0.9, 0.6], [0.72, 0.84], [0.42, 0.98], [0.001, 1],
];

const UP = new THREE.Vector3(0, 1, 0);

/**
 * A unit segment that narrows from radius 1 at y = -0.5 to `taper` at y = 0.5.
 * Scaled per instance: x and z by the start radius, y by the length.
 */
function taperedSegment(taper: number): THREE.BufferGeometry {
  return new THREE.CylinderGeometry(taper, 1, 1, 20, 1, true);
}

export class Mannequin {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(24, 1, 0.1, 50);
  private closeCamera = new THREE.PerspectiveCamera(30, 1, 0.05, 20);
  private controls: OrbitControls;
  private root = new THREE.Group();
  private key: THREE.DirectionalLight;

  private bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0 });
  private partMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0 });
  private handMats = [0, 1].map(() => new THREE.MeshStandardMaterial({ roughness: 0.48, metalness: 0, side: THREE.DoubleSide }));

  private torso: THREE.Mesh;
  private head: THREE.Mesh;
  private nose: THREE.Mesh;
  private ears: THREE.Mesh[];
  private mittens: THREE.Mesh[];
  private palms: THREE.Mesh[];
  private segments: THREE.InstancedMesh;
  private joints: THREE.InstancedMesh;
  private nSeg = 0;
  private nJoint = 0;

  private colors = {
    body: new THREE.Color(),
    right: new THREE.Color(),
    left: new THREE.Color(),
    inset: new THREE.Color(),
  };

  // Reused objects, so a frame allocates nothing.
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private v1 = new THREE.Vector3();
  private v2 = new THREE.Vector3();
  private v3 = new THREE.Vector3();
  private s = new THREE.Vector3();
  private size = new THREE.Vector2();

  /** Hand centre for the close-up camera. */
  private handFocus = new THREE.Vector3();
  private handVisible = false;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    // Above 1.5 the extra sharpness is invisible on a small player but costs a lot on laptop graphics.
    this.renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    // Soft studio light from every side (a room environment, computed once,
    // nothing downloaded), plus a key light from the front right with shadows.
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    this.scene.environmentIntensity = 0.75;
    this.key = new THREE.DirectionalLight(0xffffff, 1.6);
    this.key.position.set(2.2, 3.2, 5.5);
    this.key.castShadow = true;
    this.key.shadow.mapSize.set(1024, 1024);
    this.key.shadow.radius = 4;
    this.key.shadow.bias = -0.0004;
    this.key.shadow.normalBias = 0.02;
    const cam = this.key.shadow.camera;
    cam.left = -2.2; cam.right = 2.2; cam.top = 2.2; cam.bottom = -2.4; cam.near = 1; cam.far = 14;
    this.scene.add(this.key);

    const lathe = (profile: [number, number][], segments: number) =>
      new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), segments);
    this.torso = new THREE.Mesh(lathe(TORSO_PROFILE, 64), this.bodyMat);
    this.head = new THREE.Mesh(lathe(HEAD_PROFILE, 48), this.bodyMat);
    this.nose = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), this.bodyMat);
    this.ears = [0, 1].map(() => new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), this.bodyMat));
    this.mittens = this.handMats.map((mat) => new THREE.Mesh(new THREE.SphereGeometry(1, 28, 20), mat));
    this.palms = this.handMats.map((mat) => {
      const g = new THREE.BufferGeometry();
      const n = PALM_RING.length;
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array((2 * n + 2) * 3), 3));
      const index: number[] = [];
      for (let k = 0; k < n; k++) {
        const a = k, b = (k + 1) % n;
        index.push(2 * n, a, b); // front face around its centre
        index.push(2 * n + 1, n + b, n + a); // back face
        index.push(a, n + a, n + b, a, n + b, b); // side
      }
      g.setIndex(index);
      return new THREE.Mesh(g, mat);
    });
    this.segments = new THREE.InstancedMesh(taperedSegment(0.82), this.partMat, MAX_SEGMENTS);
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 20, 14), this.partMat, MAX_JOINTS);
    for (const mesh of [this.segments, this.joints]) {
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.setColorAt(0, new THREE.Color());
      mesh.frustumCulled = false;
    }
    const solids = [this.torso, this.head, this.nose, ...this.ears, ...this.mittens, ...this.palms, this.segments, this.joints];
    for (const o of solids) {
      o.castShadow = true;
      o.receiveShadow = true;
    }
    this.root.add(...solids);
    this.scene.add(this.root);

    this.camera.position.set(0, -0.12, 7.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, -0.22, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 3.5;
    this.controls.maxDistance = 11;
    this.controls.minAzimuthAngle = -1.3;
    this.controls.maxAzimuthAngle = 1.3;
    this.controls.minPolarAngle = 0.9;
    this.controls.maxPolarAngle = 2.0;
    this.controls.update();
    this.controls.saveState();
  }

  setPalette(p: MannequinPalette): void {
    this.bodyMat.color.set(p.body);
    this.colors.body.set(p.body);
    this.colors.right.set(p.right);
    this.colors.left.set(p.left);
    this.colors.inset.set(p.inset);
    this.handMats[0]!.color.set(p.left);
    this.handMats[1]!.color.set(p.right);
  }

  resetView(): void {
    this.controls.reset();
  }

  /** Turns the camera around the figure (radians), for previews and tests. */
  setAngle(azimuth: number): void {
    const d = this.camera.position.distanceTo(this.controls.target);
    this.camera.position.set(
      this.controls.target.x + Math.sin(azimuth) * d,
      this.camera.position.y,
      this.controls.target.z + Math.cos(azimuth) * d,
    );
    this.controls.update();
  }

  resize(width: number, height: number): void {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.controls.dispose();
    this.scene.environment?.dispose();
    this.scene.traverse((o) => {
      if (o instanceof THREE.Mesh) {
        o.geometry.dispose();
        const mats = Array.isArray(o.material) ? o.material : [o.material];
        mats.forEach((mat) => mat.dispose());
      }
    });
    this.renderer.dispose();
  }

  // ---------- building one frame ----------

  /** A segment from a (radius r) narrowing towards b. */
  private segment(a: Vec3, b: Vec3, r: number, color: THREE.Color): void {
    if (this.nSeg >= MAX_SEGMENTS) return;
    this.v1.set(a[0], a[1], a[2]);
    this.v2.set(b[0], b[1], b[2]);
    this.v3.subVectors(this.v2, this.v1);
    const len = this.v3.length();
    if (len < 1e-5) return;
    this.q.setFromUnitVectors(UP, this.v3.divideScalar(len));
    this.v1.add(this.v2).multiplyScalar(0.5);
    this.s.set(r, len, r);
    this.m.compose(this.v1, this.q, this.s);
    this.segments.setMatrixAt(this.nSeg, this.m);
    this.segments.setColorAt(this.nSeg, color);
    this.nSeg++;
  }

  private joint(p: Vec3, r: number, color: THREE.Color): void {
    if (this.nJoint >= MAX_JOINTS) return;
    this.v1.set(p[0], p[1], p[2]);
    this.q.identity();
    this.s.setScalar(r);
    this.m.compose(this.v1, this.q, this.s);
    this.joints.setMatrixAt(this.nJoint, this.m);
    this.joints.setColorAt(this.nJoint, color);
    this.nJoint++;
  }

  /** Places a mesh with three axes and three radii. */
  private orient(mesh: THREE.Object3D, center: Vec3, x: Vec3, y: Vec3, z: Vec3, size: Vec3): void {
    this.m.makeBasis(this.v1.set(...x), this.v2.set(...y), this.v3.set(...z));
    mesh.quaternion.setFromRotationMatrix(this.m);
    mesh.position.set(...center);
    mesh.scale.set(...size);
  }

  private arm(arm: Arm, side: 0 | 1, handSize: number): void {
    const handColor = side ? this.colors.right : this.colors.left;
    const { body } = this.colors;
    this.joint(arm.shoulder, R.shoulder, body);
    this.segment(arm.shoulder, arm.elbow, R.upperArm, body);
    this.joint(arm.elbow, R.elbow, body);
    this.segment(arm.elbow, arm.wrist, R.forearm, body);

    const mitten = this.mittens[side]!;
    const palm = this.palms[side]!;
    const hand = arm.hand;
    if (!hand) {
      // Hand not tracked: a relaxed closed hand continuing the forearm.
      this.joint(arm.wrist, R.wrist, body);
      const dir = normalize([arm.wrist[0] - arm.elbow[0], arm.wrist[1] - arm.elbow[1], arm.wrist[2] - arm.elbow[2]]);
      const ref: Vec3 = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      const side3 = normalize(cross(ref, dir));
      const flat = cross(side3, dir);
      const c: Vec3 = [arm.wrist[0] + dir[0] * handSize * 0.7, arm.wrist[1] + dir[1] * handSize * 0.7, arm.wrist[2] + dir[2] * handSize * 0.7];
      this.orient(mitten, c, side3, dir, flat, [handSize * 0.4, handSize * 0.78, handSize * 0.22]);
      mitten.visible = true;
      palm.visible = false;
      return;
    }
    mitten.visible = false;
    palm.visible = true;
    this.joint(hand[0]!, HAND_R[0]!, handColor);
    for (const [a, b] of FINGER_BONES) this.segment(hand[a]!, hand[b]!, HAND_R[a]! * (a === 0 ? 0.75 : 1), handColor);
    for (let i = 1; i < 21; i++) this.joint(hand[i]!, HAND_R[i]!, handColor);

    // Palm: a slab with thickness, its two faces offset along the palm's normal.
    const w = hand[0]!, ix = hand[5]!, px = hand[17]!;
    const nrm = normalize(cross([ix[0] - w[0], ix[1] - w[1], ix[2] - w[2]], [px[0] - w[0], px[1] - w[1], px[2] - w[2]]));
    const t = PALM_THICKNESS * 0.5;
    const pos = palm.geometry.getAttribute('position') as THREE.BufferAttribute;
    const n = PALM_RING.length;
    let cx = 0, cy = 0, cz = 0;
    PALM_RING.forEach((i, k) => {
      const p = hand[i]!;
      pos.setXYZ(k, p[0] + nrm[0] * t, p[1] + nrm[1] * t, p[2] + nrm[2] * t);
      pos.setXYZ(n + k, p[0] - nrm[0] * t, p[1] - nrm[1] * t, p[2] - nrm[2] * t);
      cx += p[0]; cy += p[1]; cz += p[2];
    });
    cx /= n; cy /= n; cz /= n;
    pos.setXYZ(2 * n, cx + nrm[0] * t, cy + nrm[1] * t, cz + nrm[2] * t);
    pos.setXYZ(2 * n + 1, cx - nrm[0] * t, cy - nrm[1] * t, cz - nrm[2] * t);
    pos.needsUpdate = true;
    palm.geometry.computeVertexNormals();
    palm.geometry.computeBoundingSphere();
  }

  update(fig: Figure, handSize: number): void {
    this.nSeg = 0;
    this.nJoint = 0;
    const { center, across, width } = fig.chest;

    // Torso: turned with the shoulder line.
    const up = normalize([-across[1] * across[0], 1 - across[1] * across[1], -across[1] * across[2]]);
    const fwd = cross(across, up);
    this.orient(this.torso, center, across, up, fwd, [width, width, width * 0.6]);

    this.arm(fig.left, 0, handSize);
    this.arm(fig.right, 1, handSize);

    const head = fig.head;
    this.head.visible = !!head;
    this.nose.visible = false;
    this.ears.forEach((e) => (e.visible = !!head));
    if (head) {
      const [x, y, z] = head.axes;
      const [rx, ry, rz] = head.radii;
      this.orient(this.head, head.center, x, y, z, [rx, ry, rz]);
      const at = (u: number, v: number, w: number): Vec3 => [
        head.center[0] + x[0] * u + y[0] * v + z[0] * w,
        head.center[1] + x[1] * u + y[1] * v + z[1] * w,
        head.center[2] + x[2] * u + y[2] * v + z[2] * w,
      ];
      // Two ears and no drawn features: the form alone shows which way the head faces.
      this.orient(this.ears[0]!, at(-rx * 0.97, 0, -rz * 0.05), x, y, z, [rx * 0.1, ry * 0.24, rz * 0.16]);
      this.orient(this.ears[1]!, at(rx * 0.97, 0, -rz * 0.05), x, y, z, [rx * 0.1, ry * 0.24, rz * 0.16]);
      const chin = at(0, -ry * 0.6, -rz * 0.15);
      this.segment([center[0], center[1] - 0.02, center[2]], chin, R.neck, this.colors.body);
    }

    const hand = fig.right.hand;
    this.handVisible = !!hand;
    if (hand) {
      this.handFocus.set(0, 0, 0);
      for (const p of hand) this.handFocus.add(this.v1.set(p[0], p[1], p[2]));
      this.handFocus.divideScalar(hand.length);
    }

    this.segments.count = this.nSeg;
    this.joints.count = this.nJoint;
    this.segments.instanceMatrix.needsUpdate = true;
    this.joints.instanceMatrix.needsUpdate = true;
    if (this.segments.instanceColor) this.segments.instanceColor.needsUpdate = true;
    if (this.joints.instanceColor) this.joints.instanceColor.needsUpdate = true;
  }

  /**
   * Draws the frame. With `closeUp`, a square in the top corner shows the right
   * hand from the front, for reading a letter's handshape.
   * Returns the close-up square in CSS pixels (or null), for its frame.
   */
  render(mirror: boolean, closeUp: boolean, handSize: number): { x: number; y: number; size: number } | null {
    this.root.scale.x = mirror ? -1 : 1;
    this.controls.update();
    const r = this.renderer;
    const size = r.getSize(this.size);
    const w = size.x;
    const h = size.y;
    r.autoClear = false;
    r.setScissorTest(false);
    r.setViewport(0, 0, w, h);
    r.clear();
    r.render(this.scene, this.camera);
    if (!closeUp || !this.handVisible) return null;

    const box = Math.round(Math.min(w, h) * 0.4);
    const pad = Math.round(Math.min(w, h) * 0.035);
    const x = mirror ? pad : w - box - pad;
    const yTop = pad;
    this.root.updateMatrixWorld();
    const focus = this.v1.copy(this.handFocus).applyMatrix4(this.root.matrixWorld);
    const distance = (handSize * 1.15) / Math.tan(THREE.MathUtils.degToRad(this.closeCamera.fov / 2));
    this.closeCamera.position.set(focus.x, focus.y, focus.z + distance);
    this.closeCamera.lookAt(focus);
    this.closeCamera.aspect = 1;
    this.closeCamera.updateProjectionMatrix();
    r.setScissorTest(true);
    r.setScissor(x, h - yTop - box, box, box);
    r.setViewport(x, h - yTop - box, box, box);
    r.setClearColor(this.colors.inset, 1);
    r.clear();
    r.render(this.scene, this.closeCamera);
    r.setClearColor(0x000000, 0);
    r.setScissorTest(false);
    return { x, y: yTop, size: box };
  }
}
