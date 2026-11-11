/**
 * The 3D figure, drawn with three.js.
 *
 * The figure is built from a few simple solids, like a wooden artist's
 * mannequin: a turned torso, a neck, an egg-shaped head, and limbs made of
 * cylinders joined by spheres. Every cylinder and every sphere of a frame is
 * drawn in one go (instancing), so a frame costs only a handful of draw calls.
 *
 * Each frame: solveFigure() gives joint positions, update() moves the solids,
 * render() draws the main view and, for letters, a close-up of the hand.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HAND_BONES } from '../core/tracker';
import { cross, normalize, type Arm, type Figure, type Vec3 } from '../core/learn/figure';

export interface MannequinPalette {
  body: string;
  joint: string;
  head: string;
  feature: string;
  right: string;
  left: string;
  /** Background of the hand close-up. */
  inset: string;
}

/** Thickness of each part, in shoulder widths. */
const R = {
  upperArm: 0.085,
  forearm: 0.07,
  shoulder: 0.11,
  elbow: 0.078,
  wrist: 0.06,
  finger: 0.03,
  knuckle: 0.033,
  neck: 0.1,
  eye: 0.0065,
  brow: 0.009,
  lips: 0.008,
  pupil: 0.018,
};

const MAX_CYLINDERS = 240;
const MAX_SPHERES = 160;
const PALM = [0, 1, 5, 9, 13, 17];

/** Outline of the torso, turned around its vertical axis: (radius, height). */
const TORSO_PROFILE: [number, number][] = [
  [0.001, 0.12], [0.2, 0.1], [0.42, 0.06], [0.53, -0.05], [0.56, -0.22], [0.54, -0.55], [0.47, -0.92],
  [0.42, -1.18], [0.44, -1.42], [0.4, -1.66], [0.24, -1.8], [0.001, -1.84],
]

const UP = new THREE.Vector3(0, 1, 0);

export class Mannequin {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(26, 1, 0.1, 50);
  private closeCamera = new THREE.PerspectiveCamera(30, 1, 0.05, 20);
  private controls: OrbitControls;
  private root = new THREE.Group();

  private bodyMat = new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0 });
  private headMat = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0 });
  private partMat = new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0 });
  private palmMats = [0, 1].map(() => new THREE.MeshStandardMaterial({ roughness: 0.5, side: THREE.DoubleSide }));

  private torso: THREE.Mesh;
  private head: THREE.Mesh;
  private nose: THREE.Mesh;
  private mittens: THREE.Mesh[];
  private palms: THREE.Mesh[];
  private cylinders: THREE.InstancedMesh;
  private spheres: THREE.InstancedMesh;
  private nCyl = 0;
  private nSph = 0;

  private colors = {
    body: new THREE.Color(),
    joint: new THREE.Color(),
    feature: new THREE.Color(),
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

  /** Hand centre and size for the close-up camera. */
  private handFocus = new THREE.Vector3();
  private handVisible = false;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.setClearColor(0x000000, 0);

    // Light: soft light from above, a main light from the front right, and a
    // light from behind that outlines the figure against the background.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445066, 1.5));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(2.5, 3.5, 6);
    const rim = new THREE.DirectionalLight(0xbfd4ff, 1.6);
    rim.position.set(-3, 2.5, -4);
    this.scene.add(key, rim);

    const lathe = new THREE.LatheGeometry(TORSO_PROFILE.map(([r, y]) => new THREE.Vector2(r, y)), 48);
    this.torso = new THREE.Mesh(lathe, this.bodyMat);
    this.head = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 32), this.headMat);
    this.nose = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), this.headMat);
    this.mittens = [0, 1].map(() => new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), this.partMat.clone()));
    this.palms = this.palmMats.map((mat) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(7 * 3), 3));
      g.setIndex([0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 1]);
      return new THREE.Mesh(g, mat);
    });
    this.cylinders = new THREE.InstancedMesh(new THREE.CylinderGeometry(1, 1, 1, 14, 1, true), this.partMat, MAX_CYLINDERS);
    this.spheres = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 16, 12), this.partMat, MAX_SPHERES);
    for (const mesh of [this.cylinders, this.spheres]) {
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.setColorAt(0, new THREE.Color());
      mesh.frustumCulled = false;
    }
    this.root.add(this.torso, this.head, this.nose, ...this.mittens, ...this.palms, this.cylinders, this.spheres);
    this.scene.add(this.root);

    this.camera.position.set(0, -0.2, 8.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, -0.3, 0);
    this.controls.enableDamping = true;
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
    this.headMat.color.set(p.head);
    this.colors.body.set(p.body);
    this.colors.joint.set(p.joint);
    this.colors.feature.set(p.feature);
    this.colors.right.set(p.right);
    this.colors.left.set(p.left);
    this.colors.inset.set(p.inset);
    this.palmMats[0]!.color.set(p.left);
    this.palmMats[1]!.color.set(p.right);
    (this.mittens[0]!.material as THREE.MeshStandardMaterial).color.set(p.left);
    (this.mittens[1]!.material as THREE.MeshStandardMaterial).color.set(p.right);
  }

  resetView(): void {
    this.controls.reset();
  }

  resize(width: number, height: number): void {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.controls.dispose();
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

  private cylinder(a: Vec3, b: Vec3, radius: number, color: THREE.Color): void {
    if (this.nCyl >= MAX_CYLINDERS) return;
    this.v1.set(a[0], a[1], a[2]);
    this.v2.set(b[0], b[1], b[2]);
    this.v3.subVectors(this.v2, this.v1);
    const len = this.v3.length();
    if (len < 1e-5) return;
    this.q.setFromUnitVectors(UP, this.v3.divideScalar(len));
    this.v1.add(this.v2).multiplyScalar(0.5);
    this.s.set(radius, len, radius);
    this.m.compose(this.v1, this.q, this.s);
    this.cylinders.setMatrixAt(this.nCyl, this.m);
    this.cylinders.setColorAt(this.nCyl, color);
    this.nCyl++;
  }

  private sphere(p: Vec3, radius: number, color: THREE.Color): void {
    if (this.nSph >= MAX_SPHERES) return;
    this.v1.set(p[0], p[1], p[2]);
    this.q.identity();
    this.s.setScalar(radius);
    this.m.compose(this.v1, this.q, this.s);
    this.spheres.setMatrixAt(this.nSph, this.m);
    this.spheres.setColorAt(this.nSph, color);
    this.nSph++;
  }

  /** Places a mesh with three axes and three radii (an ellipsoid). */
  private orient(mesh: THREE.Object3D, center: Vec3, x: Vec3, y: Vec3, z: Vec3, size: Vec3): void {
    this.m.makeBasis(this.v1.set(...x), this.v2.set(...y), this.v3.set(...z));
    mesh.quaternion.setFromRotationMatrix(this.m);
    mesh.position.set(...center);
    mesh.scale.set(...size);
  }

  private arm(arm: Arm, side: 0 | 1, handSize: number): void {
    const handColor = side ? this.colors.right : this.colors.left;
    const { body, joint } = this.colors;
    this.sphere(arm.shoulder, R.shoulder, body);
    this.cylinder(arm.shoulder, arm.elbow, R.upperArm, body);
    this.sphere(arm.elbow, R.elbow, joint);
    this.cylinder(arm.elbow, arm.wrist, R.forearm, body);
    this.sphere(arm.wrist, R.wrist, joint);

    const mitten = this.mittens[side]!;
    const palm = this.palms[side]!;
    const hand = arm.hand;
    if (!hand) {
      // Hand not tracked: a relaxed closed hand continuing the forearm.
      const dir = normalize([arm.wrist[0] - arm.elbow[0], arm.wrist[1] - arm.elbow[1], arm.wrist[2] - arm.elbow[2]]);
      const ref: Vec3 = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
      const side3 = normalize(cross(ref, dir));
      const flat = cross(side3, dir);
      const c: Vec3 = [arm.wrist[0] + dir[0] * handSize * 0.75, arm.wrist[1] + dir[1] * handSize * 0.75, arm.wrist[2] + dir[2] * handSize * 0.75];
      this.orient(mitten, c, side3, dir, flat, [handSize * 0.42, handSize * 0.85, handSize * 0.2]);
      mitten.visible = true;
      palm.visible = false;
      return;
    }
    mitten.visible = false;
    palm.visible = true;
    for (const [a, b] of HAND_BONES) this.cylinder(hand[a]!, hand[b]!, R.finger, handColor);
    for (let i = 1; i < 21; i++) this.sphere(hand[i]!, i % 4 === 0 ? R.finger : R.knuckle, handColor);
    // Palm: a fan of triangles around the palm's centre.
    const pos = palm.geometry.getAttribute('position') as THREE.BufferAttribute;
    let cx = 0, cy = 0, cz = 0;
    PALM.forEach((i, k) => {
      const p = hand[i]!;
      pos.setXYZ(k + 1, p[0], p[1], p[2]);
      cx += p[0]; cy += p[1]; cz += p[2];
    });
    pos.setXYZ(0, cx / 6, cy / 6, cz / 6);
    pos.needsUpdate = true;
    palm.geometry.computeVertexNormals();
    palm.geometry.computeBoundingSphere();
  }

  update(fig: Figure, handSize: number): void {
    this.nCyl = 0;
    this.nSph = 0;
    const { center, across, width } = fig.chest;

    // Torso: turned with the shoulder line.
    const up = normalize([-across[1] * across[0], 1 - across[1] * across[1], -across[1] * across[2]]);
    const fwd: Vec3 = [across[1] * up[2] - across[2] * up[1], across[2] * up[0] - across[0] * up[2], across[0] * up[1] - across[1] * up[0]];
    this.orient(this.torso, center, across, up, fwd, [width, width, width * 0.55]);

    this.arm(fig.left, 0, handSize);
    this.arm(fig.right, 1, handSize);

    const head = fig.head;
    this.head.visible = !!head;
    this.nose.visible = !!head?.nose;
    if (head) {
      const [x, y, z] = head.axes;
      this.orient(this.head, head.center, x, y, z, head.radii);
      const chin: Vec3 = [head.center[0] - y[0] * head.radii[1] * 0.55, head.center[1] - y[1] * head.radii[1] * 0.55, head.center[2] - y[2] * head.radii[1] * 0.55];
      this.cylinder([center[0], center[1] - 0.05, center[2]], chin, R.neck, this.colors.body);
      if (head.nose) this.orient(this.nose, head.nose, x, y, z, [0.035, 0.05, 0.04]);
      for (const f of head.features) {
        const r = f.kind === 'eye' ? R.eye : f.kind === 'brow' ? R.brow : R.lips;
        const n = f.points.length;
        const segments = f.closed ? n : n - 1;
        for (let i = 0; i < segments; i++) this.cylinder(f.points[i]!, f.points[(i + 1) % n]!, r, this.colors.feature);
        for (const p of f.points) this.sphere(p, r, this.colors.feature);
      }
      for (const p of head.pupils) this.sphere(p, R.pupil, this.colors.feature);
    }

    const hand = fig.right.hand;
    this.handVisible = !!hand;
    if (hand) {
      this.handFocus.set(0, 0, 0);
      for (const p of hand) this.handFocus.add(this.v1.set(p[0], p[1], p[2]));
      this.handFocus.divideScalar(hand.length);
    }

    this.cylinders.count = this.nCyl;
    this.spheres.count = this.nSph;
    this.cylinders.instanceMatrix.needsUpdate = true;
    this.spheres.instanceMatrix.needsUpdate = true;
    if (this.cylinders.instanceColor) this.cylinders.instanceColor.needsUpdate = true;
    if (this.spheres.instanceColor) this.spheres.instanceColor.needsUpdate = true;
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
