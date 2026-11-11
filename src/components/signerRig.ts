/**
 * Moves the 3D signer's skeleton from a recorded figure, like a real body would.
 *
 * Copying the direction of each recorded bone onto the model ("retargeting by
 * direction") has three problems: the model's arms do not have the signer's
 * proportions, so a hand meant to touch the chin ends up in front of it; the
 * camera's depth is noisy, so arms pass through the body; and only the arms
 * move, so the body looks stiff. Here the recording gives TARGETS, and the body
 * reaches them:
 *
 *   1. Where the hands go. Each wrist position is taken relative to the
 *      shoulders and scaled to the model. Near the face it is taken relative
 *      to the face instead, so a sign at the chin lands on the model's chin.
 *   2. Smoothing. Targets move like a mass on a spring (critically damped):
 *      they accelerate and slow down smoothly instead of jumping.
 *   3. No passing through the body. The torso and the head are measured from
 *      the model's own surface. A hand target inside them is pushed out to the
 *      surface (so a touch becomes a real contact).
 *   4. Arms by inverse kinematics (IK). With the shoulder and the wrist known,
 *      and the upper arm and forearm of fixed length, the elbow can only be on
 *      a circle; the recording says roughly where on it. If the arm would pass
 *      through the body there, the elbow is turned along the circle until it
 *      does not.
 *   5. Joint limits. The wrist bends at most as far as a real wrist; the
 *      forearm takes most of the hand's turn (as the two forearm bones do);
 *      fingers bend within their range.
 *   6. A living body. The collarbones lift when a hand goes up, the chest
 *      follows the shoulders, the head follows the face, and the chest breathes.
 *
 * Model space: x to the model's left (the viewer's right), y up, z forwards
 * (towards the viewer), in metres. The model's bones have no rotation at rest.
 */

import * as THREE from 'three';
import type { Figure, Vec3 } from '../core/learn/figure';

type Side = 'L' | 'R';

/** Finger bones and the hand points their segment goes between. */
const FINGERS: [string, number, number][] = [
  ['finger1-1', 1, 2], ['finger1-2', 2, 3], ['finger1-3', 3, 4],
  ['finger2-1', 5, 6], ['finger2-2', 6, 7], ['finger2-3', 7, 8],
  ['finger3-1', 9, 10], ['finger3-2', 10, 11], ['finger3-3', 11, 12],
  ['finger4-1', 13, 14], ['finger4-2', 14, 15], ['finger4-3', 15, 16],
  ['finger5-1', 17, 18], ['finger5-2', 18, 19], ['finger5-3', 19, 20],
];
/** A relaxed hand for when a hand was not tracked: bend of each finger joint, radians. */
const RELAXED_CURL = [0.25, 0.45, 0.35];
/** Hand points checked against the body: wrist, knuckles and fingertips. */
const HAND_CHECK = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];

const deg = THREE.MathUtils.degToRad;
const clamp = THREE.MathUtils.clamp;
const smoothstep = (a: number, b: number, x: number) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};

// ---------------------------------------------------------------- body shape

/** Torso cross-section at one height: an ellipse-like shape (see `inside`). */
interface Slice {
  halfWidth: number;
  zMin: number;
  zMax: number;
}

/** Exponent of the cross-sections: 2 is an ellipse, higher is boxier (a chest is flatter than an ellipse). */
const SHAPE_POWER = 2.5;

/**
 * "Inside-ness" of a point in a shape centred at c with half-sizes r:
 * below 1 inside, 1 on the surface. Pushing the point to the surface is then
 * a matter of dividing its offset from the centre by this number.
 */
function shapeRadius(dx: number, dy: number, dz: number, rx: number, ry: number, rz: number): number {
  const p = SHAPE_POWER;
  return (Math.abs(dx / rx) ** p + Math.abs(dy / ry) ** p + Math.abs(dz / rz) ** p) ** (1 / p);
}

/**
 * The model's torso and head, measured once from its own surface. The torso
 * is a stack of cross-sections, every 2 cm; the head an ellipsoid-like shape.
 * Both are described at rest, and turned with the body when tested.
 */
export class BodyShape {
  static readonly STEP = 0.02;

  constructor(
    /** Cross-sections from height `y0` upwards, every STEP metres. */
    readonly slices: Slice[],
    readonly y0: number,
    /** Centre and half-sizes of the head, at rest. */
    readonly headCenter: THREE.Vector3,
    readonly headRadii: THREE.Vector3,
    /** Face landmarks at rest: height of the eyes and of the mouth, distance between the eyes. */
    readonly face: { eyesY: number; mouthY: number; eyeDistance: number },
  ) {}

  /** Measures the torso and head from the skinned mesh: each vertex belongs to the bone that moves it most. */
  static measure(mesh: THREE.SkinnedMesh): BodyShape {
    const pos = mesh.geometry.getAttribute('position');
    const idx = mesh.geometry.getAttribute('skinIndex');
    const wt = mesh.geometry.getAttribute('skinWeight');
    const names = mesh.skeleton.bones.map((b) => b.name);
    const torsoBones = /^(root|spine0\d|pelvis_.|breast_.|clavicle_.)$/;
    const headBones = /^(head|jaw|eye|levator|oris|tongue|temporalis|special|risorius|orbicularis)/;
    const top = 0.68;
    const y0 = 0.02;
    const count = Math.round((top - y0) / BodyShape.STEP) + 1;
    const slices: (Slice | null)[] = new Array(count).fill(null);
    const headMin = new THREE.Vector3(Infinity, Infinity, Infinity);
    const headMax = new THREE.Vector3(-Infinity, -Infinity, -Infinity);
    const v = new THREE.Vector3();

    for (let i = 0; i < pos.count; i++) {
      let best = 0;
      for (let k = 1; k < 4; k++) if (wt.getComponent(i, k) > wt.getComponent(i, best)) best = k;
      const bone = names[idx.getComponent(i, best)] ?? '';
      v.fromBufferAttribute(pos, i);
      if (headBones.test(bone)) {
        headMin.min(v);
        headMax.max(v);
      } else if (torsoBones.test(bone)) {
        const s = Math.round((v.y - y0) / BodyShape.STEP);
        if (s < 0 || s >= count) continue;
        const slice = (slices[s] ??= { halfWidth: 0, zMin: Infinity, zMax: -Infinity });
        slice.halfWidth = Math.max(slice.halfWidth, Math.abs(v.x));
        slice.zMin = Math.min(slice.zMin, v.z);
        slice.zMax = Math.max(slice.zMax, v.z);
      }
    }
    // Fill the few heights without vertices from their neighbours.
    for (let s = 0; s < count; s++) slices[s] ??= slices[s - 1] ?? slices.find((x) => x) ?? { halfWidth: 0.15, zMin: -0.08, zMax: 0.12 };
    const headCenter = headMin.clone().add(headMax).multiplyScalar(0.5);
    const headRadii = headMax.clone().sub(headMin).multiplyScalar(0.5);
    // The face: the eyes (their bones), the chin (the lowest point of the
    // front of the head), and the mouth, a little over half way from the eyes
    // to the chin, as on most faces.
    let chin = Infinity;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i);
      if (v.z < headCenter.z + 0.3 * headRadii.z || Math.abs(v.x) > 0.02) continue;
      let best = 0;
      for (let k = 1; k < 4; k++) if (wt.getComponent(i, k) > wt.getComponent(i, best)) best = k;
      if (headBones.test(names[idx.getComponent(i, best)] ?? '')) chin = Math.min(chin, v.y);
    }
    const eye = (side: string) => (mesh.skeleton.bones.find((b) => b.name === `eye_${side}`)?.userData.head as number[] | undefined) ?? [0.03, headCenter.y, 0];
    const [eL, eR] = [eye('L'), eye('R')];
    const eyesY = (eL[1]! + eR[1]!) / 2;
    const face = { eyesY, mouthY: eyesY - 0.55 * (eyesY - chin), eyeDistance: Math.abs(eL[0]! - eR[0]!) };
    return new BodyShape(slices as Slice[], y0, headCenter, headRadii, face);
  }

  private slice(y: number): Slice | null {
    const f = (y - this.y0) / BodyShape.STEP;
    if (f < 0 || f > this.slices.length - 1) return null;
    const a = this.slices[Math.floor(f)]!;
    const b = this.slices[Math.min(this.slices.length - 1, Math.floor(f) + 1)]!;
    const t = f - Math.floor(f);
    return {
      halfWidth: a.halfWidth + (b.halfWidth - a.halfWidth) * t,
      zMin: a.zMin + (b.zMin - a.zMin) * t,
      zMax: a.zMax + (b.zMax - a.zMax) * t,
    };
  }

  /**
   * Pushes a point (rest coordinates of the torso) out of the torso, keeping
   * `margin` metres from its surface. Hands sign in front of the body, so the
   * point goes out through the FRONT: along the line from a point well behind
   * the torso's centre through it. In front of the chest that is straight
   * forwards; towards the sides it turns outwards. (Pushing to the nearest
   * surface would send a hand the recording places "in" the chest out through
   * the side, and make it jump from side to side.) Returns the distance moved.
   */
  pushOutOfTorso(p: THREE.Vector3, margin: number): number {
    const s = this.slice(p.y);
    if (!s) return 0;
    const zc = (s.zMin + s.zMax) / 2;
    const a = s.halfWidth + margin;
    const b = (s.zMax - s.zMin) / 2 + margin;
    const inside = (x: number, z: number) => shapeRadius(x, 0, z - zc, a, 1, b) < 1;
    if (!inside(p.x, p.z)) return 0;
    // Direction from the point behind, and the surface along it (by halving the interval).
    const bz = zc - 2 * b;
    const len = Math.hypot(p.x, p.z - bz);
    const dx = p.x / len;
    const dz = (p.z - bz) / len;
    let lo = 0;
    let hi = 2 * (a + b);
    for (let i = 0; i < 18; i++) {
      const mid = (lo + hi) / 2;
      if (inside(p.x + dx * mid, p.z + dz * mid)) lo = mid;
      else hi = mid;
    }
    p.x += dx * hi;
    p.z += dz * hi;
    return hi;
  }

  /** Same for the head (rest coordinates of the head). */
  pushOutOfHead(p: THREE.Vector3, margin: number): number {
    const c = this.headCenter;
    const rx = this.headRadii.x + margin;
    const ry = this.headRadii.y + margin;
    const rz = this.headRadii.z + margin;
    const r = shapeRadius(p.x - c.x, p.y - c.y, p.z - c.z, rx, ry, rz);
    if (r >= 1) return 0;
    const depth = (1 - r) * Math.min(rx, ry, rz);
    if (r < 1e-6) p.z = c.z + rz;
    else p.sub(c).divideScalar(r).add(c);
    return depth;
  }
}

// ---------------------------------------------------------------- smoothing

/**
 * A point that follows its target like a mass on a critically damped spring:
 * it speeds up and slows down smoothly and never overshoots. `omega` sets how
 * fast (rad/s); the delay behind a moving target is about 2 / omega.
 */
export class Spring {
  readonly value = new THREE.Vector3();
  private velocity = new THREE.Vector3();
  private started = false;
  private acc = new THREE.Vector3();

  constructor(private omega: number) {}

  update(target: THREE.Vector3, dtMs: number, snap = false): THREE.Vector3 {
    if (snap || !this.started) {
      this.value.copy(target);
      this.velocity.set(0, 0, 0);
      this.started = true;
      return this.value;
    }
    // Small steps keep the integration stable even when a frame is late.
    let left = Math.min(100, Math.max(0, dtMs)) / 1000;
    const w = this.omega;
    while (left > 1e-6) {
      const h = Math.min(left, 1 / 240);
      // a = w^2 (target - x) - 2 w v
      this.acc.copy(target).sub(this.value).multiplyScalar(w * w).addScaledVector(this.velocity, -2 * w);
      this.velocity.addScaledVector(this.acc, h);
      this.value.addScaledVector(this.velocity, h);
      left -= h;
    }
    return this.value;
  }
}

// ---------------------------------------------------------------- the rig

interface BoneInfo {
  bone: THREE.Bone;
  /** Joint position at rest (model space). */
  head: THREE.Vector3;
  /** Unit direction head -> tail at rest. */
  restDir: THREE.Vector3;
  /** Joint to the end of the bone (for fingertips). */
  length: number;
}

/** A hand point that should be at a given place (a touch), and how much that matters (0 to 1). */
interface Contact {
  /** Candidate touching points (hand point, its place on the face, its share; shares add up to 1). */
  points: { point: number; at: THREE.Vector3; share: number }[];
  /** Where the wrist would be if the hand had its recorded orientation. */
  wrist: THREE.Vector3;
  weight: number;
}

/** The bone and end that show each of the 21 hand points ('end' = its far end). */
export const HAND_POINT_BONE: Record<number, [string, 'head' | 'end']> = {
  0: ['wrist', 'head'], 4: ['finger1-3', 'end'], 5: ['finger2-1', 'head'], 8: ['finger2-3', 'end'],
  9: ['finger3-1', 'head'], 12: ['finger3-3', 'end'], 13: ['finger4-1', 'head'], 16: ['finger4-3', 'end'],
  17: ['finger5-1', 'head'], 20: ['finger5-3', 'end'],
};

interface ArmState {
  wrist: Spring;
  pole: Spring;
}

/** How quickly each part of the body follows its new rotation, ms: the body is heavier than the fingers. */
const LAG_MS: [RegExp, number][] = [
  [/^(spine|neck|head)/, 55],
  [/^clavicle|^shoulder/, 60],
  [/^finger|^metacarpal/, 30],
  [/./, 12],
];

export class SignerRig {
  readonly body: BodyShape;
  private bones = new Map<string, BoneInfo>();
  private worldQ = new Map<string, THREE.Quaternion>();
  private worldP = new Map<string, THREE.Vector3>();
  private shown = new Map<THREE.Bone, THREE.Quaternion>();
  private lag = new Map<THREE.Bone, number>();
  private arms: Record<Side, ArmState> = {
    L: { wrist: new Spring(38), pole: new Spring(30) },
    R: { wrist: new Spring(38), pole: new Spring(30) },
  };
  private time = 0;
  /** Where each hand touches the face this frame (a hand point and its place), if it does. */
  readonly touches: Record<Side, { point: number; at: THREE.Vector3 } | null> = { L: null, R: null };
  private torsoQ = new THREE.Quaternion();
  private torsoPivot = new THREE.Vector3();

  /** Measurements of the model, at rest. */
  readonly shoulderWidth: number;
  private shoulderMid: THREE.Vector3;
  private upperArm: number;
  private forearm: number;
  private handLength: number;

  constructor(model: THREE.Object3D) {
    let mesh: THREE.SkinnedMesh | null = null;
    model.traverse((o) => {
      const b = o as THREE.Bone;
      if (b.isBone) {
        const { head, tail } = b.userData as { head: number[]; tail: number[] };
        const h = new THREE.Vector3(head[0], head[1], head[2]);
        const dir = new THREE.Vector3(tail[0], tail[1], tail[2]).sub(h).normalize();
        const length = new THREE.Vector3(tail[0], tail[1], tail[2]).distanceTo(h);
        this.bones.set(b.name, { bone: b, head: h, restDir: dir, length });
        this.lag.set(b, LAG_MS.find(([re]) => re.test(b.name))![1]);
      }
      const m = o as THREE.SkinnedMesh;
      if (m.isSkinnedMesh && m.name === 'body') mesh = m;
    });
    if (!mesh) throw new Error('signer model has no body mesh');
    this.body = BodyShape.measure(mesh);

    // Segments measured joint to joint, so the arm bones aim exactly at the next joint.
    const head = (n: string) => this.bones.get(n)!.head;
    for (const s of ['L', 'R'] as Side[]) {
      this.setRestDir(`upperarm01_${s}`, `lowerarm01_${s}`);
      this.setRestDir(`lowerarm01_${s}`, `wrist_${s}`);
    }
    this.shoulderWidth = head('upperarm01_L').distanceTo(head('upperarm01_R'));
    this.shoulderMid = head('upperarm01_L').clone().add(head('upperarm01_R')).multiplyScalar(0.5);
    this.upperArm = head('upperarm01_L').distanceTo(head('lowerarm01_L'));
    this.forearm = head('lowerarm01_L').distanceTo(head('wrist_L'));
    this.handLength = head('wrist_L').distanceTo(head('finger3-1_L'));
  }

  private setRestDir(name: string, next: string): void {
    const a = this.bones.get(name);
    const b = this.bones.get(next);
    if (a && b) a.restDir.copy(b.head).sub(a.head).normalize();
  }

  // ---------- forward kinematics ----------

  private parentName(name: string): string | null {
    return this.bones.get(name)?.bone.parent?.name ?? null;
  }

  private parentWorld(name: string): THREE.Quaternion {
    const p = this.parentName(name);
    return (p && this.worldQ.get(p)) || new THREE.Quaternion();
  }

  /** Sets a bone's world rotation; also works out where its joint is now. */
  private setWorld(name: string, q: THREE.Quaternion): void {
    const info = this.bones.get(name);
    if (!info) return;
    const parent = this.parentName(name);
    const parentQ = this.parentWorld(name);
    this.worldQ.set(name, q.clone());
    info.bone.quaternion.copy(parentQ).invert().multiply(q);
    const parentInfo = parent ? this.bones.get(parent) : undefined;
    const parentP = parent ? this.worldP.get(parent) : undefined;
    const p = info.head.clone();
    if (parentInfo && parentP) p.sub(parentInfo.head).applyQuaternion(parentQ).add(parentP);
    this.worldP.set(name, p);
  }

  private follow(name: string): void {
    this.setWorld(name, this.parentWorld(name));
  }

  /** Smallest turn that makes the bone point along `dir`, starting from its parent's rotation. */
  private aim(name: string, dir: THREE.Vector3): void {
    const info = this.bones.get(name);
    if (!info || dir.lengthSq() < 1e-12) return this.follow(name);
    const parentQ = this.parentWorld(name);
    const current = info.restDir.clone().applyQuaternion(parentQ);
    const q = new THREE.Quaternion().setFromUnitVectors(current, dir.clone().normalize());
    this.setWorld(name, q.multiply(parentQ));
  }

  /** End of a bone this frame (a fingertip for the last finger bones). */
  private boneEnd(name: string): THREE.Vector3 | null {
    const info = this.bones.get(name);
    const p = this.worldP.get(name);
    const q = this.worldQ.get(name);
    if (!info || !p || !q) return null;
    return info.restDir.clone().applyQuaternion(q).multiplyScalar(info.length).add(p);
  }

  /** Points of a posed hand checked against the body, with their thickness. */
  private handPoints(side: Side): [THREE.Vector3, number][] {
    const out: [THREE.Vector3, number][] = [];
    const wrist = this.worldP.get(`wrist_${side}`);
    if (wrist) out.push([wrist.clone(), 0.03]);
    for (const f of ['finger1-3', 'finger2-3', 'finger3-3', 'finger4-3', 'finger5-3']) {
      const tip = this.boneEnd(`${f}_${side}`);
      if (tip) out.push([tip, 0.012]);
    }
    for (const f of ['finger2-1', 'finger5-1', 'finger1-2', 'finger3-2']) {
      const p = this.worldP.get(`${f}_${side}`);
      if (p) out.push([p.clone(), 0.018]);
    }
    return out;
  }

  /** Joint position of a bone this frame. */
  jointPosition(name: string): THREE.Vector3 | null {
    return this.worldP.get(name) ?? null;
  }

  // ---------- collisions ----------

  /**
   * Pushes a world point out of the torso, head and neck, keeping `margin`
   * from the surface. Returns the total depth it was inside.
   */
  pushOut(p: THREE.Vector3, margin: number): number {
    let depth = 0;
    // Torso: into its rest coordinates, test, and back.
    const local = p.clone().sub(this.torsoPivot).applyQuaternion(this.torsoQ.clone().invert()).add(this.bones.get('spine02')!.head);
    const d1 = this.body.pushOutOfTorso(local, margin);
    if (d1 > 0) {
      depth += d1;
      p.copy(local.sub(this.bones.get('spine02')!.head).applyQuaternion(this.torsoQ).add(this.torsoPivot));
    }
    // Head.
    const headQ = this.worldQ.get('head');
    const headP = this.worldP.get('head');
    if (headQ && headP) {
      const restHead = this.bones.get('head')!.head;
      const h = p.clone().sub(headP).applyQuaternion(headQ.clone().invert()).add(restHead);
      const d2 = this.body.pushOutOfHead(h, margin);
      if (d2 > 0) {
        depth += d2;
        p.copy(h.sub(restHead).applyQuaternion(headQ).add(headP));
      }
      // Neck: a cylinder from the base of the neck to the head joint.
      const base = this.worldP.get('neck01');
      if (base) {
        const axis = headP.clone().sub(base);
        const len = axis.length();
        axis.divideScalar(len || 1);
        const t = clamp(p.clone().sub(base).dot(axis), 0, len);
        const closest = base.clone().addScaledVector(axis, t);
        const off = p.clone().sub(closest);
        const r = 0.055 + margin;
        const dist = off.length();
        if (dist < r && t > 0 && t < len) {
          depth += r - dist;
          if (dist < 1e-6) off.set(0, 0, 1);
          p.copy(closest.addScaledVector(off.normalize(), r));
        }
      }
    }
    return depth;
  }

  /** How deep a point is inside the body (without moving it). */
  depthInside(p: THREE.Vector3, margin: number): number {
    return this.pushOut(p.clone(), margin);
  }

  // ---------- posing ----------

  /** A figure point (shoulder widths, origin between the shoulders) in model space, relative to the shoulders. */
  private fromBody(fig: Figure, p: Vec3, out = new THREE.Vector3()): THREE.Vector3 {
    const c = fig.chest.center;
    return out.set(p[0] - c[0], p[1] - c[1], p[2] - c[2]).multiplyScalar(this.shoulderWidth).add(this.shoulderMid);
  }

  /**
   * Where the wrist goes for a hand near the face, and how near it is (1 at
   * the face, fading to 0 further away), or null without a face.
   *
   * The picture shows reliably where the hand is in front of the face (up,
   * down, left, right) but not how far in front: the trackers' depth is too
   * rough. So the hand point closest to the face's centre in the picture is
   * put ON the model's face, at the same place relative to the face outline
   * (measured in face sizes, chin to forehead and cheek to cheek): a sign at
   * the chin touches the chin, one at the forehead touches the forehead.
   */
  private faceTarget(fig: Figure, hand: Vec3[]): Contact | null {
    const h = fig.head;
    const headQ = this.worldQ.get('head');
    const headP = this.worldP.get('head');
    if (!h || !headQ || !headP) return null;
    const { center, half } = h.box;
    // Distance of each hand point to the face's centre, in face sizes.
    const dist = HAND_CHECK.map((i) => Math.hypot((hand[i]![0] - center[0]) / half[0], (hand[i]![1] - center[1]) / half[1]));
    const nearest = Math.min(...dist);
    const weight = 1 - smoothstep(1.15, 1.7, nearest);
    if (weight <= 0) return null;
    // The touching point is the one nearest the face. When two are about as
    // near, both share the touch, so the hand does not jump from one to the
    // other: shares fall off quickly with the extra distance.
    const raw = dist.map((d) => Math.exp(-(d - nearest) / 0.08));
    const total = raw.reduce((a, b) => a + b, 0);
    const k = this.handLength / Math.max(1e-6, Math.hypot(hand[9]![0] - hand[0]![0], hand[9]![1] - hand[0]![1], hand[9]![2] - hand[0]![2]));
    const f = h.features;
    const face = this.body.face;
    const c = this.body.headCenter;
    const r = this.body.headRadii;
    const p = SHAPE_POWER;
    const points: Contact['points'] = [];
    const wrist = new THREE.Vector3();
    HAND_CHECK.forEach((i, n) => {
      const share = raw[n]! / total;
      if (share < 0.01) return;
      // The same place on the model's face (rest coordinates of the head),
      // measured from the eyes: sideways in eye distances, downwards in
      // eyes-to-mouth distances. Then onto the front of its head shape.
      const x = ((hand[i]![0] - f.eyes[0]) / f.eyeDistance) * face.eyeDistance;
      const y = face.eyesY + ((hand[i]![1] - f.eyes[1]) / Math.max(1e-3, f.eyes[1] - f.mouth[1])) * (face.eyesY - face.mouthY);
      const front = Math.max(0, 1 - Math.abs((x - c.x) / r.x) ** p - Math.abs((y - c.y) / r.y) ** p) ** (1 / p);
      const at = new THREE.Vector3(x, y, c.z + front * r.z + 0.012)
        .sub(this.bones.get('head')!.head).applyQuaternion(headQ).add(headP);
      points.push({ point: i, at, share });
      // The wrist is where it must be for that hand point to be there, with the hand turned as recorded.
      const offset = new THREE.Vector3(hand[i]![0] - hand[0]![0], hand[i]![1] - hand[0]![1], hand[i]![2] - hand[0]![2]).multiplyScalar(k);
      wrist.addScaledVector(at.clone().sub(offset), share);
    });
    const sum = points.reduce((a, q) => a + q.share, 0);
    wrist.divideScalar(sum);
    points.forEach((q) => (q.share /= sum));
    return { points, wrist, weight };
  }

  /** The body: chest from the shoulder line (limited), breathing, and the head from the face. */
  private poseBody(fig: Figure, lean: number): void {
    const id = new THREE.Quaternion();
    for (const name of ['root', 'spine05', 'spine04']) this.setWorld(name, id);

    // Chest turn and tilt from the recorded shoulder line, within what a torso does.
    const a = fig.chest.across;
    const yaw = clamp(Math.atan2(-a[2], a[0]), -deg(18), deg(18));
    const roll = clamp(Math.atan2(a[1], a[0]), -deg(8), deg(8));
    // Breathing: about 15 breaths a minute; the chest rises a little.
    const breath = Math.sin((this.time / 4200) * Math.PI * 2);
    const pitch = deg(1.5) * lean + deg(0.6) * breath;
    const chest = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll, 'YXZ'));
    const part = (k: number) => id.clone().slerp(chest, k);
    this.setWorld('spine03', part(0.3));
    this.setWorld('spine02', part(0.65));
    this.setWorld('spine01', chest);
    this.torsoQ.copy(part(0.65));
    this.torsoPivot.copy(this.worldP.get('spine02')!);

    // Head: the face's orientation, within the neck's range, shared over the neck.
    let head = chest.clone();
    if (fig.head) {
      const [x, y, z] = fig.head.axes;
      const m = new THREE.Matrix4().makeBasis(new THREE.Vector3(...x), new THREE.Vector3(...y), new THREE.Vector3(...z));
      const face = new THREE.Quaternion().setFromRotationMatrix(m);
      const rel = chest.clone().invert().multiply(face);
      const angle = 2 * Math.acos(clamp(Math.abs(rel.w), -1, 1));
      const max = deg(25);
      if (angle > max) rel.copy(id.clone().slerp(rel, max / angle));
      head = chest.clone().multiply(rel);
    }
    const rel = chest.clone().invert().multiply(head);
    const at = (k: number) => chest.clone().multiply(id.clone().slerp(rel, k));
    this.setWorld('neck01', at(0.25));
    this.setWorld('neck02', at(0.5));
    this.setWorld('neck03', at(0.75));
    this.setWorld('head', head);
  }

  /** Collarbone: lifts when the hand goes above the shoulder, comes forwards when it reaches forwards. */
  private poseClavicle(side: Side, wrist: THREE.Vector3): void {
    const sgn = side === 'L' ? 1 : -1;
    const shoulder = this.bones.get(`upperarm01_${side}`)!.head;
    const reach = this.upperArm + this.forearm;
    const up = clamp((wrist.y - shoulder.y) / reach + 0.15, 0, 1);
    const forward = clamp((wrist.z - shoulder.z) / reach, 0, 1);
    const across = clamp((-sgn * wrist.x) / reach, 0, 1); // hand on the other side of the body
    const breath = 0.5 + 0.5 * Math.sin((this.time / 4200) * Math.PI * 2);
    const lift = deg(16) * up + deg(0.5) * breath;
    const protract = deg(9) * forward + deg(7) * across;
    const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, -sgn * protract, sgn * lift, 'YXZ'));
    this.setWorld(`clavicle_${side}`, this.worldQ.get('spine01')!.clone().multiply(q));
    this.follow(`shoulder01_${side}`);
  }

  /**
   * Places one arm. `wrist` is the target for the wrist joint, `pole` the
   * direction the elbow should point to, `handQ` the hand's rotation.
   */
  private poseArm(side: Side, wrist: THREE.Vector3, pole: THREE.Vector3, hand: Vec3[] | null, handQ: THREE.Quaternion | null): void {
    const upper = `upperarm01_${side}`;
    const S = this.shoulderJoint(side);
    const L1 = this.upperArm;
    const L2 = this.forearm;

    // Two-bone IK. Distance shoulder-wrist, kept within what the arm can reach
    // (never fully straight: a locked elbow looks robotic).
    const toWrist = wrist.clone().sub(S);
    const dist = clamp(toWrist.length(), Math.abs(L1 - L2) + 0.02, 0.97 * (L1 + L2));
    const axis = toWrist.lengthSq() > 1e-10 ? toWrist.normalize() : new THREE.Vector3(0, -1, 0);
    const W = S.clone().addScaledVector(axis, dist);
    // Law of cosines: how far along the shoulder-wrist line the elbow is, and how far off it.
    const cosA = clamp((L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist), -1, 1);
    const along = L1 * cosA;
    const off = L1 * Math.sqrt(1 - cosA * cosA);
    // The elbow's circle: centre on the line, in the plane perpendicular to it.
    let n0 = pole.clone().addScaledVector(axis, -pole.dot(axis));
    if (n0.lengthSq() < 1e-8) n0 = new THREE.Vector3(0, -1, 0).addScaledVector(axis, axis.y);
    n0.normalize();
    const elbowAt = (theta: number) => {
      const n = n0.clone().applyAxisAngle(axis, theta);
      return S.clone().addScaledVector(axis, along).addScaledVector(n, off);
    };
    // Turn the elbow along its circle until the arm stays out of the body,
    // preferring the smallest turn from where the recording puts it.
    const cost = (E: THREE.Vector3) => {
      let c = 0;
      for (const t of [0.45, 0.75, 1]) c += this.depthInside(S.clone().lerp(E, t), 0.04);
      for (const t of [0.25, 0.5, 0.75]) c += this.depthInside(E.clone().lerp(W, t), 0.034);
      return c;
    };
    let best = 0;
    let bestCost = cost(elbowAt(0));
    if (bestCost > 1e-4) {
      for (let k = 1; k <= 12; k++) {
        for (const theta of [deg(7.5 * k), -deg(7.5 * k)]) {
          const c = cost(elbowAt(theta)) + Math.abs(theta) * 0.002;
          if (c < bestCost) {
            bestCost = c;
            best = theta;
          }
        }
      }
    }
    const E = elbowAt(best);

    // Upper arm: point at the elbow, rolled so its bend plane contains the wrist.
    const info = this.bones.get(upper)!;
    const foreRest = this.bones.get(`lowerarm01_${side}`)!.restDir;
    const a0 = info.restDir.clone();
    const b0 = new THREE.Vector3().crossVectors(a0, foreRest).normalize();
    const a1 = E.clone().sub(S).normalize();
    const b1 = new THREE.Vector3().crossVectors(a1, W.clone().sub(E)).normalize();
    if (b1.lengthSq() < 0.5) b1.crossVectors(a1, n0).normalize(); // straight arm: any plane through the pole
    const frame = (a: THREE.Vector3, b: THREE.Vector3) => new THREE.Matrix4().makeBasis(a, b, new THREE.Vector3().crossVectors(a, b));
    const upperQ = new THREE.Quaternion().setFromRotationMatrix(frame(a1, b1).multiply(frame(a0, b0).transpose()));
    this.setWorld(upper, upperQ);
    this.follow(`upperarm02_${side}`);
    this.aim(`lowerarm01_${side}`, W.clone().sub(E));

    if (!hand || !handQ) {
      this.follow(`lowerarm02_${side}`);
      this.follow(`wrist_${side}`);
      for (let m = 1; m <= 4; m++) this.follow(`metacarpal${m}_${side}`);
      for (const [f] of FINGERS) this.curl(`${f}_${side}`, RELAXED_CURL[Number(f.slice(-1)) - 1]!);
      return;
    }
    const actual = this.poseWrist(side, handQ);
    for (let m = 1; m <= 4; m++) this.follow(`metacarpal${m}_${side}`);
    // The fingers keep their shape relative to the hand, even where the wrist
    // could not turn all the way.
    const fix = actual.clone().multiply(handQ.clone().invert());
    const wristP = new THREE.Vector3(...hand[0]!);
    const local = hand.map((p) => new THREE.Vector3(...p).sub(wristP).applyQuaternion(fix));
    this.poseFingers(side, local);
  }

  /**
   * Poses an arm towards a wrist target, then checks the posed hand against
   * the body: if a part of it is inside, the target moves out by that much
   * and the arm is posed again, staying within reach. Returns the final target.
   */
  private placeArm(side: Side, wrist: THREE.Vector3, pole: THREE.Vector3, hand: Vec3[] | null, handQ: THREE.Quaternion | null, rounds: number, contact: Contact | null = null): THREE.Vector3 {
    this.withinReach(side, wrist);
    this.pushOut(wrist, 0.03);
    for (let pass = 0; pass < rounds; pass++) {
      this.withinReach(side, wrist);
      this.poseArm(side, wrist, pole, hand, handQ);
      const move = new THREE.Vector3();
      // A touch: the posed hand may be turned differently from the recording
      // (joint limits, other proportions), so move the wrist by however far the
      // touching point still is from its place.
      if (contact) {
        for (const { point, at, share } of contact.points) {
          const actual = this.handPoint(side, point);
          if (actual) move.addScaledVector(at.clone().sub(actual), share * contact.weight);
        }
      }
      // Out of the body: each posed hand point inside it says how far it must
      // move. The deeper points count more (an average weighted by depth: like
      // taking the deepest, but without jumping when another becomes deepest).
      const push = new THREE.Vector3();
      let total = 0;
      for (const [p, margin] of this.handPoints(side)) {
        const q = p.clone().add(move);
        const before = q.clone();
        if (this.pushOut(q, margin) <= 0) continue;
        const d = q.sub(before);
        const len = d.length();
        push.addScaledVector(d, len);
        total += len;
      }
      if (total > 0) move.addScaledVector(push, 1 / total);
      if (move.lengthSq() < 1e-8) break;
      wrist.add(move);
      if (pass === rounds - 1) {
        this.withinReach(side, wrist);
        this.poseArm(side, wrist, pole, hand, handQ);
      }
    }
    return wrist;
  }

  /** Where one of the 21 hand points is on the posed model. */
  handPoint(side: Side, i: number): THREE.Vector3 | null {
    const entry = HAND_POINT_BONE[i];
    if (!entry) return null;
    const [bone, at] = entry;
    return at === 'head' ? (this.worldP.get(`${bone}_${side}`)?.clone() ?? null) : this.boneEnd(`${bone}_${side}`);
  }

  /** Where the shoulder joint is this frame (it moves with the collarbone). */
  private shoulderJoint(side: Side): THREE.Vector3 {
    const sh = `shoulder01_${side}`;
    const offset = this.bones.get(`upperarm01_${side}`)!.head.clone().sub(this.bones.get(sh)!.head);
    return offset.applyQuaternion(this.worldQ.get(sh)!).add(this.worldP.get(sh)!);
  }

  /** Keeps a wrist target within the arm's reach from the shoulder (never quite straight). */
  private withinReach(side: Side, wrist: THREE.Vector3): void {
    const S = this.shoulderJoint(side);
    const d = wrist.clone().sub(S);
    const len = d.length();
    const max = 0.97 * (this.upperArm + this.forearm);
    const min = Math.abs(this.upperArm - this.forearm) + 0.02;
    if (len > max) wrist.copy(S).addScaledVector(d, max / len);
    else if (len < min && len > 1e-9) wrist.copy(S).addScaledVector(d, min / len);
  }

  /** Rotation of the hand at rest (model space -> its own frame), from its knuckles. */
  private restHandFrame(side: Side): THREE.Matrix4 {
    const h = (n: string) => this.bones.get(`${n}_${side}`)!.head;
    return handFrame(h('finger3-1').clone().sub(h('wrist')), h('finger2-1').clone().sub(h('finger5-1')));
  }

  /** The hand's rotation from recorded points: wrist to middle knuckle, and little to index knuckle. */
  private handRotation(side: Side, hand: Vec3[]): THREE.Quaternion {
    const v = (i: number) => new THREE.Vector3(...hand[i]!);
    const target = handFrame(v(9).sub(v(0)), v(5).sub(v(17)));
    return new THREE.Quaternion().setFromRotationMatrix(target.multiply(this.restHandFrame(side).transpose()));
  }

  /**
   * Wrist within a real wrist's range. The hand's rotation relative to the
   * forearm is split into a turn around the forearm (twist, done by the
   * forearm's two bones crossing) and a bend (swing):
   *   twist: up to 100 degrees either way, 60% of it shown in the forearm;
   *   bend:  70 towards the palm, 60 backwards, 25 sideways.
   * Returns the hand's actual rotation.
   */
  private poseWrist(side: Side, handQ: THREE.Quaternion): THREE.Quaternion {
    const foreQ = this.worldQ.get(`lowerarm01_${side}`)!;
    const rel = foreQ.clone().invert().multiply(handQ);
    const axis = this.bones.get(`lowerarm01_${side}`)!.restDir;
    // Swing-twist decomposition: rel = swing * twist.
    const d = rel.x * axis.x + rel.y * axis.y + rel.z * axis.z;
    const twist = new THREE.Quaternion(axis.x * d, axis.y * d, axis.z * d, rel.w);
    if (twist.lengthSq() < 1e-10) twist.identity();
    else twist.normalize();
    if (twist.w < 0) twist.set(-twist.x, -twist.y, -twist.z, -twist.w);
    const swing = rel.clone().multiply(twist.clone().invert());

    // Limit the twist.
    const twistAngle = 2 * Math.atan2(twist.x * axis.x + twist.y * axis.y + twist.z * axis.z, twist.w);
    const limitedTwist = clamp(twistAngle, -deg(100), deg(100));
    twist.setFromAxisAngle(axis, limitedTwist);

    // Limit the bend: express it as a rotation vector, split into bending
    // towards the palm (around the knuckle line) and sideways (around the palm's normal).
    if (swing.w < 0) swing.set(-swing.x, -swing.y, -swing.z, -swing.w);
    const sinHalf = Math.hypot(swing.x, swing.y, swing.z);
    const angle = 2 * Math.atan2(sinHalf, swing.w);
    if (sinHalf > 1e-8) {
      const rv = new THREE.Vector3(swing.x, swing.y, swing.z).multiplyScalar(angle / sinHalf);
      const h = (n: string) => this.bones.get(`${n}_${side}`)!.head;
      const knuckles = h('finger2-1').clone().sub(h('finger5-1')).applyQuaternion(twist);
      const u1 = knuckles.addScaledVector(axis, -knuckles.dot(axis)).normalize();
      const u2 = new THREE.Vector3().crossVectors(axis, u1);
      // Which sign of u1 bends the fingers towards the palm.
      const palm = this.palmNormal(side, h('finger3-1').clone().sub(h('wrist')), h('finger2-1').clone().sub(h('finger5-1'))).applyQuaternion(twist);
      const flexSign = Math.sign(new THREE.Vector3().crossVectors(u1, axis).dot(palm)) || 1;
      const flex = rv.dot(u1) * flexSign;
      const side2 = rv.dot(u2);
      const flexLimit = flex > 0 ? deg(70) : deg(60);
      const sideLimit = deg(25);
      const e = Math.hypot(flex / flexLimit, side2 / sideLimit);
      if (e > 1) {
        rv.copy(u1).multiplyScalar((flex / e) * flexSign).addScaledVector(u2, side2 / e);
        const a = rv.length();
        swing.setFromAxisAngle(a > 1e-9 ? rv.clone().divideScalar(a) : axis, a);
      }
    }
    const share = new THREE.Quaternion().slerp(twist, 0.6);
    this.setWorld(`lowerarm02_${side}`, foreQ.clone().multiply(share));
    const actual = foreQ.clone().multiply(swing).multiply(twist);
    this.setWorld(`wrist_${side}`, actual);
    return actual;
  }

  /** The side of the hand the fingers fold towards, from the hand's direction and its knuckle line (little to index). */
  private palmNormal(side: Side, along: THREE.Vector3, littleToIndex: THREE.Vector3): THREE.Vector3 {
    const n = new THREE.Vector3().crossVectors(along, littleToIndex).normalize();
    return side === 'L' ? n : n.negate();
  }

  /** Bends a finger bone by `angle` around its own sideways axis. */
  private curl(name: string, angle: number): void {
    const parentQ = this.parentWorld(name);
    const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), angle);
    this.setWorld(name, parentQ.clone().multiply(q));
  }

  /**
   * Fingers within what a real hand can do. The camera's depth for fingers is
   * noisy, so taken as is a finger can bend sideways or backwards. Here:
   * - the knuckle bends towards the palm by -25 to 95 degrees and spreads
   *   sideways by at most 22 degrees;
   * - the two other joints are hinges: they only bend towards the palm, up to
   *   110 and 90 degrees.
   * The thumb moves more freely and only follows the recording.
   * `hand` is the 21 points relative to the wrist, already in the hand's actual frame.
   */
  private poseFingers(side: Side, hand: THREE.Vector3[]): void {
    const wristQ = this.worldQ.get(`wrist_${side}`)!;
    const h = (n: string) => this.bones.get(`${n}_${side}`)!.head;
    const along = h('finger3-1').clone().sub(h('wrist')).applyQuaternion(wristQ);
    const across = h('finger2-1').clone().sub(h('finger5-1')).applyQuaternion(wristQ);
    const palm = this.palmNormal(side, along, across);
    const LIMITS: [number, number, number][] = [[-25, 95, 22], [-8, 110, 0], [-10, 90, 0]];

    for (const [f, a, b] of FINGERS) {
      const name = `${f}_${side}`;
      const target = hand[b]!.clone().sub(hand[a]!);
      const info = this.bones.get(name);
      if (f.startsWith('finger1') || !info || target.lengthSq() < 1e-12) {
        this.aim(name, target);
        continue;
      }
      const joint = Number(f.slice(-1)) - 1; // 0 knuckle, 1 middle, 2 end
      const [minFlex, maxFlex, maxSpread] = LIMITS[joint]!;
      // Direction the finger would have with this joint straight.
      const straight = info.restDir.clone().applyQuaternion(this.parentWorld(name)).normalize();
      const hinge = new THREE.Vector3().crossVectors(straight, palm);
      if (hinge.lengthSq() < 1e-8) {
        this.aim(name, target);
        continue;
      }
      hinge.normalize();
      const fold = new THREE.Vector3().crossVectors(hinge, straight).normalize();
      target.normalize();
      const flex = clamp(Math.atan2(target.dot(fold), target.dot(straight)), deg(minFlex), deg(maxFlex));
      const spread = clamp(Math.asin(clamp(target.dot(hinge), -1, 1)), -deg(maxSpread), deg(maxSpread));
      const dir = straight.multiplyScalar(Math.cos(flex)).addScaledVector(fold, Math.sin(flex)).multiplyScalar(Math.cos(spread)).addScaledVector(hinge, Math.sin(spread));
      this.aim(name, dir);
    }
  }

  /** Where a figure's wrist goes on the model, and which way its elbow points. */
  private armTargets(fig: Figure, side: Side): { wrist: THREE.Vector3; pole: THREE.Vector3; contact: Contact | null } {
    const arm = side === 'L' ? fig.left : fig.right;
    const sgn = side === 'L' ? 1 : -1;
    // Where the picture shows the wrist, relative to the shoulders.
    const wrist = this.fromBody(fig, arm.seen.wrist);
    // Near the face, place the hand relative to the face.
    const contact = arm.seen.hand ? this.faceTarget(fig, arm.seen.hand) : null;
    if (contact) wrist.lerp(contact.wrist, contact.weight);
    // Elbow direction: the recorded elbow's offset from the shoulder-wrist
    // line, half-trusted (its depth is a guess), and half a natural default
    // (down, out to the side and a little back).
    const S: Vec3 = arm.shoulder;
    const line = new THREE.Vector3(arm.wrist[0] - S[0], arm.wrist[1] - S[1], arm.wrist[2] - S[2]).normalize();
    const e = new THREE.Vector3(arm.elbow[0] - S[0], arm.elbow[1] - S[1], arm.elbow[2] - S[2]);
    e.addScaledVector(line, -e.dot(line));
    const recorded = e.lengthSq() > 1e-6 ? e.normalize() : new THREE.Vector3();
    const natural = new THREE.Vector3(0.45 * sgn, -1, -0.25).normalize();
    return { wrist, pole: recorded.multiplyScalar(0.5).addScaledVector(natural, 0.5), contact };
  }

  /**
   * Poses the model from a figure. `dtMs` is the time since the last call;
   * `snap` jumps straight to the pose (a new demonstration starts).
   */
  update(fig: Figure, dtMs: number, snap = false): void {
    this.time += Math.max(0, dtMs);
    for (const { bone } of this.bones.values()) bone.quaternion.identity();
    this.worldQ.clear();
    this.worldP.clear();

    // Targets first: the chest leans a little towards hands held far forwards.
    const targets = (['L', 'R'] as Side[]).map((s) => this.armTargets(fig, s));
    const forward = Math.max(...targets.map((t) => clamp((t.wrist.z - this.shoulderMid.z) / (this.upperArm + this.forearm), 0, 1)));
    this.poseBody(fig, forward);

    (['L', 'R'] as Side[]).forEach((side, i) => {
      const arm = side === 'L' ? fig.left : fig.right;
      const t = targets[i]!;
      // Face-relative targets depend on the head, posed just above: recompute.
      const again = this.armTargets(fig, side);
      const state = this.arms[side];
      const pole = state.pole.update(t.pole, dtMs, snap).clone();
      const handQ = arm.hand ? this.handRotation(side, arm.hand) : null;
      // Make the target reachable and clear of the body BEFORE smoothing: the
      // spring then glides between valid places (a correction made after it
      // would jolt). One more check after it catches the rare corner it cuts.
      this.poseClavicle(side, again.wrist);
      const clear = this.placeArm(side, again.wrist.clone(), pole, arm.hand, handQ, 5, again.contact);
      const wrist = state.wrist.update(clear, dtMs, snap).clone();
      this.poseClavicle(side, wrist);
      this.placeArm(side, wrist, pole, arm.hand, handQ, 2);
      const main = again.contact?.points.reduce((a, b) => (b.share > a.share ? b : a));
      this.touches[side] = main && again.contact!.weight > 0.9 && main.share > 0.9 ? { point: main.point, at: main.at } : null;
    });

    // Each joint glides to its new rotation; the body is heavier than the fingers.
    for (const { bone } of this.bones.values()) {
      const prev = this.shown.get(bone);
      if (!prev || snap) {
        this.shown.set(bone, bone.quaternion.clone());
        continue;
      }
      prev.slerp(bone.quaternion, 1 - Math.exp(-Math.max(0, dtMs) / this.lag.get(bone)!));
      bone.quaternion.copy(prev);
    }
  }
}


/** Rotation whose columns are a hand's own axes: along the hand, across it, and out of it. */
function handFrame(along: THREE.Vector3, across: THREE.Vector3): THREE.Matrix4 {
  const a = along.clone().normalize();
  const c = across.clone().addScaledVector(a, -across.dot(a)).normalize();
  return new THREE.Matrix4().makeBasis(a, c, new THREE.Vector3().crossVectors(a, c));
}
