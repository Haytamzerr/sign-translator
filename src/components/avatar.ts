/**
 * The 3D signer: a sculpted human model (MakeHuman, CC0) moved by the recorded
 * sign data.
 *
 * The model has a skeleton: one bone per body segment and per finger joint,
 * and its skin follows the bones (skinning). The recordings give, for every
 * frame, where the shoulders, elbows, wrists and the 21 points of each hand
 * are. For each bone we compute the rotation that makes it point the same way
 * as the matching segment of the recording ("retargeting"):
 *
 *   arm segments   shoulder -> elbow, elbow -> wrist
 *   hand           wrist -> middle knuckle, turned so the palm faces the same way
 *   finger joints  each of the 15 finger segments
 *   head and neck  the face's orientation, shared over the 3 neck bones
 *   chest          the shoulder line
 *
 * Every rotation is the smallest turn from the bone's current direction to its
 * target, starting from its parent's rotation. That keeps fingers and arms from
 * twisting around themselves.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import type { Figure, Vec3 } from '../core/learn/figure';

export interface AvatarPalette {
  skin: string;
  rim: string;
  /** Background of the hand close-up. */
  inset: string;
}

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

const base = import.meta.env.BASE_URL;
let modelPromise: Promise<THREE.Group> | null = null;
function loadModel(): Promise<THREE.Group> {
  modelPromise ??= new GLTFLoader().loadAsync(`${base}models/avatar/signer.glb`).then((g) => g.scene);
  return modelPromise;
}

interface BoneInfo {
  bone: THREE.Bone;
  /** Unit direction head -> tail at rest, in model space (the rest pose has no rotation). */
  restDir: THREE.Vector3;
}

export class Avatar {
  readonly renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(24, 1, 0.05, 20);
  private closeCamera = new THREE.PerspectiveCamera(30, 1, 0.02, 10);
  private controls: OrbitControls;
  private root = new THREE.Group();
  private rim: THREE.DirectionalLight;
  private skin = new THREE.MeshPhysicalMaterial({ roughness: 0.58, metalness: 0, sheen: 0.35, sheenRoughness: 0.55, sheenColor: new THREE.Color(0x8fa6c4) });
  private insetColor = new THREE.Color();
  private bones = new Map<string, BoneInfo>();
  private ready = false;
  /** World rotations computed this frame, by bone name. */
  private worldQ = new Map<string, THREE.Quaternion>();
  private handFocus = new THREE.Vector3();
  private handVisible = false;

  // Reused objects.
  private qa = new THREE.Quaternion();
  private va = new THREE.Vector3();
  private vb = new THREE.Vector3();
  private vc = new THREE.Vector3();
  private m = new THREE.Matrix4();
  private size = new THREE.Vector2();

  constructor(canvas: HTMLCanvasElement, onReady?: () => void) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(1.5, window.devicePixelRatio || 1));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.setClearColor(0x000000, 0);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();
    this.scene.environmentIntensity = 0.35;
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(1.5, 2.5, 3);
    this.rim = new THREE.DirectionalLight(0x6fd3ff, 4);
    this.rim.position.set(-2.5, 1.5, -2.5);
    const rim2 = new THREE.DirectionalLight(0x9ab8ff, 2);
    rim2.position.set(2.5, 1, -2);
    this.scene.add(key, this.rim, rim2);
    this.scene.add(this.root);

    this.camera.position.set(0, 0.7, 1.85);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.66, 0);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enablePan = false;
    this.controls.minDistance = 0.9;
    this.controls.maxDistance = 3.5;
    this.controls.minAzimuthAngle = -1.4;
    this.controls.maxAzimuthAngle = 1.4;
    this.controls.minPolarAngle = 0.9;
    this.controls.maxPolarAngle = 2.0;
    this.controls.update();
    this.controls.saveState();

    void loadModel().then((model) => {
      // Each player gets its own copy of the model, with its own skeleton.
      const copy = cloneSkinned(model) as THREE.Group;
      const byName = new Map<string, THREE.Bone>();
      copy.traverse((o) => {
        if ((o as THREE.Bone).isBone) byName.set(o.name, o as THREE.Bone);
        const mesh = o as THREE.SkinnedMesh;
        if (mesh.isSkinnedMesh) {
          mesh.material = this.skin;
          mesh.frustumCulled = false;
        }
      });
      for (const [name, bone] of byName) {
        const { head, tail } = bone.userData as { head: number[]; tail: number[] };
        const dir = new THREE.Vector3(tail[0]! - head[0]!, tail[1]! - head[1]!, tail[2]! - head[2]!).normalize();
        this.bones.set(name, { bone, restDir: dir });
      }
      this.root.add(copy);
      this.ready = true;
      onReady?.();
    });
  }

  get loaded(): boolean {
    return this.ready;
  }

  setPalette(p: AvatarPalette): void {
    this.skin.color.set(p.skin);
    this.rim.color.set(p.rim);
    this.insetColor.set(p.inset);
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
    this.scene.environment?.dispose();
    this.skin.dispose();
    this.renderer.dispose();
  }

  // ---------- posing ----------

  private parentWorld(name: string): THREE.Quaternion {
    const parent = this.bones.get(name)?.bone.parent;
    return (parent && this.worldQ.get(parent.name)) ?? new THREE.Quaternion();
  }

  /** Sets a bone's world rotation and converts it to the local rotation three.js needs. */
  private setWorld(name: string, q: THREE.Quaternion): void {
    const info = this.bones.get(name);
    if (!info) return;
    this.worldQ.set(name, q.clone());
    const parentQ = this.parentWorld(name);
    info.bone.quaternion.copy(parentQ).invert().multiply(q);
  }

  /** The bone keeps its parent's rotation (no bend of its own). */
  private follow(name: string): void {
    this.setWorld(name, this.parentWorld(name));
  }

  /** Smallest turn that makes the bone point along `dir`, starting from its parent's rotation. */
  private aim(name: string, dir: THREE.Vector3): void {
    const info = this.bones.get(name);
    if (!info || dir.lengthSq() < 1e-10) return this.follow(name);
    const parentQ = this.parentWorld(name);
    const current = this.va.copy(info.restDir).applyQuaternion(parentQ);
    this.qa.setFromUnitVectors(current, this.vb.copy(dir).normalize());
    this.setWorld(name, this.qa.multiply(parentQ));
  }

  /** Bends the bone by `angle` around the axis across its finger, from its parent's rotation. */
  private curl(name: string, angle: number): void {
    const info = this.bones.get(name);
    if (!info) return;
    const parentQ = this.parentWorld(name);
    this.qa.setFromAxisAngle(this.va.set(1, 0, 0), angle);
    this.setWorld(name, parentQ.clone().multiply(this.qa));
  }

  private v(p: Vec3, out = new THREE.Vector3()): THREE.Vector3 {
    return out.set(p[0], p[1], p[2]);
  }

  private poseArm(arm: Figure['left'], side: 'L' | 'R'): void {
    this.follow(`clavicle_${side}`);
    this.follow(`shoulder01_${side}`);
    this.aim(`upperarm01_${side}`, this.v(arm.elbow).sub(this.v(arm.shoulder)));
    this.follow(`upperarm02_${side}`);
    this.aim(`lowerarm01_${side}`, this.v(arm.wrist).sub(this.v(arm.elbow)));
    this.follow(`lowerarm02_${side}`);

    const hand = arm.hand;
    const wrist = `wrist_${side}`;
    if (!hand) {
      this.follow(wrist);
      for (let m = 1; m <= 4; m++) this.follow(`metacarpal${m}_${side}`);
      FINGERS.forEach(([f], i) => this.curl(`${f}_${side}`, RELAXED_CURL[i % 3]!));
      return;
    }
    // Hand: point wrist -> middle knuckle, then turn around that line so the
    // palm faces the recorded way (index and little knuckles on the right sides).
    const info = this.bones.get(wrist);
    if (info) {
      const target = this.v(hand[9]!).sub(this.v(hand[0]!)).normalize();
      const across = this.v(hand[5]!).sub(this.v(hand[17]!));
      // Little-finger knuckle to index knuckle, the same way as `across` above.
      const restAcross = this.restVector(`finger5-1_${side}`, `finger2-1_${side}`);
      const parentQ = this.parentWorld(wrist);
      const restAim = this.vc.copy(info.restDir).applyQuaternion(parentQ);
      const q1 = new THREE.Quaternion().setFromUnitVectors(restAim, target);
      // Twist around the aim so the knuckle line matches.
      const acrossNow = restAcross.applyQuaternion(parentQ).applyQuaternion(q1);
      const project = (v: THREE.Vector3) => v.sub(target.clone().multiplyScalar(v.dot(target))).normalize();
      const a = project(acrossNow);
      const b = project(across);
      const angle = Math.atan2(a.clone().cross(b).dot(target), a.dot(b));
      const twist = new THREE.Quaternion().setFromAxisAngle(target, angle);
      const wristQ = twist.multiply(q1).multiply(parentQ);
      // A real forearm shares the hand's roll along its length; putting it all
      // in the wrist would pinch the skin there. Give 60% of it to the forearm.
      const forearm = `lowerarm02_${side}`;
      const forearmInfo = this.bones.get(forearm);
      const foreParent = this.parentWorld(forearm);
      if (forearmInfo) {
        const rel = foreParent.clone().invert().multiply(wristQ);
        const axis = forearmInfo.restDir;
        const d = rel.x * axis.x + rel.y * axis.y + rel.z * axis.z;
        const roll = new THREE.Quaternion(axis.x * d, axis.y * d, axis.z * d, rel.w).normalize();
        this.setWorld(forearm, foreParent.clone().multiply(new THREE.Quaternion().slerp(roll, 0.6)));
      }
      this.setWorld(wrist, wristQ);
    }
    for (let m = 1; m <= 4; m++) this.follow(`metacarpal${m}_${side}`);
    this.poseFingers(hand, side);
  }

  /**
   * Fingers within what a real hand can do. The camera's depth for fingers is
   * noisy, so taken as is a finger can bend sideways or backwards. Here:
   * - the knuckle (first joint) bends towards the palm by -25 to 95 degrees and
   *   spreads sideways by at most 22 degrees;
   * - the two other joints are hinges: they only bend towards the palm, up to
   *   110 and 90 degrees.
   * The thumb moves more freely and only follows the recording.
   */
  private poseFingers(hand: Vec3[], side: 'L' | 'R'): void {
    const wrist = this.bones.get(`wrist_${side}`);
    const wristQ = this.worldQ.get(`wrist_${side}`);
    if (!wrist || !wristQ) return;
    // Palm frame: along the hand, across the knuckles, and the palm's normal
    // (the side the fingers fold towards). Mirrored for the left hand.
    const along = wrist.restDir.clone().applyQuaternion(wristQ);
    const across = this.restVector(`finger2-1_${side}`, `finger5-1_${side}`).applyQuaternion(wristQ);
    const palm = new THREE.Vector3().crossVectors(along, across).normalize();
    if (side === 'L') palm.negate();
    const deg = THREE.MathUtils.degToRad;
    const LIMITS: [number, number, number][] = [[-25, 95, 22], [-8, 110, 0], [-10, 90, 0]];

    for (const [f, a, b] of FINGERS) {
      const name = `${f}_${side}`;
      const target = this.v(hand[b]!).sub(this.v(hand[a]!));
      const info = this.bones.get(name);
      if (f.startsWith('finger1') || !info || target.lengthSq() < 1e-10) {
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
      const flex = THREE.MathUtils.clamp(Math.atan2(target.dot(fold), target.dot(straight)), deg(minFlex), deg(maxFlex));
      const spread = THREE.MathUtils.clamp(Math.asin(THREE.MathUtils.clamp(target.dot(hinge), -1, 1)), -deg(maxSpread), deg(maxSpread));
      const dir = straight.multiplyScalar(Math.cos(flex)).addScaledVector(fold, Math.sin(flex)).multiplyScalar(Math.cos(spread)).addScaledVector(hinge, Math.sin(spread));
      this.aim(name, dir);
    }
  }

  /** Rest-pose vector between the heads of two bones (model space). */
  private restVector(from: string, to: string): THREE.Vector3 {
    const a = this.bones.get(from)?.bone.userData.head as number[] | undefined;
    const b = this.bones.get(to)?.bone.userData.head as number[] | undefined;
    if (!a || !b) return new THREE.Vector3(1, 0, 0);
    return new THREE.Vector3(b[0]! - a[0]!, b[1]! - a[1]!, b[2]! - a[2]!).normalize();
  }

  /** Rotation of each bone shown on the previous frame, for smoothing. */
  private shown = new Map<THREE.Bone, THREE.Quaternion>();

  /**
   * Poses the figure. Each joint glides towards its new rotation with a time
   * constant of 45 ms: invisible as delay, but a single noisy frame can no
   * longer snap a finger. `snap` jumps straight there (a new sign starts).
   */
  update(fig: Figure, dtMs = 16, snap = false): void {
    if (!this.ready) return;
    for (const { bone } of this.bones.values()) bone.quaternion.identity();
    this.worldQ.clear();
    const identity = new THREE.Quaternion();
    for (const name of ['root', 'spine05', 'spine04', 'spine03', 'spine02', 'spine01']) this.worldQ.set(name, identity);

    // Chest: turned with the shoulder line, spread over the upper spine.
    const across = this.v(fig.chest.across);
    const chest = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), across);
    const part = (k: number) => identity.clone().slerp(chest, k);
    this.setWorld('spine03', part(0.35));
    this.setWorld('spine02', part(0.7));
    this.setWorld('spine01', chest);

    this.poseArm(fig.left, 'L');
    this.poseArm(fig.right, 'R');

    // Head: the face's orientation, shared over the neck.
    if (fig.head) {
      const [x, y, z] = fig.head.axes;
      this.m.makeBasis(this.v(x), this.v(y), this.v(z));
      const head = new THREE.Quaternion().setFromRotationMatrix(this.m);
      const rel = chest.clone().invert().multiply(head);
      const at = (k: number) => chest.clone().multiply(identity.clone().slerp(rel, k));
      this.setWorld('neck01', at(0.25));
      this.setWorld('neck02', at(0.5));
      this.setWorld('neck03', at(0.75));
      this.setWorld('head', head);
    } else {
      for (const n of ['neck01', 'neck02', 'neck03', 'head']) this.follow(n);
    }

    const k = snap ? 1 : 1 - Math.exp(-Math.max(0, dtMs) / 45);
    for (const { bone } of this.bones.values()) {
      const prev = this.shown.get(bone);
      if (!prev) {
        this.shown.set(bone, bone.quaternion.clone());
        continue;
      }
      prev.slerp(bone.quaternion, k);
      bone.quaternion.copy(prev);
    }
    const hand = this.bones.get('finger3-1_R')?.bone;
    this.handVisible = !!fig.right.hand && !!hand;
    if (hand) {
      this.root.updateMatrixWorld(true);
      hand.getWorldPosition(this.handFocus);
    }
  }

  render(mirror: boolean, closeUp: boolean): { x: number; y: number; size: number } | null {
    this.root.scale.x = mirror ? -1 : 1;
    this.controls.update();
    const r = this.renderer;
    const { x: w, y: h } = r.getSize(this.size);
    r.autoClear = false;
    r.setScissorTest(false);
    r.setViewport(0, 0, w, h);
    r.clear();
    r.render(this.scene, this.camera);
    if (!closeUp || !this.handVisible) return null;

    const box = Math.round(Math.min(w, h) * 0.4);
    const pad = Math.round(Math.min(w, h) * 0.035);
    const x = mirror ? pad : w - box - pad;
    this.root.updateMatrixWorld(true);
    this.bones.get('finger3-1_R')!.bone.getWorldPosition(this.handFocus);
    this.closeCamera.position.set(this.handFocus.x, this.handFocus.y, this.handFocus.z + 0.42);
    this.closeCamera.lookAt(this.handFocus);
    this.closeCamera.aspect = 1;
    this.closeCamera.updateProjectionMatrix();
    r.setScissorTest(true);
    r.setScissor(x, h - pad - box, box, box);
    r.setViewport(x, h - pad - box, box, box);
    r.setClearColor(this.insetColor, 1);
    r.clear();
    r.render(this.scene, this.closeCamera);
    r.setClearColor(0x000000, 0);
    r.setScissorTest(false);
    return { x, y: pad, size: box };
  }
}
