/**
 * The 3D signer on screen: a sculpted human model (MakeHuman, CC0), lights,
 * camera, and a close-up of the hand for fingerspelling. How the model is moved
 * by the recorded sign data is in signerRig.ts.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { clone as cloneSkinned } from 'three/addons/utils/SkeletonUtils.js';
import type { Figure } from '../core/learn/figure';
import { SignerRig } from './signerRig';

export interface AvatarPalette {
  skin: string;
  rim: string;
  /** Background of the hand close-up. */
  inset: string;
}

const base = import.meta.env.BASE_URL;
let modelPromise: Promise<THREE.Group> | null = null;
function loadModel(): Promise<THREE.Group> {
  modelPromise ??= new GLTFLoader().loadAsync(`${base}models/avatar/signer.glb`).then((g) => g.scene);
  return modelPromise;
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
  private rig: SignerRig | null = null;
  private closeUpBone: THREE.Bone | null = null;
  private handFocus = new THREE.Vector3();
  private handVisible = false;
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

    // Head to waist, with room for hands raised above the head.
    this.camera.position.set(0, 0.62, 2.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.target.set(0, 0.57, 0);
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
      copy.traverse((o) => {
        if (o.name === 'finger3-1_R') this.closeUpBone = o as THREE.Bone;
        const mesh = o as THREE.SkinnedMesh;
        if (mesh.isSkinnedMesh) {
          mesh.material = this.skin;
          mesh.frustumCulled = false;
        }
      });
      this.rig = new SignerRig(copy);
      this.root.add(copy);
      onReady?.();
    });
  }

  get loaded(): boolean {
    return this.rig !== null;
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

  /**
   * Poses the figure (see SignerRig). `dtMs` is the time since the last frame;
   * `snap` jumps straight to the pose (a new sign starts).
   */
  update(fig: Figure, dtMs = 16, snap = false): void {
    if (!this.rig) return;
    this.rig.update(fig, dtMs, snap);
    this.handVisible = !!fig.right.hand && !!this.closeUpBone;
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
    this.closeUpBone!.getWorldPosition(this.handFocus);
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
