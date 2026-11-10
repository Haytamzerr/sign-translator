/**
 * Sign demonstrations: real recordings of the 250 ASL signs, stored as body,
 * face and hand points in 3D (no images), and played back by a 3D figure.
 *
 * File format (made by training/prepare_learning_data.py):
 * - sign_demos.json: point layout, and for each sign its offset and frame count
 * - sign_demos.bin: 16-bit integers, x, y and z of 159 points per frame, in
 *   thousandths of a shoulder width, measured from the middle of the
 *   shoulders. -32768 means "not visible".
 *
 * Axes are those of the camera image: x to the right, y down, z away from the
 * camera (a smaller z is closer to the viewer). The body's depth is measured
 * from the middle of the shoulders; a hand's depth from its own wrist and the
 * face's from its own centre (only their shape is used).
 *
 * Point layout of a frame: left hand 0-20, right hand 21-41, body 42-50
 * (nose, shoulders, elbows, wrists, hips), face 51-158 (outline, lips, eyes,
 * eyebrows). "Left" and "right" are the signer's.
 */

export const DEMO_POINTS = 159;
export const DIMS = 3;

/** Index of each body point inside the "body" range. */
export const BODY = {
  nose: 0,
  leftShoulder: 1,
  rightShoulder: 2,
  leftElbow: 3,
  rightElbow: 4,
  leftWrist: 5,
  rightWrist: 6,
  leftHip: 7,
  rightHip: 8,
} as const;

export interface DemoLayout {
  leftHand: [number, number];
  rightHand: [number, number];
  body: [number, number];
  faceOval: [number, number];
  lips: [number, number];
  eyeL: [number, number];
  eyeR: [number, number];
  browL: [number, number];
  browR: [number, number];
}

/** One frame: x0, y0, z0, x1, y1, z1, ... (NaN = not visible). */
export type Pose = Float32Array;

/** A letter handshape: 21 points, wrist at the origin, hand size 1. */
export type HandShape = number[][];

export interface SignDemo {
  label: string;
  frames: Pose[];
}

/**
 * Proportions of the figure, measured once over all recordings so that the
 * figure keeps the same body from one sign to the next. Lengths are in
 * shoulder widths.
 */
export interface Rig {
  upperArm: number;
  forearm: number;
  /** Wrist to the middle finger's knuckle. */
  hand: number;
  faceWidth: number;
  faceHeight: number;
  /** +1 when the "eyeR" points are on the right of the image, -1 otherwise. */
  eyeSide: number;
}

interface DemoIndex {
  points: number;
  dims?: number;
  scale: number;
  missing: number;
  fps: number;
  layout: DemoLayout;
  signs: { label: string; offset: number; frames: number; confidence: number }[];
}

export class DemoLibrary {
  constructor(
    readonly fps: number,
    readonly layout: DemoLayout,
    private demos: Map<string, SignDemo>,
    /** A calm standing pose, used between signs and while spelling. */
    readonly neutral: Pose,
    readonly rig: Rig,
  ) {}

  get(label: string): SignDemo | undefined {
    return this.demos.get(label);
  }

  has(label: string): boolean {
    return this.demos.has(label);
  }

  get labels(): string[] {
    return [...this.demos.keys()];
  }
}

/** Reads point i of a pose. */
export function point(pose: Pose, i: number): [number, number, number] {
  return [pose[i * 3]!, pose[i * 3 + 1]!, pose[i * 3 + 2]!];
}

export function setPoint(pose: Pose, i: number, p: readonly number[]): void {
  pose[i * 3] = p[0]!;
  pose[i * 3 + 1] = p[1]!;
  pose[i * 3 + 2] = p[2] ?? 0;
}

export const visible = (pose: Pose, i: number) => !Number.isNaN(pose[i * 3]!);

/** Decodes the binary file into poses (exported for tests). */
export function decodeDemos(index: DemoIndex, data: Int16Array): Map<string, SignDemo> {
  const map = new Map<string, SignDemo>();
  const perFrame = index.points * DIMS;
  for (const s of index.signs) {
    const frames: Pose[] = [];
    for (let f = 0; f < s.frames; f++) {
      const pose = new Float32Array(perFrame);
      const start = s.offset + f * perFrame;
      for (let i = 0; i < perFrame; i++) {
        const v = data[start + i]!;
        pose[i] = v === index.missing ? NaN : v / index.scale;
      }
      frames.push(pose);
    }
    map.set(s.label, { label: s.label, frames });
  }
  return map;
}

function median(values: number[], fallback: number): number {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  return v.length ? v[Math.floor(v.length / 2)]! : fallback;
}

function percentile(values: number[], q: number): number {
  const v = values.filter(Number.isFinite).sort((a, b) => a - b);
  return v.length ? v[Math.min(v.length - 1, Math.floor(v.length * q))]! : NaN;
}

const dist2d = (a: number[], b: number[]) => Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!);
const dist3d = (a: number[], b: number[]) => Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function centroid(pose: Pose, [from, to]: [number, number]): [number, number, number] {
  let x = 0, y = 0, z = 0, n = 0;
  for (let i = from; i < to; i++) {
    if (!visible(pose, i)) continue;
    const p = point(pose, i);
    x += p[0]; y += p[1]; z += p[2]; n++;
  }
  return n ? [x / n, y / n, z / n] : [NaN, NaN, NaN];
}

/**
 * Measures the figure's proportions.
 *
 * Arm lengths: in a single picture a bone looks shorter when it points
 * towards the camera, so for each recording the length is taken when the bone
 * looks longest (90th percentile), then the median over all recordings.
 */
export function measureRig(demos: Iterable<SignDemo>, layout: DemoLayout): Rig {
  const b = layout.body[0];
  const upper: number[] = [];
  const fore: number[] = [];
  const hand: number[] = [];
  const faceW: number[] = [];
  const faceH: number[] = [];
  const eyeSide: number[] = [];
  for (const d of demos) {
    for (const [sh, el, wr, h] of [
      [BODY.leftShoulder, BODY.leftElbow, BODY.leftWrist, layout.leftHand[0]],
      [BODY.rightShoulder, BODY.rightElbow, BODY.rightWrist, layout.rightHand[0]],
    ] as const) {
      const u: number[] = [];
      const f: number[] = [];
      for (const pose of d.frames) {
        if (visible(pose, b + sh) && visible(pose, b + el)) u.push(dist2d(point(pose, b + sh), point(pose, b + el)));
        const wrist = visible(pose, h) ? point(pose, h) : point(pose, b + wr);
        if (visible(pose, b + el) && Number.isFinite(wrist[0])) f.push(dist2d(point(pose, b + el), wrist));
        if (visible(pose, h) && visible(pose, h + 9)) hand.push(dist3d(point(pose, h), point(pose, h + 9)));
      }
      if (u.length > 5) upper.push(percentile(u, 0.9));
      if (f.length > 5) fore.push(percentile(f, 0.9));
    }
    const first = d.frames[0];
    if (first) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (let i = layout.faceOval[0]; i < layout.faceOval[1]; i++) {
        if (!visible(first, i)) continue;
        const [x, y] = point(first, i);
        minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
      if (maxX > minX) {
        faceW.push(maxX - minX);
        faceH.push(maxY - minY);
      }
      const l = centroid(first, layout.eyeL);
      const r = centroid(first, layout.eyeR);
      if (Number.isFinite(l[0]) && Number.isFinite(r[0])) eyeSide.push(Math.sign(r[0] - l[0]));
    }
  }
  return {
    upperArm: clamp(median(upper, 0.85), 0.6, 1.2),
    forearm: clamp(median(fore, 0.8), 0.55, 1.15),
    hand: clamp(median(hand, 0.3), 0.2, 0.42),
    faceWidth: clamp(median(faceW, 0.55), 0.35, 0.8),
    faceHeight: clamp(median(faceH, 0.7), 0.45, 1),
    eyeSide: median(eyeSide, 1) >= 0 ? 1 : -1,
  };
}

/**
 * Neutral pose: for every body and face point, the median position at the
 * start of all recordings (signers start with their hands down). Arms resting,
 * hands hidden.
 */
export function neutralPose(demos: Iterable<SignDemo>, layout: DemoLayout): Pose {
  const n = DEMO_POINTS * DIMS;
  const samples: number[][] = Array.from({ length: n }, () => []);
  for (const d of demos) {
    const first = d.frames[0];
    if (!first) continue;
    for (let i = layout.body[0] * DIMS; i < n; i++) if (!Number.isNaN(first[i]!)) samples[i]!.push(first[i]!);
  }
  const pose = new Float32Array(n).fill(NaN);
  samples.forEach((values, i) => {
    if (values.length) pose[i] = median(values, NaN);
  });
  const b = layout.body[0];
  const [lx, ly, lz] = point(pose, b + BODY.leftShoulder);
  const [rx, ry, rz] = point(pose, b + BODY.rightShoulder);
  // Arms hanging, the hands slightly in front of the body.
  setPoint(pose, b + BODY.leftElbow, [lx + (lx - rx) * 0.08, ly + 0.8, lz]);
  setPoint(pose, b + BODY.rightElbow, [rx + (rx - lx) * 0.08, ry + 0.8, rz]);
  setPoint(pose, b + BODY.leftWrist, [lx, ly + 1.5, lz - 0.2]);
  setPoint(pose, b + BODY.rightWrist, [rx, ry + 1.5, rz - 0.2]);
  return pose;
}

const base = import.meta.env.BASE_URL;
let cached: Promise<DemoLibrary | null> | null = null;

export function loadDemos(): Promise<DemoLibrary | null> {
  cached ??= (async () => {
    try {
      const [indexRes, binRes] = await Promise.all([
        fetch(`${base}models/learn/sign_demos.json`),
        fetch(`${base}models/learn/sign_demos.bin`),
      ]);
      if (!indexRes.ok || !binRes.ok) return null;
      const index = (await indexRes.json()) as DemoIndex;
      if ((index.dims ?? 2) !== DIMS || index.points !== DEMO_POINTS) {
        console.error('Sign demonstrations are in an older format; run training/prepare_learning_data.py again');
        return null;
      }
      const data = new Int16Array(await binRes.arrayBuffer());
      const demos = decodeDemos(index, data);
      const all = [...demos.values()];
      return new DemoLibrary(index.fps, index.layout, demos, neutralPose(all, index.layout), measureRig(all, index.layout));
    } catch (err) {
      console.error('Sign demonstrations could not be loaded', err);
      cached = null;
      return null;
    }
  })();
  return cached;
}
