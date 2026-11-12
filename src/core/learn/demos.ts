/**
 * Recorded sign movements: the 250 ASL signs, each an average of up to 20
 * real recordings, stored as body, face and hand points in 3D (no images).
 * The personal recognizer uses them as examples of "other signs" (see
 * background.ts); Learn and Phrase show videos instead (signVideos.ts).
 *
 * File format (made by training/prepare_learning_data.py):
 * - sign_demos.json: point layout, and for each sign its offset and frame count
 * - sign_demos.bin: 16-bit integers, x, y and z of 159 points per frame, in
 *   thousandths of a shoulder width, measured from the middle of the
 *   shoulders. -32768 means "not visible".
 *
 * Axes are those of the camera image: x to the right, y down, z away from the
 * camera. Point layout of a frame: left hand 0-20, right hand 21-41, body
 * 42-50 (nose, shoulders, elbows, wrists, hips), face 51-158 (outline, lips,
 * eyes, eyebrows). "Left" and "right" are the signer's.
 */

export const DEMO_POINTS = 159;
export const DIMS = 3;

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

export interface SignDemo {
  label: string;
  frames: Pose[];
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
      return new DemoLibrary(index.fps, index.layout, decodeDemos(index, data));
    } catch (err) {
      console.error('Sign demonstrations could not be loaded', err);
      cached = null;
      return null;
    }
  })();
  return cached;
}
