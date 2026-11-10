/**
 * Sign demonstrations: real recordings of the 250 ASL signs, stored as body,
 * face and hand points (no images), and played back as an animated figure.
 *
 * File format (made by training/prepare_learning_data.py):
 * - sign_demos.json: point layout, and for each sign its offset and frame count
 * - sign_demos.bin: 16-bit integers, x and y of 156 points per frame, in
 *   thousandths of a shoulder width, measured from the middle of the
 *   shoulders. -32768 means "not visible".
 *
 * Point layout of a frame: left hand 0-20, right hand 21-41, arms 42-47
 * (shoulders, elbows, wrists), face 48-155 (outline, lips, eyes, eyebrows).
 */

export const DEMO_POINTS = 156;

export interface DemoLayout {
  leftHand: [number, number];
  rightHand: [number, number];
  arms: [number, number];
  faceOval: [number, number];
  lips: [number, number];
  eyeL: [number, number];
  eyeR: [number, number];
  browL: [number, number];
  browR: [number, number];
}

/** One frame: x0, y0, x1, y1, ... (NaN = not visible). */
export type Pose = Float32Array;

export interface SignDemo {
  label: string;
  frames: Pose[];
}

interface DemoIndex {
  points: number;
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
  const perFrame = index.points * 2;
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

/**
 * Neutral pose: for every arm and face point, the median position at the
 * start of all recordings (signers start with their hands down). Hands hidden.
 */
export function neutralPose(demos: Iterable<SignDemo>, layout: DemoLayout): Pose {
  const samples: number[][] = Array.from({ length: DEMO_POINTS * 2 }, () => []);
  for (const d of demos) {
    const first = d.frames[0];
    if (!first) continue;
    for (let i = layout.arms[0] * 2; i < DEMO_POINTS * 2; i++) if (!Number.isNaN(first[i]!)) samples[i]!.push(first[i]!);
  }
  const pose = new Float32Array(DEMO_POINTS * 2).fill(NaN);
  samples.forEach((values, i) => {
    if (!values.length) return;
    values.sort((a, b) => a - b);
    pose[i] = values[Math.floor(values.length / 2)]!;
  });
  // Arms resting: elbows and wrists hang below the shoulders.
  const [a] = layout.arms;
  const set = (point: number, x: number, y: number) => {
    pose[(a + point) * 2] = x;
    pose[(a + point) * 2 + 1] = y;
  };
  const lx = pose[a * 2]!;
  const rx = pose[(a + 1) * 2]!;
  set(2, lx * 1.05, 0.75);
  set(3, rx * 1.05, 0.75);
  set(4, lx * 0.95, 1.45);
  set(5, rx * 0.95, 1.45);
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
      const data = new Int16Array(await binRes.arrayBuffer());
      const demos = decodeDemos(index, data);
      return new DemoLibrary(index.fps, index.layout, demos, neutralPose(demos.values(), index.layout));
    } catch (err) {
      console.error('Sign demonstrations could not be loaded', err);
      cached = null;
      return null;
    }
  })();
  return cached;
}
