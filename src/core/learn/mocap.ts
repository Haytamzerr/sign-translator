/**
 * Motion-captured signs: 3D-LEX v1.0 (Ranum et al., 2024, CC BY 4.0).
 *
 * A signer was recorded with optical motion capture (markers on the body,
 * filmed by many cameras) and data gloves (bend sensors in each finger). Unlike
 * a single camera, this measures depth, both hands, and every finger joint
 * exactly. Where a sign was recorded this way, it is shown instead of the
 * camera recording.
 *
 * Files (made by training/lex_to_app.py):
 * - models/mocap/index.json: the list of signs, the frame layout and a face template
 * - models/mocap/signs/<sign>.bin: 16-bit integers, 162 values per frame
 *
 * Each sign is loaded only when it is first shown (about 30 kB). A frame is
 * turned into the same 159 points as a camera recording (see demos.ts), so
 * the rest of the app plays both the same way. The face is not stored point by
 * point: it is drawn from the eyes and the head's direction, with a template
 * face (the median face of the camera recordings, measured in eye distances).
 */

import { DEMO_POINTS, DIMS, setPoint, type DemoLayout, type Pose } from './demos';

interface MocapIndex {
  fps: number;
  scale: number;
  frame: { hands: [number, number]; body: [number, number]; eyeL: number; eyeR: number; headUp: number; values: number };
  /** Face points (outline, lips, eyes, brows) relative to the eyes: across, up, depth, in eye distances. */
  faceTemplate: [number, number, number][];
  signs: { name: string; frames: number; aliases: string[] }[];
}

export interface MocapClip {
  name: string;
  fps: number;
  frames: Pose[];
}

type V = [number, number, number];
const sub = (a: V, b: V): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: V, k: number): V => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V): V => scale(a, 1 / (Math.hypot(a[0], a[1], a[2]) || 1));

/** Turns stored values into a 159-point pose. */
export function decodeMocapFrame(values: Int16Array, offset: number, index: MocapIndex, layout: DemoLayout): Pose {
  const k = 1 / index.scale;
  const at = (i: number): V => [values[offset + i]! * k, values[offset + i + 1]! * k, values[offset + i + 2]! * k];
  const pose = new Float32Array(DEMO_POINTS * DIMS);
  const { hands, body, eyeL, eyeR, headUp } = index.frame;
  // Hands and body: stored in the same order as the camera recordings.
  for (let p = hands[0]; p < hands[1]; p++) setPoint(pose, layout.leftHand[0] + p, at(p * 3));
  for (let p = body[0]; p < body[1]; p++) setPoint(pose, layout.body[0] + (p - body[0]), at(p * 3));
  // Face: the template placed on the eyes, turned with the head.
  const l = at(eyeL);
  const r = at(eyeR);
  const mid = scale([l[0] + r[0], l[1] + r[1], l[2] + r[2]], 0.5);
  const d = Math.hypot(...sub(l, r));
  const across = norm(sub(l, r));
  const upRaw = at(headUp); // towards the top of the head (in the picture, y points down)
  const up = norm(sub(upRaw, scale(across, dot(upRaw, across))));
  const depth = cross(across, up);
  index.faceTemplate.forEach(([a, u, z], i) => {
    setPoint(pose, layout.faceOval[0] + i, [
      mid[0] + d * (a * across[0] + u * up[0] + z * depth[0]),
      mid[1] + d * (a * across[1] + u * up[1] + z * depth[1]),
      mid[2] + d * (a * across[2] + u * up[2] + z * depth[2]),
    ]);
  });
  return pose;
}

const base = import.meta.env.BASE_URL;
let indexPromise: Promise<MocapIndex | null> | null = null;
const clips = new Map<string, Promise<MocapClip | null>>();

export function loadMocapIndex(): Promise<MocapIndex | null> {
  indexPromise ??= fetch(`${base}models/mocap/index.json`)
    .then((r) => (r.ok ? (r.json() as Promise<MocapIndex>) : null))
    .catch(() => null);
  return indexPromise;
}

/** Name of the recorded sign for a word or an app sign label, if there is one. */
export function mocapName(index: MocapIndex, label: string): string | null {
  const key = label.toLowerCase();
  for (const s of index.signs) if (s.name === key || s.aliases.includes(label) || s.aliases.includes(key)) return s.name;
  return null;
}

/** All recorded sign names (English words), for the phrase translator. */
export function mocapWords(index: MocapIndex): string[] {
  return index.signs.map((s) => s.name);
}

/** Loads one recorded sign (cached). */
export function loadMocapClip(label: string, layout: DemoLayout): Promise<MocapClip | null> {
  const cached = clips.get(label);
  if (cached) return cached;
  const promise = loadMocapIndex().then(async (index) => {
    if (!index) return null;
    const name = mocapName(index, label);
    if (!name) return null;
    const res = await fetch(`${base}models/mocap/signs/${encodeURIComponent(name)}.bin`);
    if (!res.ok) return null;
    const values = new Int16Array(await res.arrayBuffer());
    const n = index.frame.values;
    const frames: Pose[] = [];
    for (let f = 0; f + n <= values.length; f += n) frames.push(decodeMocapFrame(values, f, index, layout));
    return { name, fps: index.fps, frames };
  }).catch(() => null);
  clips.set(label, promise);
  return promise;
}
