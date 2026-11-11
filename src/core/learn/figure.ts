/**
 * From recorded points to a 3D figure with a fixed body.
 *
 * Recorded points are noisy, and a camera sees depth badly: an arm pointing
 * at the camera looks short. Drawing the points as they are makes the arms
 * stretch and shrink. Instead, the figure has bones of fixed length (measured
 * once, see measureRig) and the recording only gives each bone its direction:
 *
 *   1. Left-right and up-down come from the picture, which is reliable.
 *   2. Depth is whatever makes the bone its true length (Pythagoras):
 *      depth = sqrt(length^2 - (seen length)^2). The recording only says
 *      whether the bone points towards the camera or away from it.
 *   3. Each hand is attached to the end of its forearm, at a fixed size.
 *
 * Output coordinates are those of the 3D scene: x to the viewer's right, y up,
 * z towards the viewer, in shoulder widths, origin between the shoulders.
 */

import type { LetterHold } from './handshapes';
import { BODY, DEMO_POINTS, DIMS, point, setPoint, visible, type DemoLayout, type HandShape, type Pose, type Rig } from './demos';

export type Vec3 = [number, number, number];

export interface Arm {
  shoulder: Vec3;
  elbow: Vec3;
  wrist: Vec3;
  /** 21 hand points, or null when the hand was not tracked (a closed hand is drawn). */
  hand: Vec3[] | null;
  /**
   * Where the picture shows the wrist and the hand (left-right and up-down as
   * recorded; depth as solved above). The arm's fixed lengths can move the
   * drawn hand a little away from there; touches need the recorded place.
   */
  seen: { wrist: Vec3; hand: Vec3[] | null };
}

export interface Head {
  center: Vec3;
  /** Unit axes of the head: to the viewer's right, up, forward (out of the face). */
  axes: [Vec3, Vec3, Vec3];
  radii: Vec3;
  /** The face outline's box as seen in the picture (x, y only): centre and half sizes. */
  box: { center: [number, number]; half: [number, number] };
  /** Eyes (their middle) and mouth as seen in the picture, and the distance between the eyes. */
  features: { eyes: [number, number]; mouth: [number, number]; eyeDistance: number };
}

export interface Figure {
  left: Arm;
  right: Arm;
  head: Head | null;
  /** Middle of the shoulders, and the shoulder line (unit vector towards the viewer's right). */
  chest: { center: Vec3; across: Vec3; width: number };
}

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const length = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
export const normalize = (a: Vec3): Vec3 => {
  const l = length(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
const finite = (a: Vec3) => Number.isFinite(a[0]) && Number.isFinite(a[1]) && Number.isFinite(a[2]);

/** Recorded axes (y down, z away) to scene axes (y up, z towards the viewer). */
const toScene = (p: readonly number[]): Vec3 => [p[0]!, -p[1]!, -(p[2] ?? 0)];

/**
 * Places the end of a bone of known length, starting at `from`, in the
 * direction of `seen`. The up-down and left-right offsets are kept; the depth
 * is solved so the bone has its true length. A bone is assumed to point
 * towards the viewer unless the recording clearly says otherwise, because
 * signing happens in front of the body.
 */
export function placeBone(from: Vec3, seen: Vec3, len: number): Vec3 {
  const dx = seen[0] - from[0];
  const dy = seen[1] - from[1];
  const dz = seen[2] - from[2];
  const flat = Math.hypot(dx, dy);
  if (flat >= len) return [from[0] + (dx / flat) * len, from[1] + (dy / flat) * len, from[2]];
  const depth = Math.sqrt(len * len - flat * flat);
  // Towards the viewer (+1) or away (-1), changing gradually rather than all
  // at once: a hard switch would make the hand jump from front to back when the
  // recorded depth wavers around the threshold.
  // The bone is then rescaled to its true length.
  const towardsViewer = Math.tanh((dz + 0.35 * len) / (0.1 * len));
  const z = towardsViewer * depth;
  const k = len / (Math.hypot(dx, dy, z) || 1);
  return [from[0] + dx * k, from[1] + dy * k, from[2] + z * k];
}

/** Hand points relative to the wrist, in scene axes, or null if not tracked. */
function handOf(pose: Pose, start: number): Vec3[] | null {
  if (!visible(pose, start) || !visible(pose, start + 9)) return null;
  const wrist = toScene(point(pose, start));
  const out: Vec3[] = [];
  for (let i = 0; i < 21; i++) {
    const p = toScene(point(pose, start + i));
    if (!finite(p)) return null;
    out.push(sub(p, wrist));
  }
  return out;
}

function solveArm(pose: Pose, layout: DemoLayout, rig: Rig, side: 'left' | 'right'): Arm {
  const b = layout.body[0];
  const [sh, el, wr] =
    side === 'left' ? [BODY.leftShoulder, BODY.leftElbow, BODY.leftWrist] : [BODY.rightShoulder, BODY.rightElbow, BODY.rightWrist];
  const handStart = side === 'left' ? layout.leftHand[0] : layout.rightHand[0];
  const shoulder = toScene(point(pose, b + sh));
  // Not tracked: hanging down, slightly forward.
  const rest = (from: Vec3, len: number): Vec3 => [from[0], from[1] - len * 0.995, from[2] + len * 0.0999];

  const seenElbow = toScene(point(pose, b + el));
  const elbow = finite(seenElbow) ? placeBone(shoulder, seenElbow, rig.upperArm) : rest(shoulder, rig.upperArm);

  // The hand's own wrist point is more precise than the body tracker's in the
  // picture. But the hand tracker's depth is only relative to the hand itself
  // (its wrist is always at depth 0), so whether the hand is in front of the
  // elbow or behind it comes from the body tracker.
  const hand = handOf(pose, handStart);
  const bodyWrist = toScene(point(pose, b + wr));
  const handWrist = toScene(point(pose, handStart));
  const seenWrist: Vec3 = hand ? [handWrist[0], handWrist[1], Number.isFinite(bodyWrist[2]) ? bodyWrist[2] : elbow[2]] : bodyWrist;
  const wrist = finite(seenWrist) ? placeBone(elbow, seenWrist, rig.forearm) : rest(elbow, rig.forearm);

  const seen: Vec3 = finite(seenWrist) ? [seenWrist[0], seenWrist[1], wrist[2]] : wrist;
  return {
    shoulder,
    elbow,
    wrist,
    hand: hand ? hand.map((p) => add(wrist, p)) : null,
    seen: { wrist: seen, hand: hand ? hand.map((p) => add(seen, p)) : null },
  };
}

function centroid(pose: Pose, [from, to]: [number, number]): Vec3 | null {
  let x = 0, y = 0, z = 0, n = 0;
  for (let i = from; i < to; i++) {
    if (!visible(pose, i)) continue;
    const p = toScene(point(pose, i));
    x += p[0]; y += p[1]; z += p[2]; n++;
  }
  return n ? [x / n, y / n, z / n] : null;
}

/**
 * The head's position and the way the face points: across from eye to eye,
 * up from the lips to the eyebrows, and forward out of the face.
 */
function solveHead(pose: Pose, layout: DemoLayout, rig: Rig, neckBase: Vec3): Head | null {
  const ovalSeen = centroid(pose, layout.faceOval);
  // The head sits above the shoulders: its depth comes from the body, not from
  // the face tracker (whose depth is only meaningful relative to the face).
  const oval: Vec3 | null = ovalSeen && [ovalSeen[0], ovalSeen[1], neckBase[2] + rig.faceWidth * 0.35];
  const eyeA = centroid(pose, layout.eyeL);
  const eyeB = centroid(pose, layout.eyeR);
  const lips = centroid(pose, layout.lips);
  const browA = centroid(pose, layout.browL);
  const browB = centroid(pose, layout.browR);
  if (!oval || !eyeA || !eyeB || !lips || !browA || !browB) return null;

  const across = normalize(scale(sub(eyeB, eyeA), rig.eyeSide));
  const upRaw = sub(scale(add(browA, browB), 0.5), lips);
  const up = normalize(sub(upRaw, scale(across, dot(upRaw, across))));
  const forward = cross(across, up);
  const radii: Vec3 = [rig.faceWidth * 0.55, rig.faceHeight * 0.6, rig.faceWidth * 0.62];
  const center = add(add(oval, scale(forward, -radii[2] * 0.55)), scale(up, radii[1] * 0.08));

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = layout.faceOval[0]; i < layout.faceOval[1]; i++) {
    if (!visible(pose, i)) continue;
    const [x, y] = toScene(point(pose, i));
    minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  return {
    center,
    axes: [across, up, forward],
    radii,
    box: { center: [(minX + maxX) / 2, (minY + maxY) / 2], half: [Math.max(1e-3, (maxX - minX) / 2), Math.max(1e-3, (maxY - minY) / 2)] },
    features: {
      eyes: [(eyeA[0] + eyeB[0]) / 2, (eyeA[1] + eyeB[1]) / 2],
      mouth: [lips[0], lips[1]],
      eyeDistance: Math.max(1e-3, Math.hypot(eyeB[0] - eyeA[0], eyeB[1] - eyeA[1])),
    },
  };
}

export function solveFigure(pose: Pose, layout: DemoLayout, rig: Rig): Figure {
  const left = solveArm(pose, layout, rig, 'left');
  const right = solveArm(pose, layout, rig, 'right');
  const center = scale(add(left.shoulder, right.shoulder), 0.5);
  const line = sub(left.shoulder, right.shoulder);
  // The shoulder line points to the viewer's right; its depth is limited, since
  // the body tracker's depth for the shoulders is the least reliable value.
  const across = normalize([Math.abs(line[0]) < 1e-6 ? 1 : line[0], line[1], Math.max(-0.35, Math.min(0.35, line[2]))]);
  const oriented: Vec3 = across[0] < 0 ? scale(across, -1) : across;
  return {
    left,
    right,
    head: solveHead(pose, layout, rig, center),
    chest: { center, across: oriented, width: Math.hypot(line[0], line[1]) || 1 },
  };
}

/**
 * Prepares a recording for playback: short tracking gaps of the hands are
 * filled with the last seen hand. (Hands keep the size they had in the
 * picture: where they are relative to the face matters for signs that touch
 * it, and the 3D model has its own hand size.)
 */
export function prepareFrames(frames: Pose[], layout: DemoLayout): Pose[] {
  const out = frames.map((f) => f.slice());
  for (const [from, to] of [layout.leftHand, layout.rightHand]) {
    let last = -1;
    out.forEach((f, i) => {
      if (!visible(f, from)) return;
      if (last >= 0 && i - last > 1 && i - last <= 6) {
        for (let j = last + 1; j < i; j++) for (let k = from * DIMS; k < to * DIMS; k++) out[j]![k] = out[last]![k]!;
      }
      last = i;
    });
  }
  return smoothFrames(out);
}

/**
 * Removes the tracker's trembling from a recording: every coordinate becomes a
 * weighted average of itself and its neighbours in time (a Gaussian, centred,
 * so nothing is delayed: the whole recording is known in advance). Points not
 * visible are left out of the average and stay not visible.
 */
export function smoothFrames(frames: Pose[], sigma = 1.4): Pose[] {
  const radius = Math.ceil(sigma * 2.5);
  const weights = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)));
  const n = frames.length;
  const out = frames.map((f) => f.slice());
  const size = frames[0]?.length ?? 0;
  for (let c = 0; c < size; c++) {
    for (let t = 0; t < n; t++) {
      if (Number.isNaN(frames[t]![c]!)) continue;
      let sum = 0;
      let total = 0;
      for (let d = -radius; d <= radius; d++) {
        const v = frames[t + d]?.[c];
        if (v === undefined || Number.isNaN(v)) continue;
        const w = weights[d + radius]!;
        sum += v * w;
        total += w;
      }
      out[t]![c] = sum / total;
    }
  }
  return out;
}

/**
 * A standing pose with the right hand forming a letter. `hold` places the arm:
 * forearm upright in front of the shoulder (most letters), across the chest
 * for letters that point sideways (G, H), or forwards for those pointing down
 * (P, Q).
 */
export function letterPose(neutral: Pose, layout: DemoLayout, rig: Rig, shape: HandShape, hold: LetterHold = 'up'): Pose {
  const pose = neutral.slice();
  const b = layout.body[0];
  const ls = point(pose, b + BODY.leftShoulder);
  const rs = point(pose, b + BODY.rightShoulder);
  const out = Math.sign(rs[0] - ls[0]) || 1; // direction away from the body on the right side
  // Recorded axes: y down, smaller z towards the viewer.
  const ARMS: Record<LetterHold, [number[], number[]]> = {
    up: [[out * 0.3, 0.75, -0.45], [out * 0.18, 0.15, -0.75]],
    side: [[out * 0.3, 0.75, -0.45], [out * 0.18, 0.15, -0.75]],
    sideways: [[out * 0.38, 0.62, -0.3], [-out * 0.12, 0.5, -0.72]],
    down: [[out * 0.3, 0.75, -0.3], [out * 0.2, 0.6, -0.88]],
  };
  const [elbowOffset, wristOffset] = ARMS[hold];
  setPoint(pose, b + BODY.rightElbow, [rs[0] + elbowOffset[0]!, rs[1] + elbowOffset[1]!, rs[2] + elbowOffset[2]!]);
  const wrist = [rs[0] + wristOffset[0]!, rs[1] + wristOffset[1]!, rs[2] + wristOffset[2]!];
  setPoint(pose, b + BODY.rightWrist, wrist);
  const [h] = layout.rightHand;
  shape.forEach((p, i) => {
    setPoint(pose, h + i, [wrist[0]! + p[0]! * rig.hand, wrist[1]! + p[1]! * rig.hand, wrist[2]! + (p[2] ?? 0) * rig.hand]);
  });
  return pose;
}

export const POSE_SIZE = DEMO_POINTS * DIMS;
