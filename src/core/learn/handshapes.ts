/**
 * The 26 handshapes of the ASL manual alphabet, defined exactly.
 *
 * Single recorded samples of letters can be wrong (a finger bent backwards
 * by tracking noise), so the demonstrations use handshapes built from joint
 * angles instead: for each finger, how much each of its three joints bends
 * towards the palm, and how far it spreads sideways; for the thumb, where it
 * points and how much it bends. The result has the same 21 points as the hand
 * tracker, so the rest of the app treats it like any recorded hand.
 *
 * Axes are those of the camera image (x right, y down, z away from the
 * camera). The hand is the signer's right hand, fingers up, palm facing the
 * camera; the wrist is at the origin and the hand size (wrist to middle
 * knuckle) is 1.
 */

import type { HandShape } from './demos';

type V = [number, number, number];
/** [knuckle bend, middle joint bend, end joint bend, sideways spread], degrees. */
type Finger = [number, number, number, number];
/** [spread away from the hand, turn towards the palm, first bend, second bend], degrees. */
type Thumb = [number, number, number, number];
/** How the whole hand is held. */
export type LetterHold = 'up' | 'side' | 'sideways' | 'down';

interface Spec {
  fingers: [Finger, Finger, Finger, Finger]; // index, middle, ring, little
  thumb: Thumb;
  hold?: LetterHold;
}

const STRAIGHT: Finger = [0, 0, 0, 0];
const CLOSED: Finger = [88, 98, 65, 0];
const THUMB_ACROSS: Thumb = [22, 8, 45, 35];

const LETTERS: Record<string, Spec> = {
  A: { fingers: [CLOSED, CLOSED, CLOSED, CLOSED], thumb: [20, 8, 0, 0] },
  B: { fingers: [[0, 0, 0, -3], STRAIGHT, [0, 0, 0, 3], [0, 0, 0, 7]], thumb: [22, 8, 45, 35] },
  C: { fingers: [[30, 40, 25, -3], [30, 40, 25, 0], [30, 40, 25, 3], [30, 40, 25, 6]], thumb: [50, 35, 10, 10], hold: 'side' },
  D: { fingers: [STRAIGHT, [55, 70, 40, 0], [55, 70, 40, 0], [55, 70, 40, 0]], thumb: [35, 45, 30, 25] },
  E: { fingers: [[15, 100, 80, 0], [15, 100, 80, 0], [15, 100, 80, 0], [15, 100, 80, 0]], thumb: [25, 30, 60, 40] },
  F: { fingers: [[50, 55, 30, 0], [0, 0, 0, -2], [0, 0, 0, 4], [0, 0, 0, 10]], thumb: [35, 40, 20, 20] },
  G: { fingers: [STRAIGHT, CLOSED, CLOSED, CLOSED], thumb: [14, 12, 0, 0], hold: 'sideways' },
  H: { fingers: [[0, 0, 0, -3], [0, 0, 0, 2], CLOSED, CLOSED], thumb: THUMB_ACROSS, hold: 'sideways' },
  I: { fingers: [CLOSED, CLOSED, CLOSED, STRAIGHT], thumb: THUMB_ACROSS },
  J: { fingers: [CLOSED, CLOSED, CLOSED, STRAIGHT], thumb: THUMB_ACROSS },
  K: { fingers: [STRAIGHT, [40, 0, 0, 0], CLOSED, CLOSED], thumb: [20, 25, 0, 0] },
  L: { fingers: [STRAIGHT, CLOSED, CLOSED, CLOSED], thumb: [85, 0, 0, 0] },
  M: { fingers: [[80, 95, 55, 0], [80, 95, 55, 0], [80, 95, 55, 0], CLOSED], thumb: [15, 25, 45, 30] },
  N: { fingers: [[80, 95, 55, 0], [80, 95, 55, 0], CLOSED, CLOSED], thumb: [15, 25, 45, 30] },
  O: { fingers: [[45, 60, 40, -2], [45, 62, 40, 0], [45, 64, 40, 2], [45, 66, 40, 4]], thumb: [40, 45, 25, 25], hold: 'side' },
  P: { fingers: [STRAIGHT, [40, 0, 0, 0], CLOSED, CLOSED], thumb: [20, 25, 0, 0], hold: 'down' },
  Q: { fingers: [STRAIGHT, CLOSED, CLOSED, CLOSED], thumb: [14, 12, 0, 0], hold: 'down' },
  R: { fingers: [[0, 0, 0, -9], [10, 0, 0, 9], CLOSED, CLOSED], thumb: THUMB_ACROSS },
  S: { fingers: [CLOSED, CLOSED, CLOSED, CLOSED], thumb: [30, 30, 50, 30] },
  T: { fingers: [[70, 90, 45, 0], CLOSED, CLOSED, CLOSED], thumb: [20, 25, 25, 10] },
  U: { fingers: [[0, 0, 0, -3], [0, 0, 0, 2], CLOSED, CLOSED], thumb: THUMB_ACROSS },
  V: { fingers: [[0, 0, 0, 10], [0, 0, 0, -7], CLOSED, CLOSED], thumb: THUMB_ACROSS },
  W: { fingers: [[0, 0, 0, 12], STRAIGHT, [0, 0, 0, -10], CLOSED], thumb: [25, 25, 50, 35] },
  X: { fingers: [[10, 90, 60, 0], CLOSED, CLOSED, CLOSED], thumb: THUMB_ACROSS },
  Y: { fingers: [CLOSED, CLOSED, CLOSED, [0, 0, 0, -18]], thumb: [85, 0, 0, 0] },
  Z: { fingers: [STRAIGHT, CLOSED, CLOSED, CLOSED], thumb: THUMB_ACROSS },
};

// ---------- a canonical right hand ----------

const ALONG: V = [0, -1, 0]; // wrist to fingers
const ACROSS: V = [1, 0, 0]; // little finger to index (the thumb's side)
const PALM: V = [0, 0, -1]; // out of the palm, towards the camera: fingers fold this way

/** Knuckle positions and segment lengths, in hand sizes. */
const KNUCKLES: V[] = [[0.31, -0.95, 0], [0.04, -1, 0], [-0.21, -0.93, 0], [-0.43, -0.8, 0.02]];
const SEGMENTS = [[0.45, 0.27, 0.22], [0.5, 0.3, 0.23], [0.47, 0.29, 0.22], [0.36, 0.22, 0.2]];
/** Natural fan of the fingers, degrees towards the thumb's side. */
const FAN = [6, 0, -5, -12];
const THUMB_BASE: V = [0.3, -0.25, -0.05];
const THUMB_SEGMENTS = [0.42, 0.33, 0.28];

const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a: V, k: number): V => [a[0] * k, a[1] * k, a[2] * k];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dot = (a: V, b: V) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: V): V => scale(a, 1 / (Math.hypot(a[0], a[1], a[2]) || 1));
const rad = (d: number) => (d * Math.PI) / 180;

/** Rotates v around a unit axis by an angle (Rodrigues' formula). */
function rotate(v: V, axis: V, angle: number): V {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  return add(add(scale(v, c), scale(cross(axis, v), s)), scale(axis, dot(axis, v) * (1 - c)));
}

/** Turns direction `from` towards direction `to` by an angle. */
function bend(from: V, to: V, angle: number): V {
  const axis = cross(from, to);
  if (Math.hypot(axis[0], axis[1], axis[2]) < 1e-9) return from;
  return rotate(from, norm(axis), angle);
}

/** The hand held in a given way: a rotation applied around the wrist. */
function holdRotation(hold: LetterHold): (v: V) => V {
  switch (hold) {
    case 'side': // turned a quarter towards the signer's left, so the curve of C and O is seen from the side
      return (v) => rotate(v, [0, 1, 0], rad(-65));
    case 'sideways': // fingers pointing to the signer's left, palm towards the signer
      // Turn the palm to face the signer, then lay the fingers flat towards the signer's left.
      return (v) => rotate(rotate(v, [0, 1, 0], rad(180)), [0, 0, 1], rad(90));
    case 'down': // fingers pointing down and forwards
      return (v) => rotate(v, [1, 0, 0], rad(130));
    default:
      return (v) => v;
  }
}

export function letterHold(letter: string): LetterHold {
  return LETTERS[letter]?.hold ?? 'up';
}

/** The 21 points of a letter's handshape, or null for an unknown letter. */
export function letterHandShape(letter: string): HandShape | null {
  const spec = LETTERS[letter];
  if (!spec) return null;
  const pts: V[] = new Array(21);
  pts[0] = [0, 0, 0];

  spec.fingers.forEach(([mcp, pip, dip, spread], f) => {
    const base = 5 + f * 4;
    // Straight direction of this finger, fanned and spread in the palm's plane.
    const dir = rotate(ALONG, PALM, -rad(FAN[f]! + spread));
    const hinge = norm(cross(dir, PALM)); // folding axis of all three joints
    let p = KNUCKLES[f]!;
    pts[base] = p;
    let d = dir;
    [mcp, pip, dip].forEach((angle, k) => {
      d = rotate(d, hinge, rad(angle));
      p = add(p, scale(d, SEGMENTS[f]![k]!));
      pts[base + k + 1] = p;
    });
  });

  // Thumb: points up and out to the side, turns towards the palm, then bends
  // its two joints towards the little finger.
  const [abd, opp, b1, b2] = spec.thumb;
  let d = bend(ALONG, ACROSS, rad(abd));
  d = bend(d, PALM, rad(opp));
  const fold = norm(add(scale(ACROSS, -1), scale(PALM, 0.4))); // mostly across the palm, slightly in front of it
  let p = THUMB_BASE;
  pts[1] = p;
  [0, b1, b2].forEach((angle, k) => {
    d = bend(d, fold, rad(angle));
    p = add(p, scale(d, THUMB_SEGMENTS[k]!));
    pts[2 + k] = p;
  });

  const turn = holdRotation(spec.hold ?? 'up');
  return pts.map((q) => turn(q));
}
