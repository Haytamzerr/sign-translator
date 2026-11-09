/**
 * Feature extraction: turning raw camera landmarks into numbers that describe
 * a sign, and measuring how different two of those descriptions are.
 */

import type { Frame, HandFeature, HandSlots, Point3, RawHand } from './types';

/** Number of landmarks MediaPipe gives per hand. */
export const LANDMARKS = 21;
/** Landmark 0 is the wrist, landmark 9 is the base of the middle finger. */
const WRIST = 0;
const MIDDLE_KNUCKLE = 9;

/** Cost added when a hand is present in one frame but missing in the other. */
export const MISSING_HAND_PENALTY = 1.0;

/**
 * Converts the tracker output for one video frame into a normalized Frame.
 *
 * @param raw     hands detected by MediaPipe
 * @param aspect  video width / height. MediaPipe gives x and y as fractions of
 *                width and height, so x is stretched by `aspect` to make one
 *                unit equal in both directions (otherwise circles look like ovals).
 * @param t       timestamp in milliseconds
 */
export function extractFrame(raw: RawHand[], aspect: number, t: number): Frame {
  const hands: HandSlots = [null, null];

  // Sort by confidence so that if MediaPipe labels both hands with the same
  // side (it happens briefly when hands cross), the surer one keeps its slot.
  const ordered = [...raw].sort((a, b) => b.score - a.score);
  for (const hand of ordered) {
    const preferred = hand.side === 'Right' ? 0 : 1;
    const slot = hands[preferred] === null ? preferred : 1 - preferred;
    if (hands[slot] === null) hands[slot] = normalizeHand(hand.landmarks, aspect);
  }
  return { t, hands };
}

/** Normalizes 21 landmarks: wrist at the origin, hand size equal to 1. */
export function normalizeHand(points: Point3[], aspect: number): HandFeature {
  const p = points.map((q) => ({ x: q.x * aspect, y: q.y, z: q.z * aspect }));
  const wrist = p[WRIST]!;
  const knuckle = p[MIDDLE_KNUCKLE]!;
  const scale = Math.max(Math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y, knuckle.z - wrist.z), 1e-6);

  const shape = new Array<number>(LANDMARKS * 3);
  for (let i = 0; i < LANDMARKS; i++) {
    const q = p[i]!;
    shape[i * 3] = (q.x - wrist.x) / scale;
    shape[i * 3 + 1] = (q.y - wrist.y) / scale;
    shape[i * 3 + 2] = (q.z - wrist.z) / scale;
  }
  return { shape, pos: [wrist.x, wrist.y], scale };
}

/**
 * How different two handshapes are: the average distance between matching
 * landmarks (root mean square), in "hand sizes". 0 means identical; values
 * around 0.3 and above usually mean a clearly different handshape.
 */
export function shapeDistance(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return Math.sqrt(sum / LANDMARKS);
}

/** Handshape distance between two frames, considering both hand slots. */
export function frameShapeDistance(a: Frame, b: Frame): number {
  let total = 0;
  let used = 0;
  for (let s = 0; s < 2; s++) {
    const ha = a.hands[s];
    const hb = b.hands[s];
    if (!ha && !hb) continue;
    used++;
    total += ha && hb ? shapeDistance(ha.shape, hb.shape) : MISSING_HAND_PENALTY;
  }
  return used === 0 ? 0 : total / used;
}

export function hasHands(frame: Frame): boolean {
  return frame.hands[0] !== null || frame.hands[1] !== null;
}

/**
 * Motion between two consecutive frames, in hand sizes per second. Combines
 * how far the wrists travelled and how much the handshape changed, so both
 * "moving the hand" and "opening and closing the fingers" count as motion.
 */
export function motionBetween(prev: Frame, cur: Frame): number {
  const dt = Math.max(cur.t - prev.t, 1) / 1000;
  let total = 0;
  let used = 0;
  for (let s = 0; s < 2; s++) {
    const a = prev.hands[s];
    const b = cur.hands[s];
    if (!a || !b) continue;
    const scale = (a.scale + b.scale) / 2;
    const travel = Math.hypot(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) / scale;
    total += travel + 0.3 * shapeDistance(a.shape, b.shape);
    used++;
  }
  return used === 0 ? 0 : total / used / dt;
}

/**
 * Mirror image of a frame: the right hand becomes the left hand. Used so a
 * sign taught with one hand is also recognized when made with the other.
 */
export function mirrorFrame(frame: Frame): Frame {
  const flip = (h: HandFeature | null): HandFeature | null =>
    h && {
      shape: h.shape.map((v, i) => (i % 3 === 0 ? -v : v)),
      pos: [-h.pos[0], h.pos[1]],
      scale: h.scale,
    };
  return { t: frame.t, hands: [flip(frame.hands[1]), flip(frame.hands[0])] };
}
