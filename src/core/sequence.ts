/**
 * Moving signs: comparing two recordings that may be performed at different
 * speeds, using Dynamic Time Warping (DTW).
 */

import { hasHands, MISSING_HAND_PENALTY, motionBetween, shapeDistance } from './features';
import type { Frame } from './types';

/** Every moving sign is resampled to this many steps before comparison. */
export const SEQUENCE_LENGTH = 24;
/** How much wrist movement counts compared with handshape. */
const MOTION_WEIGHT = 0.5;
/** DTW may only stretch time by this many steps (keeps matching fast and sane). */
const BAND = 6;

/** A hand inside a sequence: its shape plus where it is relative to where it started. */
interface SeqHand {
  shape: number[];
  rel: [number, number];
}
export type SeqStep = [SeqHand | null, SeqHand | null];

/**
 * Prepares a recording for comparison:
 * 1. Positions become relative to each hand's starting point and are measured
 *    in hand sizes, so the same sign made higher, lower or further away matches.
 * 2. The recording is resampled to a fixed number of steps, so a fast and a slow
 *    version of the same sign start out the same length. DTW handles the rest.
 */
export function toSequence(frames: Frame[], length = SEQUENCE_LENGTH): SeqStep[] {
  const origin: ([number, number] | null)[] = [null, null];
  const scaleSum = [0, 0];
  const scaleCount = [0, 0];
  for (const f of frames) {
    for (let s = 0; s < 2; s++) {
      const h = f.hands[s];
      if (!h) continue;
      if (!origin[s]) origin[s] = h.pos;
      scaleSum[s]! += h.scale;
      scaleCount[s]!++;
    }
  }
  const meanScale = [0, 1].map((s) => (scaleCount[s]! > 0 ? scaleSum[s]! / scaleCount[s]! : 1));

  const steps: SeqStep[] = [];
  for (let i = 0; i < length; i++) {
    // Pick the nearest original frame for this step.
    const idx = frames.length === 1 ? 0 : Math.round((i * (frames.length - 1)) / (length - 1));
    const f = frames[idx]!;
    const step: SeqStep = [null, null];
    for (let s = 0; s < 2; s++) {
      const h = f.hands[s];
      const o = origin[s];
      if (!h || !o) continue;
      step[s] = {
        shape: h.shape,
        rel: [(h.pos[0] - o[0]) / meanScale[s]!, (h.pos[1] - o[1]) / meanScale[s]!],
      };
    }
    steps.push(step);
  }
  return steps;
}

/** Distance between two steps: handshape difference plus position difference. */
export function stepDistance(a: SeqStep, b: SeqStep): number {
  let total = 0;
  let used = 0;
  for (let s = 0; s < 2; s++) {
    const ha = a[s];
    const hb = b[s];
    if (!ha && !hb) continue;
    used++;
    if (!ha || !hb) {
      total += MISSING_HAND_PENALTY;
      continue;
    }
    const move = Math.hypot(ha.rel[0] - hb.rel[0], ha.rel[1] - hb.rel[1]);
    total += shapeDistance(ha.shape, hb.shape) + MOTION_WEIGHT * move;
  }
  return used === 0 ? 0 : total / used;
}

/**
 * Dynamic Time Warping.
 *
 * Imagine two people saying the same word, one slowly and one quickly. DTW
 * finds the best way to line up their moments (a "warping path") and returns
 * the average distance along that path. Same sign at different speeds gives a
 * small number; a different sign gives a large one.
 *
 * `cost[i][j]` = best total cost of aligning the first i steps of `a` with the
 * first j steps of `b`. Each cell extends the cheapest of its three neighbors.
 */
export function dtw(a: SeqStep[], b: SeqStep[], band = BAND): number {
  const n = a.length;
  const m = b.length;
  const w = Math.max(band, Math.abs(n - m));
  const INF = Number.POSITIVE_INFINITY;
  let prev = new Float64Array(m + 1).fill(INF);
  let cur = new Float64Array(m + 1).fill(INF);
  prev[0] = 0;

  for (let i = 1; i <= n; i++) {
    cur.fill(INF);
    const from = Math.max(1, i - w);
    const to = Math.min(m, i + w);
    for (let j = from; j <= to; j++) {
      const best = Math.min(prev[j]!, cur[j - 1]!, prev[j - 1]!);
      cur[j] = stepDistance(a[i - 1]!, b[j - 1]!) + best;
    }
    [prev, cur] = [cur, prev];
  }
  return prev[m]! / (n + m);
}

/**
 * Removes still frames at the start and end of a recording, so the stored
 * example begins and ends with the movement itself. Keeps `pad` frames of
 * context on each side.
 */
export function trimStill(frames: Frame[], threshold: number, pad = 2): Frame[] {
  if (frames.length < 3) return frames;
  const moving = frames.map((f, i) => (i === 0 ? false : motionBetween(frames[i - 1]!, f) > threshold));
  const first = moving.indexOf(true);
  const last = moving.lastIndexOf(true);
  if (first === -1) return frames;
  return frames.slice(Math.max(0, first - pad), Math.min(frames.length, last + pad + 1));
}

/** Fewest frames with hands kept when trimming the hand's entry and exit. */
const MIN_CORE_FRAMES = 8;

/** Removes frames without hands at the start and end of a take. */
export function trimEmpty(frames: Frame[]): Frame[] {
  const first = frames.findIndex(hasHands);
  if (first === -1) return [];
  let last = frames.length - 1;
  while (last > first && !hasHands(frames[last]!)) last--;
  return frames.slice(first, last + 1);
}

/** Below this height (share of the picture from the top), a hand is near the bottom edge. */
const BOTTOM_ZONE = 0.72;

/** Wrist height of the lowest visible hand, or null. */
function wristY(frame: Frame): number | null {
  const ys = frame.hands.filter((h) => h !== null).map((h) => h!.pos[1]);
  return ys.length ? Math.max(...ys) : null;
}

/**
 * Removes the hand rising into view at the start of a take and dropping out
 * of view at the end. These are not part of the sign, but every sign you make
 * starts and ends with them: kept in an example, they would make any movement
 * look a little like that sign, and a held handshape would count as moving.
 * Only applies when the hand really enters from (or leaves by) the bottom edge.
 */
export function trimEntryExit(frames: Frame[]): Frame[] {
  const shown = trimEmpty(frames);
  if (shown.length < MIN_CORE_FRAMES) return shown;
  let start = 0;
  const y0 = wristY(shown[0]!);
  if (y0 !== null && y0 > BOTTOM_ZONE) {
    // Skip while the hand keeps rising.
    while (start < shown.length - 1) {
      const a = wristY(shown[start]!);
      const b = wristY(shown[start + 1]!);
      if (a === null || b === null || b > a - 0.004) break;
      start++;
    }
  }
  let end = shown.length - 1;
  const y1 = wristY(shown[end]!);
  if (y1 !== null && y1 > BOTTOM_ZONE) {
    // Skip back while the hand was dropping.
    while (end > start) {
      const a = wristY(shown[end - 1]!);
      const b = wristY(shown[end]!);
      if (a === null || b === null || b < a + 0.004) break;
      end--;
    }
  }
  const core = shown.slice(start, end + 1);
  // Never cut a take down to almost nothing.
  return core.filter(hasHands).length >= MIN_CORE_FRAMES ? core : shown;
}

