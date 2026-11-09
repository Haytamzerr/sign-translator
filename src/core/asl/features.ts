/**
 * Input preparation for the pre-trained ASL model.
 *
 * This file is a line-by-line copy of the Python functions `drop_empty_frames`,
 * `features` and `fit_length` in training/train_asl.py. The network only works
 * if the app prepares its input exactly as during training; the parity test
 * (asl.test.ts) checks this against data exported by the training notebook.
 *
 * Compact layout of one frame (49 points, x and y each, NaN = not visible):
 *   0-20  left hand, 21-41 right hand,
 *   42-48 body: nose, left shoulder, right shoulder, left elbow,
 *         right elbow, left wrist, right wrist.
 */

import type { Point3, RawHand } from '../types';

export const N_POINTS = 49;
export const N_FEATURES = N_POINTS * 2 + 2 * 21 * 2; // 182
/** MediaPipe pose indices of the 7 body points we keep. */
export const POSE_KEEP = [0, 11, 12, 13, 14, 15, 16];
/** Body point pairs swapped when mirroring (index inside the 7). */
const POSE_SWAP = [0, 2, 1, 4, 3, 6, 5];

/** One frame: 98 numbers (x0, y0, x1, y1, ...). */
export type RawFrame = Float64Array;

/**
 * Builds the compact frame from the trackers' output.
 * @param xScale converts the camera's x coordinates to the training data's
 *               units (see AslModel calibration).
 */
export function compactFrame(hands: RawHand[], pose: Point3[] | null, xScale: number): RawFrame {
  const f = new Float64Array(N_POINTS * 2).fill(NaN);
  const put = (index: number, p: Point3) => {
    f[index * 2] = p.x * xScale;
    f[index * 2 + 1] = p.y;
  };
  // Surest hand first, so it keeps its side if both claim the same one.
  const ordered = [...hands].sort((a, b) => b.score - a.score);
  const used = [false, false];
  for (const hand of ordered) {
    let slot = hand.side === 'Left' ? 0 : 1;
    if (used[slot]) slot = 1 - slot;
    if (used[slot]) continue;
    used[slot] = true;
    hand.landmarks.forEach((p, i) => put(slot * 21 + i, p));
  }
  if (pose) POSE_KEEP.forEach((k, i) => pose[k] && put(42 + i, pose[k]!));
  return f;
}

/** Mirror image of a frame: left and right are swapped. */
export function mirrorRaw(frame: RawFrame): RawFrame {
  const out = new Float64Array(frame.length);
  const copy = (to: number, from: number) => {
    out[to * 2] = -frame[from * 2]!;
    out[to * 2 + 1] = frame[from * 2 + 1]!;
  };
  for (let i = 0; i < 21; i++) {
    copy(i, 21 + i);
    copy(21 + i, i);
  }
  for (let i = 0; i < 7; i++) copy(42 + i, 42 + POSE_SWAP[i]!);
  return out;
}

const x = (f: RawFrame, p: number) => f[p * 2]!;
const y = (f: RawFrame, p: number) => f[p * 2 + 1]!;

/** Keeps only frames where at least one hand is visible (all frames if none). */
export function dropEmptyFrames(frames: RawFrame[]): RawFrame[] {
  const kept = frames.filter((f) => {
    for (let p = 0; p < 42; p++) if (!Number.isNaN(x(f, p))) return true;
    return false;
  });
  return kept.length > 0 ? kept : frames;
}

/** Mean of the numbers that are not NaN (NaN if there are none), like numpy.nanmean. */
function nanMean(values: number[]): number {
  let sum = 0;
  let n = 0;
  for (const v of values) {
    if (!Number.isNaN(v)) {
      sum += v;
      n++;
    }
  }
  return n ? sum / n : NaN;
}

/** Turns frames into the 182 numbers per frame the network reads. */
export function features(frames: RawFrame[]): Float32Array[] {
  // Reference point and scale: middle of the shoulders and shoulder width.
  const sx: number[] = [];
  const sy: number[] = [];
  const widths: number[] = [];
  for (const f of frames) {
    for (const p of [43, 44]) {
      sx.push(x(f, p));
      sy.push(y(f, p));
    }
    widths.push(Math.hypot(x(f, 43) - x(f, 44), y(f, 43) - y(f, 44)));
  }
  let cx = nanMean(sx);
  let cy = nanMean(sy);
  let width = nanMean(widths);
  if (Number.isNaN(cx) || Number.isNaN(cy) || !Number.isFinite(width) || width < 1e-3) {
    const ax: number[] = [];
    const ay: number[] = [];
    for (const f of frames)
      for (let p = 0; p < N_POINTS; p++) {
        ax.push(x(f, p));
        ay.push(y(f, p));
      }
    cx = nanMean(ax);
    cy = nanMean(ay);
    width = 0.3;
  }
  if (Number.isNaN(cx) || Number.isNaN(cy)) {
    cx = 0;
    cy = 0;
  }

  const clean = (v: number) => (Number.isFinite(v) ? v : 0);

  return frames.map((f) => {
    const out = new Float32Array(N_FEATURES);
    // Global: every point relative to the shoulders.
    for (let p = 0; p < N_POINTS; p++) {
      out[p * 2] = clean((x(f, p) - cx) / width);
      out[p * 2 + 1] = clean((y(f, p) - cy) / width);
    }
    // Local: each hand relative to its wrist, divided by its own size.
    let o = N_POINTS * 2;
    for (const start of [0, 21]) {
      let size = Math.hypot(x(f, start + 9) - x(f, start), y(f, start + 9) - y(f, start));
      if (!(Number.isFinite(size) && size > 1e-4)) size = 1;
      for (let i = 0; i < 21; i++) {
        out[o++] = clean((x(f, start + i) - x(f, start)) / size);
        out[o++] = clean((y(f, start + i) - y(f, start)) / size);
      }
    }
    return out;
  });
}

/** Rounds like numpy (halves go to the nearest even number). */
function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/**
 * Long recordings are resampled to `maxLen` frames, short ones padded with
 * zeros. Returns the network input and the mask (1 = real frame, 0 = padding).
 */
export function fitLength(feats: Float32Array[], maxLen: number): { x: Float32Array; mask: Float32Array } {
  const input = new Float32Array(maxLen * N_FEATURES);
  const mask = new Float32Array(maxLen);
  const t = feats.length;
  if (t > maxLen) {
    const step = (t - 1) / (maxLen - 1);
    for (let i = 0; i < maxLen; i++) {
      const index = i === maxLen - 1 ? t - 1 : roundHalfEven(i * step);
      input.set(feats[index]!, i * N_FEATURES);
      mask[i] = 1;
    }
  } else {
    feats.forEach((f, i) => {
      input.set(f, i * N_FEATURES);
      mask[i] = 1;
    });
  }
  return { x: input, mask };
}
