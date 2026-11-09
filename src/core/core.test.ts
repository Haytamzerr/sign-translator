import { describe, expect, it } from 'vitest';
import { extractFrame, frameShapeDistance, mirrorFrame, motionBetween, normalizeHand } from './features';
import { dtw, toSequence } from './sequence';
import { Recognizer } from './recognizer';
import { SignDecoder } from './decoder';
import { detectKind, findConflicts, trimEmpty } from './teach';
import type { Frame, Point3, RawHand, Sign } from './types';

// ---------- Synthetic hands -------------------------------------------------
// A simple 2D hand model: a wrist and five fingers of four joints each.
// `curl` 0 = straight finger, 1 = folded towards the palm.

function makeHand(curls: number[], cx: number, cy: number, size: number, noise = 0, seed = 1): Point3[] {
  let s = seed;
  const rand = () => {
    s = (s * 16807) % 2147483647;
    return ((s / 2147483647) * 2 - 1) * noise;
  };
  const pts: Point3[] = [{ x: cx, y: cy, z: 0 }];
  const angles = [-0.9, -0.35, 0, 0.3, 0.6];
  for (let f = 0; f < 5; f++) {
    const a = angles[f]!;
    const curl = curls[f]!;
    for (let j = 1; j <= 4; j++) {
      const reach = j * 0.25 * (1 - curl * 0.7 * (j / 4));
      pts.push({
        x: cx + Math.sin(a) * reach * size + rand() * size,
        y: cy - Math.cos(a) * reach * size + rand() * size,
        z: rand() * size,
      });
    }
  }
  return pts;
}

const OPEN = [0, 0, 0, 0, 0];
const FIST = [1, 1, 1, 1, 1];
const POINT = [1, 0, 1, 1, 1];

function frameOf(curls: number[], t: number, cx = 0.5, cy = 0.5, size = 0.2, noise = 0, seed = 1): Frame {
  const raw: RawHand = { landmarks: makeHand(curls, cx, cy, size, noise, seed), side: 'Right', score: 0.99 };
  return extractFrame([raw], 1, t);
}

function holdTake(curls: number[], seed: number, n = 20): Frame[] {
  return Array.from({ length: n }, (_, i) => frameOf(curls, i * 33, 0.5, 0.5, 0.2, 0.005, seed * 100 + i));
}

/** A hand moving in a line from (x0, y0) to (x1, y1) over `n` frames. */
function moveTake(curls: number[], x0: number, y0: number, x1: number, y1: number, seed: number, n = 30): Frame[] {
  return Array.from({ length: n }, (_, i) => {
    const k = i / (n - 1);
    return frameOf(curls, i * 33, x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, 0.2, 0.005, seed * 100 + i);
  });
}

function sign(id: string, label: string, kind: Sign['kind'], samples: Frame[][]): Sign {
  return { id, label, language: 'test', kind, category: '', samples, createdAt: 0, updatedAt: 0 };
}

// ---------- Tests -----------------------------------------------------------

describe('features', () => {
  it('ignores position and distance from the camera', () => {
    const near = normalizeHand(makeHand(OPEN, 0.3, 0.4, 0.3), 1);
    const far = normalizeHand(makeHand(OPEN, 0.7, 0.6, 0.1), 1);
    expect(frameShapeDistance({ t: 0, hands: [near, null] }, { t: 0, hands: [far, null] })).toBeLessThan(1e-9);
  });

  it('separates different handshapes', () => {
    expect(frameShapeDistance(frameOf(OPEN, 0), frameOf(FIST, 0))).toBeGreaterThan(0.2);
  });

  it('mirroring twice gives back the original', () => {
    const f = frameOf(POINT, 0);
    expect(frameShapeDistance(mirrorFrame(mirrorFrame(f)), f)).toBeLessThan(1e-9);
  });

  it('measures motion in hand sizes per second', () => {
    const a = frameOf(OPEN, 0, 0.4, 0.5);
    const b = frameOf(OPEN, 100, 0.44, 0.5); // hand size is 0.05 here, so 0.04 = 0.8 hand sizes in 0.1 s
    expect(motionBetween(a, b)).toBeCloseTo(8, 1);
  });
});

describe('dynamic time warping', () => {
  it('matches the same movement at different speeds', () => {
    const slow = toSequence(moveTake(OPEN, 0.3, 0.5, 0.7, 0.5, 1, 40));
    const fast = toSequence(moveTake(OPEN, 0.3, 0.5, 0.7, 0.5, 2, 15));
    const other = toSequence(moveTake(OPEN, 0.5, 0.3, 0.5, 0.7, 3, 30));
    expect(dtw(slow, fast)).toBeLessThan(dtw(slow, other) / 3);
  });
});

describe('recognizer', () => {
  const library = [
    sign('a', 'hello', 'static', [holdTake(OPEN, 1), holdTake(OPEN, 2), holdTake(OPEN, 3)]),
    sign('b', 'yes', 'static', [holdTake(FIST, 4), holdTake(FIST, 5), holdTake(FIST, 6)]),
    sign('c', 'go', 'dynamic', [moveTake(POINT, 0.3, 0.5, 0.7, 0.5, 7), moveTake(POINT, 0.3, 0.5, 0.7, 0.5, 8)]),
  ];
  const recognizer = new Recognizer({ sensitivity: 1, eitherHand: true });
  recognizer.setLibrary(library);

  it('recognizes still signs', () => {
    const best = recognizer.classifyStatic(frameOf(FIST, 0, 0.6, 0.4, 0.15, 0.01, 99))[0]!;
    expect(best.label).toBe('yes');
    expect(best.ratio).toBeLessThan(1);
    expect(best.confidence).toBeGreaterThan(0.6);
  });

  it('rejects an unknown handshape', () => {
    const best = recognizer.classifyStatic(frameOf(POINT, 0))[0]!;
    expect(best.ratio).toBeGreaterThan(1);
  });

  it('recognizes a moving sign, also performed faster', () => {
    const best = recognizer.classifyDynamic(moveTake(POINT, 0.2, 0.6, 0.6, 0.6, 50, 18))[0]!;
    expect(best.label).toBe('go');
    expect(best.ratio).toBeLessThan(1);
  });

  it('rejects a different movement', () => {
    const best = recognizer.classifyDynamic(moveTake(POINT, 0.5, 0.2, 0.5, 0.8, 51))[0]!;
    expect(best.ratio).toBeGreaterThan(1);
  });
});

describe('decoder', () => {
  it('writes a held sign once, then a moving sign', () => {
    const recognizer = new Recognizer({ sensitivity: 1, eitherHand: false });
    recognizer.setLibrary([
      sign('b', 'yes', 'static', [holdTake(FIST, 4), holdTake(FIST, 5)]),
      sign('c', 'go', 'dynamic', [moveTake(POINT, 0.3, 0.5, 0.7, 0.5, 7), moveTake(POINT, 0.3, 0.5, 0.7, 0.5, 8)]),
    ]);
    const decoder = new SignDecoder(recognizer, { holdMs: 300, moveThreshold: 1.2, minConfidence: 0.5 });
    const words: string[] = [];
    let t = 0;
    const feed = (f: Frame) => {
      const out = decoder.push({ ...f, t });
      t += 33;
      if (out.commit) words.push(out.commit.label);
    };
    for (let i = 0; i < 30; i++) feed(frameOf(FIST, 0, 0.5, 0.5, 0.2, 0.005, 200 + i)); // hold 1 s
    for (let i = 0; i < 5; i++) feed({ t: 0, hands: [null, null] }); // hands down
    for (const f of moveTake(POINT, 0.3, 0.5, 0.7, 0.5, 9)) feed(f); // move
    for (let i = 0; i < 12; i++) feed(frameOf(POINT, 0, 0.7, 0.5, 0.2, 0.005, 300 + i)); // stop
    expect(words).toEqual(['yes', 'go']);
  });
});

describe('teaching', () => {
  it('detects still and moving signs', () => {
    expect(detectKind([holdTake(FIST, 1), holdTake(FIST, 2)])).toBe('static');
    expect(detectKind([moveTake(OPEN, 0.3, 0.5, 0.7, 0.5, 1), moveTake(OPEN, 0.3, 0.5, 0.7, 0.5, 2)])).toBe('dynamic');
  });

  it('trims frames without hands', () => {
    const empty: Frame = { t: 0, hands: [null, null] };
    expect(trimEmpty([empty, ...holdTake(OPEN, 1, 5), empty]).length).toBe(5);
  });

  it('warns when a new sign looks like an existing one', () => {
    const existing = [sign('a', 'hello', 'static', [holdTake(OPEN, 1), holdTake(OPEN, 2)])];
    const conflicts = findConflicts({ kind: 'static', samples: [holdTake(OPEN, 7)] }, existing, {
      sensitivity: 1,
      eitherHand: false,
    });
    expect(conflicts[0]?.label).toBe('hello');
  });
});
