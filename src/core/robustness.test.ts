import { describe, expect, it } from 'vitest';
import { HandContinuity } from './continuity';
import { framingHint, HintDebouncer } from './framing';
import { boostFilter, meanLuma, nextBoostState } from './lighting';
import type { Point3, RawHand } from './types';

/** A simple hand: wrist at (x, y), middle knuckle `size` above it, other points around. */
function hand(side: RawHand['side'], x: number, y: number, size = 0.12): RawHand {
  const landmarks: Point3[] = Array.from({ length: 21 }, (_, i) => ({
    x: x + ((i % 5) - 2) * size * 0.2,
    y: y - (i / 20) * size * 1.8,
    z: 0,
  }));
  landmarks[0] = { x, y, z: 0 };
  landmarks[9] = { x, y: y - size, z: 0 };
  return { landmarks, side, score: 0.9 };
}

describe('hand continuity', () => {
  it('keeps a hand through a short gap', () => {
    const c = new HandContinuity(150);
    c.apply([hand('Right', 0.5, 0.6)], 0, 2);
    const out = c.apply([], 66, 2);
    expect(out).toHaveLength(1);
    expect(out[0]!.bridged).toBe(true);
  });

  it('lets a hand go after a long gap', () => {
    const c = new HandContinuity(150);
    c.apply([hand('Right', 0.5, 0.6)], 0, 2);
    expect(c.apply([], 200, 2)).toHaveLength(0);
  });

  it('does not duplicate a hand whose left/right label flipped', () => {
    const c = new HandContinuity(150);
    c.apply([hand('Right', 0.5, 0.6)], 0, 2);
    const out = c.apply([hand('Left', 0.51, 0.6)], 33, 2);
    expect(out).toHaveLength(1);
    expect(out[0]!.bridged).toBe(false);
  });

  it('never returns more hands than allowed', () => {
    const c = new HandContinuity(150);
    c.apply([hand('Right', 0.3, 0.6)], 0, 1);
    expect(c.apply([hand('Left', 0.7, 0.6)], 33, 1)).toHaveLength(1);
  });
});

describe('lighting', () => {
  it('measures brightness', () => {
    expect(meanLuma(new Uint8ClampedArray([255, 255, 255, 255, 0, 0, 0, 255]))).toBeCloseTo(127.5);
  });

  it('switches the boost with hysteresis', () => {
    expect(nextBoostState(false, 80)).toBe(false);
    expect(nextBoostState(false, 60)).toBe(true);
    expect(nextBoostState(true, 85)).toBe(true);
    expect(nextBoostState(true, 110)).toBe(false);
  });

  it('limits the brightness gain', () => {
    expect(boostFilter(5)).toContain('brightness(2.40)');
    expect(boostFilter(200)).toContain('brightness(1.00)');
  });
});

describe('framing hints', () => {
  const opts = { aspect: 4 / 3, dark: false, needShoulders: false };

  it('is quiet for a well framed hand', () => {
    expect(framingHint([hand('Right', 0.5, 0.7)], null, opts)).toBeNull();
  });

  it('reports dark images first', () => {
    expect(framingHint([hand('Right', 0.5, 0.7)], null, { ...opts, dark: true })).toBe('dark');
  });

  it('reports hands at the edge, too far or too close', () => {
    expect(framingHint([hand('Right', 0.99, 0.7)], null, opts)).toBe('edge');
    expect(framingHint([hand('Right', 0.5, 0.7, 0.03)], null, opts)).toBe('far');
    expect(framingHint([hand('Right', 0.5, 0.95, 0.4)], null, opts)).toBe('close');
  });

  it('asks for the shoulders when the ASL model needs them', () => {
    expect(framingHint([hand('Right', 0.5, 0.7)], null, { ...opts, needShoulders: true })).toBe('shoulders');
  });

  it('shows a hint only after it persists', () => {
    const d = new HintDebouncer(500);
    expect(d.update('far', 0)).toBeNull();
    expect(d.update('far', 600)).toBe('far');
    expect(d.update(null, 650)).toBeNull();
  });
});
