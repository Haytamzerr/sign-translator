import { describe, expect, it } from 'vitest';
import { letterHandShape, letterHold } from './handshapes';

const dist = (a: number[], b: number[]) => Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);
/** How far a fingertip is from its knuckle, compared with the finger's full length. */
const reach = (shape: number[][], finger: number) => {
  const base = 5 + finger * 4;
  const length = [0, 1, 2].reduce((s, k) => s + dist(shape[base + k]!, shape[base + k + 1]!), 0);
  return dist(shape[base]!, shape[base + 3]!) / length;
};

describe('letter handshapes', () => {
  const letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

  it('defines all 26 letters with 21 points and a hand size of 1', () => {
    for (const l of letters) {
      const shape = letterHandShape(l)!;
      expect(shape).toHaveLength(21);
      expect(dist(shape[0]!, shape[9]!)).toBeCloseTo(1, 1);
    }
    expect(letterHandShape('?')).toBeNull();
  });

  it('extends exactly the right fingers', () => {
    // index, middle, ring, little: true = straight
    const open: Record<string, boolean[]> = {
      A: [false, false, false, false],
      B: [true, true, true, true],
      D: [true, false, false, false],
      I: [false, false, false, true],
      L: [true, false, false, false],
      V: [true, true, false, false],
      W: [true, true, true, false],
      Y: [false, false, false, true],
    };
    for (const [l, fingers] of Object.entries(open)) {
      const shape = letterHandShape(l)!;
      fingers.forEach((straight, f) => {
        if (straight) expect(reach(shape, f), `${l} finger ${f}`).toBeGreaterThan(0.95);
        else expect(reach(shape, f), `${l} finger ${f}`).toBeLessThan(0.75);
      });
    }
  });

  it('puts the thumb out for L and Y, and alongside the fist for A', () => {
    const l = letterHandShape('L')!;
    const a = letterHandShape('A')!;
    expect(l[4]![0]! - l[2]![0]!).toBeGreaterThan(0.5); // L thumb points to the side
    expect(a[4]![1]!).toBeLessThan(a[2]![1]!); // A thumb points up
  });

  it('holds G and H sideways and P and Q pointing down', () => {
    expect(letterHold('G')).toBe('sideways');
    expect(letterHold('H')).toBe('sideways');
    expect(letterHold('P')).toBe('down');
    expect(letterHold('B')).toBe('up');
    const p = letterHandShape('P')!;
    expect(p[8]![1]!).toBeGreaterThan(p[0]![1]!); // index tip below the wrist (y grows downwards)
  });
});
