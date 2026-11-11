/**
 * Teaching a sign must not make the app write that word for other signs.
 *
 * Uses the 250 real recordings of the Learn data. A person is simulated:
 * the hand rises into view, signs at their own speed with some trembling,
 * holds, and drops. One sign is taught from 3 such takes, then other real
 * signs are performed and every word the app writes is counted.
 *
 * Half of the recordings serve as the recognizer's background of "other
 * signs"; mistakes are counted on the other half, never seen by it.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { SignDecoder } from '../decoder';
import { extractFrame } from '../features';
import { Recognizer } from '../recognizer';
import { detectKind, prepareTake } from '../teach';
import type { Frame, RawHand, Sign } from '../types';
import { demoToFrames } from './background';
import { decodeDemos } from './demos';

const index = JSON.parse(readFileSync('public/models/learn/sign_demos.json', 'utf8'));
const buf = readFileSync('public/models/learn/sign_demos.bin');
const demos = decodeDemos(index, new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2));
const labels = [...demos.keys()];
const backgroundLabels = labels.filter((_, i) => i % 2 === 0);
const unseenLabels = labels.filter((_, i) => i % 2 === 1);
const background = backgroundLabels.map((l) => demoToFrames(demos.get(l)!.frames, index.layout).filter((f) => f.hands[0] || f.hands[1]));

let seed = 3;
const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());

/** One performance of a recorded sign by a simulated person. `variation` scales speed changes and trembling. */
function perform(label: string, t0: number, variation: number): Frame[] {
  const poses = demos.get(label)!.frames;
  const speed = 1 + (rnd() - 0.5) * 0.3 * variation;
  const n = Math.max(4, Math.round(poses.length / speed));
  const at = (k: number) => poses[Math.min(poses.length - 1, Math.round((k * (poses.length - 1)) / (n - 1)))]!;
  const hands = (p: Float32Array, dy: number): RawHand[] => {
    const out: RawHand[] = [];
    for (const [start, side] of [[21, 'Right'], [0, 'Left']] as const) {
      if (Number.isNaN(p[start * 3]!)) continue;
      const landmarks = Array.from({ length: 21 }, (_, i) => ({
        x: 0.5 + p[(start + i) * 3]! * 0.25 + gauss() * 0.0015 * variation,
        y: 0.35 + (p[(start + i) * 3 + 1]! + dy) * 0.25 + gauss() * 0.0015 * variation,
        z: p[(start + i) * 3 + 2]! * 0.25,
      }));
      out.push({ landmarks, side, score: 1 });
    }
    return out;
  };
  const frames: Frame[] = [];
  let t = t0;
  const push = (h: RawHand[]) => frames.push(extractFrame(h, 1, (t += 33)));
  for (let k = 0; k < 8; k++) push([]);
  for (let k = 10; k > 0; k--) push(hands(at(0), k * 0.15)); // rises into view
  for (let k = 0; k < n; k++) push(hands(at(k), 0));
  for (let k = 0; k < 12; k++) push(hands(at(n - 1), 0)); // holds
  for (let k = 1; k <= 10; k++) push(hands(at(n - 1), k * 0.15)); // drops
  for (let k = 0; k < 12; k++) push([]);
  return frames;
}

function trial(taught: string, variation: number, withBackground: boolean) {
  const takes = [0, 1, 2].map(() => perform(taught, 0, variation));
  const kind = detectKind(takes);
  const sign: Sign = {
    id: 'mine', label: taught, language: 'asl', kind, category: '',
    samples: takes.map((f) => prepareTake(f, kind, 1.5)), createdAt: 0, updatedAt: 0,
  };
  const recognizer = new Recognizer({ sensitivity: 1, eitherHand: true });
  recognizer.setLibrary([sign]);
  if (withBackground) recognizer.setBackground(background);
  const decoder = new SignDecoder(recognizer, { holdMs: 400, moveThreshold: 1.5, minConfidence: 0.55 });
  const wrote = (frames: Frame[]) => frames.map((f) => decoder.push(f).commit !== null).some(Boolean);
  const others = unseenLabels.filter((l) => l !== taught);
  const falseWords = others.filter((l, i) => wrote(perform(l, (i + 1) * 1e4, variation))).length;
  const found = Array.from({ length: 10 }, (_, r) => wrote(perform(taught, 1e7 + r * 1e4, variation))).filter(Boolean).length;
  return { falseRate: falseWords / others.length, found };
}

describe('teaching a sign does not take over other signs', () => {
  const taughtSigns = ['hello', 'mom', 'cat', 'happy', 'water', 'yes'].filter((l) => demos.has(l));

  it('rejects other real signs, even with sloppy takes', () => {
    for (const variation of [1, 4, 8]) {
      let before = 0;
      let after = 0;
      let foundAfter = 0;
      for (const taught of taughtSigns) {
        before += trial(taught, variation, false).falseRate;
        const r = trial(taught, variation, true);
        after += r.falseRate;
        foundAfter += r.found;
      }
      const n = taughtSigns.length;
      console.log(`variation ${variation}: wrong words ${((before / n) * 100).toFixed(1)}% -> ${((after / n) * 100).toFixed(1)}% with background; taught sign found ${foundAfter}/${n * 10}`);
      expect(after / n).toBeLessThanOrEqual(0.01);
      expect(foundAfter / (n * 10)).toBeGreaterThanOrEqual(0.7);
    }
  }, 120_000);
});
