import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { LetterDecoder, LetterModel, type LetterModelData } from '../letters';
import { TOPICS } from './categories';
import { decodeDemos, neutralPose, setPoint, DEMO_POINTS, type DemoLayout, type Rig } from './demos';
import { placeBone, prepareFrames, solveFigure, type Vec3 } from './figure';
import { phraseToItems } from './phrase';

const LABELS = 'public/models/asl/asl_labels.json';

describe('lesson topics', () => {
  it.skipIf(!existsSync(LABELS))('cover every built-in sign exactly once', () => {
    const labels = JSON.parse(readFileSync(LABELS, 'utf8')) as string[];
    const used = TOPICS.flatMap((t) => t.signs);
    expect(new Set(used).size).toBe(used.length);
    expect([...used].sort()).toEqual([...labels].sort());
  });
});

describe('phrase to signs', () => {
  it('matches multi-word signs first', () => {
    const items = phraseToItems('Thank you, see you tomorrow!');
    expect(items.map((i) => i.kind)).toEqual(['sign', 'sign', 'spell', 'sign']);
    expect(items[0]).toMatchObject({ kind: 'sign', label: 'thankyou' });
    expect(items[3]).toMatchObject({ kind: 'sign', label: 'tomorrow' });
  });

  it('spells unknown words and names', () => {
    expect(phraseToItems('Haytam')[0]).toEqual({ kind: 'spell', text: 'Haytam', letters: ['H', 'A', 'Y', 'T', 'A', 'M'] });
  });

  it('understands French and Arabic words', () => {
    expect(phraseToItems('bonjour maman')).toMatchObject([
      { kind: 'sign', label: 'hello' },
      { kind: 'sign', label: 'mom' },
    ]);
    expect(phraseToItems('مرحبا')[0]).toMatchObject({ kind: 'sign', label: 'hello' });
  });

  it('removes accents before spelling', () => {
    expect(phraseToItems('Zoé')[0]).toMatchObject({ kind: 'spell', letters: ['Z', 'O', 'E'] });
  });

  it('skips words that cannot be spelled with the ASL alphabet', () => {
    expect(phraseToItems('٣٤٥')[0]).toMatchObject({ kind: 'skip', reason: 'unavailable' });
  });

  it('leaves out words ASL does not sign', () => {
    expect(phraseToItems('the dog is happy').map((i) => i.kind)).toEqual(['skip', 'sign', 'skip', 'sign']);
  });
});

describe('letter decoder', () => {
  const guess = (letter: string, confidence = 0.9) => ({ letter, confidence });

  it('writes a letter once it is held', () => {
    const d = new LetterDecoder({ holdMs: 300, minConfidence: 0.7 });
    expect(d.push(0, guess('A'), false)).toBeNull();
    expect(d.push(200, guess('A'), false)).toBeNull();
    expect(d.push(350, guess('A'), false)).toBe('A');
    expect(d.push(800, guess('A'), false)).toBeNull();
  });

  it('allows a double letter after a small movement', () => {
    const d = new LetterDecoder({ holdMs: 300, minConfidence: 0.7 });
    d.push(0, guess('L'), false);
    expect(d.push(400, guess('L'), false)).toBe('L');
    d.push(450, guess('L'), true);
    d.push(500, guess('L'), false);
    expect(d.push(850, guess('L'), false)).toBe('L');
  });

  it('ignores unsure frames', () => {
    const d = new LetterDecoder({ holdMs: 300, minConfidence: 0.7 });
    d.push(0, guess('B', 0.5), false);
    expect(d.push(400, guess('B', 0.5), false)).toBeNull();
  });
});

const LETTER_MODEL = 'public/models/letters/letters_model.json';
const LETTER_PARITY = 'public/models/letters/letters_parity.json';

describe.skipIf(!existsSync(LETTER_MODEL) || !existsSync(LETTER_PARITY))('letter model matches the training notebook', () => {
  it('gives the same probabilities', () => {
    const model = new LetterModel(JSON.parse(readFileSync(LETTER_MODEL, 'utf8')) as LetterModelData);
    const parity = JSON.parse(readFileSync(LETTER_PARITY, 'utf8')) as { points: number[][]; probs: number[] }[];
    for (const item of parity) {
      const points = item.points.map(([x, y, z]) => ({ x: x!, y: y!, z: z ?? 0 }));
      const probs = model.probabilities(points, (model as unknown as { data: LetterModelData }).data.aspect);
      probs.forEach((p, i) => expect(Math.abs(p - item.probs[i]!)).toBeLessThan(2e-3));
    }
  });
});

const LAYOUT = {
  leftHand: [0, 21], rightHand: [21, 42], body: [42, 51], faceOval: [51, 87], lips: [87, 107],
  eyeL: [107, 123], eyeR: [123, 139], browL: [139, 149], browR: [149, 159],
} as DemoLayout;

describe('demo decoding', () => {
  it('reads 3D points and missing values', () => {
    const perFrame = DEMO_POINTS * 3;
    const data = new Int16Array(perFrame * 2).fill(500);
    data[0] = -32768;
    const demos = decodeDemos(
      { points: DEMO_POINTS, dims: 3, scale: 1000, missing: -32768, fps: 30, layout: LAYOUT, signs: [{ label: 'hello', offset: 0, frames: 2, confidence: 1 }] },
      data,
    );
    const frames = demos.get('hello')!.frames;
    expect(frames).toHaveLength(2);
    expect(frames[0]).toHaveLength(perFrame);
    expect(Number.isNaN(frames[0]![0]!)).toBe(true);
    expect(frames[0]![2]).toBeCloseTo(0.5); // z of the first point
    const neutral = neutralPose(demos.values(), LAYOUT);
    expect(Number.isNaN(neutral[0]!)).toBe(true); // hands hidden in the neutral pose
    expect(neutral[60 * 3]).toBeCloseTo(0.5); // face kept
  });
});

describe('3D figure', () => {
  const rig: Rig = { upperArm: 0.8, forearm: 0.75, hand: 0.3, faceWidth: 0.5, faceHeight: 0.65, eyeSide: 1 };
  const len = (a: number[], b: number[]) => Math.hypot(a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!);

  it('keeps bone lengths fixed, whatever the recording says', () => {
    for (const seen of [[0.1, 0.2, 0], [2, 2, 0], [0.3, 0.3, -0.5], [0.05, 0.05, 0.3]] as Vec3[]) {
      const end = placeBone([0, 0, 0], seen, 0.8);
      expect(len([0, 0, 0], end)).toBeCloseTo(0.8, 5);
    }
  });

  it('keeps what the picture shows and only solves depth', () => {
    const end = placeBone([0, 0, 0], [0.3, -0.4, 0], 1);
    expect(end[0]).toBeCloseTo(0.3);
    expect(end[1]).toBeCloseTo(-0.4);
    expect(end[2]).toBeGreaterThan(0); // towards the viewer by default
    expect(placeBone([0, 0, 0], [0.3, -0.4, -0.9], 1)[2]).toBeLessThan(0); // clearly behind
  });

  it('attaches the hand to the end of the forearm', () => {
    const pose = new Float32Array(DEMO_POINTS * 3).fill(NaN);
    const b = LAYOUT.body[0];
    setPoint(pose, b + 1, [0.5, 0, 0]);
    setPoint(pose, b + 2, [-0.5, 0, 0]);
    setPoint(pose, b + 4, [-0.7, 0.6, -0.2]);
    for (let i = 0; i < 21; i++) setPoint(pose, 21 + i, [-0.6 + i * 0.01, 0.2 - i * 0.02, -0.5]);
    const fig = solveFigure(pose, LAYOUT, rig);
    expect(len(fig.right.shoulder, fig.right.elbow)).toBeCloseTo(rig.upperArm, 4);
    expect(len(fig.right.elbow, fig.right.wrist)).toBeCloseTo(rig.forearm, 4);
    expect(fig.right.hand![0]).toEqual(fig.right.wrist);
    expect(fig.left.hand).toBeNull(); // not tracked: a closed hand is drawn
    expect(len(fig.left.elbow, fig.left.wrist)).toBeCloseTo(rig.forearm, 4);
  });

  it('fills short gaps in hand tracking', () => {
    const seen = new Float32Array(DEMO_POINTS * 3).fill(NaN);
    for (let i = 0; i < 21; i++) setPoint(seen, i, [i * 0.05, 0, 0]);
    const gap = new Float32Array(DEMO_POINTS * 3).fill(NaN);
    const out = prepareFrames([seen, gap, gap, seen], LAYOUT);
    expect(out[1]![27]).toBeCloseTo(0.45, 4); // knuckle 9 kept from the frame before
  });
});
