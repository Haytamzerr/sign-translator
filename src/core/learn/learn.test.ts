import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { LetterDecoder, LetterModel, type LetterModelData } from '../letters';
import { TOPICS } from './categories';
import { decodeDemos, neutralPose, DEMO_POINTS } from './demos';
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

describe('demo decoding', () => {
  it('reads points and missing values', () => {
    const perFrame = DEMO_POINTS * 2;
    const data = new Int16Array(perFrame * 2).fill(500);
    data[0] = -32768;
    const layout = {
      leftHand: [0, 21], rightHand: [21, 42], arms: [42, 48], faceOval: [48, 84], lips: [84, 104],
      eyeL: [104, 120], eyeR: [120, 136], browL: [136, 146], browR: [146, 156],
    } as const;
    const demos = decodeDemos(
      { points: DEMO_POINTS, scale: 1000, missing: -32768, fps: 30, layout: layout as never, signs: [{ label: 'hello', offset: 0, frames: 2, confidence: 1 }] },
      data,
    );
    const frames = demos.get('hello')!.frames;
    expect(frames).toHaveLength(2);
    expect(Number.isNaN(frames[0]![0]!)).toBe(true);
    expect(frames[0]![1]).toBeCloseTo(0.5);
    const neutral = neutralPose(demos.values(), layout as never);
    expect(Number.isNaN(neutral[0]!)).toBe(true); // hands hidden in the neutral pose
    expect(neutral[48 * 2]).toBeCloseTo(0.5); // face kept
  });
});
