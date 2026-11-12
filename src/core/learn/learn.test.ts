import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { LetterDecoder, LetterModel, type LetterModelData } from '../letters';
import { TOPICS } from './categories';
import { decodeDemos, DEMO_POINTS, type DemoLayout } from './demos';
import { addPhraseWords, phraseToItems } from './phrase';
import { SignVideos, type SignVideoIndex } from './signVideos';

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
  });
});

describe('sign videos', () => {
  const videos = new SignVideos({
    source: '',
    license: '',
    glosses: { '1': 'MOTHER', '2': 'LETTER-A', '3': 'LOVE' },
    learn: { mom: '1' },
    letters: { A: '2' },
    words: { love: '3', 'good morning': '3' },
  });

  it('finds the clip of a built-in sign, a letter or a word', () => {
    expect(videos.clipId('mom')).toBe('1');
    expect(videos.clipId('A')).toBe('2');
    expect(videos.clipId('Love')).toBe('3');
    expect(videos.clipId('chin')).toBeNull();
    expect(videos.gloss('mom')).toBe('MOTHER');
  });

  it('lets Phrase to signs use video words and spell signs without a video', () => {
    addPhraseWords(videos.words(), (label) => videos.has(label));
    try {
      expect(phraseToItems('love chin')).toMatchObject([
        { kind: 'sign', label: 'love' },
        { kind: 'spell', letters: ['C', 'H', 'I', 'N'] },
      ]);
      expect(phraseToItems('Good morning mother')).toMatchObject([
        { kind: 'sign', label: 'good morning' },
        { kind: 'sign', label: 'mom' },
      ]);
    } finally {
      addPhraseWords([], () => true);
    }
  });
});

const VIDEO_INDEX = 'public/models/signs/index.json';

describe.skipIf(!existsSync(VIDEO_INDEX))('installed sign videos', () => {
  const index = JSON.parse(readFileSync(VIDEO_INDEX, 'utf8')) as SignVideoIndex;
  const clips = [...Object.values(index.learn), ...Object.values(index.letters), ...Object.values(index.words)];

  it('have a file for every clip they name', () => {
    for (const id of new Set(clips)) expect(existsSync(`public/models/signs/${id}.mp4`), id).toBe(true);
  });

  it('cover the alphabet and almost every built-in sign', () => {
    expect(Object.keys(index.letters)).toHaveLength(26);
    expect(Object.keys(index.learn).length).toBeGreaterThanOrEqual(240);
    for (const label of Object.keys(index.learn)) expect(TOPICS.some((t) => t.signs.includes(label)), label).toBe(true);
  });
});
