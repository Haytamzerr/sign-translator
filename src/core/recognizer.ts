/**
 * The recognizer: compares what the camera sees now with the examples in the
 * user's sign library and says which sign it is (or that it is none of them).
 *
 * It learns instantly: adding an example only means storing it. Recognition is
 * "nearest neighbors": find the stored examples most similar to the input.
 */

import { frameShapeDistance, hasHands, mirrorFrame } from './features';
import { dtw, toSequence, type SeqStep } from './sequence';
import type { Candidate, Frame, Sign, SignKind } from './types';

export interface RecognizerOptions {
  /** Multiplies every acceptance threshold: above 1 accepts more, below 1 is stricter. */
  sensitivity: number;
  /** Also recognize the mirror image of each sign (made with the other hand). */
  eitherHand: boolean;
}

/** Default and allowed range for each sign's acceptance threshold. */
const THRESHOLDS: Record<SignKind, { initial: number; min: number; max: number }> = {
  static: { initial: 0.16, min: 0.07, max: 0.3 },
  dynamic: { initial: 0.18, min: 0.08, max: 0.4 },
};
/** How sharply confidence drops as distance grows (softmax temperature). */
const SHARPNESS = 5;
/** Number of frames taken from each still example. */
const FRAMES_PER_STATIC_SAMPLE = 6;
/** For still signs, the score of a sign is the average of its k best matches. */
const K_NEAREST = 3;

interface Prototype {
  sign: Sign;
  sample: number;
  frame: Frame;
}
interface Template {
  sign: Sign;
  sample: number;
  seq: SeqStep[];
}

/** Picks evenly spaced frames from the middle 60% of a still example. */
export function staticKeyFrames(sample: Frame[], count = FRAMES_PER_STATIC_SAMPLE): Frame[] {
  const withHands = sample.filter(hasHands);
  if (withHands.length === 0) return [];
  const start = Math.floor(withHands.length * 0.2);
  const end = Math.max(start + 1, Math.ceil(withHands.length * 0.8));
  const middle = withHands.slice(start, end);
  if (middle.length <= count) return middle;
  return Array.from({ length: count }, (_, i) => middle[Math.round((i * (middle.length - 1)) / (count - 1))]!);
}

export class Recognizer {
  private prototypes: Prototype[] = [];
  private templates: Template[] = [];
  private thresholds = new Map<string, number>();
  private signs: Sign[] = [];

  constructor(private options: RecognizerOptions) {}

  setOptions(options: RecognizerOptions): void {
    const rebuild = options.eitherHand !== this.options.eitherHand;
    this.options = options;
    if (rebuild) this.setLibrary(this.signs);
  }

  get size(): number {
    return this.signs.length;
  }

  get hasStatic(): boolean {
    return this.prototypes.length > 0;
  }

  get hasDynamic(): boolean {
    return this.templates.length > 0;
  }

  /** Prepares every example once, so recognition per frame stays fast. */
  setLibrary(signs: Sign[]): void {
    this.signs = signs;
    this.prototypes = [];
    this.templates = [];
    for (const sign of signs) {
      sign.samples.forEach((sample, index) => {
        const variants = this.options.eitherHand ? [sample, sample.map(mirrorFrame)] : [sample];
        for (const v of variants) {
          if (sign.kind === 'static') {
            for (const frame of staticKeyFrames(v)) this.prototypes.push({ sign, sample: index, frame });
          } else if (v.some(hasHands)) {
            this.templates.push({ sign, sample: index, seq: toSequence(v.filter(hasHands)) });
          }
        }
      });
    }
    this.calibrate();
  }

  /** The acceptance threshold for a sign, including the sensitivity setting. */
  threshold(signId: string, kind: SignKind): number {
    return (this.thresholds.get(signId) ?? THRESHOLDS[kind].initial) * this.options.sensitivity;
  }

  /**
   * Automatic calibration. For each sign we check how different its own
   * examples are from each other ("leave one out"). A sign you perform very
   * consistently gets a strict threshold; a looser sign gets a wider one.
   */
  private calibrate(): void {
    this.thresholds.clear();
    for (const sign of this.signs) {
      const limits = THRESHOLDS[sign.kind];
      if (sign.samples.length < 2) {
        this.thresholds.set(sign.id, limits.initial);
        continue;
      }
      let worst = 0;
      if (sign.kind === 'static') {
        const own = this.prototypes.filter((p) => p.sign.id === sign.id);
        for (const p of own) {
          let best = Infinity;
          for (const q of own) if (q.sample !== p.sample) best = Math.min(best, frameShapeDistance(p.frame, q.frame));
          if (best < Infinity) worst = Math.max(worst, best);
        }
      } else {
        const own = this.templates.filter((t) => t.sign.id === sign.id);
        for (const a of own) {
          let best = Infinity;
          for (const b of own) if (b.sample !== a.sample) best = Math.min(best, dtw(a.seq, b.seq));
          if (best < Infinity) worst = Math.max(worst, best);
        }
      }
      const value = worst * 1.6 + limits.min * 0.5;
      this.thresholds.set(sign.id, Math.min(limits.max, Math.max(limits.min, value)));
    }
  }

  /** Recognizes a held handshape from a single frame. */
  classifyStatic(frame: Frame): Candidate[] {
    if (!hasHands(frame) || this.prototypes.length === 0) return [];
    const perSign = new Map<string, { sign: Sign; distances: number[] }>();
    for (const p of this.prototypes) {
      const d = frameShapeDistance(frame, p.frame);
      const entry = perSign.get(p.sign.id) ?? { sign: p.sign, distances: [] };
      entry.distances.push(d);
      perSign.set(p.sign.id, entry);
    }
    const scored = [...perSign.values()].map(({ sign, distances }) => {
      distances.sort((a, b) => a - b);
      const k = distances.slice(0, K_NEAREST);
      return { sign, distance: k.reduce((s, v) => s + v, 0) / k.length };
    });
    return this.rank(scored);
  }

  /** Recognizes a moving sign from a segment of frames. */
  classifyDynamic(frames: Frame[]): Candidate[] {
    const withHands = frames.filter(hasHands);
    if (withHands.length < 4 || this.templates.length === 0) return [];
    const seq = toSequence(withHands);
    const best = new Map<string, { sign: Sign; distance: number }>();
    for (const t of this.templates) {
      const d = dtw(seq, t.seq);
      const cur = best.get(t.sign.id);
      if (!cur || d < cur.distance) best.set(t.sign.id, { sign: t.sign, distance: d });
    }
    return this.rank([...best.values()]);
  }

  /**
   * Turns distances into probabilities. Each sign's distance is divided by its
   * threshold (ratio). An extra "unknown" option always sits at ratio 1, so if
   * nothing is clearly closer than its threshold, "unknown" wins and the
   * confidence of every real sign stays low. That is how random hand movements
   * are rejected instead of being forced into the nearest word.
   */
  private rank(scored: { sign: Sign; distance: number }[]): Candidate[] {
    const withRatio = scored.map(({ sign, distance }) => ({
      sign,
      distance,
      ratio: distance / this.threshold(sign.id, sign.kind),
    }));
    const unknownWeight = Math.exp(-SHARPNESS);
    const weights = withRatio.map((c) => Math.exp(-SHARPNESS * c.ratio));
    const total = weights.reduce((s, w) => s + w, unknownWeight);
    return withRatio
      .map((c, i) => ({
        signId: c.sign.id,
        label: c.sign.label,
        kind: c.sign.kind,
        distance: c.distance,
        ratio: c.ratio,
        confidence: weights[i]! / total,
      }))
      .sort((a, b) => a.ratio - b.ratio);
  }
}
