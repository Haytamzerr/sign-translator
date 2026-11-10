/**
 * Fingerspelling: recognizing the 26 letters of the ASL manual alphabet.
 *
 * A letter is a handshape held still, so one frame is enough to recognize it.
 * The classifier is a small neural network (multi-layer perceptron) trained
 * on about 44,000 hand landmark samples recorded with the same MediaPipe
 * tracker this app uses (ASLNow! fingerspelling data, MIT license). It is so
 * small that it runs in plain TypeScript in a fraction of a millisecond.
 *
 * The decoder turns per-frame guesses into letters: a letter is written when
 * the same confident answer is held for a short time, and repeated only after
 * the hand changes or moves (that is how "LL" in "HELLO" is spelled).
 */

import { normalizeHand } from './features';
import type { Point3 } from './types';

interface Layer {
  w: number[][];
  b: number[];
}

export interface LetterModelData {
  labels: string[];
  aspect: number;
  valAccuracy?: number;
  layers: Layer[];
}

export interface LetterGuess {
  letter: string;
  confidence: number;
}

export class LetterModel {
  constructor(private data: LetterModelData) {}

  get labels(): string[] {
    return this.data.labels;
  }

  get accuracy(): number | undefined {
    return this.data.valAccuracy;
  }

  /** Probabilities for each letter, from 21 hand landmarks. */
  probabilities(points: Point3[], aspect: number): number[] {
    let x = normalizeHand(points, aspect).shape;
    this.data.layers.forEach((layer, i) => {
      const out = layer.b.slice();
      for (let o = 0; o < out.length; o++) {
        const row = layer.w[o]!;
        let sum = out[o]!;
        for (let k = 0; k < row.length; k++) sum += row[k]! * x[k]!;
        // ReLU on hidden layers, raw scores on the last one.
        out[o] = i < this.data.layers.length - 1 ? Math.max(0, sum) : sum;
      }
      x = out;
    });
    const max = Math.max(...x);
    const exp = x.map((v) => Math.exp(v - max));
    const total = exp.reduce((s, v) => s + v, 0);
    return exp.map((v) => v / total);
  }

  /** Best guesses, most likely first. */
  classify(points: Point3[], aspect: number, top = 3): LetterGuess[] {
    return this.probabilities(points, aspect)
      .map((confidence, i) => ({ letter: this.data.labels[i]!, confidence }))
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, top);
  }
}

const base = import.meta.env.BASE_URL;

export async function loadLetterModel(): Promise<LetterModel | null> {
  try {
    const res = await fetch(`${base}models/letters/letters_model.json`);
    if (!res.ok) return null;
    return new LetterModel((await res.json()) as LetterModelData);
  } catch {
    return null;
  }
}

/** Reference handshapes (21 points x, y, wrist at origin, hand size 1) per letter. */
export async function loadLetterShapes(): Promise<Record<string, [number, number][]> | null> {
  try {
    const res = await fetch(`${base}models/letters/letters_shapes.json`);
    return res.ok ? ((await res.json()) as Record<string, [number, number][]>) : null;
  } catch {
    return null;
  }
}

export interface LetterDecoderOptions {
  /** How long a letter must be held before it is written, ms. */
  holdMs: number;
  /** Minimum probability to accept a letter. */
  minConfidence: number;
}

export class LetterDecoder {
  private candidate: string | null = null;
  private since = 0;
  private lastEmitted: string | null = null;
  /** True once the hand has changed or moved since the last letter (allows double letters). */
  private released = true;

  constructor(private options: LetterDecoderOptions = { holdMs: 350, minConfidence: 0.7 }) {}

  setOptions(options: LetterDecoderOptions): void {
    this.options = options;
  }

  /**
   * @param guess   best letter for this frame, or null if no usable hand
   * @param moving  true when the hand is moving (between letters)
   * @returns the letter to write now, if any
   */
  push(t: number, guess: LetterGuess | null, moving: boolean): string | null {
    if (!guess || moving || guess.confidence < this.options.minConfidence) {
      this.candidate = null;
      this.released = true;
      return null;
    }
    if (guess.letter !== this.candidate) {
      if (this.candidate !== null) this.released = true;
      this.candidate = guess.letter;
      this.since = t;
      return null;
    }
    if (t - this.since < this.options.holdMs) return null;
    if (guess.letter === this.lastEmitted && !this.released) return null;
    this.lastEmitted = guess.letter;
    this.released = false;
    return guess.letter;
  }

  reset(): void {
    this.candidate = null;
    this.lastEmitted = null;
    this.released = true;
  }
}
