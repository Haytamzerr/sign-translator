/**
 * Helpers for teaching a new sign: cleaning recorded takes, deciding whether
 * the sign is still or moving, and warning when it looks like an existing sign.
 */

import { frameShapeDistance, hasHands, motionBetween } from './features';
import { Recognizer, staticKeyFrames } from './recognizer';
import { trimEmpty, trimEntryExit, trimStill } from './sequence';
import type { Frame, Sign, SignKind } from './types';

/** Fewest frames with visible hands for a take to be usable. */
export const MIN_TAKE_FRAMES = 8;

export { trimEmpty, trimEntryExit } from './sequence';

/**
 * Measures how much a take moves: the largest wrist travel from the start
 * (in hand sizes) and the largest handshape change from the first frame.
 */
export function measureMovement(frames: Frame[]): { travel: number; shapeChange: number } {
  const withHands = frames.filter(hasHands);
  const first = withHands[0];
  if (!first) return { travel: 0, shapeChange: 0 };
  let travel = 0;
  let shapeChange = 0;
  for (const f of withHands) {
    shapeChange = Math.max(shapeChange, frameShapeDistance(first, f));
    for (let s = 0; s < 2; s++) {
      const a = first.hands[s];
      const b = f.hands[s];
      if (a && b) travel = Math.max(travel, Math.hypot(b.pos[0] - a.pos[0], b.pos[1] - a.pos[1]) / a.scale);
    }
  }
  return { travel, shapeChange };
}

/** A take is "moving" if the hand travels or the handshape changes noticeably. */
export function detectKind(takes: Frame[][]): SignKind {
  if (takes.length === 0) return 'static';
  const moving = takes.filter((t) => {
    const m = measureMovement(t);
    return m.travel > 0.6 || m.shapeChange > 0.3;
  }).length;
  return moving * 2 > takes.length ? 'dynamic' : 'static';
}

/** Average speed of a take, used to show a hint while recording. */
export function averageMotion(frames: Frame[]): number {
  if (frames.length < 2) return 0;
  let sum = 0;
  for (let i = 1; i < frames.length; i++) sum += motionBetween(frames[i - 1]!, frames[i]!);
  return sum / (frames.length - 1);
}

/** Final cleaning of a take before saving, depending on the sign type. */
export function prepareTake(frames: Frame[], kind: SignKind, moveThreshold: number): Frame[] {
  // A held sign keeps only the hold; a moving sign keeps its whole movement
  // (live movements are compared whole too), without still frames at the ends.
  return kind === 'dynamic' ? trimStill(trimEmpty(frames), moveThreshold * 0.6) : trimEntryExit(frames);
}

/**
 * Checks a new sign against the rest of the library. Returns the labels of
 * existing signs that the new examples would be mistaken for, with how many
 * of the examples were confused.
 */
export function findConflicts(
  draft: Pick<Sign, 'kind' | 'samples'>,
  others: Sign[],
  options: { sensitivity: number; eitherHand: boolean },
): { label: string; count: number }[] {
  const recognizer = new Recognizer(options);
  recognizer.setLibrary(others.filter((s) => s.kind === draft.kind));
  const counts = new Map<string, number>();
  for (const sample of draft.samples) {
    const candidates =
      draft.kind === 'static'
        ? recognizer.classifyStatic(staticKeyFrames(sample)[0] ?? sample[0]!)
        : recognizer.classifyDynamic(sample);
    const best = candidates[0];
    if (best && best.ratio < 1) counts.set(best.label, (counts.get(best.label) ?? 0) + 1);
  }
  return [...counts.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count);
}
