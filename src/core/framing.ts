/**
 * Framing advice: tells the signer what to change when tracking struggles.
 *
 * The palm detector needs the whole palm in the picture and a hand that is
 * neither tiny nor filling the screen. The built-in ASL model also needs the
 * shoulders. These checks run on the landmarks already found, so they cost
 * nothing.
 */

import type { Point3, RawHand } from './types';

export type FramingHint = 'dark' | 'edge' | 'far' | 'close' | 'shoulders';

/** A fingertip or wrist closer than this to the border (fraction of the image) is "at the edge". */
const EDGE = 0.02;
/** Hand size (wrist to middle knuckle, fraction of image height) below which the hand is far. */
const FAR = 0.07;
/** Hand size above which the hand is too close. */
const CLOSE = 0.33;

export function framingHint(
  hands: RawHand[],
  pose: Point3[] | null,
  opts: { aspect: number; dark: boolean; needShoulders: boolean },
): FramingHint | null {
  if (opts.dark) return 'dark';
  for (const h of hands) {
    const p = h.landmarks;
    if (p.some((q) => q.x < EDGE || q.x > 1 - EDGE || q.y < EDGE || q.y > 1 - EDGE)) return 'edge';
    const size = Math.hypot((p[9]!.x - p[0]!.x) * opts.aspect, p[9]!.y - p[0]!.y);
    if (size < FAR) return 'far';
    if (size > CLOSE) return 'close';
  }
  if (opts.needShoulders) {
    const visible = (i: number) => {
      const q = pose?.[i] as (Point3 & { visibility?: number }) | undefined;
      return !!q && (q.visibility ?? 1) > 0.5 && q.y < 1 && q.y > 0;
    };
    if (!visible(11) || !visible(12)) return 'shoulders';
  }
  return null;
}

/** Shows a hint only after it has been true for a while, and hides it quickly. */
export class HintDebouncer {
  private candidate: FramingHint | null = null;
  private since = 0;
  current: FramingHint | null = null;

  constructor(private showAfterMs = 700) {}

  update(hint: FramingHint | null, t: number): FramingHint | null {
    if (hint !== this.candidate) {
      this.candidate = hint;
      this.since = t;
    }
    if (hint === null) this.current = null;
    else if (t - this.since >= this.showAfterMs) this.current = hint;
    return this.current;
  }
}
