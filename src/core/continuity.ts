/**
 * Bridging short tracking gaps.
 *
 * During a fast movement the camera image blurs and MediaPipe can lose a hand
 * for one or two frames. Without help, the app would believe the hands went
 * down: a moving sign would be cut in two and the smoothing filter would
 * restart. Here a hand that disappears is kept at its last position for a
 * short time (150 ms by default, about 4 frames). If it comes back within
 * that time, nothing downstream notices the gap.
 */

import type { RawHand } from './types';

export interface BridgedHand extends RawHand {
  /** True when this hand was not detected in this frame but kept from the last one. */
  bridged: boolean;
}

/** Two hands closer than this (wrist to wrist, in image fractions) are the same hand. */
const SAME_HAND_DISTANCE = 0.12;

export class HandContinuity {
  private last = new Map<RawHand['side'], { hand: RawHand; t: number }>();

  constructor(private maxGapMs = 150) {}

  apply(hands: RawHand[], t: number, maxHands: number): BridgedHand[] {
    const out: BridgedHand[] = hands.map((h) => ({ ...h, bridged: false }));
    for (const h of hands) this.last.set(h.side, { hand: h, t });

    for (const [side, memory] of this.last) {
      if (t - memory.t > this.maxGapMs) {
        this.last.delete(side);
        continue;
      }
      if (out.length >= maxHands || out.some((h) => h.side === side)) continue;
      // MediaPipe sometimes flips the left/right label; do not bridge a "missing"
      // hand that is in fact the same hand detected under the other label.
      const wrist = memory.hand.landmarks[0]!;
      const duplicate = out.some((h) => {
        const w = h.landmarks[0]!;
        return Math.hypot(w.x - wrist.x, w.y - wrist.y) < SAME_HAND_DISTANCE;
      });
      if (!duplicate) out.push({ ...memory.hand, bridged: true });
    }
    return out;
  }

  reset(): void {
    this.last.clear();
  }
}
