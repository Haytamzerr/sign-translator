/**
 * Jitter removal with the "One Euro Filter" (Casiez, Roussel and Vogel, 2012).
 *
 * Camera landmarks tremble slightly even when the hand is still. A normal
 * average removes the trembling but makes everything lag behind. The One Euro
 * Filter adapts: when the hand is slow it smooths strongly (no jitter), when
 * the hand moves fast it smooths very little (no lag). That is exactly what
 * real-time sign recognition needs.
 */

import type { Frame, HandFeature } from './types';

/** Smoothing factor for a given cutoff frequency and time step. */
function alpha(cutoff: number, dt: number): number {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuro {
  private x: number | null = null;
  private dx = 0;

  constructor(
    private minCutoff: number,
    private beta: number,
  ) {}

  filter(value: number, dt: number): number {
    if (this.x === null) {
      this.x = value;
      return value;
    }
    // 1. Estimate the speed of change (itself lightly smoothed).
    const rawDx = (value - this.x) / dt;
    this.dx += alpha(1, dt) * (rawDx - this.dx);
    // 2. Faster change -> higher cutoff -> less smoothing.
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }
}

/** Smooths every number of one hand slot (63 shape values + 2 position values). */
class HandFilter {
  private filters: OneEuro[];

  constructor(minCutoff: number, beta: number) {
    this.filters = Array.from({ length: 65 }, () => new OneEuro(minCutoff, beta));
  }

  apply(hand: HandFeature, dt: number): HandFeature {
    const shape = hand.shape.map((v, i) => this.filters[i]!.filter(v, dt));
    const pos: [number, number] = [this.filters[63]!.filter(hand.pos[0], dt), this.filters[64]!.filter(hand.pos[1], dt)];
    return { shape, pos, scale: hand.scale };
  }
}

export class FrameSmoother {
  private slots: (HandFilter | null)[] = [null, null];
  private lastT = 0;

  constructor(
    private minCutoff = 1.5,
    private beta = 0.4,
  ) {}

  smooth(frame: Frame): Frame {
    const dt = this.lastT ? Math.max((frame.t - this.lastT) / 1000, 1 / 240) : 1 / 30;
    this.lastT = frame.t;
    const hands = frame.hands.map((hand, s) => {
      if (!hand) {
        this.slots[s] = null; // hand left: start fresh when it comes back
        return null;
      }
      const filter = (this.slots[s] ??= new HandFilter(this.minCutoff, this.beta));
      return filter.apply(hand, dt);
    }) as Frame['hands'];
    return { t: frame.t, hands };
  }

  reset(): void {
    this.slots = [null, null];
    this.lastT = 0;
  }
}
