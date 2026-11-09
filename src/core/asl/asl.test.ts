/**
 * Parity test: the app must prepare the network's input exactly like the
 * training notebook. asl_parity.json (exported by the notebook) holds raw
 * recordings and the features Python computed for them.
 */

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { dropEmptyFrames, features, fitLength, mirrorRaw, N_FEATURES, type RawFrame } from './features';

const PARITY = process.env.ASL_PARITY ?? 'public/models/asl/asl_parity.json';

interface ParityItem {
  raw: (number | null)[][][];
  features: number[][];
  label: string;
}

function toFrames(raw: ParityItem['raw']): RawFrame[] {
  return raw.map((points) => Float64Array.from(points.flatMap(([x, y]) => [x ?? NaN, y ?? NaN])));
}

describe.skipIf(!existsSync(PARITY))('ASL features match the training notebook', () => {
  const items = existsSync(PARITY) ? (JSON.parse(readFileSync(PARITY, 'utf8')) as ParityItem[]) : [];

  it.each(items.map((item, i) => [i, item] as const))('recording %i', (_, item) => {
    const feats = features(dropEmptyFrames(toFrames(item.raw)));
    expect(feats.length).toBe(item.features.length);
    let worst = 0;
    feats.forEach((row, t) => row.forEach((v, j) => (worst = Math.max(worst, Math.abs(v - item.features[t]![j]!)))));
    expect(worst).toBeLessThan(2e-3);
  });
});

describe('ASL input shaping', () => {
  const frame = (v: number): RawFrame => new Float64Array(98).fill(v);

  it('mirroring twice gives back the original', () => {
    const f = Float64Array.from({ length: 98 }, (_, i) => i / 100);
    expect(Array.from(mirrorRaw(mirrorRaw(f)))).toEqual(Array.from(f));
  });

  it('pads short recordings and marks real frames', () => {
    const { x, mask } = fitLength(features([frame(0.5), frame(0.6)]), 48);
    expect(x.length).toBe(48 * N_FEATURES);
    expect(Array.from(mask.slice(0, 3))).toEqual([1, 1, 0]);
  });

  it('resamples long recordings to the maximum length', () => {
    const long = Array.from({ length: 100 }, (_, i) => frame(i / 100));
    const { mask } = fitLength(features(long), 48);
    expect(mask.every((m) => m === 1)).toBe(true);
  });

  it('never outputs NaN, even without any body or hand', () => {
    const { x } = fitLength(features([frame(NaN), frame(NaN)]), 48);
    expect(x.every((v) => Number.isFinite(v))).toBe(true);
  });
});
