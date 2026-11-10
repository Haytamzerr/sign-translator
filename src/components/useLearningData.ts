/** Loads the sign demonstrations and letter handshapes once, for Learn and Phrase views. */

import { useEffect, useState } from 'react';
import { loadDemos, type DemoLibrary } from '../core/learn/demos';
import { loadLetterShapes } from '../core/letters';
import type { Clip } from './SignPlayer';

export interface LearningData {
  status: 'loading' | 'ready' | 'missing';
  demos: DemoLibrary | null;
  shapes: Record<string, [number, number][]> | null;
}

let shared: Promise<LearningData> | null = null;

function load(): Promise<LearningData> {
  shared ??= Promise.all([loadDemos(), loadLetterShapes()]).then(([demos, shapes]) => ({
    status: demos ? 'ready' : 'missing',
    demos,
    shapes,
  }));
  return shared;
}

export function useLearningData(): LearningData {
  const [data, setData] = useState<LearningData>({ status: 'loading', demos: null, shapes: null });
  useEffect(() => {
    let active = true;
    void load().then((d) => active && setData(d));
    return () => {
      active = false;
    };
  }, []);
  return data;
}

/** The clip for a sign label or a single letter, if its data is available. */
export function clipFor(id: string, data: LearningData): Clip | null {
  if (id.length === 1 && id >= 'A' && id <= 'Z') {
    const shape = data.shapes?.[id];
    return shape ? { kind: 'letter', letter: id, shape } : null;
  }
  const demo = data.demos?.get(id);
  return demo ? { kind: 'sign', label: id, frames: demo.frames } : null;
}

/** Address of an external video dictionary page for an English word. */
export function videoLink(english: string): string {
  return `https://www.signasl.org/sign/${encodeURIComponent(english.toLowerCase().replace(/\s+/g, '-'))}`;
}
