/** Loads the sign demonstrations and letter handshapes once, for Learn and Phrase views. */

import { useEffect, useState } from 'react';
import { loadDemos, type DemoLibrary, type HandShape } from '../core/learn/demos';
import { letterHandShape, letterHold } from '../core/learn/handshapes';
import { loadMocapClip, loadMocapIndex, mocapWords, type MocapClip } from '../core/learn/mocap';
import { addPhraseWords } from '../core/learn/phrase';
import { loadLetterShapes } from '../core/letters';
import type { Clip } from './SignPlayer';

export interface LearningData {
  status: 'loading' | 'ready' | 'missing';
  demos: DemoLibrary | null;
  shapes: Record<string, HandShape> | null;
}

let shared: Promise<LearningData> | null = null;

function load(): Promise<LearningData> {
  shared ??= Promise.all([loadDemos(), loadLetterShapes(), loadMocapIndex()]).then(([demos, shapes, mocap]) => {
    // Every motion-captured sign can be used in Phrase to signs (by its English word).
    if (mocap) addPhraseWords(mocapWords(mocap));
    return { status: demos ? 'ready' : 'missing', demos, shapes };
  });
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

/**
 * Motion-captured versions of the given signs, loaded in the background.
 * `ready` turns true once every one is loaded (or known to be missing).
 */
export function useMocap(ids: string[], data: LearningData): { ready: boolean; clips: Map<string, MocapClip> } {
  const key = ids.join('|');
  const [state, setState] = useState<{ key: string; clips: Map<string, MocapClip> } | null>(null);
  useEffect(() => {
    const layout = data.demos?.layout;
    if (!layout) return;
    let active = true;
    const wanted = key ? key.split('|').filter((id) => id.length > 1) : [];
    void Promise.all(wanted.map((id) => loadMocapClip(id, layout).then((c) => [id, c] as const))).then((pairs) => {
      if (!active) return;
      const clips = new Map<string, MocapClip>();
      for (const [id, c] of pairs) if (c) clips.set(id, c);
      setState({ key, clips });
    });
    return () => {
      active = false;
    };
  }, [key, data.demos]);
  return state?.key === key ? { ready: true, clips: state.clips } : { ready: false, clips: new Map() };
}

/**
 * The clip for a sign label or a single letter, if its data is available:
 * the motion-captured sign when there is one, otherwise the camera recording.
 */
export function clipFor(id: string, data: LearningData, mocap?: Map<string, MocapClip>): Clip | null {
  if (id.length === 1 && id >= 'A' && id <= 'Z') {
    // Exact handshape when defined; the recorded one otherwise.
    const shape = letterHandShape(id) ?? data.shapes?.[id];
    return shape ? { kind: 'letter', letter: id, shape, hold: letterHold(id) } : null;
  }
  const captured = mocap?.get(id);
  if (captured) return { kind: 'sign', label: id, frames: captured.frames, measured: true };
  const demo = data.demos?.get(id);
  return demo ? { kind: 'sign', label: id, frames: demo.frames } : null;
}

/** Address of an external video dictionary page for an English word. */
export function videoLink(english: string): string {
  return `https://www.signasl.org/sign/${encodeURIComponent(english.toLowerCase().replace(/\s+/g, '-'))}`;
}
