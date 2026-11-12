/** Loads the sign videos once, for the Learn and Phrase views. */

import { useEffect, useState } from 'react';
import { addPhraseWords } from '../core/learn/phrase';
import { loadSignVideos, type SignVideos } from '../core/learn/signVideos';
import type { VideoClip } from './SignVideoPlayer';

export interface LearningData {
  status: 'loading' | 'ready' | 'missing';
  videos: SignVideos | null;
}

let shared: Promise<LearningData> | null = null;

function load(): Promise<LearningData> {
  shared ??= loadSignVideos().then((videos) => {
    // Every English word with a video can be used in Phrase to signs.
    if (videos) addPhraseWords(videos.words(), (label) => videos.has(label));
    return { status: videos ? 'ready' : 'missing', videos };
  });
  return shared;
}

export function useLearningData(): LearningData {
  const [data, setData] = useState<LearningData>({ status: 'loading', videos: null });
  useEffect(() => {
    let active = true;
    void load().then((d) => active && setData(d));
    return () => {
      active = false;
    };
  }, []);
  return data;
}

/** The clip for a sign label, an English word or a single letter, if there is one. */
export function clipFor(id: string, data: LearningData): VideoClip | null {
  const src = data.videos?.url(id);
  return src ? { src, letter: id.length === 1 } : null;
}

/** Address of an external video dictionary page for an English word. */
export function videoLink(english: string): string {
  return `https://www.signasl.org/sign/${encodeURIComponent(english.toLowerCase().replace(/\s+/g, '-'))}`;
}
