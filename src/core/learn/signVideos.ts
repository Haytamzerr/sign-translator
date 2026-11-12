/**
 * Sign videos for Learn and Phrase to signs: short clips of a native signer
 * from ASL Signbank (Haskins Laboratories, Yale; CC BY-NC-SA 4.0).
 *
 * Files (made by training/signbank_videos.py):
 * - models/signs/index.json: which clip shows each built-in sign, each letter
 *   of the manual alphabet and each English word
 * - models/signs/<id>.mp4: one clip per sign (silent, 360 pixels high)
 */

export interface SignVideoIndex {
  source: string;
  license: string;
  /** Clip id -> Signbank gloss (the sign's name in capitals). */
  glosses: Record<string, string>;
  /** Built-in sign label -> clip id. */
  learn: Record<string, string>;
  /** Letter A-Z -> clip id. */
  letters: Record<string, string>;
  /** Lowercase English word or phrase -> clip id. */
  words: Record<string, string>;
}

export class SignVideos {
  constructor(readonly index: SignVideoIndex) {}

  /** Clip id for a built-in sign label, a letter, or an English word. */
  clipId(id: string): string | null {
    if (id.length === 1 && id >= 'A' && id <= 'Z') return this.index.letters[id] ?? null;
    return this.index.learn[id] ?? this.index.words[id.toLowerCase()] ?? null;
  }

  has(id: string): boolean {
    return this.clipId(id) !== null;
  }

  /** Address of the clip file, or null when there is no clip. */
  url(id: string): string | null {
    const clip = this.clipId(id);
    return clip ? `${import.meta.env.BASE_URL}models/signs/${clip}.mp4` : null;
  }

  /** The Signbank name of the sign shown, e.g. "MOTHER". */
  gloss(id: string): string {
    const clip = this.clipId(id);
    return clip ? (this.index.glosses[clip] ?? '') : '';
  }

  /** English words (and short phrases) that have a clip. */
  words(): string[] {
    return Object.keys(this.index.words);
  }
}

let shared: Promise<SignVideos | null> | null = null;

export function loadSignVideos(): Promise<SignVideos | null> {
  shared ??= fetch(`${import.meta.env.BASE_URL}models/signs/index.json`)
    .then((res) => (res.ok ? (res.json() as Promise<SignVideoIndex>) : null))
    .then((index) => (index ? new SignVideos(index) : null))
    .catch(() => null);
  return shared;
}
