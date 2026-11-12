/**
 * Phrase to signs: turns typed text into a sequence of signs to show.
 *
 * Each word (or group of words such as "thank you" or "ice cream") is matched
 * against the 250 built-in signs, in English, French or Arabic, and against
 * the motion-captured signs (English words). A word with no
 * sign is spelled with the manual alphabet, letter by letter, which is also
 * what signers do for names.
 */

import { WORDS } from '../asl/labels';

export type PhraseItem =
  | { kind: 'sign'; text: string; label: string }
  | { kind: 'spell'; text: string; letters: string[] }
  | { kind: 'skip'; text: string; reason: 'omitted' | 'unavailable' };

/**
 * Extra ways of writing a sign's meaning. Only true equivalents: for example
 * "me" is NOT mapped to MY, because ASL has a different sign for it; an
 * unmatched word is spelled instead of being shown with the wrong sign.
 */
const ALIASES: Record<string, string[]> = {
  hello: ['hi', 'hey', 'salut', 'salam', 'السلام'],
  bye: ['goodbye', 'bye-bye', 'ciao'],
  thankyou: ['thanks', 'merci beaucoup', 'شكرا'],
  mom: ['mother', 'mommy', 'mum', 'mère', 'ماما'],
  dad: ['father', 'daddy', 'père', 'بابا'],
  kitty: ['kitten'],
  puppy: ['pup'],
  minemy: ['my', 'mine', 'mon', 'ma', 'mes'],
  weus: ['we', 'us', 'nous'],
  hesheit: ['he', 'she', 'it', 'him', 'her', 'il', 'elle'],
  icecream: ['ice-cream'],
  TV: ['television', 'télévision', 'tv'],
  fireman: ['firefighter', 'pompier'],
  glasswindow: ['window'],
  potty: ['toilet', 'bathroom'],
  frenchfries: ['fries', 'chips'],
  callonphone: ['phone', 'call', 'telephone', 'téléphone'],
  feet: ['foot'],
  tooth: ['teeth'],
  shoe: ['shoes'],
  yucky: ['gross', 'disgusting'],
  wake: ['wake up'],
  haveto: ['must', 'need to'],
  finish: ['done', 'finished'],
  happy: ['glad'],
  mad: ['angry', 'fâché', 'en colère'],
};

/**
 * Words ASL normally leaves out: articles and the verb "to be" (ASL says
 * "ME HAPPY", not "I am happy"). They are shown as skipped, not spelled.
 */
const OMITTED = new Set([
  'a', 'an', 'the', 'is', 'am', 'are', 'was', 'were', 'be', 'been',
  'le', 'la', 'les', "l'", 'un', 'une', 'des', 'du', 'est', 'suis', 'es', 'sont',
  'ال',
]);

/** Lowercase, without accents or punctuation, so "Mère" and "mere" match. */
export function normalizeWord(word: string): string {
  return word
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[ً-ْ]/g, '') // Arabic short vowel marks
    .toLowerCase()
    .replace(/[^\p{L}\p{N}' -]/gu, '')
    .trim();
}

let index: Map<string, string> | null = null;
let longest = 1;
/** Extra English words with a sign (motion-captured signs), added once loaded. */
let extraWords: string[] = [];

export function addPhraseWords(words: string[]): void {
  extraWords = words;
  index = null;
}

/** Builds the lookup table: normalized phrase -> sign label. */
function buildIndex(): Map<string, string> {
  const map = new Map<string, string>();
  const add = (phrase: string, label: string) => {
    const key = normalizeWord(phrase);
    if (!key || map.has(key)) return;
    map.set(key, label);
    longest = Math.max(longest, key.split(' ').length);
  };
  for (const [label, words] of Object.entries(WORDS)) {
    add(label, label);
    for (const w of words) for (const part of w.split('/')) add(part, label);
  }
  for (const [label, extra] of Object.entries(ALIASES)) for (const a of extra) add(a, label);
  for (const w of extraWords) add(w, w);
  return map;
}

/** Splits text into words, matching the longest known phrase first. */
export function phraseToItems(text: string): PhraseItem[] {
  index ??= buildIndex();
  const words = text.split(/\s+/).filter(Boolean);
  const items: PhraseItem[] = [];
  let i = 0;
  while (i < words.length) {
    let matched = false;
    for (let n = Math.min(longest, words.length - i); n >= 1; n--) {
      const original = words.slice(i, i + n).join(' ');
      const key = normalizeWord(original);
      const label = key ? index.get(key) : undefined;
      if (label) {
        items.push({ kind: 'sign', text: original.replace(/[.,!?;:]+$/, ''), label });
        i += n;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    const original = words[i]!.replace(/[.,!?;:"«»()]+/g, '');
    if (OMITTED.has(normalizeWord(original))) {
      items.push({ kind: 'skip', text: original, reason: 'omitted' });
      i++;
      continue;
    }
    const letters = normalizeWord(original)
      .toUpperCase()
      .split('')
      .filter((c) => c >= 'A' && c <= 'Z');
    if (original) {
      items.push(letters.length ? { kind: 'spell', text: original, letters } : { kind: 'skip', text: original, reason: 'unavailable' });
    }
    i++;
  }
  return items;
}
