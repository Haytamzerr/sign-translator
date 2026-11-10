/**
 * Learning progress: the best score (1 to 3 stars) reached for each sign and
 * letter. Kept in this browser only.
 */

const KEY = 'signal.progress.v1';

export type Progress = Record<string, number>;

export function loadProgress(): Progress {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? '{}') as Progress;
  } catch {
    return {};
  }
}

export function saveProgress(progress: Progress): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(progress));
  } catch {
    // Storage unavailable: progress lasts for this session only.
  }
}

/** Keeps the best score: a weaker attempt never lowers it. */
export function recordScore(progress: Progress, id: string, stars: number): Progress {
  return stars > (progress[id] ?? 0) ? { ...progress, [id]: stars } : progress;
}

/**
 * Score for a practice attempt, from the model's ranked guesses:
 * 3 = recognized first and clearly, 2 = recognized first,
 * 1 = among the top three guesses, 0 = not recognized.
 */
export function scoreAttempt(target: string, guesses: { label: string; confidence: number }[]): number {
  const rank = guesses.findIndex((g) => g.label === target);
  if (rank === 0) return guesses[0]!.confidence >= 0.5 ? 3 : 2;
  return rank > 0 && rank < 3 ? 1 : 0;
}
