/** Shows what the recognizer is thinking right now, plus the transcript. */

import type { Translate } from '../i18n/strings';
import type { EngineSnapshot } from '../state/engine';
import { IconCopy } from './icons';

export interface TranscriptLine {
  id: number;
  text: string;
  time: number;
}

export function LivePanel({
  snap,
  librarySize,
  transcript,
  moveThreshold,
  t,
  locale,
  onTeach,
}: {
  snap: EngineSnapshot;
  librarySize: number;
  transcript: TranscriptLine[];
  moveThreshold: number;
  t: Translate;
  locale: string;
  onTeach: () => void;
}) {
  // Motion meter: full bar at three times the threshold, with a tick at the threshold.
  const motionFill = Math.min(snap.motion / (moveThreshold * 3), 1);

  return (
    <div className="panel-body">
      {librarySize === 0 && (
        <div className="callout">
          <h3>{t('emptyLibraryTitle')}</h3>
          <p>{t('emptyLibraryBody')}</p>
          <button className="btn btn-primary" onClick={onTeach}>
            {t('teachFirst')}
          </button>
        </div>
      )}

      <section className="block">
        <h3 className="block-title">{t('liveGuesses')}</h3>
        {snap.candidates.length === 0 ? (
          <p className="muted">{t('liveGuessesEmpty')}</p>
        ) : (
          <ol className="guesses">
            {snap.candidates.map((c) => {
              const accepted = c.ratio < 1;
              return (
                <li key={c.signId} className={accepted ? 'is-accepted' : ''}>
                  <span className="guess-label">{c.label}</span>
                  <span className="guess-kind">{c.kind === 'static' ? t('stillSign') : t('movingSign')}</span>
                  <span className="guess-value">{Math.round(c.confidence * 100)}%</span>
                  <span className="guess-bar" style={{ transform: `scaleX(${c.confidence})` }} aria-hidden="true" />
                </li>
              );
            })}
          </ol>
        )}
      </section>

      <section className="block">
        <h3 className="block-title">{t('liveMotion')}</h3>
        <div className="meter" role="meter" aria-valuenow={snap.motion} aria-valuemin={0} aria-valuemax={moveThreshold * 3}>
          <span className="meter-fill" style={{ transform: `scaleX(${motionFill})` }} />
          <span className="meter-tick" style={{ insetInlineStart: `${100 / 3}%` }} />
        </div>
      </section>

      <section className="block">
        <h3 className="block-title">{t('liveTranscript')}</h3>
        {transcript.length === 0 ? (
          <p className="muted">{t('liveTranscriptEmpty')}</p>
        ) : (
          <ul className="transcript">
            {[...transcript].reverse().map((line) => (
              <li key={line.id}>
                <time>{new Date(line.time).toLocaleTimeString(locale, { hour: '2-digit', minute: '2-digit' })}</time>
                <span>{line.text}</span>
                <button
                  className="icon-btn small"
                  aria-label={t('copy')}
                  onClick={() => void navigator.clipboard.writeText(line.text).catch(() => undefined)}
                >
                  <IconCopy size={15} />
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
