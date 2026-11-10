/**
 * Learn: lessons by topic. Watch an animated demonstration, then make the sign
 * in front of the camera; the built-in model checks it and gives 1 to 3 stars.
 * Quiz mode hides the demonstration until you ask for it.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { translateAslLabel, WORDS } from '../core/asl/labels';
import { ALPHABET, TOPICS } from '../core/learn/categories';
import type { Translate, UiLanguage } from '../i18n/strings';
import type { EngineSnapshot, SignEngine } from '../state/engine';
import { loadProgress, recordScore, saveProgress, scoreAttempt } from '../state/progress';
import type { Settings } from '../state/settings';
import { Segmented, Toggle } from './controls';
import { IconExternal, IconNext, IconPrev, IconReplay, IconStar } from './icons';
import { SignPlayer } from './SignPlayer';
import { Stage } from './Stage';
import { clipFor, useLearningData, videoLink } from './useLearningData';

type Mode = 'learn' | 'quiz';
interface Feedback {
  stars: number;
  text: string;
}

const isLetter = (id: string) => id.length === 1;

function shuffle<T>(items: T[]): T[] {
  const a = [...items];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j]!, a[i]!];
  }
  return a;
}

export function Stars({ count, size = 16 }: { count: number; size?: number }) {
  return (
    <span className="stars" aria-label={`${count}/3`}>
      {[1, 2, 3].map((n) => (
        <span key={n} className={n <= count ? 'is-on' : ''}>
          <IconStar size={size} />
        </span>
      ))}
    </span>
  );
}

export function LearnView({
  engine,
  snap,
  settings,
  t,
}: {
  engine: SignEngine;
  snap: EngineSnapshot;
  settings: Settings;
  t: Translate;
}) {
  const data = useLearningData();
  const lang: UiLanguage = settings.uiLanguage;
  const [topicId, setTopicId] = useState('greetings');
  const [index, setIndex] = useState(0);
  const [mode, setMode] = useState<Mode>('learn');
  const [progress, setProgress] = useState(loadProgress);
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [speed, setSpeed] = useState(0.75);
  const [mirror, setMirror] = useState(true);
  const [replay, setReplay] = useState(0);
  const [quiz, setQuiz] = useState<{ order: string[]; pos: number; score: number; revealed: boolean }>({
    order: [],
    pos: 0,
    score: 0,
    revealed: false,
  });

  const items = useMemo(() => {
    if (topicId === 'alphabet') return ALPHABET.filter((l) => data.shapes?.[l]);
    const topic = TOPICS.find((x) => x.id === topicId);
    return (topic?.signs ?? []).filter((s) => data.demos?.has(s));
  }, [topicId, data]);

  const current = mode === 'quiz' ? quiz.order[quiz.pos] : items[Math.min(index, items.length - 1)];
  const currentRef = useRef(current);
  currentRef.current = current;

  const startQuiz = () => setQuiz({ order: shuffle(items).slice(0, 10), pos: 0, score: 0, revealed: false });

  useEffect(() => {
    setIndex(0);
    if (mode === 'quiz') startQuiz();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [topicId, mode, items.length]);

  useEffect(() => setFeedback(null), [current]);

  const label = (id: string) => (isLetter(id) ? id : translateAslLabel(id, lang));
  const gloss = (id: string) => (isLetter(id) ? '' : (WORDS[id]?.[0] ?? id).toUpperCase());

  // Score an attempt, keep the best result, and move on in a quiz.
  const award = (stars: number, text: string) => {
    const id = currentRef.current;
    if (!id) return;
    setFeedback({ stars, text });
    setProgress((p) => {
      const next = recordScore(p, id, stars);
      saveProgress(next);
      return next;
    });
    if (mode === 'quiz' && stars > 0) {
      setQuiz((q) => ({ ...q, score: q.score + (q.revealed ? 0 : 1) }));
      setTimeout(() => setQuiz((q) => (q.order[q.pos] === id ? { ...q, pos: q.pos + 1, revealed: false } : q)), 1400);
    }
  };
  const awardRef = useRef(award);
  awardRef.current = award;

  // Signs: judged by the built-in model when a movement ends.
  useEffect(
    () =>
      engine.onSegment(({ guesses }) => {
        const id = currentRef.current;
        if (!id || isLetter(id)) return;
        const stars = scoreAttempt(id, guesses);
        const seen = guesses[0] ? label(guesses[0].label) : '';
        const text = stars === 3 ? t('learnPerfect') : stars === 2 ? t('learnGood') : stars === 1 ? t('learnAlmost') : t('learnNotYet', { label: seen });
        awardRef.current(stars, text);
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [engine, t, lang],
  );

  // Letters: judged by the letter recognizer when a handshape is held.
  useEffect(
    () =>
      engine.onLetter((letter) => {
        const id = currentRef.current;
        if (!id || !isLetter(id)) return;
        if (letter === id) awardRef.current(3, t('learnPerfect'));
        else setFeedback({ stars: 0, text: t('learnLetterSeen', { letter }) });
      }),
    [engine, t],
  );

  const clip = current ? clipFor(current, data) : null;
  const clips = useMemo(() => (clip ? [clip] : []), [clip]);
  const learned = (ids: string[]) => ids.filter((id) => (progress[id] ?? 0) > 0).length;
  const topicItems = (id: string) =>
    id === 'alphabet' ? ALPHABET : (TOPICS.find((x) => x.id === id)?.signs ?? []).filter((s) => data.demos?.has(s));
  const quizDone = mode === 'quiz' && quiz.order.length > 0 && quiz.pos >= quiz.order.length;
  const showDemo = mode === 'learn' || quiz.revealed;

  if (data.status !== 'ready' || !data.demos) {
    return (
      <div className="view-message">
        {data.status === 'loading' ? (
          <>
            <div className="spinner" aria-hidden="true" />
            <p>{t('learnLoading')}</p>
          </>
        ) : (
          <p>{t('learnMissing')}</p>
        )}
      </div>
    );
  }

  return (
    <div className="learn">
      <nav className="learn-topics" aria-label={t('learnTopics')}>
        <h2 className="block-title">{t('learnTopics')}</h2>
        {[...TOPICS.map((x) => x.id), 'alphabet'].map((id) => {
          const all = topicItems(id);
          const done = learned(all);
          return (
            <button key={id} className={`topic ${id === topicId ? 'is-active' : ''}`} onClick={() => setTopicId(id)}>
              <span className="topic-name">{t(`topic_${id}` as Parameters<Translate>[0])}</span>
              <span className="topic-meter" aria-hidden="true">
                <span style={{ transform: `scaleX(${all.length ? done / all.length : 0})` }} />
              </span>
              <span className="topic-count">
                {done}/{all.length}
              </span>
            </button>
          );
        })}
      </nav>

      <section className="learn-main">
        <div className="learn-toolbar">
          <Segmented
            label={t('learnModeLearn')}
            value={mode}
            options={[
              { value: 'learn' as Mode, label: t('learnModeLearn') },
              { value: 'quiz' as Mode, label: t('learnModeQuiz') },
            ]}
            onChange={setMode}
          />
          <span className="muted">
            {mode === 'quiz'
              ? t('quizScore', { n: quiz.score, total: quiz.order.length })
              : t('learnProgress', { n: learned(items), total: items.length })}
          </span>
        </div>

        <div className="learn-stage">
          <div className="learn-card">
            <div className="learn-card-head">
              <span className="block-title">{mode === 'quiz' ? t('quizPrompt') : t('learnWatch')}</span>
              {current && <Stars count={progress[current] ?? 0} />}
            </div>
            {quizDone ? (
              <div className="quiz-done">
                <p className="quiz-final">{t('quizScore', { n: quiz.score, total: quiz.order.length })}</p>
                <button className="btn btn-primary" onClick={startQuiz}>
                  {t('learnReplay')}
                </button>
              </div>
            ) : current ? (
              <>
                <div className="learn-word">
                  <span className="learn-word-main">{label(current)}</span>
                  {gloss(current) && <span className="learn-word-gloss">{gloss(current)}</span>}
                </div>
                {showDemo ? (
                  <SignPlayer
                    clips={clips}
                    neutral={data.demos.neutral}
                    layout={data.demos.layout}
                    rig={data.demos.rig}
                    resetLabel={t('playerFront')}
                    failedLabel={t('player3dMissing')}
                    speed={speed}
                    mirror={mirror}
                    loop
                    replayKey={replay}
                    label={label(current)}
                  />
                ) : (
                  <div className="player player-hidden">
                    <button className="btn" onClick={() => setQuiz((q) => ({ ...q, revealed: true }))}>
                      {t('quizShow')}
                    </button>
                  </div>
                )}
                <div className="player-controls">
                  <button className="icon-btn" onClick={() => setReplay((r) => r + 1)} aria-label={t('learnReplay')} title={t('learnReplay')}>
                    <IconReplay />
                  </button>
                  <Segmented
                    label={t('learnSpeed')}
                    value={speed}
                    options={[0.5, 0.75, 1].map((v) => ({ value: v, label: `${v}×` }))}
                    onChange={setSpeed}
                  />
                  <Toggle label={t('learnMirror')} checked={mirror} onChange={setMirror} />
                  {!isLetter(current) && (
                    <a className="link" href={videoLink(WORDS[current]?.[0] ?? current)} target="_blank" rel="noreferrer">
                      {t('learnVideo')} <IconExternal size={14} />
                    </a>
                  )}
                </div>
              </>
            ) : (
              <div className="player">
                <p className="muted">{t('learnMissing')}</p>
              </div>
            )}
          </div>

          <div className="learn-card">
            <div className="learn-card-head">
              <span className="block-title">{t('learnYourTurn')}</span>
              {current && isLetter(current) && snap.letter && (
                <span className="muted">{t('learnLetterSeen', { letter: snap.letter.letter })}</span>
              )}
            </div>
            <Stage engine={engine} snap={snap} settings={settings} teach={null} t={t} />
            <div className={`feedback ${feedback ? (feedback.stars > 0 ? 'is-good' : 'is-bad') : ''}`} role="status">
              {feedback ? (
                <>
                  <Stars count={feedback.stars} size={20} />
                  <span>{feedback.text}</span>
                </>
              ) : (
                <span className="muted">{current && isLetter(current) ? t('learnTryLetter') : t('learnTrySign')}</span>
              )}
            </div>
          </div>
        </div>

        {mode === 'learn' ? (
          <div className="learn-nav">
            <button className="btn" onClick={() => setIndex((i) => Math.max(0, i - 1))} disabled={index === 0}>
              <IconPrev /> {t('learnPrev')}
            </button>
            <div className="word-strip">
              {items.map((id, i) => (
                <button key={id} className={`word-chip ${i === index ? 'is-active' : ''}`} onClick={() => setIndex(i)}>
                  {label(id)}
                  {(progress[id] ?? 0) > 0 && <Stars count={progress[id]!} size={10} />}
                </button>
              ))}
            </div>
            <button className="btn btn-primary" onClick={() => setIndex((i) => Math.min(items.length - 1, i + 1))} disabled={index >= items.length - 1}>
              {t('learnNext')} <IconNext />
            </button>
          </div>
        ) : (
          !quizDone && (
            <div className="learn-nav">
              <button className="btn" onClick={() => setQuiz((q) => ({ ...q, pos: q.pos + 1, revealed: false }))}>
                {t('quizSkip')}
              </button>
            </div>
          )
        )}
      </section>
    </div>
  );
}
