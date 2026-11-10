/**
 * Phrase to signs: type a sentence, watch it signed. Known words use their
 * recorded sign; names and other words are fingerspelled letter by letter;
 * words ASL leaves out are shown greyed.
 */

import { useMemo, useState } from 'react';
import { translateAslLabel, WORDS } from '../core/asl/labels';
import { phraseToItems, type PhraseItem } from '../core/learn/phrase';
import type { Translate } from '../i18n/strings';
import type { Settings } from '../state/settings';
import { Segmented, Toggle } from './controls';
import { IconExternal, IconPause, IconPlay, IconReplay } from './icons';
import { SignPlayer, type Clip } from './SignPlayer';
import { clipFor, useLearningData, videoLink } from './useLearningData';

const EXAMPLES: Record<Settings['uiLanguage'], string[]> = {
  en: ['Hello, my name is Sara', 'Thank you, see you tomorrow', 'I like pizza and ice cream', 'Where is my dog?'],
  fr: ["Bonjour, je m'appelle Sara", 'Merci, à demain', 'Maman aime la pizza', 'Où est mon chien ?'],
  ar: ['مرحبا، اسمي سارة', 'شكرا', 'أمي وأبي', 'أين كلبي'],
};

export function PhraseView({ settings, t }: { settings: Settings; t: Translate }) {
  const data = useLearningData();
  const [text, setText] = useState('');
  const [items, setItems] = useState<PhraseItem[]>([]);
  const [only, setOnly] = useState<number | null>(null);
  const [activeClip, setActiveClip] = useState(-1);
  const [playing, setPlaying] = useState(true);
  const [loop, setLoop] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [mirror, setMirror] = useState(true);
  const [replay, setReplay] = useState(0);

  // Clips to play, and which phrase item (and letter) each clip belongs to.
  const { clips, owners } = useMemo(() => {
    const clips: Clip[] = [];
    const owners: { item: number; letter: number }[] = [];
    items.forEach((item, i) => {
      if (only !== null && only !== i) return;
      if (item.kind === 'sign') {
        const c = clipFor(item.label, data);
        if (c) {
          clips.push(c);
          owners.push({ item: i, letter: -1 });
        }
      } else if (item.kind === 'spell') {
        item.letters.forEach((l, li) => {
          const c = clipFor(l, data);
          if (c) {
            clips.push(c);
            owners.push({ item: i, letter: li });
          }
        });
      }
    });
    return { clips, owners };
  }, [items, only, data]);

  const show = (value: string) => {
    setText(value);
    setItems(phraseToItems(value));
    setOnly(null);
    setActiveClip(-1);
    setPlaying(true);
    setReplay((r) => r + 1);
  };

  const owner = activeClip >= 0 ? owners[activeClip] : undefined;
  const nowShowing = (() => {
    if (!owner) return '';
    const item = items[owner.item]!;
    if (item.kind === 'sign') return translateAslLabel(item.label, settings.uiLanguage);
    if (item.kind === 'spell') return item.letters[owner.letter] ?? '';
    return '';
  })();

  return (
    <div className="phrase">
      <form
        className="phrase-form"
        onSubmit={(e) => {
          e.preventDefault();
          show(text);
        }}
      >
        <label className="phrase-label" htmlFor="phrase-input">
          {t('phraseTitle')}
        </label>
        <div className="phrase-input-row">
          <input
            id="phrase-input"
            className="input phrase-input"
            value={text}
            maxLength={200}
            placeholder={t('phrasePlaceholder')}
            onChange={(e) => setText(e.target.value)}
            dir="auto"
          />
          <button className="btn btn-primary" type="submit" disabled={!text.trim()}>
            {t('phraseShow')}
          </button>
        </div>
        <div className="phrase-examples">
          <span className="muted">{t('phraseExamples')}</span>
          {EXAMPLES[settings.uiLanguage].map((ex) => (
            <button key={ex} type="button" className="word-chip" onClick={() => show(ex)} dir="auto">
              {ex}
            </button>
          ))}
        </div>
      </form>

      {items.length === 0 ? (
        <p className="empty">{t('phraseEmpty')}</p>
      ) : (
        <div className="phrase-body">
          <div className="learn-card phrase-player">
            {data.demos ? (
              <SignPlayer
                clips={clips}
                neutral={data.demos.neutral}
                layout={data.demos.layout}
                speed={speed}
                mirror={mirror}
                loop={loop}
                playing={playing}
                replayKey={replay}
                onClipChange={setActiveClip}
                onEnd={() => setPlaying(false)}
                label={text}
              />
            ) : (
              <div className="player">
                <div className="spinner" aria-hidden="true" />
              </div>
            )}
            <p className="now-showing" aria-live="polite" dir="auto">
              {nowShowing || ' '}
            </p>
            <div className="player-controls">
              <button
                className="icon-btn"
                onClick={() => {
                  if (!playing) setReplay((r) => r + 1);
                  setPlaying(!playing);
                }}
                aria-label={playing ? t('phrasePause') : t('phrasePlay')}
                title={playing ? t('phrasePause') : t('phrasePlay')}
              >
                {playing ? <IconPause /> : <IconPlay />}
              </button>
              <button
                className="icon-btn"
                onClick={() => {
                  setPlaying(true);
                  setReplay((r) => r + 1);
                }}
                aria-label={t('learnReplay')}
                title={t('learnReplay')}
              >
                <IconReplay />
              </button>
              <Segmented
                label={t('learnSpeed')}
                value={speed}
                options={[0.5, 0.75, 1, 1.25].map((v) => ({ value: v, label: `${v}×` }))}
                onChange={setSpeed}
              />
              <Toggle label={t('phraseLoop')} checked={loop} onChange={setLoop} />
              <Toggle label={t('learnMirror')} checked={mirror} onChange={setMirror} />
            </div>
          </div>

          <ol className="phrase-items">
            {items.map((item, i) => {
              const active = owner?.item === i;
              const playable = item.kind !== 'skip';
              return (
                <li key={i} className={`phrase-item kind-${item.kind} ${active ? 'is-active' : ''}`}>
                  <button
                    className="phrase-item-main"
                    disabled={!playable}
                    onClick={() => {
                      setOnly(only === i ? null : i);
                      setPlaying(true);
                      setReplay((r) => r + 1);
                    }}
                  >
                    <span className="phrase-item-text" dir="auto">
                      {item.text}
                    </span>
                    <span className="phrase-item-sub">
                      {item.kind === 'sign' &&
                        `${translateAslLabel(item.label, settings.uiLanguage)} · ${(WORDS[item.label]?.[0] ?? item.label).toUpperCase()}`}
                      {item.kind === 'spell' && (
                        <>
                          {t('phraseSpelled')}:{' '}
                          {item.letters.map((l, li) => (
                            <span key={li} className={active && owner?.letter === li ? 'letter is-active' : 'letter'}>
                              {l}
                            </span>
                          ))}
                        </>
                      )}
                      {item.kind === 'skip' && (item.reason === 'omitted' ? t('phraseOmitted') : t('phraseUnavailable'))}
                    </span>
                  </button>
                  {item.kind === 'sign' && (
                    <a
                      className="icon-btn small"
                      href={videoLink(WORDS[item.label]?.[0] ?? item.label)}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={t('learnVideo')}
                      title={t('learnVideo')}
                    >
                      <IconExternal size={15} />
                    </a>
                  )}
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </div>
  );
}
