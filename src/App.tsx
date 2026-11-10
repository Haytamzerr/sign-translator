/**
 * Three screens, chosen in the top bar:
 * - Translate: camera and captions on the left, a panel with tabs on the right
 *   (Live, Teach, Library, Settings)
 * - Learn: lessons with demonstrations and live practice
 * - Phrase to signs: type text, watch it signed
 * App owns the shared state and passes it down.
 */

import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { CaptionBar, type CaptionWord } from './components/CaptionBar';
import { IconCamera, IconCameraOff } from './components/icons';
import { LibraryPanel } from './components/LibraryPanel';
import { LivePanel, type TranscriptLine } from './components/LivePanel';
import { SettingsPanel } from './components/SettingsPanel';
import { Stage, type TeachOverlay } from './components/Stage';
import { TeachPanel } from './components/TeachPanel';
import { translateAslLabel } from './core/asl/labels';
import { speak } from './core/speech';
import { deleteSign, importLibrary, loadSigns, saveSign } from './core/storage';
import { prepareTake } from './core/teach';
import type { Sign } from './core/types';
import { SIGN_LANGUAGES, speechLang, translator, type StringKey } from './i18n/strings';
import { SignEngine } from './state/engine';

// The Learn and Phrase views include the 3D figure (three.js); they load on first use.
const LearnView = lazy(() => import('./components/LearnView').then((m) => ({ default: m.LearnView })));
const PhraseView = lazy(() => import('./components/PhraseView').then((m) => ({ default: m.PhraseView })));
const viewLoading = (
  <div className="view-message">
    <div className="spinner" aria-hidden="true" />
  </div>
);
import { loadSettings, saveSettings, type Settings } from './state/settings';

type View = 'translate' | 'learn' | 'phrase';
const VIEWS: { id: View; key: StringKey }[] = [
  { id: 'translate', key: 'navTranslate' },
  { id: 'learn', key: 'navLearn' },
  { id: 'phrase', key: 'navPhrase' },
];
/** A spelled word is finished when no new letter comes for this long. */
const SPELLING_PAUSE_MS = 1600;

type Tab = 'live' | 'teach' | 'library' | 'settings';
const TABS: { id: Tab; key: StringKey }[] = [
  { id: 'live', key: 'tabLive' },
  { id: 'teach', key: 'tabTeach' },
  { id: 'library', key: 'tabLibrary' },
  { id: 'settings', key: 'tabSettings' },
];

export default function App() {
  const [settings, setSettings] = useState<Settings>(loadSettings);
  const [engine] = useState(() => new SignEngine(settings));
  const snap = useSyncExternalStore(engine.subscribe, engine.getSnapshot);

  const [signs, setSigns] = useState<Sign[]>([]);
  const [view, setView] = useState<View>('translate');
  const [tab, setTab] = useState<Tab>('live');
  const [spelling, setSpelling] = useState('');
  const [teachTarget, setTeachTarget] = useState<Sign | null>(null);
  const [overlay, setOverlay] = useState<TeachOverlay | null>(null);
  const [words, setWords] = useState<CaptionWord[]>([]);
  const [previous, setPrevious] = useState('');
  const [transcript, setTranscript] = useState<TranscriptLine[]>([]);
  const [toast, setToast] = useState('');
  const nextId = useRef(1);

  const t = useMemo(() => translator(settings.uiLanguage), [settings.uiLanguage]);
  const lang = speechLang[settings.uiLanguage];

  // Refs hold the latest values for callbacks that must not re-run (no side
  // effects inside React state updaters, which may run twice in development).
  const settingsRef = useRef(settings);
  const wordsRef = useRef(words);
  wordsRef.current = words;
  const spellingRef = useRef(spelling);
  spellingRef.current = spelling;
  const viewRef = useRef(view);
  viewRef.current = view;

  const update = useCallback(
    (patch: Partial<Settings>) => {
      const next = { ...settingsRef.current, ...patch };
      if (!SIGN_LANGUAGES.includes(next.signLanguage)) next.signLanguage = 'custom';
      settingsRef.current = next;
      saveSettings(next);
      engine.setSettings(next);
      setSettings(next);
    },
    [engine],
  );

  const notify = useCallback((message: string) => setToast(message), []);
  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(''), 3200);
    return () => clearTimeout(id);
  }, [toast]);

  // Page language, reading direction (Arabic is right-to-left) and theme.
  useEffect(() => {
    const root = document.documentElement;
    root.lang = settings.uiLanguage;
    root.dir = settings.uiLanguage === 'ar' ? 'rtl' : 'ltr';
    root.dataset.theme = settings.theme;
  }, [settings.uiLanguage, settings.theme]);

  // Load the library of the chosen sign language.
  useEffect(() => {
    let active = true;
    loadSigns(settings.signLanguage)
      .then((list) => active && setSigns(list))
      .catch((err) => console.error('Could not load the sign library', err));
    return () => {
      active = false;
    };
  }, [settings.signLanguage]);

  useEffect(() => engine.setLibrary(signs), [engine, signs]);
  useEffect(() => {
    engine.translateLabel = (label) => translateAslLabel(label, settings.uiLanguage);
  }, [engine, settings.uiLanguage]);
  useEffect(() => () => engine.stop(), [engine]);

  // Learn uses the camera for practice: nothing is written as captions there.
  useEffect(() => engine.setMode(view === 'learn' ? 'practice' : 'translate'), [engine, view]);

  /** Turns the letters spelled so far into one caption word ("haytam" -> "Haytam"). */
  const flushSpelling = useCallback(() => {
    const letters = spellingRef.current;
    if (!letters) return;
    spellingRef.current = '';
    setSpelling('');
    const label = letters.charAt(0) + letters.slice(1).toLowerCase();
    setWords((prev) => [...prev, { id: nextId.current++, signId: 'spell', label, confidence: 1, kind: 'static', frames: [], t: performance.now() }]);
  }, []);

  // New word recognized -> add it to the caption (after any word being spelled).
  useEffect(
    () =>
      engine.onCommit((commit) => {
        flushSpelling();
        setWords((prev) => [...prev, { ...commit, id: nextId.current++ }]);
      }),
    [engine, flushSpelling],
  );

  // Fingerspelled letters build up a word in the caption.
  useEffect(
    () =>
      engine.onLetter((letter) => {
        if (viewRef.current !== 'translate') return;
        setSpelling((s) => s + letter);
      }),
    [engine],
  );
  useEffect(() => {
    if (!spelling) return;
    const id = setTimeout(flushSpelling, SPELLING_PAUSE_MS);
    return () => clearTimeout(id);
  }, [spelling, flushSpelling]);

  const finishSentence = useCallback(() => {
    const current = wordsRef.current;
    if (current.length === 0) return;
    const text = current.map((w) => w.label).join(' ');
    const line = { id: nextId.current++, text, time: Date.now() };
    setWords([]);
    setPrevious(text);
    setTranscript((list) => [...list.slice(-49), line]);
    if (settings.autoSpeak) speak(text, lang, settings.voiceURI, settings.speechRate);
  }, [settings.autoSpeak, settings.voiceURI, settings.speechRate, lang]);

  // A sentence ends when the hands have been down for the chosen pause.
  useEffect(() => {
    if (snap.handsVisible || words.length === 0 || overlay || spelling) return;
    const id = setTimeout(finishSentence, settings.sentencePauseMs);
    return () => clearTimeout(id);
  }, [snap.handsVisible, words.length, overlay, spelling, settings.sentencePauseMs, finishSentence]);

  // ---------- Library changes ----------

  const storeSign = async (sign: Sign) => {
    await saveSign(sign);
    setSigns((list) => [...list.filter((s) => s.id !== sign.id), sign].sort((a, b) => a.label.localeCompare(b.label)));
  };

  const correctWord = async (wordId: number, sign: Sign) => {
    const word = words.find((w) => w.id === wordId);
    if (!word) return;
    const sample = prepareTake(word.frames, sign.kind, settings.moveThreshold);
    if (sample.length > 0) await storeSign({ ...sign, samples: [...sign.samples, sample], updatedAt: Date.now() });
    setWords((list) => list.map((w) => (w.id === wordId ? { ...w, label: sign.label, signId: sign.id, confidence: 1 } : w)));
    notify(t('correctedTo', { label: sign.label }));
  };

  const cameraOn = snap.status === 'running' || snap.status === 'loading';

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 64 64" aria-hidden="true">
            <g fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round">
              <path d="M18 50V26M30 50V14M42 50V18M54 50V30" />
            </g>
          </svg>
          <div>
            <h1>{t('appName')}</h1>
            <p>{t('appTagline')}</p>
          </div>
        </div>
        <nav className="views" aria-label="Main">
          {VIEWS.map(({ id, key }) => (
            <button key={id} className={view === id ? 'is-active' : ''} aria-current={view === id ? 'page' : undefined} onClick={() => setView(id)}>
              {t(key)}
            </button>
          ))}
        </nav>
        <div className="topbar-actions">
          <select
            className="select select-compact"
            aria-label={t('settingSignLanguage')}
            value={settings.signLanguage}
            onChange={(e) => update({ signLanguage: e.target.value as Settings['signLanguage'] })}
          >
            {SIGN_LANGUAGES.map((value) => (
              <option key={value} value={value}>
                {t(`sl_${value}`)}
              </option>
            ))}
          </select>
          <button
            className={`btn ${cameraOn ? '' : 'btn-primary'}`}
            onClick={() => (cameraOn ? engine.stop() : void engine.start())}
          >
            {cameraOn ? <IconCameraOff /> : <IconCamera />}
            <span className="btn-text">{cameraOn ? t('cameraStop') : t('cameraStart')}</span>
          </button>
        </div>
      </header>

      {view === 'learn' && (
        <main className="workspace workspace-full">
          <Suspense fallback={viewLoading}>
            <LearnView engine={engine} snap={snap} settings={settings} t={t} />
          </Suspense>
        </main>
      )}
      {view === 'phrase' && (
        <main className="workspace workspace-full">
          <Suspense fallback={viewLoading}>
            <PhraseView settings={settings} t={t} />
          </Suspense>
        </main>
      )}

      {view === 'translate' && (
      <main className="workspace">
        <div className="main-column">
          <Stage engine={engine} snap={snap} settings={settings} teach={overlay} t={t} />
          <CaptionBar
            words={words}
            spelling={spelling}
            previous={previous}
            size={settings.captionSize}
            signs={signs}
            t={t}
            onSpeak={() => speak(words.map((w) => w.label).join(' ') || previous, lang, settings.voiceURI, settings.speechRate)}
            onUndo={() => setWords((list) => list.slice(0, -1))}
            onClear={() => {
              setWords([]);
              setSpelling('');
              setPrevious('');
            }}
            onEndSentence={finishSentence}
            onCorrect={(id, sign) => void correctWord(id, sign)}
            onRemove={(id) => setWords((list) => list.filter((w) => w.id !== id))}
          />
        </div>

        <aside className="side">
          <nav className="tabs" role="tablist">
            {TABS.map(({ id, key }) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                className={tab === id ? 'is-active' : ''}
                onClick={() => {
                  setTab(id);
                  if (id !== 'teach') setTeachTarget(null);
                }}
              >
                {t(key)}
                {id === 'library' && signs.length > 0 && <span className="count">{signs.length}</span>}
              </button>
            ))}
          </nav>

          {tab === 'live' && (
            <LivePanel
              snap={snap}
              librarySize={signs.length}
              transcript={transcript}
              moveThreshold={settings.moveThreshold}
              t={t}
              locale={settings.uiLanguage}
              pretrainedCount={engine.pretrainedLabels.length}
              onTeach={() => setTab('teach')}
            />
          )}
          {tab === 'teach' && (
            <TeachPanel
              engine={engine}
              snap={snap}
              signs={signs}
              language={settings.signLanguage}
              settings={settings}
              target={teachTarget}
              t={t}
              setOverlay={setOverlay}
              onSaved={(sign) => {
                void storeSign(sign);
                notify(t('teachSaved', { label: sign.label }));
                setTeachTarget(null);
                setTab('live');
              }}
              onCancel={() => {
                setTeachTarget(null);
                setTab('library');
              }}
            />
          )}
          {tab === 'library' && (
            <LibraryPanel
              signs={signs}
              language={settings.signLanguage}
              t={t}
              onAddExamples={(sign) => {
                setTeachTarget(sign);
                setTab('teach');
              }}
              onRename={(sign, label) => void storeSign({ ...sign, label, updatedAt: Date.now() })}
              onDelete={(sign) => {
                void deleteSign(sign.id);
                setSigns((list) => list.filter((s) => s.id !== sign.id));
              }}
              onImport={(text) => {
                importLibrary(text, settings.signLanguage, signs)
                  .then(async (n) => {
                    setSigns(await loadSigns(settings.signLanguage));
                    notify(t('imported', { n }));
                  })
                  .catch(() => notify(t('importFailed')));
              }}
            />
          )}
          {tab === 'settings' && <SettingsPanel settings={settings} update={update} cameraOn={cameraOn} t={t} />}
        </aside>
      </main>
      )}

      {toast && (
        <div className="toast" role="status">
          {toast}
        </div>
      )}
    </div>
  );
}
