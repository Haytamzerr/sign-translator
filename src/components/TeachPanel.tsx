/**
 * Teach mode: record a few takes of a sign, check them, save them.
 *
 * Flow: form -> recording (countdown, then each take) -> review -> saved.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { newId } from '../core/storage';
import { detectKind, findConflicts, MIN_TAKE_FRAMES, prepareTake, trimEmpty } from '../core/teach';
import type { Frame, Sign, SignKind } from '../core/types';
import type { Translate } from '../i18n/strings';
import type { EngineSnapshot, SignEngine } from '../state/engine';
import type { Settings } from '../state/settings';
import { Field, Segmented } from './controls';
import { IconAlert, IconCheck, IconRecord } from './icons';
import { SignThumb } from './SignThumb';
import type { TeachOverlay } from './Stage';

type Phase = 'form' | 'recording' | 'review';

const wait = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function TeachPanel({
  engine,
  snap,
  signs,
  language,
  settings,
  target,
  t,
  setOverlay,
  onSaved,
  onCancel,
}: {
  engine: SignEngine;
  snap: EngineSnapshot;
  signs: Sign[];
  language: string;
  settings: Settings;
  /** When set, new takes are added to this existing sign. */
  target: Sign | null;
  t: Translate;
  setOverlay: (o: TeachOverlay | null) => void;
  onSaved: (sign: Sign) => void;
  onCancel: () => void;
}) {
  const [label, setLabel] = useState(target?.label ?? '');
  const [category, setCategory] = useState(target?.category ?? '');
  const [takeCount, setTakeCount] = useState(5);
  const [durationMs, setDurationMs] = useState(2000);
  const [phase, setPhase] = useState<Phase>('form');
  const [takes, setTakes] = useState<(Frame[] | null)[]>([]);
  const [kindOverride, setKindOverride] = useState<SignKind | null>(null);
  const [error, setError] = useState('');
  const cancelled = useRef(false);

  useEffect(() => {
    setLabel(target?.label ?? '');
    setCategory(target?.category ?? '');
    setPhase('form');
    setTakes([]);
  }, [target]);

  // Stop everything if the panel closes mid-recording.
  useEffect(
    () => () => {
      cancelled.current = true;
      setOverlay(null);
      engine.setPaused(false);
    },
    [engine, setOverlay],
  );

  /** Countdown, then collect frames for `durationMs`. Returns null if no hands were seen. */
  async function recordTake(take: number, total: number, countdown: number): Promise<Frame[] | null> {
    for (let c = countdown; c > 0; c--) {
      if (cancelled.current) return null;
      setOverlay({ phase: 'countdown', count: c, progress: 0, take, total });
      await wait(700);
    }
    const frames: Frame[] = [];
    const stop = engine.onFrame((f) => frames.push(f));
    const start = performance.now();
    while (performance.now() - start < durationMs && !cancelled.current) {
      setOverlay({ phase: 'recording', count: 0, progress: (performance.now() - start) / durationMs, take, total });
      await wait(50);
    }
    stop();
    setOverlay(null);
    const trimmed = trimEmpty(frames);
    return trimmed.filter((f) => f.hands[0] || f.hands[1]).length >= MIN_TAKE_FRAMES ? trimmed : null;
  }

  async function startRecording() {
    if (snap.status !== 'running') return setError(t('teachNeedCamera'));
    if (!label.trim()) return setError(t('teachNeedLabel'));
    setError('');
    cancelled.current = false;
    engine.setPaused(true);
    setPhase('recording');
    setKindOverride(null);
    const result: (Frame[] | null)[] = [];
    setTakes([]);
    for (let i = 0; i < takeCount && !cancelled.current; i++) {
      const frames = await recordTake(i + 1, takeCount, i === 0 ? 3 : 2);
      result.push(frames);
      setTakes([...result]);
      await wait(250);
    }
    engine.setPaused(false);
    if (!cancelled.current) setPhase('review');
  }

  async function redo(index: number) {
    cancelled.current = false;
    engine.setPaused(true);
    setPhase('recording');
    const frames = await recordTake(index + 1, takes.length, 2);
    engine.setPaused(false);
    setTakes((prev) => prev.map((f, i) => (i === index ? frames : f)));
    if (!cancelled.current) setPhase('review');
  }

  function stopRecording() {
    cancelled.current = true;
    setOverlay(null);
    engine.setPaused(false);
    setPhase(takes.length ? 'review' : 'form');
  }

  const valid = takes.filter((f): f is Frame[] => f !== null);
  const detected = target?.kind ?? detectKind(valid);
  const kind = target?.kind ?? kindOverride ?? detected;
  const samples = useMemo(
    () => takes.filter((f): f is Frame[] => f !== null).map((f) => prepareTake(f, kind, settings.moveThreshold)),
    [takes, kind, settings.moveThreshold],
  );
  const conflicts = useMemo(
    () =>
      phase === 'review' && samples.length
        ? findConflicts(
            { kind, samples },
            signs.filter((s) => s.id !== target?.id),
            { sensitivity: settings.sensitivity, eitherHand: settings.eitherHand },
          )
        : [],
    [phase, samples, kind, signs, target, settings.sensitivity, settings.eitherHand],
  );

  function save() {
    const now = Date.now();
    const sign: Sign = target
      ? { ...target, samples: [...target.samples, ...samples], updatedAt: now }
      : {
          id: newId(),
          label: label.trim(),
          language,
          kind,
          category: category.trim(),
          samples,
          createdAt: now,
          updatedAt: now,
        };
    setPhase('form');
    setTakes([]);
    if (!target) {
      setLabel('');
    }
    onSaved(sign);
  }

  return (
    <div className="panel-body">
      <h3 className="panel-heading">{target ? t('teachAddTo', { label: target.label }) : t('teachTitle')}</h3>
      <p className="muted">{t('teachIntro')}</p>

      {phase === 'form' && (
        <form
          className="form"
          onSubmit={(e) => {
            e.preventDefault();
            void startRecording();
          }}
        >
          {!target && (
            <>
              <Field label={t('teachLabel')}>
                <input
                  className="input"
                  value={label}
                  maxLength={60}
                  placeholder={t('teachLabelPlaceholder')}
                  onChange={(e) => {
                    setLabel(e.target.value);
                    setError('');
                  }}
                  autoFocus
                />
              </Field>
              <Field label={t('teachCategory')}>
                <input
                  className="input"
                  value={category}
                  maxLength={40}
                  placeholder={t('teachCategoryPlaceholder')}
                  onChange={(e) => setCategory(e.target.value)}
                />
              </Field>
            </>
          )}
          <Field label={t('teachTakes')}>
            <Segmented
              label={t('teachTakes')}
              value={takeCount}
              options={[3, 5, 8].map((n) => ({ value: n, label: String(n) }))}
              onChange={setTakeCount}
            />
          </Field>
          <Field label={t('teachDuration')}>
            <Segmented
              label={t('teachDuration')}
              value={durationMs}
              options={[1500, 2000, 3000].map((ms) => ({ value: ms, label: t('seconds', { n: ms / 1000 }) }))}
              onChange={setDurationMs}
            />
          </Field>
          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}
          <div className="form-actions">
            <button type="submit" className="btn btn-primary">
              <IconRecord /> {t('teachStart')}
            </button>
            {target && (
              <button type="button" className="btn" onClick={onCancel}>
                {t('teachCancel')}
              </button>
            )}
          </div>
          <p className="tip">{t('teachTip')}</p>
        </form>
      )}

      {phase !== 'form' && (
        <ul className="takes">
          {takes.map((frames, i) => (
            <li key={i} className={frames ? 'is-ok' : 'is-bad'}>
              <span className="take-index">{i + 1}</span>
              {frames ? <SignThumb sample={frames} moving={kind === 'dynamic'} size={44} /> : <IconAlert />}
              <span className="take-status">{frames ? <IconCheck size={16} /> : t('teachNoHands')}</span>
              {phase === 'review' && (
                <button className="btn btn-small" onClick={() => void redo(i)}>
                  {t('teachRedo')}
                </button>
              )}
            </li>
          ))}
        </ul>
      )}

      {phase === 'recording' && (
        <button className="btn" onClick={stopRecording}>
          {t('teachCancel')}
        </button>
      )}

      {phase === 'review' && (
        <div className="form">
          {!target && valid.length > 0 && (
            <Field label={t('teachType')} help={t('teachTypeHelp')}>
              <Segmented
                label={t('teachType')}
                value={kind}
                options={[
                  { value: 'static' as SignKind, label: t('stillSign') },
                  { value: 'dynamic' as SignKind, label: t('movingSign') },
                ]}
                onChange={setKindOverride}
              />
            </Field>
          )}
          {conflicts[0] && (
            <p className="warning" role="status">
              <IconAlert /> {t('teachConflict', { label: conflicts[0].label })}
            </p>
          )}
          <div className="form-actions">
            <button className="btn btn-primary" onClick={save} disabled={valid.length === 0}>
              <IconCheck /> {t('teachSave')}
            </button>
            <button
              className="btn"
              onClick={() => {
                setPhase('form');
                setTakes([]);
                if (target) onCancel();
              }}
            >
              {t('teachDiscard')}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
