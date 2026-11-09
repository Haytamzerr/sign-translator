/**
 * Live captions under the camera. Each word can be clicked to correct it:
 * picking the right sign saves that performance as a new example, so the app
 * makes the same mistake less often.
 */

import { useEffect, useRef, useState } from 'react';
import type { Commit } from '../core/decoder';
import type { Sign } from '../core/types';
import type { Translate } from '../i18n/strings';
import { IconClear, IconCopy, IconReturn, IconSpeaker, IconUndo } from './icons';

export interface CaptionWord extends Commit {
  id: number;
}

export function CaptionBar({
  words,
  previous,
  size,
  signs,
  t,
  onSpeak,
  onUndo,
  onClear,
  onEndSentence,
  onCorrect,
  onRemove,
}: {
  words: CaptionWord[];
  previous: string;
  size: 'm' | 'l' | 'xl';
  signs: Sign[];
  t: Translate;
  onSpeak: () => void;
  onUndo: () => void;
  onClear: () => void;
  onEndSentence: () => void;
  onCorrect: (wordId: number, sign: Sign) => void;
  onRemove: (wordId: number) => void;
}) {
  const [open, setOpen] = useState<number | null>(null);
  const [copied, setCopied] = useState(false);
  const popRef = useRef<HTMLDivElement>(null);
  const text = words.map((w) => w.label).join(' ');

  // Close the correction menu when clicking elsewhere or pressing Escape.
  useEffect(() => {
    if (open === null) return;
    const onDown = (e: PointerEvent) => {
      if (!popRef.current?.contains(e.target as Node)) setOpen(null);
    };
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(null);
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const copy = async () => {
    const value = text || previous;
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // Clipboard can be refused by the browser; nothing else to do.
    }
  };

  const openWord = words.find((w) => w.id === open);
  const choices = openWord ? signs.filter((s) => s.kind === openWord.kind && s.id !== openWord.signId) : [];

  return (
    <section className={`captions size-${size}`} aria-label="Captions">
      <p className="caption-previous">{previous || ' '}</p>
      <div className="caption-line" aria-live="polite">
        {words.length === 0 && <span className="caption-placeholder">{t('captionPlaceholder')}</span>}
        {words.map((w) => (
          <span key={w.id} className="word-wrap">
            <button
              className={`word ${w.confidence < 0.75 ? 'is-unsure' : ''}`}
              onClick={() => setOpen(open === w.id ? null : w.id)}
              aria-haspopup="menu"
              aria-expanded={open === w.id}
              title={`${Math.round(w.confidence * 100)}%`}
            >
              {w.label}
            </button>
            {open === w.id && (
              <div className="word-menu" role="menu" ref={popRef}>
                <p className="word-menu-title">{t('correctTitle')}</p>
                <p className="word-menu-hint">{t('correctHint')}</p>
                <div className="word-menu-list">
                  {choices.map((s) => (
                    <button
                      key={s.id}
                      role="menuitem"
                      onClick={() => {
                        onCorrect(w.id, s);
                        setOpen(null);
                      }}
                    >
                      {s.label}
                    </button>
                  ))}
                </div>
                <button
                  className="word-menu-remove"
                  role="menuitem"
                  onClick={() => {
                    onRemove(w.id);
                    setOpen(null);
                  }}
                >
                  {t('removeWord')}
                </button>
              </div>
            )}
          </span>
        ))}
      </div>
      <div className="caption-actions">
        <button className="icon-btn" onClick={onSpeak} aria-label={t('speak')} title={t('speak')}>
          <IconSpeaker />
        </button>
        <button className="icon-btn" onClick={() => void copy()} aria-label={t('copy')} title={copied ? t('copied') : t('copy')}>
          <IconCopy />
        </button>
        <button className="icon-btn" onClick={onUndo} disabled={!words.length} aria-label={t('undo')} title={t('undo')}>
          <IconUndo />
        </button>
        <button className="icon-btn" onClick={onEndSentence} disabled={!words.length} aria-label={t('endSentence')} title={t('endSentence')}>
          <IconReturn />
        </button>
        <button className="icon-btn" onClick={onClear} disabled={!words.length && !previous} aria-label={t('clear')} title={t('clear')}>
          <IconClear />
        </button>
        {copied && <span className="caption-flash">{t('copied')}</span>}
      </div>
    </section>
  );
}
