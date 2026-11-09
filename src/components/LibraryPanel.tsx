/** The sign library: search, rename, delete, add examples, export and import. */

import { useMemo, useRef, useState } from 'react';
import { exportLibrary } from '../core/storage';
import type { Sign } from '../core/types';
import type { Translate } from '../i18n/strings';
import { IconDownload, IconEdit, IconPlus, IconSearch, IconTrash, IconUpload } from './icons';
import { SignThumb } from './SignThumb';

export function LibraryPanel({
  signs,
  language,
  t,
  onAddExamples,
  onRename,
  onDelete,
  onImport,
}: {
  signs: Sign[];
  language: string;
  t: Translate;
  onAddExamples: (sign: Sign) => void;
  onRename: (sign: Sign, label: string) => void;
  onDelete: (sign: Sign) => void;
  onImport: (text: string) => void;
}) {
  const [query, setQuery] = useState('');
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState('');
  const fileRef = useRef<HTMLInputElement>(null);

  // Group by category, filtered by the search text.
  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = signs.filter((s) => !q || s.label.toLowerCase().includes(q) || s.category.toLowerCase().includes(q));
    const map = new Map<string, Sign[]>();
    for (const s of filtered) map.set(s.category, [...(map.get(s.category) ?? []), s]);
    return [...map.entries()].sort(([a], [b]) => (a === '' ? 1 : b === '' ? -1 : a.localeCompare(b)));
  }, [signs, query]);

  const download = () => {
    const url = URL.createObjectURL(exportLibrary(language, signs));
    const a = document.createElement('a');
    a.href = url;
    a.download = `signs-${language}-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="panel-body">
      <div className="library-tools">
        <label className="search">
          <IconSearch />
          <input
            className="input"
            type="search"
            placeholder={t('librarySearch')}
            aria-label={t('librarySearch')}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <button className="icon-btn" onClick={download} disabled={!signs.length} aria-label={t('exportLibrary')} title={t('exportLibrary')}>
          <IconDownload />
        </button>
        <button className="icon-btn" onClick={() => fileRef.current?.click()} aria-label={t('importLibrary')} title={t('importLibrary')}>
          <IconUpload />
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={async (e) => {
            const file = e.target.files?.[0];
            e.target.value = '';
            if (file) onImport(await file.text());
          }}
        />
      </div>

      <p className="muted">{t('libraryCount', { n: signs.length })}</p>

      {signs.length === 0 && <p className="empty">{t('libraryEmpty')}</p>}
      {signs.length > 0 && groups.length === 0 && <p className="empty">{t('libraryNoMatch')}</p>}

      {groups.map(([category, items]) => (
        <section key={category || '-'} className="block">
          {category && <h3 className="block-title">{category}</h3>}
          <ul className="sign-list">
            {items.map((s) => (
              <li key={s.id} className="sign-row">
                <SignThumb sample={s.samples[0]} moving={s.kind === 'dynamic'} />
                <div className="sign-info">
                  {editing === s.id ? (
                    <form
                      onSubmit={(e) => {
                        e.preventDefault();
                        if (draft.trim()) onRename(s, draft.trim());
                        setEditing(null);
                      }}
                    >
                      <input
                        className="input input-small"
                        value={draft}
                        maxLength={60}
                        autoFocus
                        onChange={(e) => setDraft(e.target.value)}
                        onBlur={() => setEditing(null)}
                        onKeyDown={(e) => e.key === 'Escape' && setEditing(null)}
                      />
                    </form>
                  ) : (
                    <span className="sign-label">{s.label}</span>
                  )}
                  <span className="sign-meta">
                    <span className={`kind kind-${s.kind}`}>{s.kind === 'static' ? t('stillSign') : t('movingSign')}</span>
                    {t('examples', { n: s.samples.length })}
                  </span>
                </div>
                <div className="sign-actions">
                  <button className="icon-btn small" onClick={() => onAddExamples(s)} aria-label={t('addExamples')} title={t('addExamples')}>
                    <IconPlus size={16} />
                  </button>
                  <button
                    className="icon-btn small"
                    onMouseDown={(e) => e.preventDefault()}
                    onClick={() => {
                      setDraft(s.label);
                      setEditing(s.id);
                    }}
                    aria-label={t('rename')}
                    title={t('rename')}
                  >
                    <IconEdit size={16} />
                  </button>
                  <button
                    className="icon-btn small danger"
                    onClick={() => window.confirm(t('deleteConfirm', { label: s.label })) && onDelete(s)}
                    aria-label={t('delete')}
                    title={t('delete')}
                  >
                    <IconTrash size={16} />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
