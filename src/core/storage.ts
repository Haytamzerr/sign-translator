/**
 * The sign library is saved in the browser's IndexedDB database, on this
 * computer only. Nothing is uploaded anywhere.
 */

import { openDB, type IDBPDatabase } from 'idb';
import type { Sign } from './types';

const DB_NAME = 'signal';
const STORE = 'signs';

let dbPromise: Promise<IDBPDatabase> | null = null;

function db(): Promise<IDBPDatabase> {
  dbPromise ??= openDB(DB_NAME, 1, {
    upgrade(database) {
      const store = database.createObjectStore(STORE, { keyPath: 'id' });
      store.createIndex('language', 'language');
    },
  });
  return dbPromise;
}

export async function loadSigns(language: string): Promise<Sign[]> {
  const all = (await (await db()).getAllFromIndex(STORE, 'language', language)) as Sign[];
  return all.sort((a, b) => a.label.localeCompare(b.label));
}

export async function saveSign(sign: Sign): Promise<void> {
  await (await db()).put(STORE, sign);
}

export async function deleteSign(id: string): Promise<void> {
  await (await db()).delete(STORE, id);
}

export function newId(): string {
  return crypto.randomUUID();
}

// ---------- Export / import -------------------------------------------------

interface LibraryFile {
  format: 'signal-library';
  version: 1;
  language: string;
  exportedAt: string;
  signs: Sign[];
}

export function exportLibrary(language: string, signs: Sign[]): Blob {
  const file: LibraryFile = {
    format: 'signal-library',
    version: 1,
    language,
    exportedAt: new Date().toISOString(),
    signs,
  };
  return new Blob([JSON.stringify(file)], { type: 'application/json' });
}

/**
 * Reads a library file and adds its signs to `language`. A sign whose label
 * already exists receives the imported examples instead of being duplicated.
 * Returns the number of signs added or updated.
 */
export async function importLibrary(text: string, language: string, current: Sign[]): Promise<number> {
  const data = JSON.parse(text) as Partial<LibraryFile>;
  if (data.format !== 'signal-library' || !Array.isArray(data.signs)) {
    throw new Error('not-a-library');
  }
  let count = 0;
  for (const incoming of data.signs) {
    if (typeof incoming.label !== 'string' || !Array.isArray(incoming.samples)) continue;
    const existing = current.find((s) => s.label.toLowerCase() === incoming.label.toLowerCase() && s.kind === incoming.kind);
    const sign: Sign = existing
      ? { ...existing, samples: [...existing.samples, ...incoming.samples], updatedAt: Date.now() }
      : {
          id: newId(),
          label: incoming.label,
          language,
          kind: incoming.kind === 'dynamic' ? 'dynamic' : 'static',
          category: incoming.category ?? '',
          samples: incoming.samples,
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
    await saveSign(sign);
    count++;
  }
  return count;
}
