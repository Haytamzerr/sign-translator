// Copies the MediaPipe runtime files into /public and downloads the hand
// tracking model, so the app can run fully offline after `npm install`.
//
// This script never fails the install: if the download is not possible
// (no internet), the app falls back to loading the same files from a CDN.

import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

function copyWasm() {
  const src = join(root, 'node_modules', '@mediapipe', 'tasks-vision', 'wasm');
  const dest = join(root, 'public', 'mediapipe', 'wasm');
  if (!existsSync(src)) {
    console.warn('[assets] MediaPipe package not found, skipping runtime copy.');
    return;
  }
  mkdirSync(dest, { recursive: true });
  for (const file of readdirSync(src)) copyFileSync(join(src, file), join(dest, file));
  console.log('[assets] MediaPipe runtime copied to public/mediapipe/wasm');
}

async function downloadModel() {
  const dest = join(root, 'public', 'models', 'hand_landmarker.task');
  if (existsSync(dest)) return;
  try {
    const res = await fetch(MODEL_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
    console.log('[assets] Hand model downloaded to public/models');
  } catch (err) {
    console.warn(`[assets] Could not download the hand model (${err.message}). The app will load it online instead.`);
  }
}

copyWasm();
await downloadModel();
