// Copies the MediaPipe and ONNX Runtime files into /public and downloads the
// hand and body tracking models, so the app can run fully offline after `npm install`.
//
// This script never fails the install: if the download is not possible
// (no internet), the app falls back to loading the same files from a CDN.

import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const MODELS = {
  'hand_landmarker.task':
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
  'pose_landmarker_lite.task':
    'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
};

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

/** Copies the ONNX Runtime files used by the built-in ASL model. */
function copyOnnxRuntime() {
  const src = join(root, 'node_modules', 'onnxruntime-web', 'dist');
  const dest = join(root, 'public', 'ort');
  if (!existsSync(src)) return;
  mkdirSync(dest, { recursive: true });
  for (const file of ['ort-wasm-simd-threaded.wasm']) {
    copyFileSync(join(src, file), join(dest, file));
  }
  console.log('[assets] ONNX Runtime copied to public/ort');
}

async function downloadModels() {
  for (const [name, url] of Object.entries(MODELS)) {
    const dest = join(root, 'public', 'models', name);
    if (existsSync(dest)) continue;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      mkdirSync(dirname(dest), { recursive: true });
      writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
      console.log(`[assets] ${name} downloaded to public/models`);
    } catch (err) {
      console.warn(`[assets] Could not download ${name} (${err.message}). The app will load it online instead.`);
    }
  }
}

copyWasm();
copyOnnxRuntime();
await downloadModels();
