/**
 * The pre-trained ASL recognizer: a neural network trained on about 94,000
 * recordings of 250 American Sign Language signs (PopSign ASL / Google ISLR).
 *
 * It runs in the browser with ONNX Runtime (WebAssembly), on this computer,
 * without internet once the files are loaded.
 */

import * as ort from 'onnxruntime-web/wasm';
import { dropEmptyFrames, features, fitLength, mirrorRaw, N_FEATURES, type RawFrame } from './features';

const base = import.meta.env.BASE_URL;
const MODEL_DIR = `${base}models/asl/`;

export interface AslConfig {
  maxLen: number;
  features: number;
  valAccuracy?: number;
}

export interface AslGuess {
  /** Index of the sign in the model's vocabulary. */
  index: number;
  /** English label as trained (translated for display elsewhere). */
  label: string;
  /** Probability 0..1. */
  confidence: number;
}

/**
 * Body proportion used to match the camera's shape to the training videos:
 * (height from the shoulder line to the nose) / (shoulder width).
 * `TRAINING_RATIO` is that value measured on the training data (in its own
 * units); the same value is measured live on the camera. Their ratio tells how
 * much to stretch x so both look alike.
 */
const TRAINING_RATIO_DEFAULT = 0.55;
const TYPICAL_PERSON_RATIO = 0.55;

export class AslModel {
  private session: ort.InferenceSession | null = null;
  private config: AslConfig | null = null;
  labels: string[] = [];
  private trainingRatio = TRAINING_RATIO_DEFAULT;
  private liveRatios: number[] = [];

  get ready(): boolean {
    return this.session !== null;
  }

  get accuracy(): number | undefined {
    return this.config?.valAccuracy;
  }

  /** Loads the model files. Returns false if they are not installed. */
  async load(): Promise<boolean> {
    try {
      const [config, labels] = await Promise.all([
        fetch(`${MODEL_DIR}asl_config.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('missing')))),
        fetch(`${MODEL_DIR}asl_labels.json`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('missing')))),
      ]);
      this.config = config as AslConfig;
      this.labels = labels as string[];
      if (typeof (config as AslConfig & { bodyRatio?: number }).bodyRatio === 'number') {
        this.trainingRatio = (config as AslConfig & { bodyRatio: number }).bodyRatio;
      }
      ort.env.wasm.wasmPaths = `${base}ort/`;
      ort.env.wasm.numThreads = 1;
      this.session = await ort.InferenceSession.create(`${MODEL_DIR}asl_model.onnx`, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
      return true;
    } catch (err) {
      console.info('Pre-trained ASL model not available:', err);
      this.session = null;
      return false;
    }
  }

  /**
   * Live calibration of the camera's x stretch, from the body proportions.
   * Call with each frame's pose (nose, left shoulder, right shoulder) in
   * square units (x already multiplied by the video's width/height).
   */
  observePose(nose: [number, number], left: [number, number], right: [number, number]): void {
    const width = Math.hypot(left[0] - right[0], left[1] - right[1]);
    if (width < 1e-3) return;
    const height = (left[1] + right[1]) / 2 - nose[1];
    const ratio = height / width;
    if (ratio < 0.2 || ratio > 1.5) return; // unusual posture, ignore
    this.liveRatios.push(ratio);
    if (this.liveRatios.length > 300) this.liveRatios.shift();
  }

  /** Factor applied to x (in square units) to match the training data. */
  get xStretch(): number {
    const live =
      this.liveRatios.length >= 15
        ? [...this.liveRatios].sort((a, b) => a - b)[Math.floor(this.liveRatios.length / 2)]!
        : TYPICAL_PERSON_RATIO;
    return Math.min(2.5, Math.max(0.4, live / this.trainingRatio));
  }

  /**
   * Recognizes one sign from its frames. The mirrored version is also scored
   * and both are averaged, so left- and right-handed signing both work.
   */
  async classify(frames: RawFrame[], top = 3): Promise<AslGuess[]> {
    if (!this.session || !this.config) return [];
    const kept = dropEmptyFrames(frames);
    const [a, b] = await Promise.all([this.logits(kept), this.logits(kept.map(mirrorRaw))]);
    const scores = softmax(a.map((v, i) => (v + b[i]!) / 2));
    return scores
      .map((confidence, index) => ({ index, label: this.labels[index] ?? String(index), confidence }))
      .sort((p, q) => q.confidence - p.confidence)
      .slice(0, top);
  }

  private async logits(frames: RawFrame[]): Promise<number[]> {
    const { x, mask } = fitLength(features(frames), this.config!.maxLen);
    const out = await this.session!.run({
      x: new ort.Tensor('float32', x, [1, this.config!.maxLen, N_FEATURES]),
      mask: new ort.Tensor('float32', mask, [1, this.config!.maxLen]),
    });
    return Array.from(out.logits!.data as Float32Array);
  }
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exp = values.map((v) => Math.exp(v - max));
  const sum = exp.reduce((s, v) => s + v, 0);
  return exp.map((v) => v / sum);
}
