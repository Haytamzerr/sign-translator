/**
 * The engine connects everything in real time:
 *
 *   camera frame -> MediaPipe (hand points) -> features -> smoothing
 *                -> decoder (when did a sign happen?) -> recognizer (which sign?)
 *                -> committed word
 *
 * It runs outside React on purpose: React only receives a small summary about
 * ten times per second, so the interface never slows down the video loop.
 */

import { SignDecoder, type Commit, type DecoderState } from '../core/decoder';
import { extractFrame, hasHands } from '../core/features';
import { Recognizer } from '../core/recognizer';
import { FrameSmoother } from '../core/smoothing';
import { HAND_BONES, HandTracker } from '../core/tracker';
import type { Candidate, Frame, RawHand, Sign } from '../core/types';
import type { Settings } from './settings';

export type EngineStatus = 'off' | 'loading' | 'running' | 'error';
export type EngineError = 'denied' | 'missing' | 'busy' | 'model' | null;

export interface EngineSnapshot {
  status: EngineStatus;
  error: EngineError;
  fps: number;
  /** Time spent detecting hands per frame, in ms. */
  inferenceMs: number;
  state: DecoderState;
  motion: number;
  candidates: Candidate[];
  handsVisible: boolean;
}

/** Colors of the drawn skeleton: right hand and left hand of the signer. */
const COLORS = { Right: '#f2b544', Left: '#5ec8e0' } as const;
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
const UI_UPDATE_MS = 100;

export class SignEngine {
  private tracker = new HandTracker();
  private smoother = new FrameSmoother();
  private recognizer: Recognizer;
  private decoder: SignDecoder;
  private stream: MediaStream | null = null;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private running = false;
  private settings: Settings;
  private lastUiUpdate = 0;
  private lastFrameTime = 0;
  private flashUntil = 0;
  private paused = false;

  private snapshot: EngineSnapshot = {
    status: 'off',
    error: null,
    fps: 0,
    inferenceMs: 0,
    state: 'idle',
    motion: 0,
    candidates: [],
    handsVisible: false,
  };

  private subscribers = new Set<() => void>();
  private commitListeners = new Set<(commit: Commit) => void>();
  private frameListeners = new Set<(frame: Frame) => void>();

  constructor(settings: Settings) {
    this.settings = settings;
    this.recognizer = new Recognizer({ sensitivity: settings.sensitivity, eitherHand: settings.eitherHand });
    this.decoder = new SignDecoder(this.recognizer, this.decoderSettings(settings));
  }

  // ---------- Subscriptions (used by React) ----------

  subscribe = (callback: () => void): (() => void) => {
    this.subscribers.add(callback);
    return () => this.subscribers.delete(callback);
  };

  getSnapshot = (): EngineSnapshot => this.snapshot;

  onCommit(callback: (commit: Commit) => void): () => void {
    this.commitListeners.add(callback);
    return () => this.commitListeners.delete(callback);
  }

  /** Receives every processed frame. Used by teach mode to record examples. */
  onFrame(callback: (frame: Frame) => void): () => void {
    this.frameListeners.add(callback);
    return () => this.frameListeners.delete(callback);
  }

  private update(patch: Partial<EngineSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...patch };
    for (const s of this.subscribers) s();
  }

  // ---------- Configuration ----------

  attach(video: HTMLVideoElement, canvas: HTMLCanvasElement): void {
    this.video = video;
    this.canvas = canvas;
  }

  setLibrary(signs: Sign[]): void {
    this.recognizer.setLibrary(signs);
    this.decoder.reset();
  }

  /** While teaching, recognition is paused so no words are written. */
  setPaused(paused: boolean): void {
    this.paused = paused;
    this.decoder.reset();
  }

  setSettings(settings: Settings): void {
    const previous = this.settings;
    this.settings = settings;
    this.recognizer.setOptions({ sensitivity: settings.sensitivity, eitherHand: settings.eitherHand });
    this.decoder.setSettings(this.decoderSettings(settings));
    if (previous.numHands !== settings.numHands) void this.tracker.setNumHands(settings.numHands);
    if (previous.delegate !== settings.delegate) this.tracker.close();
    const restart =
      previous.cameraId !== settings.cameraId ||
      previous.resolution !== settings.resolution ||
      previous.delegate !== settings.delegate;
    if (restart && this.running) void this.start();
  }

  private decoderSettings(s: Settings) {
    return { holdMs: s.holdMs, moveThreshold: s.moveThreshold, minConfidence: s.minConfidence };
  }

  // ---------- Camera ----------

  async start(): Promise<void> {
    this.stop();
    if (!this.video) return;
    this.update({ status: 'loading', error: null });

    const size = this.settings.resolution === 'hd' ? { width: 1280, height: 720 } : { width: 640, height: 480 };
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          ...(this.settings.cameraId ? { deviceId: { exact: this.settings.cameraId } } : { facingMode: 'user' }),
          width: { ideal: size.width },
          height: { ideal: size.height },
          frameRate: { ideal: 30 },
        },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : '';
      const error: EngineError =
        name === 'NotAllowedError' || name === 'SecurityError'
          ? 'denied'
          : name === 'NotReadableError' || name === 'AbortError'
            ? 'busy'
            : 'missing';
      this.update({ status: 'error', error });
      return;
    }

    this.video.srcObject = this.stream;
    await this.video.play().catch(() => undefined);

    try {
      if (!this.tracker.ready) await this.tracker.init(this.settings.delegate, this.settings.numHands);
    } catch (err) {
      console.error(err);
      this.stopStream();
      this.update({ status: 'error', error: 'model' });
      return;
    }

    this.smoother.reset();
    this.decoder.reset();
    this.running = true;
    this.update({ status: 'running' });
    this.scheduleNext();
  }

  stop(): void {
    this.running = false;
    this.stopStream();
    this.clearCanvas();
    this.update({ status: 'off', state: 'idle', candidates: [], handsVisible: false, fps: 0 });
  }

  private stopStream(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    if (this.video) this.video.srcObject = null;
  }

  // ---------- Real-time loop ----------

  private scheduleNext(): void {
    const video = this.video;
    if (!this.running || !video) return;
    // requestVideoFrameCallback fires exactly once per new camera frame,
    // so no frame is processed twice and none is waited for needlessly.
    if ('requestVideoFrameCallback' in video) {
      video.requestVideoFrameCallback(() => this.tick());
    } else {
      requestAnimationFrame(() => this.tick());
    }
  }

  private tick(): void {
    const video = this.video;
    if (!this.running || !video) return;
    if (video.readyState >= 2 && video.videoWidth > 0) {
      try {
        this.processFrame(video);
      } catch (err) {
        console.error('Frame skipped:', err);
      }
    }
    this.scheduleNext();
  }

  private processFrame(video: HTMLVideoElement): void {
    const now = performance.now();
    const t0 = performance.now();
    const raw = this.tracker.detect(video, now);
    const inference = performance.now() - t0;

    const aspect = video.videoWidth / video.videoHeight;
    let frame = extractFrame(raw, aspect, now);
    if (this.settings.smoothing) frame = this.smoother.smooth(frame);

    for (const listener of this.frameListeners) listener(frame);

    let output = null;
    if (!this.paused) {
      output = this.decoder.push(frame);
      if (output.commit) {
        this.flashUntil = now + 300;
        for (const listener of this.commitListeners) listener(output.commit);
      }
    }

    this.draw(raw, video, now);

    // Frame rate and inference time as smoothed averages.
    const dt = this.lastFrameTime ? now - this.lastFrameTime : 33;
    this.lastFrameTime = now;
    const fps = this.snapshot.fps ? this.snapshot.fps * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
    const inferenceMs = this.snapshot.inferenceMs ? this.snapshot.inferenceMs * 0.9 + inference * 0.1 : inference;

    if (now - this.lastUiUpdate > UI_UPDATE_MS || output?.commit) {
      this.lastUiUpdate = now;
      this.update({
        fps,
        inferenceMs,
        state: output?.state ?? (hasHands(frame) ? 'hold' : 'idle'),
        motion: output?.motion ?? 0,
        candidates: output?.candidates ?? [],
        handsVisible: hasHands(frame),
      });
    } else {
      this.snapshot = { ...this.snapshot, fps, inferenceMs };
    }
  }

  // ---------- Drawing ----------

  private clearCanvas(): void {
    const ctx = this.canvas?.getContext('2d');
    if (ctx && this.canvas) ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private draw(hands: RawHand[], video: HTMLVideoElement, now: number): void {
    const canvas = this.canvas;
    if (!canvas) return;
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.settings.showSkeleton) return;

    const w = canvas.width;
    const h = canvas.height;
    const flash = now < this.flashUntil;
    const unit = Math.max(w, h) / 640;

    for (const hand of hands) {
      const color = COLORS[hand.side];
      const p = hand.landmarks;
      ctx.strokeStyle = color;
      ctx.lineCap = 'round';
      ctx.lineWidth = (flash ? 5 : 3) * unit;
      ctx.globalAlpha = flash ? 1 : 0.85;
      ctx.beginPath();
      for (const [a, b] of HAND_BONES) {
        const pa = p[a];
        const pb = p[b];
        if (!pa || !pb) continue;
        ctx.moveTo(pa.x * w, pa.y * h);
        ctx.lineTo(pb.x * w, pb.y * h);
      }
      ctx.stroke();

      ctx.globalAlpha = 1;
      p.forEach((point, i) => {
        const r = (FINGERTIPS.has(i) ? 5 : 3) * unit;
        ctx.beginPath();
        ctx.arc(point.x * w, point.y * h, r, 0, Math.PI * 2);
        ctx.fillStyle = FINGERTIPS.has(i) ? color : '#ffffff';
        ctx.fill();
      });
    }
    ctx.globalAlpha = 1;
  }
}
