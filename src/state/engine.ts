/**
 * The engine connects everything in real time:
 *
 *   camera frame -> MediaPipe (hand points) -> features -> smoothing
 *                -> decoder (when did a sign happen?) -> recognizer (which sign?)
 *                -> committed word
 *
 * For ASL, a pre-trained neural network also checks every finished movement,
 * so the 250 signs it knows are recognized without any teaching.
 *
 * It runs outside React on purpose: React only receives a small summary about
 * ten times per second, so the interface never slows down the video loop.
 *
 * Robustness measures (why tracking keeps working in real conditions):
 * - short tracking gaps (motion blur) are bridged, see core/continuity.ts
 * - in a dark room the tracker receives a brightened frame, see core/lighting.ts
 * - framing advice is shown on screen, see core/framing.ts
 * - if the graphics card or the camera fails, the engine restarts itself
 */

import type { AslModel } from '../core/asl/model';
import { compactFrame, type RawFrame } from '../core/asl/features';
import { SignDecoder, type Commit, type DecoderState } from '../core/decoder';
import { HandContinuity, type BridgedHand } from '../core/continuity';
import { extractFrame, hasHands } from '../core/features';
import { framingHint, HintDebouncer, type FramingHint } from '../core/framing';
import { boostFilter, FrameBooster, LightMeter, nextBoostState } from '../core/lighting';
import { LetterDecoder, loadLetterModel, type LetterGuess, type LetterModel } from '../core/letters';
import { Recognizer } from '../core/recognizer';
import { FrameSmoother } from '../core/smoothing';
import { HAND_BONES, HandTracker } from '../core/tracker';
import type { Candidate, Frame, Sign } from '../core/types';
import type { Settings } from './settings';

export type EngineStatus = 'off' | 'loading' | 'running' | 'error';
export type EngineError = 'denied' | 'missing' | 'busy' | 'model' | null;
/** State of the pre-trained ASL model. */
export type PretrainedStatus = 'off' | 'loading' | 'ready' | 'missing';
/**
 * translate: recognized signs are written as captions.
 * practice:  nothing is written; the Learn view receives every guess instead.
 */
export type EngineMode = 'translate' | 'practice';

/** What the built-in model thought of one finished movement (for practice). */
export interface SegmentResult {
  /** Model labels (English, untranslated), best first, with probabilities. */
  guesses: { label: string; confidence: number }[];
  t: number;
}

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
  pretrained: PretrainedStatus;
  /** Advice shown on the camera view, if any. */
  hint: FramingHint | null;
  /** True while the low-light boost is brightening the tracker's input. */
  boosted: boolean;
  /** Set briefly while the engine restarts the tracker after a failure. */
  recovering: boolean;
  /** Current fingerspelling guess for the visible hand, if any. */
  letter: LetterGuess | null;
}

/** Colors of the drawn skeleton: right hand and left hand of the signer. */
const COLORS = { Right: '#f2b544', Left: '#5ec8e0' } as const;
const FINGERTIPS = new Set([4, 8, 12, 16, 20]);
const UI_UPDATE_MS = 100;
/** Recent frames kept for the pre-trained model, in ms. */
const RAW_BUFFER_MS = 6000;
/** Time added before a movement, so the network sees the start of the sign. */
const SEGMENT_LEAD_MS = 200;
/** How often the image brightness is measured. */
const LIGHT_CHECK_MS = 500;
/** Consecutive failed frames before the tracker is rebuilt. */
const MAX_FRAME_ERRORS = 8;
/** After a spelled letter, movements are not read as whole signs for this long. */
const SPELLING_GUARD_MS = 1500;

export class SignEngine {
  private tracker = new HandTracker();
  private smoother = new FrameSmoother();
  private continuity = new HandContinuity();
  private lightMeter = new LightMeter();
  private booster = new FrameBooster();
  private boostActive = false;
  private lastLightCheck = 0;
  private hints = new HintDebouncer();
  private frameErrors = 0;
  private recovering = false;
  private lastRecovery = -Infinity;
  private mode: EngineMode = 'translate';
  /** Increases when the camera view moves to another screen, to stop the old loop. */
  private loopToken = 0;
  private letters: LetterModel | null = null;
  private lettersLoad: Promise<LetterModel | null> | null = null;
  private letterDecoder = new LetterDecoder();
  private lastLetterAt = -Infinity;
  private letterListeners = new Set<(letter: string, t: number) => void>();
  private segmentListeners = new Set<(result: SegmentResult) => void>();
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
  /** Loaded only when needed, so the app starts fast without it. */
  private asl: AslModel | null = null;
  private aslLoad: Promise<boolean> | null = null;
  private rawBuffer: { t: number; raw: RawFrame; frame: Frame }[] = [];
  /** Turns a model label (English) into the interface language. */
  translateLabel: (label: string) => string = (label) => label;

  private snapshot: EngineSnapshot = {
    status: 'off',
    error: null,
    fps: 0,
    inferenceMs: 0,
    state: 'idle',
    motion: 0,
    candidates: [],
    handsVisible: false,
    pretrained: 'off',
    hint: null,
    boosted: false,
    recovering: false,
    letter: null,
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

  /** Receives every fingerspelled letter. */
  onLetter(callback: (letter: string, t: number) => void): () => void {
    this.letterListeners.add(callback);
    return () => this.letterListeners.delete(callback);
  }

  /** Receives the built-in model's opinion of every finished movement. */
  onSegment(callback: (result: SegmentResult) => void): () => void {
    this.segmentListeners.add(callback);
    return () => this.segmentListeners.delete(callback);
  }

  /**
   * Connects the engine to a camera view. The Translate and Learn screens each
   * have their own; when the view changes while the camera is on, the stream
   * moves to the new video element and the loop restarts on it.
   */
  attach(video: HTMLVideoElement, canvas: HTMLCanvasElement): void {
    if (this.video === video && this.canvas === canvas) return;
    this.video = video;
    this.canvas = canvas;
    if (this.running && this.stream) {
      video.srcObject = this.stream;
      void video.play().catch(() => undefined);
      this.loopToken++;
      this.scheduleNext();
    }
  }

  setMode(mode: EngineMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.decoder.reset();
    this.letterDecoder.reset();
    void this.syncPretrained();
    void this.syncLetters();
  }

  /** Loads the letter recognizer when fingerspelling is wanted. */
  private async syncLetters(): Promise<void> {
    if (this.settings.fingerspelling === 'off' && this.mode === 'translate') return;
    this.lettersLoad ??= loadLetterModel();
    this.letters = await this.lettersLoad;
    if (!this.letters) this.lettersLoad = null;
  }

  get letterLabels(): string[] {
    return this.letters?.labels ?? [];
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
    if (previous.signLanguage !== settings.signLanguage || previous.pretrained !== settings.pretrained) {
      void this.syncPretrained();
    }
    if (previous.fingerspelling !== settings.fingerspelling) void this.syncLetters();
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
    // If the camera disconnects or another app takes it, try to reconnect once.
    const track = this.stream.getVideoTracks()[0];
    track?.addEventListener('ended', () => {
      if (!this.running || this.stream?.getVideoTracks()[0] !== track) return;
      console.warn('Camera stream ended, reconnecting.');
      setTimeout(() => void this.start(), 1000);
    });

    try {
      if (!this.tracker.ready) await this.tracker.init(this.settings.delegate, this.settings.numHands);
    } catch (err) {
      console.error(err);
      this.stopStream();
      this.update({ status: 'error', error: 'model' });
      return;
    }

    this.smoother.reset();
    this.continuity.reset();
    this.decoder.reset();
    this.frameErrors = 0;
    this.running = true;
    this.update({ status: 'running' });
    this.loopToken++;
    this.scheduleNext();
    void this.syncPretrained();
    void this.syncLetters();
  }

  /** Turns the pre-trained ASL model (and body tracking) on or off. */
  private async syncPretrained(): Promise<void> {
    // Practice always uses the built-in vocabulary: the lessons teach ASL.
    const wanted = (this.settings.pretrained && this.settings.signLanguage === 'asl') || this.mode === 'practice';
    if (!wanted) {
      await this.tracker.enablePose(false);
      this.update({ pretrained: 'off' });
      return;
    }
    if (!this.asl?.ready) {
      this.update({ pretrained: 'loading' });
      this.aslLoad ??= import('../core/asl/model').then(({ AslModel }) => {
        this.asl ??= new AslModel();
        return this.asl.load();
      });
      const ok = await this.aslLoad;
      if (!ok) {
        this.aslLoad = null;
        this.update({ pretrained: 'missing' });
        return;
      }
    }
    try {
      if (this.tracker.ready) await this.tracker.enablePose(true);
      this.update({ pretrained: 'ready' });
    } catch (err) {
      console.error('Body tracking could not start', err);
      this.update({ pretrained: 'missing' });
    }
  }

  get pretrainedLabels(): string[] {
    return this.asl?.labels ?? [];
  }

  stop(): void {
    this.running = false;
    this.stopStream();
    this.clearCanvas();
    this.update({ status: 'off', state: 'idle', candidates: [], handsVisible: false, fps: 0, hint: null, boosted: false, letter: null });
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
    const token = this.loopToken;
    if ('requestVideoFrameCallback' in video) {
      video.requestVideoFrameCallback(() => this.tick(token));
    } else {
      requestAnimationFrame(() => this.tick(token));
    }
  }

  private tick(token: number): void {
    const video = this.video;
    if (!this.running || !video || token !== this.loopToken) return;
    if (video.readyState >= 2 && video.videoWidth > 0 && !this.recovering) {
      try {
        this.processFrame(video);
        this.frameErrors = 0;
      } catch (err) {
        console.error('Frame skipped:', err);
        if (++this.frameErrors >= MAX_FRAME_ERRORS) void this.recover();
      }
    }
    this.scheduleNext();
  }

  /**
   * The tracker failed several frames in a row (typically the graphics card
   * was reset, which happens on laptops after sleep or a driver hiccup).
   * Rebuild it, on the processor if the graphics card keeps failing.
   */
  private async recover(): Promise<void> {
    if (this.recovering) return;
    this.recovering = true;
    this.update({ recovering: true });
    // A second failure within a minute means the graphics card is not reliable: use the processor.
    const repeated = performance.now() - this.lastRecovery < 60_000;
    this.lastRecovery = performance.now();
    const fallback = repeated ? 'CPU' : this.tracker.delegate;
    try {
      this.tracker.close();
      await this.tracker.init(fallback, this.settings.numHands);
      if (this.snapshot.pretrained === 'ready') await this.tracker.enablePose(true);
      this.smoother.reset();
      this.continuity.reset();
      this.frameErrors = 0;
    } catch (err) {
      console.error('Tracker could not be rebuilt', err);
      this.frameErrors = 0;
    } finally {
      this.recovering = false;
      this.update({ recovering: false });
    }
  }

  private processFrame(video: HTMLVideoElement): void {
    const now = performance.now();

    // Light check a few times per second; decide whether to brighten the input.
    if (now - this.lastLightCheck > LIGHT_CHECK_MS) {
      this.lastLightCheck = now;
      const brightness = this.lightMeter.measure(video);
      const mode = this.settings.lowLightBoost;
      this.boostActive = mode === 'on' || (mode === 'auto' && nextBoostState(this.boostActive, brightness));
    }
    const source = this.boostActive ? (this.booster.apply(video, boostFilter(this.lightMeter.brightness)) ?? video) : video;

    const t0 = performance.now();
    const detected = this.tracker.detect(source, now);
    const inference = performance.now() - t0;
    const pose = detected.pose;
    const raw = this.continuity.apply(detected.hands, now, this.settings.numHands);

    const aspect = video.videoWidth / video.videoHeight;
    let frame = extractFrame(raw, aspect, now);
    if (this.settings.smoothing) frame = this.smoother.smooth(frame);

    const asl = this.snapshot.pretrained === 'ready' ? this.asl : null;
    const aslOn = asl !== null;
    if (asl) {
      if (pose?.[0] && pose[11] && pose[12]) {
        const sq = (i: number): [number, number] => [pose[i]!.x * aspect, pose[i]!.y];
        asl.observePose(sq(0), sq(11), sq(12));
      }
      this.rawBuffer.push({ t: now, raw: compactFrame(raw, pose, aspect * asl.xStretch), frame });
      while (this.rawBuffer.length && now - this.rawBuffer[0]!.t > RAW_BUFFER_MS) this.rawBuffer.shift();
    }

    for (const listener of this.frameListeners) listener(frame);

    let output = null;
    let letter: LetterGuess | null = null;
    if (!this.paused) {
      output = this.decoder.push(frame);
      const practice = this.mode === 'practice';
      if (output.commit && !practice && now - this.lastLetterAt > SPELLING_GUARD_MS) {
        this.flashUntil = now + 300;
        for (const listener of this.commitListeners) listener(output.commit);
      }
      if (aslOn && output.segment && (practice || !output.segment.committed)) void this.classifySegment(output.segment);
      letter = this.readLetter(raw, aspect, now, output.state === 'moving');
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
        hint: this.currentHint(raw, pose, aspect, now),
        boosted: this.boostActive,
        letter,
      });
    } else {
      this.snapshot = { ...this.snapshot, fps, inferenceMs };
    }
  }

  /**
   * Fingerspelling: classify the main hand's shape while it is still, and
   * write a letter when it has been held long enough.
   */
  private readLetter(hands: BridgedHand[], aspect: number, now: number, moving: boolean): LetterGuess | null {
    const model = this.letters;
    const enabled = this.mode === 'practice' || this.settings.fingerspelling !== 'off';
    if (!model || !enabled) return null;
    // The main hand: the largest one that was really detected in this frame.
    let main: BridgedHand | null = null;
    let biggest = 0;
    for (const h of hands) {
      if (h.bridged) continue;
      const p = h.landmarks;
      const size = Math.hypot((p[9]!.x - p[0]!.x) * aspect, p[9]!.y - p[0]!.y);
      if (size > biggest) {
        biggest = size;
        main = h;
      }
    }
    // Spelling happens with the hand raised; a hand resting low is ignored.
    if (!main || main.landmarks[0]!.y > 0.85) {
      this.letterDecoder.push(now, null, moving);
      return null;
    }
    const guess = model.classify(main.landmarks, aspect, 1)[0] ?? null;
    const written = this.letterDecoder.push(now, guess, moving);
    if (written) {
      this.lastLetterAt = now;
      for (const listener of this.letterListeners) listener(written, now);
    }
    return guess;
  }

  private currentHint(hands: BridgedHand[], pose: Parameters<typeof framingHint>[1], aspect: number, now: number) {
    if (!this.settings.trackingHints) return null;
    // "Too dark" is only shown when even the boosted image is very dim.
    const dark = this.lightMeter.brightness >= 0 && this.lightMeter.brightness < 35;
    const hint = framingHint(
      hands.filter((h) => !h.bridged),
      pose,
      { aspect, dark, needShoulders: this.snapshot.pretrained === 'ready' && hands.length > 0 },
    );
    return this.hints.update(hint, now);
  }

  /** Runs the pre-trained model on a finished movement and writes the word if sure. */
  private async classifySegment(segment: { start: number; end: number }): Promise<void> {
    const part = this.rawBuffer.filter((r) => r.t >= segment.start - SEGMENT_LEAD_MS && r.t <= segment.end);
    if (part.length < 6 || !this.asl) return;
    try {
      const guesses = await this.asl.classify(part.map((p) => p.raw), 5);
      const result: SegmentResult = { guesses: guesses.map((g) => ({ label: g.label, confidence: g.confidence })), t: segment.end };
      for (const listener of this.segmentListeners) listener(result);
      if (this.mode === 'practice') return;
      const threshold = this.settings.pretrainedConfidence;
      const candidates: Candidate[] = guesses.map((g) => ({
        signId: `asl:${g.index}`,
        label: this.translateLabel(g.label),
        kind: 'dynamic',
        distance: 1 - g.confidence,
        ratio: g.confidence >= threshold ? 0.5 : 1.5,
        confidence: g.confidence,
      }));
      this.update({ candidates });
      const best = candidates[0];
      if (!best || best.confidence < threshold || !this.running || this.paused) return;
      // Movements between spelled letters are not whole signs.
      if (performance.now() - this.lastLetterAt < SPELLING_GUARD_MS) return;
      const commit: Commit = {
        signId: best.signId,
        label: best.label,
        confidence: best.confidence,
        kind: 'dynamic',
        frames: part.map((p) => p.frame),
        t: segment.end,
      };
      this.decoder.noteExternalCommit(segment.end);
      this.flashUntil = performance.now() + 300;
      for (const listener of this.commitListeners) listener(commit);
    } catch (err) {
      console.error('Pre-trained model failed on this sign', err);
    }
  }

  // ---------- Drawing ----------

  private clearCanvas(): void {
    const ctx = this.canvas?.getContext('2d');
    if (ctx && this.canvas) ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private draw(hands: BridgedHand[], video: HTMLVideoElement, now: number): void {
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
      // A bridged hand (kept from the previous frame) is drawn faded.
      ctx.globalAlpha = hand.bridged ? 0.35 : flash ? 1 : 0.85;
      ctx.beginPath();
      for (const [a, b] of HAND_BONES) {
        const pa = p[a];
        const pb = p[b];
        if (!pa || !pb) continue;
        ctx.moveTo(pa.x * w, pa.y * h);
        ctx.lineTo(pb.x * w, pb.y * h);
      }
      ctx.stroke();

      ctx.globalAlpha = hand.bridged ? 0.35 : 1;
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
