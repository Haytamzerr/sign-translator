/**
 * The decoder watches the stream of frames and decides WHEN a sign happened.
 *
 * In continuous signing there is no "start" button, so it uses three states:
 * - idle:   no hands in view
 * - hold:   hands visible and still  -> check still signs (handshapes)
 * - moving: hands moving             -> collect the movement; when the hands
 *           stop or leave, check moving signs on the collected segment
 *
 * A word is only written ("committed") when the recognizer is confident AND,
 * for still signs, the same answer has been held for a short time. This is
 * what stops the app from writing words when you scratch your nose.
 */

import { hasHands, motionBetween } from './features';
import type { Recognizer } from './recognizer';
import type { Candidate, Frame } from './types';

export type DecoderState = 'idle' | 'hold' | 'moving';

export interface DecoderSettings {
  /** How long a still sign must be held before it is written, in ms. */
  holdMs: number;
  /** Speed (hand sizes per second) above which the hands count as moving. */
  moveThreshold: number;
  /** Minimum confidence (0..1) required to write a word. */
  minConfidence: number;
}

export interface Commit {
  signId: string;
  label: string;
  confidence: number;
  kind: Candidate['kind'];
  /** The frames that produced this word, kept so a correction can teach the app. */
  frames: Frame[];
  t: number;
}

export interface DecoderOutput {
  state: DecoderState;
  /** Smoothed motion, hand sizes per second. */
  motion: number;
  /** Current best guesses, best first. */
  candidates: Candidate[];
  commit: Commit | null;
  /**
   * Set when a movement has just ended: its time span, and whether the personal
   * library already wrote a word for it. Used by the pre-trained model.
   */
  segment: { start: number; end: number; committed: boolean } | null;
}

/** Frames of the moving segment kept from before motion was detected. */
const LEAD_IN = 3;
/** Still frames needed to decide that a movement has ended. */
const STILL_TO_END = 4;
/** Ignore movements shorter than this many frames. */
const MIN_SEGMENT = 6;
/** Longest movement considered one sign. */
const MAX_SEGMENT_MS = 4000;
/** After a moving sign is written, still signs are paused briefly (avoids a double word). */
const COOLDOWN_MS = 600;
/** How often a live guess is computed while moving (every N frames). */
const PREVIEW_EVERY = 5;

export class SignDecoder {
  private history: Frame[] = [];
  private state: DecoderState = 'idle';
  private motion = 0;
  private segmentStart = -1;
  private stillCount = 0;
  private holdLabel: string | null = null;
  private holdSince = 0;
  private lastHoldCommit: string | null = null;
  private cooldownUntil = 0;
  private frameCount = 0;
  private lastCandidates: Candidate[] = [];
  private finished: DecoderOutput['segment'] = null;

  constructor(
    private recognizer: Recognizer,
    private settings: DecoderSettings,
  ) {}

  setSettings(settings: DecoderSettings): void {
    this.settings = settings;
  }

  reset(): void {
    this.history = [];
    this.state = 'idle';
    this.motion = 0;
    this.segmentStart = -1;
    this.stillCount = 0;
    this.holdLabel = null;
    this.lastHoldCommit = null;
    this.lastCandidates = [];
  }

  push(frame: Frame): DecoderOutput {
    this.frameCount++;
    const prev = this.history[this.history.length - 1];
    this.history.push(frame);
    // Keep only the recent past.
    while (this.history.length > 0 && frame.t - this.history[0]!.t > MAX_SEGMENT_MS + 1000) {
      this.history.shift();
      if (this.segmentStart > 0) this.segmentStart--;
    }

    // Smoothed motion (exponential moving average), so one shaky frame
    // does not flip the state.
    const raw = prev ? motionBetween(prev, frame) : 0;
    this.motion = this.motion * 0.5 + raw * 0.5;

    let commit: Commit | null = null;

    if (!hasHands(frame)) {
      if (this.state === 'moving') commit = this.finishSegment(this.history.length - 1);
      this.state = 'idle';
      this.holdLabel = null;
      this.lastHoldCommit = null;
      this.lastCandidates = [];
      return this.output(commit);
    }

    // Hysteresis: it takes more motion to start "moving" than to stay moving.
    const threshold = this.settings.moveThreshold;
    const isMoving = this.state === 'moving' ? this.motion > threshold * 0.6 : this.motion > threshold;

    if (isMoving) {
      if (this.state !== 'moving') {
        this.segmentStart = Math.max(0, this.history.length - 1 - LEAD_IN);
        this.state = 'moving';
      }
      this.stillCount = 0;
      this.holdLabel = null;
      this.lastHoldCommit = null;
      const tooLong = frame.t - this.history[this.segmentStart]!.t > MAX_SEGMENT_MS;
      if (tooLong) {
        commit = this.finishSegment(this.history.length);
        this.state = 'hold';
      } else if (this.frameCount % PREVIEW_EVERY === 0 && this.recognizer.hasDynamic) {
        this.lastCandidates = this.recognizer.classifyDynamic(this.history.slice(this.segmentStart));
      }
      return this.output(commit);
    }

    if (this.state === 'moving') {
      this.stillCount++;
      if (this.stillCount < STILL_TO_END) return this.output(null);
      commit = this.finishSegment(this.history.length - STILL_TO_END + 1);
      this.state = 'hold';
      if (commit) return this.output(commit);
    }

    this.state = 'hold';
    return this.output(this.checkHold(frame));
  }

  /** Still sign logic: the same confident answer must be held for `holdMs`. */
  private checkHold(frame: Frame): Commit | null {
    const candidates = this.recognizer.classifyStatic(frame);
    this.lastCandidates = candidates;
    const best = candidates[0];
    if (!best || best.ratio >= 1 || best.confidence < this.settings.minConfidence) {
      this.holdLabel = null;
      return null;
    }
    if (best.signId !== this.holdLabel) {
      this.holdLabel = best.signId;
      this.holdSince = frame.t;
      return null;
    }
    const held = frame.t - this.holdSince >= this.settings.holdMs;
    if (!held || this.lastHoldCommit === best.signId || frame.t < this.cooldownUntil) return null;

    this.lastHoldCommit = best.signId;
    const frames = this.history.filter((f) => f.t >= this.holdSince);
    return { signId: best.signId, label: best.label, confidence: best.confidence, kind: best.kind, frames, t: frame.t };
  }

  /** Moving sign logic: classify the collected segment once the movement ends. */
  private finishSegment(end: number): Commit | null {
    const segment = this.history.slice(this.segmentStart, end);
    this.segmentStart = -1;
    this.stillCount = 0;
    if (segment.length >= MIN_SEGMENT) {
      this.finished = { start: segment[0]!.t, end: segment[segment.length - 1]!.t, committed: false };
    }
    if (segment.length < MIN_SEGMENT || !this.recognizer.hasDynamic) return null;

    const candidates = this.recognizer.classifyDynamic(segment);
    this.lastCandidates = candidates;
    const best = candidates[0];
    if (!best || best.ratio >= 1 || best.confidence < this.settings.minConfidence) return null;

    const last = segment[segment.length - 1]!;
    this.cooldownUntil = last.t + COOLDOWN_MS;
    if (this.finished) this.finished.committed = true;
    return { signId: best.signId, label: best.label, confidence: best.confidence, kind: best.kind, frames: segment, t: last.t };
  }

  /** Tells the decoder that a word was written from outside (pre-trained model). */
  noteExternalCommit(t: number): void {
    this.cooldownUntil = t + COOLDOWN_MS;
  }

  private output(commit: Commit | null): DecoderOutput {
    const segment = this.finished;
    this.finished = null;
    return { state: this.state, motion: this.motion, candidates: this.lastCandidates.slice(0, 3), commit, segment };
  }
}
