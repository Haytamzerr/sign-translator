/**
 * Hand tracking with Google's MediaPipe Hand Landmarker.
 *
 * MediaPipe finds hands in each video frame and returns 21 points per hand
 * (wrist, knuckles, finger joints and fingertips). It runs on the graphics
 * card (GPU) through WebGL, which is what makes 30+ frames per second possible.
 *
 * Files are loaded from the app itself when available (fully offline), and
 * from the internet otherwise.
 */

import { FilesetResolver, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { Point3, RawHand } from './types';

const VERSION = '1.2.2';
const CDN_WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${VERSION}/wasm`;
const CDN_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';

const CDN_POSE_MODEL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';

const base = import.meta.env.BASE_URL;
const LOCAL_WASM = `${base}mediapipe/wasm`;
const LOCAL_MODEL = `${base}models/hand_landmarker.task`;
const LOCAL_POSE_MODEL = `${base}models/pose_landmarker_lite.task`;

/** Pairs of landmark indices that form the bones drawn on screen. */
export const HAND_BONES: [number, number][] = HandLandmarker.HAND_CONNECTIONS.map((c) => [c.start, c.end]);

export type Delegate = 'GPU' | 'CPU';

async function exists(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { method: 'HEAD' });
    // Vite's dev server answers unknown paths with index.html, so check the type too.
    return res.ok && !(res.headers.get('content-type') ?? '').includes('text/html');
  } catch {
    return false;
  }
}

/** What the trackers found in one video frame. */
export interface Detection {
  hands: RawHand[];
  /** The 33 body points, or null when body tracking is off or no body is visible. */
  pose: Point3[] | null;
}

export class HandTracker {
  private landmarker: HandLandmarker | null = null;
  private poseLandmarker: PoseLandmarker | null = null;
  private fileset: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>> | null = null;
  private lastTimestamp = 0;
  delegate: Delegate = 'GPU';

  /** Loads the hand model. Falls back to the CPU if the GPU cannot be used. */
  async init(preferred: Delegate, numHands: number): Promise<void> {
    const wasmPath = (await exists(`${LOCAL_WASM}/vision_wasm_internal.js`)) ? LOCAL_WASM : CDN_WASM;
    const modelPath = (await exists(LOCAL_MODEL)) ? LOCAL_MODEL : CDN_MODEL;
    const fileset = await FilesetResolver.forVisionTasks(wasmPath);
    this.fileset = fileset;

    const create = (delegate: Delegate) =>
      HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: modelPath, delegate },
        runningMode: 'VIDEO',
        numHands,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });

    this.close();
    try {
      this.landmarker = await create(preferred);
      this.delegate = preferred;
    } catch (err) {
      if (preferred === 'CPU') throw err;
      console.warn('GPU not available, using the CPU instead.', err);
      this.landmarker = await create('CPU');
      this.delegate = 'CPU';
    }
  }

  /**
   * Body tracking (shoulders, elbows, face position). Needed by the pre-trained
   * ASL model, which uses where the hands are relative to the body.
   */
  async enablePose(enabled: boolean): Promise<void> {
    if (!enabled) {
      this.poseLandmarker?.close();
      this.poseLandmarker = null;
      return;
    }
    if (this.poseLandmarker || !this.fileset) return;
    const modelPath = (await exists(LOCAL_POSE_MODEL)) ? LOCAL_POSE_MODEL : CDN_POSE_MODEL;
    this.poseLandmarker = await PoseLandmarker.createFromOptions(this.fileset, {
      baseOptions: { modelAssetPath: modelPath, delegate: this.delegate },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
  }

  get poseEnabled(): boolean {
    return this.poseLandmarker !== null;
  }

  async setNumHands(numHands: number): Promise<void> {
    await this.landmarker?.setOptions({ numHands });
  }

  get ready(): boolean {
    return this.landmarker !== null;
  }

  /** Finds hands (and the body, if enabled) in the current video frame. */
  detect(video: HTMLVideoElement, now: number): Detection {
    if (!this.landmarker) return { hands: [], pose: null };
    // MediaPipe requires strictly increasing timestamps.
    const ts = Math.max(now, this.lastTimestamp + 1);
    this.lastTimestamp = ts;
    const result = this.landmarker.detectForVideo(video, ts);
    const pose = this.poseLandmarker?.detectForVideo(video, ts).landmarks[0] ?? null;
    const hands: RawHand[] = result.landmarks.map((landmarks, i) => {
      const category = result.handedness[i]?.[0];
      return {
        landmarks,
        side: category?.categoryName === 'Left' ? 'Left' : 'Right',
        score: category?.score ?? 0,
      };
    });
    return { hands, pose };
  }

  close(): void {
    this.landmarker?.close();
    this.landmarker = null;
    this.poseLandmarker?.close();
    this.poseLandmarker = null;
  }
}
