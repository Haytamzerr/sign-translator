/**
 * Shared data types.
 *
 * The whole pipeline is built on one idea: a sign is a short sequence of
 * "frames", and every frame describes up to two hands in a way that does not
 * depend on where you stand or how close you are to the camera.
 */

/** A raw landmark from MediaPipe: x and y are 0..1 of the image, z is depth. */
export interface Point3 {
  x: number;
  y: number;
  z: number;
}

/** One hand as returned by the tracker, before any processing. */
export interface RawHand {
  /** The 21 hand landmarks (wrist, then 4 points per finger). */
  landmarks: Point3[];
  /** Which hand of the signer this is, as reported by MediaPipe. */
  side: 'Left' | 'Right';
  /** How sure MediaPipe is about `side`, 0..1. */
  score: number;
}

/**
 * One hand after normalization.
 * - `shape`: 63 numbers (21 points x 3), measured from the wrist and divided
 *   by the hand's own size, so a big hand close to the camera and a small hand
 *   far away give the same numbers for the same handshape.
 * - `pos`: where the wrist is in the picture (used only to measure movement).
 * - `scale`: the hand's size in the picture (wrist to middle-finger knuckle).
 */
export interface HandFeature {
  shape: number[];
  pos: [number, number];
  scale: number;
}

/**
 * A frame holds two hand "slots": slot 0 is the signer's right hand, slot 1
 * the left. An empty slot is `null`.
 */
export type HandSlots = [HandFeature | null, HandFeature | null];

export interface Frame {
  /** Time in milliseconds. */
  t: number;
  hands: HandSlots;
}

/** A still sign is a held handshape; a moving sign involves motion over time. */
export type SignKind = 'static' | 'dynamic';

/** A sign the user has taught, as stored in the library. */
export interface Sign {
  id: string;
  label: string;
  /** Which sign language library this sign belongs to (for example "asl"). */
  language: string;
  kind: SignKind;
  category: string;
  /** Each sample is one recorded take: a list of frames. */
  samples: Frame[][];
  createdAt: number;
  updatedAt: number;
}

/** One possible answer from the recognizer. */
export interface Candidate {
  signId: string;
  label: string;
  kind: SignKind;
  /** Raw distance to the closest example (smaller is more similar). */
  distance: number;
  /** distance / this sign's acceptance threshold. Below 1 means "accepted". */
  ratio: number;
  /** Probability 0..1 compared with the other signs and with "unknown". */
  confidence: number;
}
