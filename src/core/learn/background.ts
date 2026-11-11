/**
 * Real signs as "negative examples" for the personal recognizer.
 *
 * A sign you teach comes with only 3 or so takes, all of it. Its acceptance
 * threshold is set from how much those takes vary, which says nothing about
 * what OTHER signs look like: a loose threshold can then accept almost any
 * movement. The 250 real recordings used by Learn mode are a ready set of
 * other signs, by other people. The recognizer uses them to make sure each
 * taught sign's threshold leaves real other signs outside (see Recognizer.calibrate).
 */

import { extractFrame } from '../features';
import type { Frame, RawHand } from '../types';
import type { DemoLibrary, Pose } from './demos';

/** Converts a recording to frames, as if it came from the camera (shoulder width = a quarter of the picture). */
export function demoToFrames(poses: Pose[], layout: DemoLibrary['layout']): Frame[] {
  return poses.map((p, t) => {
    const hands: RawHand[] = [];
    for (const [start, side] of [[layout.rightHand[0], 'Right'], [layout.leftHand[0], 'Left']] as const) {
      if (Number.isNaN(p[start * 3]!)) continue;
      const landmarks = Array.from({ length: 21 }, (_, i) => ({
        x: 0.5 + p[(start + i) * 3]! * 0.25,
        y: 0.35 + p[(start + i) * 3 + 1]! * 0.25,
        z: p[(start + i) * 3 + 2]! * 0.25,
      }));
      hands.push({ landmarks, side, score: 1 });
    }
    return extractFrame(hands, 1, t * 33);
  });
}

/** Every recording of the library as frames (only frames with hands). */
export function backgroundFromDemos(library: DemoLibrary): Frame[][] {
  return library.labels
    .map((label) => demoToFrames(library.get(label)!.frames, library.layout).filter((f) => f.hands[0] || f.hands[1]))
    .filter((frames) => frames.length >= 4);
}
