/**
 * Animated sign demonstration: a 3D figure moved by recorded points, played at
 * the chosen speed. Drag to turn the figure; the button puts it back.
 *
 * A "clip" is either a recorded sign or a fingerspelled letter. Clips are
 * turned into a timeline of key poses; between key poses the points are
 * interpolated, so playback stays smooth at the screen's frame rate.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import type { DemoLayout, HandShape, Pose, Rig } from '../core/learn/demos';
import type { LetterHold } from '../core/learn/handshapes';
import { letterPose, POSE_SIZE, prepareFrames, solveFigure } from '../core/learn/figure';
import { IconFront } from './icons';
import { Avatar, type AvatarPalette } from './avatar';

export type Clip =
  | { kind: 'sign'; label: string; frames: Pose[] }
  | { kind: 'letter'; letter: string; shape: HandShape; hold?: LetterHold };

interface Key {
  pose: Pose;
  /** Time to move from the previous key to this one, ms (at speed 1). */
  blend: number;
  /** Time to stay on this key afterwards, ms (at speed 1). */
  hold: number;
  clip: number;
  /** Letters get a close-up of the hand. */
  letter: boolean;
}

const FRAME_MS = 1000 / 30;
const LETTER_HOLD_MS = 520;
const LETTER_BLEND_MS = 180;
const PAUSE_BETWEEN_CLIPS_MS = 260;

export function buildTimeline(clips: Clip[], neutral: Pose, layout: DemoLayout, rig: Rig): Key[] {
  const keys: Key[] = [];
  clips.forEach((clip, ci) => {
    if (clip.kind === 'sign') {
      const frames = prepareFrames(clip.frames, layout, rig);
      frames.forEach((pose, fi) => keys.push({ pose, blend: fi === 0 ? 240 : FRAME_MS, hold: 0, clip: ci, letter: false }));
      const last = keys[keys.length - 1];
      if (last) last.hold = PAUSE_BETWEEN_CLIPS_MS;
    } else {
      keys.push({ pose: letterPose(neutral, layout, rig, clip.shape, clip.hold), blend: LETTER_BLEND_MS, hold: LETTER_HOLD_MS, clip: ci, letter: true });
    }
  });
  return keys;
}

/**
 * Interpolates two poses. A point visible in only one of them (a hand
 * appearing) jumps to its new place instead of flying in from nowhere.
 */
function lerpPose(a: Pose, b: Pose, t: number, out: Pose): Pose {
  for (let i = 0; i < out.length; i++) {
    const va = a[i]!;
    const vb = b[i]!;
    out[i] = Number.isNaN(vb) ? (t < 0.5 ? va : NaN) : Number.isNaN(va) ? vb : va + (vb - va) * t;
  }
  return out;
}

/**
 * Curved interpolation between recorded frames b and c (Catmull-Rom spline),
 * using the frames before (a) and after (d) to keep the speed continuous.
 * Straight-line interpolation makes visible corners at every recorded frame;
 * the curve passes through the same points without them.
 */
function splinePose(a: Pose, b: Pose, c: Pose, d: Pose, t: number, out: Pose): Pose {
  const t2 = t * t;
  const t3 = t2 * t;
  for (let i = 0; i < out.length; i++) {
    const p1 = b[i]!;
    const p2 = c[i]!;
    if (Number.isNaN(p1) || Number.isNaN(p2)) {
      out[i] = Number.isNaN(p2) ? (t < 0.5 ? p1 : NaN) : p2;
      continue;
    }
    const p0 = Number.isNaN(a[i]!) ? p1 : a[i]!;
    const p3 = Number.isNaN(d[i]!) ? p2 : d[i]!;
    out[i] = 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (3 * p1 - p0 - 3 * p2 + p3) * t3);
  }
  return out;
}

function readPalette(el: HTMLElement): AvatarPalette {
  const css = getComputedStyle(el);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    skin: v('--avatar-skin', '#2e333c'),
    rim: v('--avatar-rim', '#6fd3ff'),
    inset: v('--mannequin-inset', '#1c2230'),
  };
}

export function SignPlayer({
  clips,
  neutral,
  layout,
  rig,
  speed = 1,
  mirror = true,
  loop = false,
  playing = true,
  replayKey = 0,
  onClipChange,
  onEnd,
  label,
  resetLabel,
  failedLabel,
}: {
  clips: Clip[];
  neutral: Pose;
  layout: DemoLayout;
  rig: Rig;
  speed?: number;
  mirror?: boolean;
  loop?: boolean;
  playing?: boolean;
  /** Changing this number restarts playback from the beginning. */
  replayKey?: number;
  onClipChange?: (index: number) => void;
  onEnd?: () => void;
  label?: string;
  /** Text of the button that turns the figure back to face the viewer. */
  resetLabel?: string;
  /** Message shown when the browser cannot draw 3D. */
  failedLabel?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const insetRef = useRef<HTMLDivElement>(null);
  const mannequinRef = useRef<Avatar | null>(null);
  const [failed, setFailed] = useState(false);
  const keys = useMemo(() => buildTimeline(clips, neutral, layout, rig), [clips, neutral, layout, rig]);
  const settings = useRef({ speed, mirror, loop, playing, onClipChange, onEnd });
  settings.current = { speed, mirror, loop, playing, onClipChange, onEnd };

  // The 3D scene lives as long as the player.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let mannequin: Avatar;
    try {
      mannequin = new Avatar(canvas);
    } catch (err) {
      console.error('3D view unavailable', err);
      setFailed(true);
      return;
    }
    mannequinRef.current = mannequin;
    const applyPalette = () => mannequin.setPalette(readPalette(canvas));
    applyPalette();
    // Follow theme changes (the theme is an attribute on <html>, or the system setting).
    const themeObserver = new MutationObserver(applyPalette);
    themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    media.addEventListener('change', applyPalette);
    const resize = () => {
      const { width, height } = canvas.getBoundingClientRect();
      mannequin.resize(Math.max(1, width), Math.max(1, height));
    };
    resize();
    const sizeObserver = new ResizeObserver(resize);
    sizeObserver.observe(canvas);
    return () => {
      themeObserver.disconnect();
      media.removeEventListener('change', applyPalette);
      sizeObserver.disconnect();
      mannequin.dispose();
      mannequinRef.current = null;
    };
  }, []);

  // Playback.
  useEffect(() => {
    const mannequin = mannequinRef.current;
    if (!mannequin) return;
    const scratch = new Float32Array(POSE_SIZE);
    let k = 0; // current key
    let elapsed = 0; // ms since key k started (at speed 1)
    let last = performance.now();
    let clip = -1;
    let ended = false;
    let frame = 0;
    let insetShown = false;
    let lastDraw = performance.now();
    let snapNext = true; // the first frame of a new demonstration jumps straight to its pose
    const loopFrom = new Float32Array(POSE_SIZE);
    let looped = false;

    const step = (now: number) => {
      const s = settings.current;
      const dt = Math.min(100, now - last) * s.speed;
      last = now;
      if (s.playing && !ended && keys.length) {
        elapsed += dt;
        while (k < keys.length && elapsed >= keys[k]!.blend + keys[k]!.hold) {
          elapsed -= keys[k]!.blend + keys[k]!.hold;
          k++;
        }
        if (k >= keys.length) {
          if (s.loop) {
            // Start again from where the figure is now, not from the rest pose:
            // jumping back to rest would show as a snap at every repeat.
            loopFrom.set(scratch);
            looped = true;
            k = 0;
            elapsed = 0;
          } else {
            k = keys.length - 1;
            elapsed = keys[k]!.blend + keys[k]!.hold;
            ended = true;
            s.onEnd?.();
          }
        }
      }
      let pose: Pose = neutral;
      let closeUp = false;
      if (keys.length) {
        const cur = keys[k]!;
        const prev = k > 0 ? keys[k - 1]!.pose : looped ? loopFrom : neutral;
        const t = cur.blend > 0 ? Math.min(1, elapsed / cur.blend) : 1;
        if (cur.blend > FRAME_MS || k === 0) {
          // Transitions and letters: ease in and out.
          pose = lerpPose(prev, cur.pose, t * t * (3 - 2 * t), scratch);
        } else {
          // Between recorded frames of the same sign: a smooth curve.
          const before = k > 1 && keys[k - 2]!.clip === cur.clip ? keys[k - 2]!.pose : prev;
          const after = k + 1 < keys.length && keys[k + 1]!.clip === cur.clip && keys[k + 1]!.blend <= FRAME_MS ? keys[k + 1]!.pose : cur.pose;
          pose = splinePose(before, prev, cur.pose, after, t, scratch);
        }
        closeUp = cur.letter;
        if (cur.clip !== clip) {
          clip = cur.clip;
          s.onClipChange?.(clip);
        }
      }
      mannequin.update(solveFigure(pose, layout, rig), now - lastDraw, snapNext);
      lastDraw = now;
      snapNext = false;
      const inset = mannequin.render(s.mirror, closeUp);
      const box = insetRef.current;
      if (box && (inset || insetShown)) {
        insetShown = !!inset;
        box.hidden = !inset;
        if (inset) Object.assign(box.style, { left: `${inset.x}px`, top: `${inset.y}px`, width: `${inset.size}px`, height: `${inset.size}px` });
      }
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frame);
  }, [keys, neutral, layout, rig, replayKey]);

  return (
    <div className="player">
      <canvas ref={canvasRef} className="player-canvas" role="img" aria-label={label} />
      <div ref={insetRef} className="player-inset" hidden aria-hidden="true" />
      {failed ? (
        <p className="muted player-error">{failedLabel}</p>
      ) : (
        <button
          className="icon-btn player-reset"
          onClick={() => mannequinRef.current?.resetView()}
          aria-label={resetLabel}
          title={resetLabel}
        >
          <IconFront size={18} />
        </button>
      )}
    </div>
  );
}
