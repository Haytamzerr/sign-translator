/**
 * Animated sign demonstration: a flat figure (face, shoulders, arms, hands)
 * drawn from recorded points, played at the chosen speed.
 *
 * A "clip" is either a recorded sign or a fingerspelled letter. Clips are
 * turned into a timeline of key poses; between key poses the points are
 * interpolated, so playback stays smooth at the screen's frame rate.
 */

import { useEffect, useMemo, useRef } from 'react';
import { HAND_BONES } from '../core/tracker';
import { DEMO_POINTS, type DemoLayout, type Pose } from '../core/learn/demos';

export type Clip =
  | { kind: 'sign'; label: string; frames: Pose[] }
  | { kind: 'letter'; letter: string; shape: [number, number][] };

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
const LETTER_BLEND_MS = 160;
const PAUSE_BETWEEN_CLIPS_MS = 260;
/** Hand size (wrist to middle knuckle) in shoulder widths, for drawn letters. */
const HAND_UNIT = 0.3;

/** Fills short gaps where a hand was not tracked, so the drawing does not flicker. */
function fillGaps(frames: Pose[], layout: DemoLayout): Pose[] {
  const out = frames.map((f) => f.slice());
  for (const [from, to] of [layout.leftHand, layout.rightHand]) {
    let last = -1;
    out.forEach((f, i) => {
      if (!Number.isNaN(f[from * 2]!)) {
        if (last >= 0 && i - last > 1 && i - last <= 6) {
          for (let j = last + 1; j < i; j++) for (let k = from * 2; k < to * 2; k++) out[j]![k] = out[last]![k]!;
        }
        last = i;
      }
    });
  }
  return out;
}

/** A standing pose with the right hand raised in front of the shoulder, forming a letter. */
export function letterPose(neutral: Pose, layout: DemoLayout, shape: [number, number][]): Pose {
  const pose = neutral.slice();
  const a = layout.arms[0];
  const lsx = pose[a * 2]!;
  const rsx = pose[(a + 1) * 2]!;
  const rsy = pose[(a + 1) * 2 + 1]!;
  const out = Math.sign(rsx - lsx) || 1; // direction away from the body on the right side
  const wrist: [number, number] = [rsx + out * 0.12, rsy + 0.3];
  pose[(a + 3) * 2] = rsx + out * 0.3; // right elbow
  pose[(a + 3) * 2 + 1] = rsy + 0.85;
  pose[(a + 5) * 2] = wrist[0]; // right wrist
  pose[(a + 5) * 2 + 1] = wrist[1];
  const [h] = layout.rightHand;
  shape.forEach(([x, y], i) => {
    pose[(h + i) * 2] = wrist[0] + x * HAND_UNIT;
    pose[(h + i) * 2 + 1] = wrist[1] + y * HAND_UNIT;
  });
  return pose;
}

export function buildTimeline(clips: Clip[], neutral: Pose, layout: DemoLayout): Key[] {
  const keys: Key[] = [];
  clips.forEach((clip, ci) => {
    if (clip.kind === 'sign') {
      const frames = fillGaps(clip.frames, layout);
      frames.forEach((pose, fi) => keys.push({ pose, blend: fi === 0 ? 220 : FRAME_MS, hold: 0, clip: ci, letter: false }));
      const last = keys[keys.length - 1];
      if (last) last.hold = PAUSE_BETWEEN_CLIPS_MS;
    } else {
      keys.push({ pose: letterPose(neutral, layout, clip.shape), blend: LETTER_BLEND_MS, hold: LETTER_HOLD_MS, clip: ci, letter: true });
    }
  });
  return keys;
}

function lerpPose(a: Pose, b: Pose, t: number, out: Pose): Pose {
  for (let i = 0; i < out.length; i++) {
    const va = a[i]!;
    const vb = b[i]!;
    out[i] = Number.isNaN(vb) ? NaN : Number.isNaN(va) ? vb : va + (vb - va) * t;
  }
  return out;
}

interface Palette {
  body: string;
  bodyLine: string;
  face: string;
  faceLine: string;
  right: string;
  left: string;
}

function readPalette(el: HTMLElement): Palette {
  const css = getComputedStyle(el);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return {
    body: v('--figure-body', '#2b3446'),
    bodyLine: v('--figure-line', '#46526b'),
    face: v('--figure-face', '#323c51'),
    faceLine: v('--figure-face-line', '#8a96ad'),
    right: '#f2b544',
    left: '#5ec8e0',
  };
}

/** Draws one hand (21 points) with a filled palm, bones and fingertips. */
function drawHand(ctx: CanvasRenderingContext2D, P: (i: number) => readonly [number, number], color: string, width: number, tip: number) {
  ctx.fillStyle = color + '40';
  ctx.beginPath();
  [0, 1, 5, 9, 13, 17].forEach((i, k) => (k ? ctx.lineTo : ctx.moveTo).call(ctx, ...P(i)));
  ctx.closePath();
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  for (const [p, q] of HAND_BONES) {
    ctx.moveTo(...P(p));
    ctx.lineTo(...P(q));
  }
  ctx.stroke();
  ctx.fillStyle = color;
  for (const t of [4, 8, 12, 16, 20]) {
    ctx.beginPath();
    ctx.arc(...P(t), tip, 0, Math.PI * 2);
    ctx.fill();
  }
}

/** Close-up of the right hand in a corner, so a letter's handshape is easy to read. */
function drawCloseUp(ctx: CanvasRenderingContext2D, pose: Pose, layout: DemoLayout, w: number, h: number, mirror: boolean, pal: Palette) {
  const [rh] = layout.rightHand;
  if (Number.isNaN(pose[rh * 2]!)) return;
  const size = Math.min(w, h) * 0.42;
  const pad = Math.min(w, h) * 0.04;
  const x0 = mirror ? pad : w - size - pad;
  const y0 = pad;
  ctx.fillStyle = pal.body;
  ctx.strokeStyle = pal.bodyLine;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x0, y0, size, size, size * 0.08);
  ctx.fill();
  ctx.stroke();
  // Fit the hand's bounding box into the panel.
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < 21; i++) {
    const x = pose[(rh + i) * 2]!, y = pose[(rh + i) * 2 + 1]!;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x); minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
  const span = Math.max(maxX - minX, maxY - minY, 1e-3);
  const k = (size * 0.74) / span;
  const cx = x0 + size / 2, cy = y0 + size / 2;
  const mx = (minX + maxX) / 2, my = (minY + maxY) / 2;
  const P = (i: number) => [cx + (mirror ? -1 : 1) * (pose[(rh + i) * 2]! - mx) * k, cy + (pose[(rh + i) * 2 + 1]! - my) * k] as const;
  drawHand(ctx, P, pal.right, Math.max(3, size * 0.03), Math.max(4, size * 0.028));
}

function draw(ctx: CanvasRenderingContext2D, pose: Pose, layout: DemoLayout, w: number, h: number, mirror: boolean, pal: Palette, closeUp = false) {
  ctx.clearRect(0, 0, w, h);
  const unit = Math.min(w / 3.2, h / 3.1);
  const cx = w / 2;
  const cy = h * 0.5;
  const X = (i: number) => cx + (mirror ? -1 : 1) * pose[i * 2]! * unit;
  const Y = (i: number) => cy + pose[i * 2 + 1]! * unit;
  const ok = (i: number) => !Number.isNaN(pose[i * 2]!);
  const a = layout.arms[0];
  const [lh] = layout.leftHand;
  const [rh] = layout.rightHand;

  // Torso: a rounded shape hanging from the shoulders.
  if (ok(a) && ok(a + 1)) {
    const lx = X(a), ly = Y(a), rx = X(a + 1), ry = Y(a + 1);
    const pad = unit * 0.12;
    ctx.fillStyle = pal.body;
    ctx.beginPath();
    ctx.moveTo(lx, ly);
    ctx.quadraticCurveTo((lx + rx) / 2, Math.min(ly, ry) - pad * 0.6, rx, ry);
    ctx.lineTo(rx + Math.sign(rx - lx) * pad * 0.4, h + 10);
    ctx.lineTo(lx - Math.sign(rx - lx) * pad * 0.4, h + 10);
    ctx.closePath();
    ctx.fill();
  }

  // Neck and head.
  const [fo0, fo1] = layout.faceOval;
  if (ok(fo0)) {
    let sx = 0, sy = 0, n = 0;
    for (let i = fo0; i < fo1; i++) if (ok(i)) { sx += X(i); sy += Y(i); n++; }
    if (ok(a) && ok(a + 1) && n) {
      ctx.strokeStyle = pal.body;
      ctx.lineWidth = unit * 0.22;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo((X(a) + X(a + 1)) / 2, (Y(a) + Y(a + 1)) / 2);
      ctx.lineTo(sx / n, sy / n);
      ctx.stroke();
    }
    ctx.fillStyle = pal.face;
    ctx.strokeStyle = pal.faceLine;
    ctx.lineWidth = Math.max(1, unit * 0.012);
    ctx.beginPath();
    for (let i = fo0; i < fo1; i++) if (ok(i)) (i === fo0 ? ctx.moveTo : ctx.lineTo).call(ctx, X(i), Y(i));
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    // Eyes, eyebrows and lips as thin outlines.
    for (const [s, e, closed] of [
      [...layout.eyeL, true], [...layout.eyeR, true], [...layout.lips, true], [...layout.browL, false], [...layout.browR, false],
    ] as [number, number, boolean][]) {
      ctx.beginPath();
      let started = false;
      for (let i = s; i < e; i++) {
        if (!ok(i)) continue;
        if (!started) { ctx.moveTo(X(i), Y(i)); started = true; } else ctx.lineTo(X(i), Y(i));
      }
      if (closed) ctx.closePath();
      ctx.stroke();
    }
  }

  // Arms: shoulder -> elbow -> wrist (the hand's own wrist when visible).
  ctx.strokeStyle = pal.bodyLine;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = unit * 0.13;
  for (const [sh, el, wr, hand] of [[a, a + 2, a + 4, lh], [a + 1, a + 3, a + 5, rh]] as const) {
    if (!ok(sh) || !ok(el)) continue;
    const end = ok(hand) ? hand : ok(wr) ? wr : -1;
    ctx.beginPath();
    ctx.moveTo(X(sh), Y(sh));
    ctx.lineTo(X(el), Y(el));
    if (end >= 0) ctx.lineTo(X(end), Y(end));
    ctx.stroke();
  }

  // Hands: filled palm, bones, fingertips.
  for (const [start, color] of [[lh, pal.left], [rh, pal.right]] as const) {
    if (!ok(start)) continue;
    drawHand(ctx, (i) => [X(start + i), Y(start + i)] as const, color, Math.max(2, unit * 0.028), Math.max(2.5, unit * 0.03));
  }
  if (closeUp) drawCloseUp(ctx, pose, layout, w, h, mirror, pal);
}

export function SignPlayer({
  clips,
  neutral,
  layout,
  speed = 1,
  mirror = true,
  loop = false,
  playing = true,
  replayKey = 0,
  onClipChange,
  onEnd,
  label,
}: {
  clips: Clip[];
  neutral: Pose;
  layout: DemoLayout;
  speed?: number;
  mirror?: boolean;
  loop?: boolean;
  playing?: boolean;
  /** Changing this number restarts playback from the beginning. */
  replayKey?: number;
  onClipChange?: (index: number) => void;
  onEnd?: () => void;
  label?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const keys = useMemo(() => buildTimeline(clips, neutral, layout), [clips, neutral, layout]);
  const settings = useRef({ speed, mirror, loop, playing, onClipChange, onEnd });
  settings.current = { speed, mirror, loop, playing, onClipChange, onEnd };

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const palette = readPalette(canvas);
    const scratch = new Float32Array(DEMO_POINTS * 2);
    let k = 0; // current key
    let elapsed = 0; // ms since key k started (at speed 1)
    let last = performance.now();
    let clip = -1;
    let ended = false;
    let frame = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const { width, height } = canvas.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

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
        const prev = k > 0 ? keys[k - 1]!.pose : neutral;
        const t = cur.blend > 0 ? Math.min(1, elapsed / cur.blend) : 1;
        // Ease in and out for letters and transitions; linear between recorded frames.
        const eased = cur.blend > FRAME_MS ? t * t * (3 - 2 * t) : t;
        pose = lerpPose(prev, cur.pose, eased, scratch);
        closeUp = cur.letter;
        if (cur.clip !== clip) {
          clip = cur.clip;
          s.onClipChange?.(clip);
        }
      }
      draw(ctx, pose, layout, canvas.width, canvas.height, s.mirror, palette, closeUp);
      frame = requestAnimationFrame(step);
    };
    frame = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
    };
  }, [keys, neutral, layout, replayKey]);

  return (
    <div className="player">
      <canvas ref={canvasRef} className="player-canvas" role="img" aria-label={label} />
    </div>
  );
}
