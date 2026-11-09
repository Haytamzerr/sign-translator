/**
 * A small drawing of a sign, built from the stored hand points (no photo is
 * ever stored). Moving signs also show the path of the wrist as dots.
 */

import { useEffect, useRef } from 'react';
import { HAND_BONES } from '../core/tracker';
import type { Frame } from '../core/types';
import { hasHands } from '../core/features';

const COLORS = ['#f2b544', '#5ec8e0'];

export function SignThumb({ sample, moving, size = 56 }: { sample: Frame[] | undefined; moving: boolean; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx || !sample) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const frames = sample.filter(hasHands);
    const key = frames[Math.floor(frames.length / 2)];
    if (!key) return;

    // Everything is placed in "hand size" units around the key frame's first hand.
    const ref0 = key.hands[0] ?? key.hands[1]!;
    const toUnits = (pos: [number, number]) => [(pos[0] - ref0.pos[0]) / ref0.scale, (pos[1] - ref0.pos[1]) / ref0.scale];

    const hands = key.hands.map((h) => {
      if (!h) return null;
      const [ox, oy] = toUnits(h.pos);
      const pts: [number, number][] = [];
      for (let i = 0; i < 21; i++) pts.push([ox! + h.shape[i * 3]!, oy! + h.shape[i * 3 + 1]!]);
      return pts;
    });
    const path: [number, number][][] = [0, 1].map((s) =>
      moving ? frames.flatMap((f) => (f.hands[s] ? [toUnits(f.hands[s]!.pos) as [number, number]] : [])) : [],
    );

    const all = [...hands.flatMap((p) => p ?? []), ...path.flat()];
    const xs = all.map((p) => p[0]);
    const ys = all.map((p) => p[1]);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const span = Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY, 1e-3);
    const pad = size * 0.12;
    const k = (size - pad * 2) / span;
    const offX = pad + (size - pad * 2 - (Math.max(...xs) - minX) * k) / 2;
    const offY = pad + (size - pad * 2 - (Math.max(...ys) - minY) * k) / 2;
    // Mirror horizontally so the drawing matches the mirrored camera view.
    const X = (x: number) => size - (offX + (x - minX) * k);
    const Y = (y: number) => offY + (y - minY) * k;

    path.forEach((points, s) => {
      ctx.fillStyle = COLORS[s]!;
      ctx.globalAlpha = 0.45;
      points.forEach(([x, y], i) => {
        if (i % 2) return;
        ctx.beginPath();
        ctx.arc(X(x), Y(y), 1.2, 0, Math.PI * 2);
        ctx.fill();
      });
    });
    ctx.globalAlpha = 1;
    hands.forEach((pts, s) => {
      if (!pts) return;
      ctx.strokeStyle = COLORS[s]!;
      ctx.lineWidth = 1.6;
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (const [a, b] of HAND_BONES) {
        ctx.moveTo(X(pts[a]![0]), Y(pts[a]![1]));
        ctx.lineTo(X(pts[b]![0]), Y(pts[b]![1]));
      }
      ctx.stroke();
    });
  }, [sample, moving, size]);

  return <canvas ref={ref} className="thumb" style={{ width: size, height: size }} aria-hidden="true" />;
}
