/**
 * Quality checks of the signer's movement on all 250 recorded signs, with the
 * real model: no part of the arms or hands inside the body, joints within
 * their range, motion without jolts, and hands that reach the face where the
 * sign touches it.
 */
import { readFileSync } from 'node:fs';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { beforeAll, describe, expect, it } from 'vitest';
import { decodeDemos, measureRig, type DemoLayout, type Pose, type Rig } from '../core/learn/demos';
import { prepareFrames, solveFigure } from '../core/learn/figure';
import { decodeMocapFrame } from '../core/learn/mocap';
import { HAND_POINT_BONE, SignerRig } from './signerRig';

const index = JSON.parse(readFileSync('public/models/learn/sign_demos.json', 'utf8'));
const buf = readFileSync('public/models/learn/sign_demos.bin');
const demos = decodeDemos(index, new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2));
const layout: DemoLayout = index.layout;
const rigMeasure: Rig = measureRig(demos.values(), layout);

// Motion-captured signs (3D-LEX), decoded as the app does.
const mocapIndex = JSON.parse(readFileSync('public/models/mocap/index.json', 'utf8'));
function mocapFrames(name: string): Pose[] {
  const b = readFileSync(`public/models/mocap/signs/${name}.bin`);
  const values = new Int16Array(b.buffer, b.byteOffset, b.byteLength / 2);
  const out: Pose[] = [];
  for (let f = 0; f + mocapIndex.frame.values <= values.length; f += mocapIndex.frame.values) out.push(decodeMocapFrame(values, f, mocapIndex, layout));
  return out;
}

let model: THREE.Group;
beforeAll(async () => {
  const glb = readFileSync('public/models/avatar/signer.glb');
  const gltf = await new GLTFLoader().parseAsync(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength), '');
  model = gltf.scene;
});

const FRAME_MS = 1000 / 30;
const DRAW_MS = 1000 / 60;

/** Plays a recording at 60 frames per second and calls `check` after each drawn frame. */
function play(rig: SignerRig, frames: Pose[], check: (t: number) => void, measured = false): void {
  const prepared = prepareFrames(frames, layout);
  const scratch = new Float32Array(prepared[0]!.length);
  const total = (prepared.length - 1) * FRAME_MS;
  for (let t = 0, first = true; t <= total; t += DRAW_MS, first = false) {
    const f = t / FRAME_MS;
    const a = prepared[Math.floor(f)]!;
    const b = prepared[Math.min(prepared.length - 1, Math.floor(f) + 1)]!;
    const k = f - Math.floor(f);
    for (let i = 0; i < scratch.length; i++) scratch[i] = Number.isNaN(b[i]!) ? (k < 0.5 ? a[i]! : NaN) : Number.isNaN(a[i]!) ? b[i]! : a[i]! + (b[i]! - a[i]!) * k;
    rig.update(solveFigure(scratch, layout, rigMeasure, measured), DRAW_MS, first);
    model.updateMatrixWorld(true);
    check(t);
  }
}

const bone = (name: string) => model.getObjectByName(name) as THREE.Bone;
const pos = (name: string) => bone(name).getWorldPosition(new THREE.Vector3());

/** Points along the arms and hands as shown, with their thickness. */
function armPoints(side: 'L' | 'R'): [THREE.Vector3, number][] {
  const out: [THREE.Vector3, number][] = [];
  const e = pos(`lowerarm01_${side}`);
  const w = pos(`wrist_${side}`);
  for (const t of [0.25, 0.5, 0.75]) out.push([e.clone().lerp(w, t), 0.025]);
  out.push([w, 0.02]);
  for (const f of ['finger1-3', 'finger2-3', 'finger3-3', 'finger4-3', 'finger5-3', 'finger2-1', 'finger5-1']) out.push([pos(`${f}_${side}`), 0.005]);
  return out;
}

describe('signer movement', () => {
  it('keeps arms and hands out of the body, and moves without jolts', () => {
    const rig = new SignerRig(model);
    let frames = 0;
    let inside = 0;
    let worstDepth = 0;
    let worstSign = '';
    const accels: [number, string][] = [];
    for (const label of demos.keys()) {
      let prev: THREE.Vector3[] | null = null;
      let prevV: THREE.Vector3[] | null = null;
      play(rig, demos.get(label)!.frames, (t) => {
        frames++;
        let deepest = 0;
        for (const side of ['L', 'R'] as const) for (const [p, r] of armPoints(side)) deepest = Math.max(deepest, rig.depthInside(p, r));
        if (deepest > 0.01) inside++;
        if (deepest > worstDepth) [worstDepth, worstSign] = [deepest, label];
        const now = [pos('wrist_L'), pos('wrist_R')];
        if (prev) {
          const v = now.map((p, i) => p.clone().sub(prev![i]!).divideScalar(DRAW_MS / 1000));
          // The first moments catch up with the sign's first pose (in the app, a blend from the rest pose).
          if (prevV && t > 300) accels.push([Math.max(...v.map((x, i) => x.clone().sub(prevV![i]!).length() / (DRAW_MS / 1000))), label]);
          prevV = v;
        }
        prev = now;
      });
    }
    accels.sort((a, b) => a[0] - b[0]);
    const p99 = accels[Math.floor(accels.length * 0.99)]![0];
    const [worstAccel, accelSign] = accels[accels.length - 1]!;
    console.log(`frames ${frames}; more than 1 cm inside the body: ${((inside / frames) * 100).toFixed(2)}%; deepest ${(worstDepth * 100).toFixed(1)} cm (${worstSign}); wrist acceleration: 99% of frames below ${p99.toFixed(1)} m/s², highest ${worstAccel.toFixed(1)} (${accelSign})`);
    expect(inside / frames).toBeLessThan(0.005);
    expect(p99).toBeLessThan(40);
    expect(worstAccel).toBeLessThan(200);
  }, 300_000);

  it('touches the face where signs touch it', () => {
    const rig = new SignerRig(model);
    const report: string[] = [];
    let worst = 0;
    for (const label of ['mom', 'nose', 'think', 'lips', 'eye', 'chin', 'cheek', 'tooth'].filter((l) => demos.has(l))) {
      let nearest = Infinity;
      play(rig, demos.get(label)!.frames, (t) => {
        const touch = rig.touches.R;
        if (!touch || t < 300) return;
        const [name, at] = HAND_POINT_BONE[touch.point]!;
        const b = bone(`${name}_R`);
        const p = b.getWorldPosition(new THREE.Vector3());
        if (at === 'end') {
          const { head, tail } = b.userData as { head: number[]; tail: number[] };
          const len = Math.hypot(tail[0]! - head[0]!, tail[1]! - head[1]!, tail[2]! - head[2]!);
          const dir = new THREE.Vector3(tail[0]! - head[0]!, tail[1]! - head[1]!, tail[2]! - head[2]!).normalize();
          p.add(dir.applyQuaternion(b.getWorldQuaternion(new THREE.Quaternion())).multiplyScalar(len));
        }
        nearest = Math.min(nearest, p.distanceTo(touch.at));
      });
      report.push(`${label} ${Number.isFinite(nearest) ? (nearest * 100).toFixed(1) + ' cm' : 'no touch'}`);
      worst = Math.max(worst, nearest);
    }
    console.log(`closest approach of the touching point to its place on the face: ${report.join(', ')}`);
    expect(worst).toBeLessThan(0.02);
  });

  it('plays the motion-captured signs cleanly', () => {
    const rig = new SignerRig(model);
    let frames = 0;
    let inside = 0;
    let worstDepth = 0;
    let worstSign = '';
    const accels: number[] = [];
    for (const { name } of mocapIndex.signs as { name: string }[]) {
      let prev: THREE.Vector3[] | null = null;
      let prevV: THREE.Vector3[] | null = null;
      play(rig, mocapFrames(name), (t) => {
        frames++;
        let deepest = 0;
        for (const side of ['L', 'R'] as const) for (const [p, r] of armPoints(side)) deepest = Math.max(deepest, rig.depthInside(p, r));
        if (deepest > 0.01) inside++;
        if (deepest > worstDepth) [worstDepth, worstSign] = [deepest, name];
        const now = [pos('wrist_L'), pos('wrist_R')];
        if (prev) {
          const v = now.map((p, i) => p.clone().sub(prev![i]!).divideScalar(DRAW_MS / 1000));
          if (prevV && t > 300) accels.push(Math.max(...v.map((x, i) => x.clone().sub(prevV![i]!).length() / (DRAW_MS / 1000))));
          prevV = v;
        }
        prev = now;
      }, true);
    }
    accels.sort((a, b) => a - b);
    const p99 = accels[Math.floor(accels.length * 0.99)]!;
    console.log(`motion capture: ${mocapIndex.signs.length} signs, ${frames} frames; more than 1 cm inside the body: ${((inside / frames) * 100).toFixed(2)}%; deepest ${(worstDepth * 100).toFixed(1)} cm (${worstSign}); wrist acceleration: 99% below ${p99.toFixed(1)} m/s², highest ${accels.at(-1)!.toFixed(1)}`);
    expect(inside / frames).toBeLessThan(0.005);
    expect(p99).toBeLessThan(40);
  }, 600_000);
});
