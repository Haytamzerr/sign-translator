# %% [markdown]
# # 3D-LEX motion capture: files for the app
#
# Turns the joint positions extracted by lex_extract.py into the app's sign
# format, one small file per sign, loaded only when that sign is shown.
#
# The app's recordings describe a signer as seen by a camera in front of them:
# x to the right of the picture, y down, z away from the camera, in shoulder
# widths, centred between the shoulders. The motion capture is in the room's
# 3D coordinates, so each clip is first turned to face the camera, using the
# actor's own body: "up" from the hips to the head, "left" from the right
# shoulder to the left one. Unlike a camera, motion capture measures depth,
# so the app can use it as is.
#
# Stored per frame (16-bit integers, 1/1000 of a shoulder width):
#   42 hand points (left hand, right hand; MediaPipe's 21-point order),
#   9 body points (nose, shoulders, elbows, wrists, hips),
#   the two eyes and the head's "up" direction, from which the app draws the
#   face points (outline, lips, eyes, brows) with a face template.
#
# Usage: python lex_to_app.py JOINTS_DIR DEMOS_DIR OUT_DIR

# %%
import json, sys
from pathlib import Path
import numpy as np

JOINTS = Path(sys.argv[1]); DEMOS = Path(sys.argv[2]); OUT = Path(sys.argv[3])
(OUT / 'signs').mkdir(parents=True, exist_ok=True)
SCALE = 1000
FPS = 30

FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
def hand_joints(side):
    return [f'{side}Hand'] + [f'{side}Hand{f}{k}' for f in FINGERS for k in range(1, 5)]
BODY = ['Nose', 'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand', 'LeftUpLeg', 'RightUpLeg']

# Names the app's built-in signs use for the same ASL sign.
ALIASES = {
    'mother': ['mom'], 'grandmother': ['grandma'], 'grandfather': ['grandpa'], 'eyes': ['eye'],
    'shoes': ['shoe'], 'policeman': ['police'], 'teeth': ['tooth'], 'cat': ['kitty'], 'my': ['minemy'],
    'we': ['weus'], 'telephone': ['callonphone'],
}

# %%
def to_camera(pos, names):
    """Room coordinates -> picture coordinates (x right, y down, z away), shoulder widths."""
    j = {n: pos[:, i] for i, n in enumerate(names)}
    up = (j['Head'] - j['Hips']).mean(0); up /= np.linalg.norm(up)
    left = (j['LeftArm'] - j['RightArm']).mean(0); left -= up * left.dot(up); left /= np.linalg.norm(left)
    forward = np.cross(left, up)  # the way the actor faces
    center = ((j['LeftArm'] + j['RightArm']) / 2).mean(0)
    # The actor's left appears on the right of the picture; the camera looks at their front.
    basis = np.stack([left, -up, -forward])
    cam = (pos - center) @ basis.T
    width = np.linalg.norm(cam[:, names.index('LeftArm'), :2] - cam[:, names.index('RightArm'), :2], axis=-1).mean()
    return cam / width, basis

def finger_chains(names):
    for side in ('Left', 'Right'):
        for f in FINGERS:
            yield side, [names.index(n) for n in [f'{side}Hand'] + [f'{side}Hand{f}{k}' for k in range(1, 5)]]

def hand_size(pos, names, side):
    return np.linalg.norm(pos[:, names.index(f'{side}HandMiddle1')] - pos[:, names.index(f'{side}Hand')], axis=-1)

def usual_bones(clips):
    """Each finger bone's usual length, in hand sizes, over all recordings (the median)."""
    ratios = {}
    for pos, names in clips:
        for side, chain in finger_chains(names):
            size = np.median(hand_size(pos, names, side))
            for k, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
                ratios.setdefault((side, chain[1], k), []).append(np.median(np.linalg.norm(pos[:, b] - pos[:, a], axis=-1)) / size)
    return {key: float(np.median(v)) for key, v in ratios.items()}

def fix_fingers(pos, names, usual):
    """In some recordings a fingertip marker was lost and the tip sits far from
    its finger, in a few frames or in all of them. A finger bone keeps its
    length, so a bone more than 30% shorter or longer than usual (over all
    recordings, in hand sizes) is repaired: a tip continues the direction of
    the bone before it, at its usual length; an inner joint (rare) takes its
    offset from the nearest good frame."""
    pos = pos.copy()
    for side, chain in finger_chains(names):
        size = np.median(hand_size(pos, names, side))
        for k, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
            want = usual[(side, chain[1], k)] * size
            length = np.linalg.norm(pos[:, b] - pos[:, a], axis=-1)
            bad = np.abs(length - want) > 0.3 * want
            if not bad.any():
                continue
            good = np.where(~bad)[0]
            if b == chain[-1] or not len(good):
                before = pos[:, a] - pos[:, chain[-3] if b == chain[-1] else chain[max(0, k - 1)]]
                before /= np.maximum(np.linalg.norm(before, axis=-1, keepdims=True), 1e-9)
                pos[bad, b] = pos[bad, a] + before[bad] * want
            else:
                for t in np.where(bad)[0]:
                    g = good[np.argmin(np.abs(good - t))]
                    pos[t, b] = pos[t, a] + (pos[g, b] - pos[g, a])
    return pos

def cut_jumps(cam, names):
    """Some recordings end (or start) with a frame where the skeleton snaps back
    to its reference pose. A wrist cannot move half a shoulder width in 1/30 s,
    so the clip is cut at such a jump, keeping the longest part without one."""
    wrists = cam[:, [names.index('LeftHand'), names.index('RightHand')]]
    step = np.linalg.norm(np.diff(wrists, axis=0), axis=-1).max(-1)
    breaks = [0] + [i + 1 for i in np.where(step > 0.5)[0]] + [len(cam)]
    a, b = max(zip(breaks[:-1], breaks[1:]), key=lambda ab: ab[1] - ab[0])
    return cam[a:b]

def trim(cam, names):
    """Drops the still frames before and after the sign (hands resting), keeping a few."""
    wrists = cam[:, [names.index('LeftHand'), names.index('RightHand')]]
    rest = wrists[0]
    moved = np.linalg.norm(wrists - rest, axis=-1).max(-1) > 0.12
    end_rest = wrists[-1]
    moved_end = np.linalg.norm(wrists - end_rest, axis=-1).max(-1) > 0.12
    if not moved.any() or not moved_end.any():
        return cam
    a = max(0, int(np.argmax(moved)) - 4)
    b = min(len(cam), len(cam) - int(np.argmax(moved_end[::-1])) + 4)
    return cam[a:b]

def active_side(cam, names):
    """The hand that moves more (path length of the wrist): the one signing."""
    path = lambda n: np.linalg.norm(np.diff(cam[:, names.index(n)], axis=0), axis=-1).sum()
    return 'Left' if path('LeftHand') > 1.3 * path('RightHand') else 'Right'

def mirror(cam, names):
    """As if signed with the other hand: left and right swap, the picture flips."""
    cam = cam.copy(); cam[..., 0] *= -1
    swapped = [n.replace('Left', '#').replace('Right', 'Left').replace('#', 'Right') for n in names]
    return cam, swapped

def frame_values(cam, names):
    """[T, C] values in the stored order."""
    def P(n): return cam[:, names.index(n)]
    hands = []
    for side in ('Left', 'Right'):
        pts = np.stack([P(n) for n in hand_joints(side)], 1)
        pts = pts.copy(); pts[..., 2] -= pts[:, :1, 2]  # hand depth relative to its wrist, as in the recordings
        hands.append(pts)
    eyes_l, eyes_r = P('LeftEye'), P('RightEye')
    head_up = P('HeadTop_End') - P('Head')
    head_up /= np.linalg.norm(head_up, axis=-1, keepdims=True)
    mid = (eyes_l + eyes_r) / 2
    # Nose tip: below the eyes and in front of them, as on an average face.
    eye_dist = np.linalg.norm(eyes_l - eyes_r, axis=-1, keepdims=True)
    toward_camera = np.cross(eyes_l - eyes_r, head_up); toward_camera /= np.linalg.norm(toward_camera, axis=-1, keepdims=True)
    if toward_camera[:, 2].mean() > 0: toward_camera = -toward_camera
    nose = mid - head_up * eye_dist * 0.75 + toward_camera * eye_dist * 0.6
    body = np.stack([nose] + [P(n) for n in BODY[1:]], 1)
    head = np.concatenate([eyes_l, eyes_r, head_up], -1)
    return np.concatenate([np.concatenate(hands, 1).reshape(len(cam), -1), body.reshape(len(cam), -1), head], -1)

# %%
def face_template(demos_dir):
    """Face points relative to the eyes, in eye distances, from the recorded signs (median face).
    Axes: across (from the right eye to the left eye, as seen), up, and depth."""
    index = json.load(open(demos_dir / 'sign_demos.json'))
    data = np.fromfile(demos_dir / 'sign_demos.bin', '<i2').astype(np.float32)
    L = index['layout']; P = index['points']
    face0 = L['faceOval'][0]
    samples = []
    for s in index['signs']:
        f = data[s['offset']:s['offset'] + P * 3].reshape(P, 3)
        if (f[face0:] == index['missing']).any(): continue
        f = f / index['scale']
        eye_l = f[L['eyeL'][0]:L['eyeL'][1]].mean(0); eye_r = f[L['eyeR'][0]:L['eyeR'][1]].mean(0)
        # The data's 'eyeL' group is the eye on the picture's right (the signer's left eye).
        across = eye_l - eye_r; d = np.linalg.norm(across[:2]); across = across / np.linalg.norm(across)
        up = np.array([across[1], -across[0], 0.0])  # 90 degrees from across in the picture, pointing up (y down)
        if up[1] > 0: up = -up
        depth = np.cross(across, up)
        mid = (eye_l + eye_r) / 2
        rel = f[face0:] - mid
        samples.append(np.stack([rel @ across, rel @ up, rel @ depth], -1) / d)
    return np.median(np.stack(samples), 0)

# %%
if __name__ == '__main__':
    clips = []
    for f in sorted(JOINTS.glob('*.npz')):
        d = np.load(f)
        if len(d['pos']) >= 8:
            clips.append((f.stem, d['pos'].astype(np.float64), [str(n) for n in d['joints']]))
    usual = usual_bones([(p, n) for _, p, n in clips])
    signs = []
    total = 0
    n_mirrored = 0
    skipped = []
    for stem, pos, names in clips:
        cam, _ = to_camera(fix_fingers(pos, names, usual), names)
        cam = trim(cut_jumps(cam, names), names)
        # The app shows every sign with the right hand (the most common); a few
        # recordings were signed with the left hand.
        mirrored = active_side(cam, names) == 'Left'
        if mirrored:
            cam, names = mirror(cam, names)
            n_mirrored += 1
        # A few recordings are cut or frozen (the hands never leave the rest position).
        wrists = cam[:, [names.index('LeftHand'), names.index('RightHand')]]
        if len(cam) < 15 or np.linalg.norm(wrists - wrists[:1], axis=-1).max() < 0.4:
            skipped.append(stem)
            continue
        values = frame_values(cam, names)
        q = np.clip(np.round(values * SCALE), -32767, 32767).astype('<i2')
        q.tofile(OUT / 'signs' / f'{stem}.bin')
        signs.append({'name': stem, 'frames': len(q), 'aliases': ALIASES.get(stem, [])})
        total += q.nbytes
    template = face_template(DEMOS)
    index = {
        'source': '3D-LEX v1.0 (Ranum, Otterspeer, Andersen, Belleman, Roelofsen, 2024), CC BY 4.0, https://osf.io/cv276/',
        'fps': FPS, 'scale': SCALE,
        'frame': {'hands': [0, 42], 'body': [42, 51], 'eyeL': 153, 'eyeR': 156, 'headUp': 159, 'values': 162},
        'faceTemplate': np.round(template, 4).tolist(),
        'signs': signs,
    }
    json.dump(index, open(OUT / 'index.json', 'w'), separators=(',', ':'))
    print(f'{len(signs)} signs ({n_mirrored} mirrored to the right hand), {total / 1e6:.1f} MB; left out (cut or frozen): {skipped}')
