# %% [markdown]
# # 3D-LEX motion capture: joint positions
#
# 3D-LEX v1.0 (Ranum et al., 2024, CC BY 4.0) recorded 1,000 ASL signs with
# optical motion capture for the body and data gloves for the fingers. Its
# animation files (glTF binary, .glb) hold a skeleton and, for each joint, its
# rotation over time. This script plays each animation (forward kinematics:
# each joint's position = its parent's position + the parent's rotation applied
# to the bone) and saves the 3D positions of the joints the app uses.
#
# Input:  a folder of .glb files (ASL_ANIM_GLB.zip from osf.io/cv276, unzipped)
# Output: one compressed .npz per sign with positions (frames x joints x 3),
#         head rotations (frames x 4), the frame rate and the joint names.
#
# Only NumPy is needed.

# %%
import json, os, struct, sys
from pathlib import Path
import numpy as np

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else 'ANIM_GLB')
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else 'lex_joints')
OUT.mkdir(parents=True, exist_ok=True)

FINGERS = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
JOINTS = ['Hips', 'Spine2', 'Neck', 'Head', 'HeadTop_End', 'LeftEye', 'RightEye',
          'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
          'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
          'LeftUpLeg', 'RightUpLeg', 'LeftFoot', 'LeftToeBase']
for side in ('Left', 'Right'):
    for f in FINGERS:
        for k in range(1, 5):
            JOINTS.append(f'{side}Hand{f}{k}')

# %%
def read_glb(path):
    data = path.read_bytes()
    n = struct.unpack('<I', data[12:16])[0]
    gltf = json.loads(data[20:20 + n])
    binary = data[20 + n + 8:]
    return gltf, binary

COMPONENTS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
TYPES = {5126: np.float32, 5123: np.uint16, 5125: np.uint32, 5121: np.uint8, 5122: np.int16, 5120: np.int8}

def accessor(gltf, binary, i):
    a = gltf['accessors'][i]
    view = gltf['bufferViews'][a['bufferView']]
    dtype = TYPES[a['componentType']]
    k = COMPONENTS[a['type']]
    start = view.get('byteOffset', 0) + a.get('byteOffset', 0)
    stride = view.get('byteStride')
    item = np.dtype(dtype).itemsize * k
    if stride and stride != item:
        raw = np.frombuffer(binary, np.uint8, count=stride * a['count'], offset=start).reshape(a['count'], stride)[:, :item]
        out = np.frombuffer(raw.tobytes(), dtype).reshape(a['count'], k)
    else:
        out = np.frombuffer(binary, dtype, count=a['count'] * k, offset=start).reshape(a['count'], k)
    out = out.astype(np.float64)
    if a.get('normalized') and dtype != np.float32:
        out /= np.iinfo(dtype).max
    return out

def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 3], a[..., 0], a[..., 1], a[..., 2]
    w2, x2, y2, z2 = b[..., 3], b[..., 0], b[..., 1], b[..., 2]
    return np.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                     w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)

def quat_apply(q, v):
    u = q[..., :3]; w = q[..., 3:4]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)

def slerp(q0, q1, t):
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1); d = np.abs(d)
    lin = q0 + (q1 - q0) * t
    theta = np.arccos(np.clip(d, -1, 1)); s = np.sin(theta)
    out = np.where(s > 1e-6, (np.sin((1 - t) * theta) * q0 + np.sin(t * theta) * q1) / np.where(s > 1e-6, s, 1), lin)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)

def sample(times, values, at, path):
    """Linear interpolation of a channel at the given times (slerp for rotations)."""
    i = np.clip(np.searchsorted(times, at, side='right') - 1, 0, len(times) - 1)
    j = np.clip(i + 1, 0, len(times) - 1)
    span = np.where(times[j] > times[i], times[j] - times[i], 1)
    t = np.clip((at - times[i]) / span, 0, 1)[:, None]
    if path == 'rotation':
        return slerp(values[i], values[j], t)
    return values[i] + (values[j] - values[i]) * t

# %%
def extract(path, fps=30.0):
    gltf, binary = read_glb(path)
    if not gltf.get('animations'):
        return None
    nodes = gltf['nodes']
    names = [n.get('name', '') for n in nodes]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
    # Files converted from FBX have two animations; the skeleton's has the most channels.
    anim = max(gltf['animations'], key=lambda a: len(a['channels']))
    end = max(accessor(gltf, binary, s['input']).max() for s in anim['samplers'])
    at = np.arange(0, end + 1e-6, 1 / fps)
    T = len(at)
    # Rest transforms, replaced by the animated ones where a channel exists.
    trans = np.array([n.get('translation', [0, 0, 0]) for n in nodes], np.float64)[None].repeat(T, 0)
    rot = np.array([n.get('rotation', [0, 0, 0, 1]) for n in nodes], np.float64)[None].repeat(T, 0)
    scale = np.array([n.get('scale', [1, 1, 1]) for n in nodes], np.float64)[None].repeat(T, 0)
    for ch in anim['channels']:
        s = anim['samplers'][ch['sampler']]
        node, p = ch['target']['node'], ch['target']['path']
        times = accessor(gltf, binary, s['input'])[:, 0]
        values = accessor(gltf, binary, s['output'])
        v = sample(times, values, at, p)
        {'translation': trans, 'rotation': rot, 'scale': scale}[p][:, node] = v
    # Forward kinematics, from the root down.
    order = []
    def visit(i):
        order.append(i)
        for c in nodes[i].get('children', []): visit(c)
    for i in range(len(nodes)):
        if i not in parent: visit(i)
    wpos = np.zeros((T, len(nodes), 3)); wrot = np.zeros((T, len(nodes), 4)); wscale = np.ones((T, len(nodes), 3))
    for i in order:
        if i in parent:
            p = parent[i]
            wpos[:, i] = wpos[:, p] + quat_apply(wrot[:, p], trans[:, i] * wscale[:, p])
            wrot[:, i] = quat_mul(wrot[:, p], rot[:, i])
            wscale[:, i] = wscale[:, p] * scale[:, i]
        else:
            wpos[:, i] = trans[:, i]; wrot[:, i] = rot[:, i]; wscale[:, i] = scale[:, i]
    idx = [names.index(j) for j in JOINTS]
    return {'pos': wpos[:, idx].astype(np.float32), 'head': wrot[:, names.index('Head')].astype(np.float32), 'fps': fps}

# %%
if __name__ == '__main__':
    files = sorted(SRC.glob('*.glb'))
    done = skipped = 0
    for f in files:
        if (OUT / (f.stem + '.npz')).exists():
            done += 1
            continue
        r = extract(f)
        if r is None:
            skipped += 1
            continue
        np.savez_compressed(OUT / (f.stem + '.npz'), pos=r['pos'], head=r['head'], fps=r['fps'], joints=np.array(JOINTS))
        done += 1
        if done % 100 == 0:
            print(done, 'signs', flush=True)
    print(f'{done} signs extracted, {skipped} files without animation')
