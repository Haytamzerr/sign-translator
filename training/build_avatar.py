"""
Builds the 3D signer used by Learn mode (public/models/avatar/signer.glb).

Source: the MakeHuman 2 assets (https://github.com/makehumancommunity/makehuman2),
released under CC0: the base human mesh, body-shape targets, the low-poly
eyes, the default skeleton (163 bones, every finger joint) and its skin weights.

What it does:
1. Loads the base mesh and blends body-shape targets into it: an adult male
   (mixed phenotype), athletic build.
2. Places every bone from the mesh's joint helpers (a joint = the centre of a
   small group of vertices, so joints follow the body shape).
3. Keeps only the body surface and the eyes, and attaches each vertex to at
   most 4 bones with the published skin weights.
4. Writes a binary glTF (.glb) that three.js loads directly. Bones have no rest
   rotation (only positions), which makes driving them from recorded points
   simple; each bone's rest direction (head -> tail) is stored in its extras.

Usage:  python training/build_avatar.py /path/to/makehuman2 public/models/avatar/signer.glb
"""

import json
import os
import struct
import sys

import numpy as np

MH = sys.argv[1] if len(sys.argv) > 1 else 'makehuman2'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'public/models/avatar/signer.glb'
DATA = os.path.join(MH, 'data')

# Body shape: weights of MakeHuman's shape targets (each moves vertices from the base).
TARGETS = {
    'macro_age/african-male-young.target': 1 / 3,
    'macro_age/asian-male-young.target': 1 / 3,
    'macro_age/caucasian-male-young.target': 1 / 3,
    'macro_muscle/male-young-maxmuscle-averageweight.target': 0.45,
}

# Bones the app drives, plus their parents up to the root. Others are merged into their parent.
KEEP_PREFIXES = ('root', 'spine', 'neck', 'head', 'clavicle', 'shoulder01', 'upperarm', 'lowerarm', 'wrist',
                 'metacarpal', 'finger', 'pelvis', 'upperleg', 'breast', 'jaw', 'eye.')


def load_obj(path):
    verts, faces, groups = [], [], []
    group = None
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('g '):
                group = line.split()[1]
            elif line.startswith('f '):
                idx = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
                faces.append(idx)
                groups.append(group)
    return np.array(verts, np.float64), faces, groups


def apply_target(verts, path, weight):
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith('#'):
                continue
            i, dx, dy, dz = line.split()
            verts[int(i)] += weight * np.array([float(dx), float(dy), float(dz)])


def load_proxy(path, base_verts):
    """A MakeHuman proxy (.mhclo): each vertex follows a triangle of the base mesh."""
    obj_file, refs, scales = None, [], {}
    in_verts = False
    with open(path) as f:
        for line in f:
            s = line.split()
            if not s or line.startswith('#'):
                continue
            if s[0] == 'obj_file':
                obj_file = s[1]
            elif s[0] in ('x_scale', 'y_scale', 'z_scale'):
                scales[s[0][0]] = (int(s[1]), int(s[2]), float(s[3]))
            elif s[0] == 'verts':
                in_verts = True
            elif in_verts and len(s) in (1, 9):
                refs.append(s)
            elif in_verts:
                in_verts = False
    scale = []
    for axis, k in (('x', 0), ('y', 1), ('z', 2)):
        a, b, ref = scales[axis]
        scale.append(abs(base_verts[a][k] - base_verts[b][k]) / ref)
    pverts = []
    for r in refs:
        if len(r) == 1:
            pverts.append(base_verts[int(r[0])])
        else:
            v = [int(x) for x in r[:3]]
            w = [float(x) for x in r[3:6]]
            d = np.array([float(x) for x in r[6:9]]) * scale
            pverts.append(w[0] * base_verts[v[0]] + w[1] * base_verts[v[1]] + w[2] * base_verts[v[2]] + d)
    pv, pf, _ = load_obj(os.path.join(os.path.dirname(path), obj_file))
    return np.array(pverts), pf


def triangulate(faces):
    tris = []
    for f in faces:
        for k in range(1, len(f) - 1):
            tris.append([f[0], f[k], f[k + 1]])
    return np.array(tris, np.uint32)


def vertex_normals(v, tris):
    n = np.zeros_like(v)
    fn = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    for k in range(3):
        np.add.at(n, tris[:, k], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def main():
    base_dir = os.path.join(DATA, 'base', 'hm08')
    verts, faces, groups = load_obj(os.path.join(base_dir, 'base.obj'))
    for rel, w in TARGETS.items():
        apply_target(verts, os.path.join(DATA, 'target', 'hm08', rel), w)

    # Skeleton: joint positions from helper vertex groups.
    skel = json.load(open(os.path.join(DATA, 'rigs', 'hm08', 'default.mhskel')))
    joint_pos = {name: verts[idx].mean(axis=0) for name, idx in skel['joints'].items()}
    bones = skel['bones']

    def kept(name):
        return name.startswith(KEEP_PREFIXES)

    def kept_ancestor(name):
        while name is not None and not kept(name):
            name = bones[name]['parent']
        return name

    order = []  # parents before children

    def visit(name):
        if name in order:
            return
        parent = bones[name]['parent']
        if parent is not None and kept(parent):
            visit(parent)
        order.append(name)

    for name in bones:
        if kept(name):
            visit(name)
    index = {name: i for i, name in enumerate(order)}

    # Weights: each vertex keeps its 4 strongest bones (weights of dropped bones go to their kept parent).
    weights_file = json.load(open(os.path.join(DATA, 'rigs', 'hm08', 'default_weights.mhw')))['weights']
    per_vertex = [dict() for _ in range(len(verts))]
    for bone, pairs in weights_file.items():
        target = kept_ancestor(bone)
        if target is None:
            continue
        for vi, w in pairs:
            per_vertex[vi][target] = per_vertex[vi].get(target, 0) + w

    # Body surface only (no helper geometry), re-indexed.
    body_faces = [f for f, g in zip(faces, groups) if g == 'body']
    used = sorted({i for f in body_faces for i in f})
    remap = {old: new for new, old in enumerate(used)}
    body_v = verts[used]
    body_t = triangulate([[remap[i] for i in f] for f in body_faces])

    def skin_of(vertex_ids):
        joints = np.zeros((len(vertex_ids), 4), np.uint16)
        wts = np.zeros((len(vertex_ids), 4), np.float32)
        for k, vi in enumerate(vertex_ids):
            items = sorted(per_vertex[vi].items(), key=lambda kv: -kv[1])[:4]
            if not items:
                items = [('spine01', 1.0)]
            total = sum(w for _, w in items)
            for j, (b, w) in enumerate(items):
                joints[k, j] = index[b]
                wts[k, j] = w / total
        return joints, wts

    body_j, body_w = skin_of(used)

    # Eyes: a proxy fitted to the face, attached to the head.
    eye_v, eye_f = load_proxy(os.path.join(DATA, 'eyes', 'hm08', 'low-poly', 'low-poly.mhclo'), verts)
    eye_t = triangulate(eye_f)
    eye_j = np.zeros((len(eye_v), 4), np.uint16)
    eye_j[:, 0] = index['head']
    eye_w = np.zeros((len(eye_v), 4), np.float32)
    eye_w[:, 0] = 1

    # MakeHuman units are decimetres; the app works in metres.
    unit = 0.1
    meshes = [
        ('body', body_v * unit, body_t, body_j, body_w),
        ('eyes', eye_v * unit, eye_t, eye_j, eye_w),
    ]

    # ---------- binary glTF ----------
    blob = bytearray()
    views, accessors = [], []

    def add(array, target=None, comp=None, typ=None, minmax=False):
        data = array.tobytes()
        while len(blob) % 4:
            blob.append(0)
        offset = len(blob)
        blob.extend(data)
        view = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(data)}
        if target:
            view['target'] = target
        views.append(view)
        acc = {'bufferView': len(views) - 1, 'componentType': comp, 'count': int(array.shape[0]), 'type': typ}
        if minmax:
            acc['min'] = array.min(axis=0).tolist()
            acc['max'] = array.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    gl_meshes = []
    for name, v, t, j, w in meshes:
        v = v.astype(np.float32)
        n = vertex_normals(v.astype(np.float64), t).astype(np.float32)
        prim = {
            'attributes': {
                'POSITION': add(v, 34962, 5126, 'VEC3', True),
                'NORMAL': add(n, 34962, 5126, 'VEC3'),
                'JOINTS_0': add(j, 34962, 5123, 'VEC4'),
                'WEIGHTS_0': add(w, 34962, 5126, 'VEC4'),
            },
            'indices': add(t.reshape(-1).astype(np.uint32), 34963, 5125, 'SCALAR'),
            'material': 0 if name == 'body' else 1,
        }
        gl_meshes.append({'name': name, 'primitives': [prim]})

    heads = {b: joint_pos[bones[b]['head']] * unit for b in order}
    tails = {b: joint_pos[bones[b]['tail']] * unit for b in order}
    nodes = []
    for b in order:
        parent = bones[b]['parent']
        p = heads[parent] if parent in index else np.zeros(3)
        nodes.append({
            'name': b.replace('.', '_'),  # loaders drop dots from names
            'translation': (heads[b] - p).tolist(),
            'extras': {'head': heads[b].tolist(), 'tail': tails[b].tolist()},
        })
    for b in order:
        children = [index[c] for c in order if bones[c]['parent'] == b]
        if children:
            nodes[index[b]]['children'] = children
    ibm = np.stack([np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-heads[b][0], -heads[b][1], -heads[b][2], 1]], np.float32)
                    for b in order]).reshape(len(order), 16)
    skin = {'joints': list(range(len(order))), 'inverseBindMatrices': add(ibm, None, 5126, 'MAT4'), 'skeleton': index['root']}
    first_mesh = len(nodes)
    for k in range(len(gl_meshes)):
        nodes.append({'name': gl_meshes[k]['name'], 'mesh': k, 'skin': 0})
    roots = [index[b] for b in order if bones[b]['parent'] not in index]

    gltf = {
        'asset': {'version': '2.0', 'generator': 'build_avatar.py', 'copyright': 'MakeHuman assets, CC0 1.0'},
        'scene': 0,
        'scenes': [{'nodes': roots + [first_mesh, first_mesh + 1]}],
        'nodes': nodes,
        'meshes': gl_meshes,
        'skins': [skin],
        'materials': [
            {'name': 'skin', 'pbrMetallicRoughness': {'baseColorFactor': [0.25, 0.27, 0.31, 1], 'metallicFactor': 0, 'roughnessFactor': 0.62}},
            {'name': 'eyes', 'pbrMetallicRoughness': {'baseColorFactor': [0.25, 0.27, 0.31, 1], 'metallicFactor': 0, 'roughnessFactor': 0.4}},
        ],
        'accessors': accessors,
        'bufferViews': views,
        'buffers': [{'byteLength': len(blob)}],
    }
    js = json.dumps(gltf, separators=(',', ':')).encode()
    while len(js) % 4:
        js += b' '
    while len(blob) % 4:
        blob.append(0)
    os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
    with open(OUT, 'wb') as f:
        f.write(struct.pack('<III', 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        f.write(struct.pack('<II', len(js), 0x4E4F534A) + js)
        f.write(struct.pack('<II', len(blob), 0x004E4942) + bytes(blob))
    print(f'{OUT}: {len(order)} bones, {len(body_v)} body vertices, {len(body_t)} triangles, '
          f'{os.path.getsize(OUT) / 1e6:.2f} MB, height {np.ptp(body_v[:, 1]) * unit:.2f} m')


if __name__ == '__main__':
    main()
