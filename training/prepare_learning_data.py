# %% [markdown]
# # Learning data: sign demonstrations and the fingerspelling alphabet
#
# Produces the data behind the Learn and "Phrase to signs" views, and the
# letter recognizer used to spell names.
#
# Part 1, sign demonstrations: for each of the 250 ASL signs, build one clean
# motion from many real recordings in the Google ISLR / PopSign data:
#   - keep recordings with face, shoulders and hands visible that the trained
#     model recognizes (probability at least 0.4);
#   - un-mirror the selfie videos and mirror left-handed signers, so every
#     demonstration is signed with the right hand;
#   - start from the most typical recording, keep the 20 closest to it (one way
#     of doing the sign, not a blend of variants), and average them after
#     aligning them in time (DTW Barycenter Averaging). Hand shapes are averaged
#     relative to the wrist, so the fingers stay sharp.
#
# Part 2, alphabet: download the ASLNow! fingerspelling landmarks (MIT
# license, recorded with MediaPipe's web hand tracker), train a small letter
# classifier, and export one reference handshape per letter.
#
# Inputs: the "asl-signs" competition data and the output of the training
# notebook (asl_model.onnx). Internet on. CPU is enough.
#
# Output (in /kaggle/working):
#   sign_demos.bin / sign_demos.json   animations of the 250 signs
#   letters_model.json                 letter classifier weights
#   letters_shapes.json                reference handshape per letter
#   letters_parity.json                test data for the app

# %%
import json, math, os, random, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=RuntimeWarning)
OUT = Path(os.environ.get('OUT', '/kaggle/working'))
SEED = 7
random.seed(SEED); np.random.seed(SEED)

def find(name, root=os.environ.get('INPUT', '/kaggle/input')):
    hits = list(Path(root).rglob(name))
    return hits[0] if hits else None

# %% [markdown]
# ## Part 1: sign demonstrations

# %%
# Face points drawn by the avatar (MediaPipe face mesh indices).
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
FACE = FACE_OVAL + LIPS + EYE_L + EYE_R + BROW_L + BROW_R            # 108 points
LEFT_HAND = list(range(468, 489)); POSE0 = 489; RIGHT_HAND = list(range(522, 543))
POSE_ARMS = [POSE0 + i for i in (11, 12, 13, 14, 15, 16)]            # shoulders, elbows, wrists
# Body for the 3D figure: nose, shoulders, elbows, wrists, hips.
POSE_BODY = [POSE0 + i for i in (0, 11, 12, 13, 14, 15, 16, 23, 24)]
# Layout of one demo frame: left hand 21, right hand 21, body 9, face 108 = 159 points, x y z each.
DEMO_POINTS = LEFT_HAND + RIGHT_HAND + POSE_BODY + FACE
MAX_DEMO_FRAMES = 60
TYPICAL_BODY_RATIO = 0.55   # (nose to shoulder line) / shoulder width for a webcam

def run_demos():
    data = find('sign_to_prediction_index_map.json').parent
    model_path = find('asl_model.onnx')
    train = pd.read_csv(data / 'train.csv')
    labels = [s for s, _ in sorted(json.load(open(data / 'sign_to_prediction_index_map.json')).items(), key=lambda kv: kv[1])]
    print(len(train), 'recordings;', 'model:', model_path)

    try:
        import onnxruntime as ort
    except ImportError:
        os.system('pip install -q onnxruntime')
        import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path)) if model_path else None
    cfg_path = find('asl_config.json')
    max_len = json.load(open(cfg_path))['maxLen'] if cfg_path else 48

    # Same input preparation as the training notebook.
    COMPACT = LEFT_HAND + RIGHT_HAND + [POSE0 + i for i in (0, 11, 12, 13, 14, 15, 16)]
    def features(raw):
        shoulders = raw[:, 43:45, :]
        center = np.nanmean(shoulders.reshape(-1, 2), axis=0)
        width = np.nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1))
        if np.any(np.isnan(center)) or not np.isfinite(width) or width < 1e-3:
            center = np.nanmean(raw.reshape(-1, 2), axis=0); width = 0.3
        if np.any(np.isnan(center)): center = np.zeros(2, np.float32)
        glob = (raw - center) / width
        local = []
        for s in (0, 21):
            hand = raw[:, s:s + 21]; wrist = hand[:, :1]
            size = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=-1)[:, None, None]
            size = np.where(np.isfinite(size) & (size > 1e-4), size, 1.0)
            local.append((hand - wrist) / size)
        f = np.concatenate([glob.reshape(len(raw), -1), np.concatenate(local, 1).reshape(len(raw), -1)], 1)
        return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    def prepare(raw):
        keep = ~np.all(np.isnan(raw[:, :42, 0]), axis=1)
        raw = raw[keep] if keep.any() else raw
        f = features(raw); t = len(f)
        if t > max_len:
            idx = np.round(np.linspace(0, t - 1, max_len)).astype(int); return f[idx], np.ones(max_len, np.float32)
        x = np.zeros((max_len, f.shape[1]), np.float32); m = np.zeros(max_len, np.float32); x[:t] = f; m[:t] = 1
        return x, m

    def load(path):
        xy = pd.read_parquet(data / path, columns=['x', 'y']).values.reshape(-1, 543, 2).astype(np.float32)
        return xy

    from joblib import Parallel, delayed
    per_sign = int(os.environ.get('PER_SIGN', 60))
    cands = train.sample(frac=1, random_state=SEED).groupby('sign').head(per_sign)

    def load(path):
        return pd.read_parquet(data / path, columns=['x', 'y', 'z']).values.reshape(-1, 543, 3).astype(np.float32)

    t0 = time.time()
    arrays = Parallel(n_jobs=os.cpu_count(), batch_size=64)(delayed(load)(p) for p in cands.path)
    print(f'loaded {len(arrays)} candidates in {time.time() - t0:.0f} s')

    # 1. Keep recordings that are complete and that the trained model recognizes.
    pool = {}
    for row, raw in zip(cands.itertuples(), arrays):
        xy = raw[..., :2]
        t = len(xy)
        face_ok = np.mean(~np.isnan(xy[:, 1, 0]))
        sh_ok = np.mean(~np.isnan(xy[:, POSE0 + 11, 0]) & ~np.isnan(xy[:, POSE0 + 12, 0]))
        lh = np.mean(~np.isnan(xy[:, 468, 0])); rh = np.mean(~np.isnan(xy[:, 522, 0]))
        if face_ok < 0.9 or sh_ok < 0.9 or max(lh, rh) < 0.6 or not (12 <= t <= 120):
            continue
        x, m = prepare(xy[:, COMPACT, :])
        logits = sess.run(None, {'x': x[None], 'mask': m[None]})[0][0]
        e = np.exp(logits - logits.max()); prob = float(e[labels.index(row.sign)] / e.sum())
        if prob < float(os.environ.get('MIN_PROB', 0.4)):
            continue
        pool.setdefault(row.sign, []).append((prob, raw))
    print('signs with usable recordings:', len(pool), '| missing:', [s for s in labels if s not in pool])

    # Body proportions of the phone videos give their sideways stretch.
    ratios = []
    for recs in pool.values():
        for _, raw in recs[:5]:
            nose, ls, rs = raw[:, POSE0, :2], raw[:, POSE0 + 11, :2], raw[:, POSE0 + 12, :2]
            w = np.linalg.norm(ls - rs, axis=-1); h = (ls[:, 1] + rs[:, 1]) / 2 - nose[:, 1]
            r = h / w; ratios.extend(r[np.isfinite(r) & (r > 0.1) & (r < 2)].tolist())
    x_fix = float(np.median(ratios)) / TYPICAL_BODY_RATIO
    print(f'body ratio {np.median(ratios):.3f} -> x correction {x_fix:.3f}')

    face0 = 21 * 2 + len(POSE_BODY)
    BODY_SL = slice(42, face0)

    def normalize_rec(raw):
        """Shoulder-centred, shoulder-width units; hand and face depth relative to themselves."""
        pts = raw[:, DEMO_POINTS, :].copy()
        pts[..., 0] *= x_fix
        pts[..., 2] *= x_fix
        ls, rs = pts[:, 43, :], pts[:, 44, :]
        center = np.nanmean((ls + rs) / 2, axis=0)
        width = np.nanmean(np.linalg.norm(ls[:, :2] - rs[:, :2], axis=-1))
        pts[..., :2] = (pts[..., :2] - center[:2]) / width
        pts[:, BODY_SL, 2] = (pts[:, BODY_SL, 2] - center[2]) / width
        pts[:, :42, 2] /= width
        pts[:, face0:, 2] /= width
        pts[..., 2] = np.clip(pts[..., 2], -2.0, 2.0)
        return pts

    # Mirroring: every point takes the place of its mirror image. Body points
    # swap sides; eye and brow groups swap; the face outline and the lips are
    # rings, and each point swaps with the one at the same place on the other
    # side (the outline starts at the top of the forehead, the lips at a corner).
    # Without this, averaging mirrored and unmirrored recordings would squash
    # the face to a vertical line.
    body_swap = [0, 2, 1, 4, 3, 6, 5, 8, 7]
    no, nl = len(FACE_OVAL), len(LIPS)
    fl = no + nl
    eyes = list(range(fl, fl + 16)); eyes_r = list(range(fl + 16, fl + 32))
    brows = list(range(fl + 32, fl + 42)); brows_r = list(range(fl + 42, fl + 52))
    face_perm = list(range(len(FACE)))
    for a_, b_ in list(zip(eyes, eyes_r)) + list(zip(brows, brows_r)):
        face_perm[a_], face_perm[b_] = b_, a_
    for i in range(no):
        face_perm[i] = (no - i) % no
    half = nl // 2  # LIPS: corner, lower lip to the other corner, upper lip back
    for i in range(nl):
        face_perm[no + i] = no + (half - i if i <= half else 3 * half - i)
    PERM = list(range(21, 42)) + list(range(0, 21)) + [42 + i for i in body_swap] + [face0 + i for i in face_perm]

    def mirror(pts):
        out = pts[:, PERM, :].copy()
        out[..., 0] *= -1
        return out

    def dominant_slot(pts):
        """The hand that moves and shows the most (0: left-hand slot, 1: right-hand slot)."""
        scores = []
        for s in (0, 21):
            w = pts[:, s, :2]; ok = ~np.isnan(w[:, 0])
            move = np.nansum(np.linalg.norm(np.diff(w, axis=0), axis=-1)) if ok.sum() > 1 else 0
            scores.append(ok.mean() + move)
        return int(np.argmax(scores))

    def fill(series):
        """Fills missing frames with the nearest seen value (for alignment only)."""
        out = series.copy(); ok = ~np.isnan(out[:, 0])
        if not ok.any():
            return np.zeros_like(out)
        idx = np.where(ok, np.arange(len(out)), -1); np.maximum.accumulate(idx, out=idx)
        first = np.argmax(ok); idx[idx < 0] = first
        return out[idx]

    def align_features(pts):
        dom = fill(pts[:, 21, :2]); off = fill(pts[:, 0, :2])
        shape = pts[:, 21:42, :2] - pts[:, 21:22, :2]
        size = np.linalg.norm(shape[:, 9], axis=-1)[:, None, None]
        shape = fill((shape / np.where(size > 1e-4, size, 1)).reshape(len(pts), -1))
        return np.concatenate([dom * 2.0, off, shape * 0.15], 1).astype(np.float64)

    try:
        from numba import njit
    except ImportError:
        os.system('pip install -q numba'); from numba import njit

    @njit(cache=True)
    def dtw_path(a, b, band):
        n, m = len(a), len(b)
        cost = np.full((n + 1, m + 1), np.inf); cost[0, 0] = 0.0
        for i in range(1, n + 1):
            lo = max(1, int(i * m / n) - band); hi = min(m, int(i * m / n) + band)
            for j in range(lo, hi + 1):
                d = 0.0
                for k in range(a.shape[1]):
                    t = a[i - 1, k] - b[j - 1, k]; d += t * t
                cost[i, j] = np.sqrt(d) + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
        i, j = n, m; path = []
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            c = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            if c[0] <= c[1] and c[0] <= c[2]: i -= 1; j -= 1
            elif c[1] <= c[2]: i -= 1
            else: j -= 1
        return cost[n, m] / (n + m), path[::-1]

    def average(recs, iters=6):
        """DTW Barycenter Averaging over the full point arrays."""
        feats = [align_features(p) for p in recs]
        band = 20
        # Start from the most typical recording (smallest total distance to the others).
        k = len(recs)
        dist = np.zeros((k, k))
        for i in range(k):
            for j in range(i + 1, k):
                dist[i, j] = dist[j, i] = dtw_path(feats[i], feats[j], band)[0]
        medoid = int(np.argmin(dist.sum(1)))
        # Keep the recordings closest to it: one way of doing the sign, not a blend of variants.
        keep = list(np.argsort(dist[medoid])[:min(k, 20)])
        recs = [recs[i] for i in keep]; feats = [feats[i] for i in keep]
        avg = recs[0].copy() if keep[0] == medoid else recs[keep.index(medoid)].copy()
        avg_f = align_features(avg)
        for _ in range(iters):
            sums = np.zeros(avg.shape, np.float64); counts = np.zeros(avg.shape[:2], np.float64)
            shape_sum = np.zeros((len(avg), 2, 21, 3)); size_sum = np.zeros((len(avg), 2)); shape_n = np.zeros((len(avg), 2))
            for p, f in zip(recs, feats):
                _, path = dtw_path(avg_f, f, band)
                for i, j in path:
                    frame = p[j]; ok = ~np.isnan(frame[:, 0])
                    sums[i][ok] += frame[ok]; counts[i][ok] += 1
                    # Hands: average the shape (relative to the wrist, unit size) and the size separately.
                    for h, s0 in enumerate((0, 21)):
                        if ok[s0] and ok[s0 + 9]:
                            rel = frame[s0:s0 + 21] - frame[s0]
                            size = np.linalg.norm(rel[9])
                            if size > 1e-4:
                                shape_sum[i, h] += rel / size; size_sum[i, h] += size; shape_n[i, h] += 1
            new = np.where(counts[..., None] > 0, sums / np.maximum(counts[..., None], 1), np.nan)
            for h, s0 in enumerate((0, 21)):
                # A hand is shown where most aligned recordings have it.
                for i in range(len(avg)):
                    if shape_n[i, h] > 0 and shape_n[i, h] >= 0.4 * counts[i].max():
                        wrist = new[i, s0]
                        new[i, s0:s0 + 21] = wrist + shape_sum[i, h] / shape_n[i, h] * (size_sum[i, h] / shape_n[i, h])
                    else:
                        new[i, s0:s0 + 21] = np.nan
            avg = new.astype(np.float32); avg_f = align_features(avg)
        return avg, len(recs)

    from scipy.signal import savgol_filter

    def smooth(series):
        """Fills gaps of up to 8 frames, then removes the remaining jitter."""
        out = series.copy(); t = np.arange(len(series)); ok = ~np.isnan(series)
        if ok.sum() >= 2:
            filled = np.interp(t, t[ok], series[ok])
            i = 0
            while i < len(out):
                if not ok[i]:
                    j = i
                    while j < len(out) and not ok[j]: j += 1
                    if i > 0 and j < len(out) and j - i <= 8: out[i:j] = filled[i:j]
                    i = j
                else:
                    i += 1
        good = ~np.isnan(out); start = None
        for i in range(len(out) + 1):
            if i < len(out) and good[i]:
                if start is None: start = i
            elif start is not None:
                if i - start >= 9: out[start:i] = savgol_filter(out[start:i], 9, 2)
                start = None
        return out

    # Most recordings use one hand slot as the dominant hand; mirror the others to it.
    doms = [dominant_slot(normalize_rec(raw)) for recs in pool.values() for _, raw in recs[:3]]
    DOM = 1 if np.mean(doms) >= 0.5 else 0
    print('dominant hand slot:', 'right_hand' if DOM else 'left_hand', f'({np.mean(doms):.2f})')

    index = []; chunks = []; offset = 0; t0 = time.time()
    for sign in labels:
        if sign not in pool:
            continue
        recs = sorted(pool[sign], key=lambda r: -r[0])[:40]
        normed = []
        for _, raw in recs:
            pts = normalize_rec(raw)
            if dominant_slot(pts) != DOM:
                pts = mirror(pts)
            if DOM == 0:
                pts = mirror(pts)  # the app expects the dominant hand in the right-hand slot
            normed.append(pts)
        if len(normed) >= 3:
            pts, used = average(normed)
        else:
            pts, used = normed[0], 1
        for k in range(pts.shape[1]):
            for c in range(3):
                pts[:, k, c] = smooth(pts[:, k, c])
        if len(pts) > MAX_DEMO_FRAMES:
            pts = pts[np.round(np.linspace(0, len(pts) - 1, MAX_DEMO_FRAMES)).astype(int)]
        q = np.where(np.isnan(pts), -32768, np.clip(np.round(pts * 1000), -32767, 32767)).astype('<i2')
        chunks.append(q.tobytes())
        prob = float(np.mean([r[0] for r in recs]))
        index.append({'label': sign, 'offset': offset, 'frames': int(len(pts)), 'confidence': round(prob, 3), 'recordings': int(used)})
        offset += q.size
        if len(index) % 25 == 0:
            print(f'{len(index)} signs averaged ({time.time() - t0:.0f} s)', flush=True)
    with open(OUT / 'sign_demos.bin', 'wb') as f:
        for c in chunks: f.write(c)
    b = 42
    json.dump({'points': len(DEMO_POINTS), 'dims': 3, 'scale': 1000, 'missing': -32768, 'fps': 30,
               'layout': {'leftHand': [0, 21], 'rightHand': [21, 42], 'body': [b, b + 9],
                          'faceOval': [b + 9, b + 45], 'lips': [b + 45, b + 65], 'eyeL': [b + 65, b + 81], 'eyeR': [b + 81, b + 97],
                          'browL': [b + 97, b + 107], 'browR': [b + 107, b + 117]},
               'body': ['nose', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip'],
               'signs': index}, open(OUT / 'sign_demos.json', 'w'))
    print('sign_demos.bin', round(os.path.getsize(OUT / 'sign_demos.bin') / 1e6, 2), 'MB')

# %% [markdown]
# ## Part 2: fingerspelling alphabet

# %%
LETTERS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]

def as_points(obj):
    """Finds 21 landmarks with x, y (and z) inside any JSON structure."""
    if isinstance(obj, list) and len(obj) == 21:
        if all(isinstance(p, dict) and 'x' in p and 'y' in p for p in obj):
            return [[float(p['x']), float(p['y']), float(p.get('z', 0.0))] for p in obj]
        if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in obj):
            return [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in obj]
    if isinstance(obj, list) and len(obj) in (42, 63) and all(isinstance(v, (int, float)) for v in obj):
        k = len(obj) // 21
        return [[float(obj[i * k]), float(obj[i * k + 1]), float(obj[i * k + 2]) if k == 3 else 0.0] for i in range(21)]
    if isinstance(obj, dict):
        for key in ('landmarks', 'hand', 'keypoints', 'points', 'data'):
            if key in obj:
                r = as_points(obj[key])
                if r: return r
        for v in obj.values():
            r = as_points(v)
            if r: return r
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, dict)):
        r = as_points(obj[0])
        if r: return r
    return None

def all_point_sets(obj, depth=0):
    """Every set of 21 landmarks inside a JSON structure (a file can hold a whole recording)."""
    pts = as_points(obj) if not isinstance(obj, dict) else None
    if pts and isinstance(obj, list) and len(obj) in (21, 42, 63):
        yield pts
        return
    if depth > 6:
        return
    if isinstance(obj, dict):
        for v in obj.values():
            yield from all_point_sets(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, (list, dict)):
                yield from all_point_sets(v, depth + 1)

def find_label(obj, path):
    if isinstance(obj, dict):
        for key in ('label', 'letter', 'class', 'sign', 'y'):
            v = obj.get(key)
            if isinstance(v, str) and v.strip().upper() in LETTERS: return v.strip().upper()
    for part in reversed(path.parts[:-1]):
        if part.strip().upper() in LETTERS: return part.strip().upper()
    stem = path.stem.split('_')[0].split('-')[0].upper()
    return stem if stem in LETTERS else None

def label_names():
    """Class names when the label column is stored as numbers (Hugging Face ClassLabel)."""
    import requests
    try:
        info = requests.get('https://datasets-server.huggingface.co/info?dataset=sid220/asl-now-fingerspelling', timeout=30).json()
        for cfg in info.get('dataset_info', {}).values():
            for name, feat in cfg.get('features', {}).items():
                if isinstance(feat, dict) and feat.get('names'):
                    print('label column', name, feat['names'][:30]); return name, feat['names']
    except Exception as e:
        print('no label info:', e)
    return None, None

def records_to_samples(records, path, names_col=None, names=None):
    X, Y, G = [], [], []
    for rec in records:
        rec = {k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in rec.items()}
        if names and names_col in rec and isinstance(rec[names_col], int):
            rec[names_col] = names[rec[names_col]]
        pts, lab = as_points(rec), find_label(rec, path)
        if pts and lab:
            X.append(pts); Y.append(lab); G.append(str(rec.get('user', rec.get('participant', rec.get('session', 'all')))))
    return X, Y, G

def load_alphabet():
    X, Y, G = [], [], []
    if os.environ.get('ASLNOW_DIR'):
        root = Path(os.environ['ASLNOW_DIR'])
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        repo = 'sid220/asl-now-fingerspelling'
        print('downloading the alphabet data', flush=True)
        # Hugging Face keeps a Parquet copy of every dataset: one file instead of thousands.
        try:
            files = [f for f in list_repo_files(repo, repo_type='dataset', revision='refs/convert/parquet') if f.endswith('.parquet')]
            print('parquet files:', files, flush=True)
            col, names = label_names()
            for f in files:
                local = Path(hf_hub_download(repo, f, repo_type='dataset', revision='refs/convert/parquet'))
                df = pd.read_parquet(local)
                print('columns:', list(df.columns), '| first row:', str(df.iloc[0].to_dict())[:400], flush=True)
                x, y, g = records_to_samples(df.to_dict('records'), local, col, names)
                X += x; Y += y; G += g
        except Exception as e:
            print('parquet route failed:', e, flush=True)
        if X:
            print('samples:', len(X), '| per letter:', pd.Series(Y).value_counts().sort_index().to_dict())
            return np.array(X, np.float32), np.array(Y), np.array(G)
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(repo, repo_type='dataset', local_dir='/kaggle/temp/aslnow', max_workers=16))
    files = [p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.json', '.parquet', '.jsonl')]
    print(len(files), 'data files; examples:', [str(p.relative_to(root)) for p in files[:5]], flush=True)
    for p in files:
        try:
            if p.suffix == '.parquet':
                x, y, g = records_to_samples(pd.read_parquet(p).to_dict('records'), p)
                X += x; Y += y; G += g
                continue
            text = p.read_text()
            items = [json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix == '.jsonl' else [json.loads(text)]
            if not X:
                first = items[0]
                print('structure of', p.name, ':', type(first).__name__,
                      list(first.keys())[:10] if isinstance(first, dict) else len(first), flush=True)
            for obj in items:
                lab = find_label(obj if isinstance(obj, dict) else {}, p)
                if not lab: continue
                for pts in all_point_sets(obj):
                    X.append(pts); Y.append(lab); G.append(p.stem)
        except Exception as e:
            print('skip', p.name, e)
    print('samples:', len(X), '| per letter:', pd.Series(Y).value_counts().sort_index().to_dict())
    return np.array(X, np.float32), np.array(Y), np.array(G)

ASPECT = 4 / 3   # webcam images; x is multiplied by this to get square units

def normalize(points, aspect=ASPECT):
    """Same as normalizeHand() in the app: wrist at origin, divided by wrist-to-middle-knuckle distance."""
    p = points.copy(); p[..., 0] *= aspect; p[..., 2] *= aspect
    p = p - p[..., :1, :]
    size = np.linalg.norm(p[..., 9, :], axis=-1)[..., None, None]
    return p / np.maximum(size, 1e-6)

def augment(points):
    p = points.copy()
    if random.random() < 0.5: p[:, 0] = 1 - p[:, 0]                       # other hand
    stretch = random.uniform(0.8, 1.25); p[:, 0] = 0.5 + (p[:, 0] - 0.5) * stretch
    a = math.radians(random.uniform(-15, 15)); c, s = math.cos(a), math.sin(a)
    cx, cy = p[:, 0].mean(), p[:, 1].mean()
    x, y = p[:, 0] - cx, p[:, 1] - cy
    p[:, 0], p[:, 1] = cx + c * x - s * y, cy + s * x + c * y
    p[:, :2] += np.random.normal(0, 0.002, (21, 2))
    return p

def run_letters():
    import torch, torch.nn as nn
    X, Y, G = load_alphabet()
    if len(X) == 0:
        print('No alphabet data found.'); return
    letters = sorted(str(l) for l in set(Y)); yi = np.array([letters.index(l) for l in Y])
    groups = np.unique(G)
    if len(groups) >= 5:
        rng = np.random.default_rng(SEED); val_groups = set(rng.choice(groups, max(1, len(groups) // 6), replace=False))
        val = np.isin(G, list(val_groups))
    else:
        val = np.random.default_rng(SEED).random(len(X)) < 0.15
    print('groups:', len(groups), '| validation samples:', int(val.sum()))

    def batch(idx, aug):
        pts = np.stack([augment(X[i]) if aug else X[i] for i in idx])
        return torch.from_numpy(normalize(pts).reshape(len(idx), -1).astype(np.float32)), torch.from_numpy(yi[idx])

    model = nn.Sequential(nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.15), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, len(letters)))
    opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=1e-4)
    tr = np.where(~val)[0]; va = np.where(val)[0]
    epochs = 40
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * math.ceil(len(tr) / 256))
    for ep in range(epochs):
        model.train(); np.random.shuffle(tr)
        for i in range(0, len(tr), 256):
            xb, yb = batch(tr[i:i + 256], True)
            loss = nn.functional.cross_entropy(model(xb), yb, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                xb, yb = batch(va, False); acc = (model(xb).argmax(1) == yb).float().mean().item()
            print(f'epoch {ep + 1} loss {loss.item():.3f} val accuracy {acc:.3f}')
    model.eval()
    layers = [m for m in model if isinstance(m, nn.Linear)]
    json.dump({'labels': letters, 'aspect': ASPECT, 'valAccuracy': round(acc, 4),
               'layers': [{'w': l.weight.detach().numpy().round(4).tolist(), 'b': l.bias.detach().numpy().round(4).tolist()} for l in layers]},
              open(OUT / 'letters_model.json', 'w'))

    # Reference handshape: the sample closest to the average of its letter (right hand, as recorded).
    shapes = {}
    norm = normalize(X)
    for li, l in enumerate(letters):
        idx = np.where(yi == li)[0]
        mean = norm[idx].mean(0)
        best = idx[np.argmin(np.linalg.norm((norm[idx] - mean).reshape(len(idx), -1), axis=1))]
        shapes[l] = np.round(norm[best], 4).tolist()
    json.dump(shapes, open(OUT / 'letters_shapes.json', 'w'))

    sample = np.random.default_rng(1).choice(va if len(va) else np.arange(len(X)), 20, replace=False)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(normalize(X[sample]).reshape(20, -1))), 1).numpy()
    json.dump([{'points': X[i].round(6).tolist(), 'label': str(Y[i]), 'probs': p.round(5).tolist()} for i, p in zip(sample, probs)],
              open(OUT / 'letters_parity.json', 'w'))
    print('letters done:', letters)

# %%
PARTS = os.environ.get('PARTS', 'alphabet,demos').split(',')
for name, fn in (('alphabet', run_letters), ('demos', run_demos)):
    if name not in PARTS: continue
    try:
        t0 = time.time(); fn(); print(f'{name} finished in {time.time() - t0:.0f} s')
    except Exception as e:
        import traceback; traceback.print_exc(); print(f'{name} FAILED: {e}')
print(sorted(os.listdir(OUT)))
