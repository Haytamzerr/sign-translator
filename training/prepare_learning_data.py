# %% [markdown]
# # Learning data: sign demonstrations and the fingerspelling alphabet
#
# Produces the data behind the Learn and "Phrase to signs" views, and the
# letter recognizer used to spell names.
#
# Part 1, sign demonstrations: for each of the 250 ASL signs, choose the
# clearest real recording in the Google ISLR / PopSign data (face, shoulders
# and hands visible, recognized most confidently by the trained model) and pack
# it as a small animation.
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
# Layout of one demo frame: left hand 21, right hand 21, arms 6, face 108 = 156 points.
DEMO_POINTS = LEFT_HAND + RIGHT_HAND + POSE_ARMS + FACE
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
    per_sign = 30
    cands = train.sample(frac=1, random_state=SEED).groupby('sign').head(per_sign)
    t0 = time.time()
    arrays = Parallel(n_jobs=os.cpu_count(), batch_size=64)(delayed(load)(p) for p in cands.path)
    print(f'loaded {len(arrays)} candidates in {time.time() - t0:.0f} s')

    chosen = {}
    for (row, xy) in zip(cands.itertuples(), arrays):
        t = len(xy)
        face_ok = np.mean(~np.isnan(xy[:, 1, 0]))
        sh_ok = np.mean(~np.isnan(xy[:, POSE0 + 11, 0]) & ~np.isnan(xy[:, POSE0 + 12, 0]))
        lh = np.mean(~np.isnan(xy[:, 468, 0])); rh = np.mean(~np.isnan(xy[:, 522, 0]))
        if face_ok < 0.9 or sh_ok < 0.9 or max(lh, rh) < 0.6 or not (12 <= t <= 120):
            continue
        prob = 0.5
        if sess is not None:
            x, m = prepare(xy[:, COMPACT, :])
            logits = sess.run(None, {'x': x[None], 'mask': m[None]})[0][0]
            e = np.exp(logits - logits.max()); prob = float(e[labels.index(row.sign)] / e.sum())
        score = prob + 0.3 * max(lh, rh) + 0.1 * face_ok
        if row.sign not in chosen or score > chosen[row.sign][0]:
            chosen[row.sign] = (score, prob, xy)
    missing = [s for s in labels if s not in chosen]
    print('demos chosen:', len(chosen), '| missing:', missing)

    # Body proportions of the chosen recordings give the x stretch of the phone videos.
    ratios = []
    for _, _, xy in chosen.values():
        nose, ls, rs = xy[:, POSE0], xy[:, POSE0 + 11], xy[:, POSE0 + 12]
        w = np.linalg.norm(ls - rs, axis=-1); h = (ls[:, 1] + rs[:, 1]) / 2 - nose[:, 1]
        r = h / w; ratios.extend(r[np.isfinite(r) & (r > 0.1) & (r < 2)].tolist())
    x_fix = float(np.median(ratios)) / TYPICAL_BODY_RATIO   # multiply x by this to get square units
    print(f'body ratio {np.median(ratios):.3f} -> x correction {x_fix:.3f}')

    index = []; chunks = []; offset = 0
    for sign in labels:
        if sign not in chosen:
            continue
        _, prob, xy = chosen[sign]
        pts = xy[:, DEMO_POINTS, :].copy()
        pts[..., 0] *= x_fix
        ls, rs = pts[:, 42], pts[:, 43]
        center = np.nanmean((ls + rs) / 2, axis=0)
        width = np.nanmean(np.linalg.norm(ls - rs, axis=-1))
        pts = (pts - center) / width
        if len(pts) > MAX_DEMO_FRAMES:
            pts = pts[np.round(np.linspace(0, len(pts) - 1, MAX_DEMO_FRAMES)).astype(int)]
        q = np.where(np.isnan(pts), -32768, np.clip(np.round(pts * 1000), -32767, 32767)).astype('<i2')
        chunks.append(q.tobytes())
        index.append({'label': sign, 'offset': offset, 'frames': int(len(pts)), 'confidence': round(prob, 3)})
        offset += q.size
    with open(OUT / 'sign_demos.bin', 'wb') as f:
        for c in chunks: f.write(c)
    json.dump({'points': len(DEMO_POINTS), 'scale': 1000, 'missing': -32768, 'fps': 30,
               'layout': {'leftHand': [0, 21], 'rightHand': [21, 42], 'arms': [42, 48],
                          'faceOval': [48, 84], 'lips': [84, 104], 'eyeL': [104, 120], 'eyeR': [120, 136],
                          'browL': [136, 146], 'browR': [146, 156]},
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
        shapes[l] = np.round(norm[best][:, :2], 4).tolist()
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
