# %% [markdown]
# # ASL sign recognizer: training
#
# Trains the neural network that lets the app recognize 250 American Sign
# Language words without any teaching.
#
# Data: Google "Isolated Sign Language Recognition" (PopSign ASL), about 94,000
# recordings of 250 signs by 21 signers, already converted to MediaPipe points.
#
# Output (in /kaggle/working):
# - asl_model.onnx    the trained network, run by the app in the browser
# - asl_labels.json   the 250 words, in the order of the network's outputs
# - asl_parity.json   test data proving the app prepares input exactly like here
#
# Works with or without a GPU. With a GPU it takes about 30-60 minutes; on CPU
# a smaller network is trained within a time budget (default 7 hours).

# %%
import json, math, os, random, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed

def find_data():
    # The competition folder can be mounted at different paths; find it by its files.
    if os.environ.get('ASL_DATA'):
        return Path(os.environ['ASL_DATA'])
    for csv in Path('/kaggle/input').rglob('train.csv'):
        if (csv.parent / 'sign_to_prediction_index_map.json').exists():
            return csv.parent
    raise FileNotFoundError('Add the "asl-signs" competition data to this notebook (Add Input).')

DATA = find_data()
OUT = Path(os.environ.get('ASL_OUT', '/kaggle/working'))
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
warnings.filterwarnings('ignore', category=RuntimeWarning)  # 'mean of empty slice' when a part is missing
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE, '| torch', torch.__version__)

# %% [markdown]
# ## 1. Which points we use
#
# MediaPipe gives 543 points per frame (468 face, 33 body, 2 x 21 hands).
# We keep 49: both hands (42) and 7 upper-body points (nose, shoulders,
# elbows, wrists). The body points tell the network WHERE the hands are
# (near the face, at the chest...), which matters a lot in sign language.
#
# Compact layout used everywhere (here AND in the app):
#   0-20  left hand, 21-41 right hand, 42-48 body points below.

# %%
POSE_KEEP = [0, 11, 12, 13, 14, 15, 16]          # nose, shoulders, elbows, wrists
LEFT_HAND = list(range(468, 489))
POSE = list(range(489, 522))
RIGHT_HAND = list(range(522, 543))
COMPACT = LEFT_HAND + RIGHT_HAND + [POSE[i] for i in POSE_KEEP]
N_POINTS = len(COMPACT)                           # 49
MAX_LEN = 48                                      # frames given to the network

# Body pairs swapped when mirroring (index inside the 7 body points).
POSE_SWAP = [0, 2, 1, 4, 3, 6, 5]

# %% [markdown]
# ## 2. Preparing a recording (feature extraction)
#
# This function is written again, line by line, in TypeScript in the app
# (src/core/asl/features.ts). Both must stay identical; asl_parity.json checks it.
#
# For every frame:
# - global: every point relative to the middle of the shoulders, divided by
#   shoulder width (so distance to the camera and position do not matter)
# - local: each hand relative to its own wrist, divided by its own size
#   (the handshape, independent of where the hand is)
# Missing hands become zeros.

# %%
N_FEATURES = N_POINTS * 2 + 2 * 21 * 2            # 98 + 84 = 182

def drop_empty_frames(raw):
    """Keeps only frames where at least one hand is visible."""
    hands = raw[:, :42, 0]
    keep = ~np.all(np.isnan(hands), axis=1)
    return raw[keep] if keep.any() else raw

def features(raw):
    """raw: [T, 49, 2] x,y in image units, NaN = missing -> [T, 182]."""
    raw = raw.astype(np.float32)
    shoulders = raw[:, 43:45, :]                               # body points 1 and 2
    with np.errstate(all='ignore'):
        center = np.nanmean(shoulders.reshape(-1, 2), axis=0)
        width = np.nanmean(np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1))
    if np.any(np.isnan(center)) or not np.isfinite(width) or width < 1e-3:
        with np.errstate(all='ignore'):
            center = np.nanmean(raw.reshape(-1, 2), axis=0)
        width = 0.3
    if np.any(np.isnan(center)):
        center = np.zeros(2, np.float32)
    glob = (raw - center) / width                               # [T, 49, 2]

    local = []
    for start in (0, 21):
        hand = raw[:, start:start + 21, :]
        wrist = hand[:, 0:1, :]
        size = np.linalg.norm(hand[:, 9, :] - hand[:, 0, :], axis=-1)[:, None, None]
        size = np.where(np.isfinite(size) & (size > 1e-4), size, 1.0)
        local.append((hand - wrist) / size)
    feats = np.concatenate([glob.reshape(len(raw), -1), np.concatenate(local, 1).reshape(len(raw), -1)], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

def fit_length(feats):
    """Long recordings are resampled to MAX_LEN frames; short ones are padded. Returns (x, mask)."""
    t = len(feats)
    if t > MAX_LEN:
        idx = np.round(np.linspace(0, t - 1, MAX_LEN)).astype(int)
        return feats[idx], np.ones(MAX_LEN, np.float32)
    x = np.zeros((MAX_LEN, N_FEATURES), np.float32)
    m = np.zeros(MAX_LEN, np.float32)
    x[:t] = feats; m[:t] = 1
    return x, m

def prepare(raw):
    return fit_length(features(drop_empty_frames(raw)))

# %% [markdown]
# ## 3. Loading the recordings

# %%
train = pd.read_csv(DATA / 'train.csv')
label_map = json.load(open(DATA / 'sign_to_prediction_index_map.json'))
labels = [s for s, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
print(len(train), 'recordings,', len(labels), 'signs,', train.participant_id.nunique(), 'signers')

def load(path):
    df = pd.read_parquet(DATA / path, columns=['x', 'y'])
    data = df.values.reshape(-1, 543, 2)[:, COMPACT, :].astype(np.float32)
    return data.astype(np.float16)

t0 = time.time()
RAW = Parallel(n_jobs=os.cpu_count(), batch_size=256)(delayed(load)(p) for p in train.path)
Y = train.sign.map(label_map).values
print(f'Loaded in {time.time() - t0:.0f} s')

# %% [markdown]
# ## 4. Augmentation
#
# Each time a recording is used for training it is changed slightly: mirrored
# (left-handed signers), rotated, stretched, shifted, sped up or slowed down,
# with frames dropped. The network then learns the sign, not one exact video.
# The x/y stretch also covers different camera shapes (phone vs webcam).

# %%
def augment(raw):
    raw = raw.astype(np.float32).copy()
    t = len(raw)
    if random.random() < 0.5:                                   # mirror
        raw[..., 0] = -raw[..., 0]
        raw = np.concatenate([raw[:, 21:42], raw[:, 0:21], raw[:, 42:][:, POSE_SWAP]], axis=1)
    if random.random() < 0.8:                                   # rotate, scale, stretch, shift
        a = math.radians(random.uniform(-15, 15))
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
        s = random.uniform(0.85, 1.15)
        stretch = np.array([random.uniform(0.75, 1.25), 1.0], np.float32)
        with np.errstate(all='ignore'):
            c = np.nanmean(raw.reshape(-1, 2), axis=0)
        if np.all(np.isfinite(c)):
            raw = ((raw - c) @ rot.T) * s * stretch + c + np.random.uniform(-0.05, 0.05, 2).astype(np.float32)
    if t > 4 and random.random() < 0.6:                         # speed change
        speed = random.uniform(0.7, 1.3)
        n = max(3, int(round(t / speed)))
        raw = raw[np.round(np.linspace(0, t - 1, n)).astype(int)]
    if len(raw) > 8 and random.random() < 0.3:                  # drop frames
        keep = np.random.rand(len(raw)) > 0.15
        if keep.sum() > 4: raw = raw[keep]
    if len(raw) > 10 and random.random() < 0.3:                 # trim start or end
        a = random.randint(0, len(raw) // 8); b = len(raw) - random.randint(0, len(raw) // 8)
        raw = raw[a:b]
    if random.random() < 0.5:                                   # each hand turned and resized a little
        for start in (0, 21):
            hand = raw[:, start:start + 21]
            wrist = hand[:, :1]
            if np.isnan(wrist).all(): continue
            a = math.radians(random.uniform(-12, 12)); s = random.uniform(0.9, 1.1)
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32)
            raw[:, start:start + 21] = ((hand - wrist) @ rot.T) * s + wrist
    if len(raw) > 12 and random.random() < 0.3:                 # hide a short stretch of time
        n = random.randint(1, max(1, len(raw) // 8)); a = random.randint(0, len(raw) - n)
        raw[a:a + n] = np.nan
    if random.random() < 0.15:                                  # hide the body points
        raw[:, 42:] = np.nan
    return raw

class SignData(torch.utils.data.Dataset):
    def __init__(self, idx, train):
        self.idx, self.train = idx, train
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        raw = RAW[j].astype(np.float32)
        if self.train: raw = augment(raw)
        x, m = prepare(raw)
        return torch.from_numpy(x), torch.from_numpy(m), int(Y[j])

# Validation: 2 signers never seen in training (honest accuracy for new people).
signers = sorted(train.participant_id.unique())
rng = random.Random(SEED)
val_signers = set(rng.sample(signers, 2))
val_idx = np.where(train.participant_id.isin(val_signers))[0]
tr_idx = np.where(~train.participant_id.isin(val_signers))[0]
print(len(tr_idx), 'training /', len(val_idx), 'validation recordings')

# %% [markdown]
# ## 5. The network
#
# A small and fast design (about 1.6 million numbers to learn):
# - convolution blocks look at short windows of frames (local motion)
# - attention blocks let every frame look at every other frame (whole sign)
# - the result is averaged over time and turned into 250 scores
# Only standard operations, so it exports cleanly to the browser.

# %%
# Smaller network on CPU so training finishes in a few hours. ASL_DIM and
# ASL_DEPTH make it wider (numbers per frame) and deeper (blocks).
D = int(os.environ.get('ASL_DIM', 192 if DEVICE == 'cuda' else 128))
DEPTH = int(os.environ.get('ASL_DEPTH', 2))
DROP_PATH = float(os.environ.get('ASL_DROP_PATH', 0.0))

def drop_path(h, training):
    """Stochastic depth: during training a block is sometimes skipped for a
    whole recording, so the network cannot rely on any single block."""
    if not training or DROP_PATH <= 0:
        return h
    keep = (torch.rand(h.shape[0], 1, 1, device=h.device) > DROP_PATH).to(h.dtype)
    return h * keep / (1 - DROP_PATH)

class ConvBlock(nn.Module):
    def __init__(self, d, k=17, expand=2, drop=0.2):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.up = nn.Linear(d, d * expand)
        self.dw = nn.Conv1d(d * expand, d * expand, k, padding=k // 2, groups=d * expand)
        self.down = nn.Linear(d * expand, d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, m):
        h = F.silu(self.up(self.ln(x))) * m[..., None]
        h = F.silu(self.dw(h.transpose(1, 2)).transpose(1, 2))
        return (x + drop_path(self.drop(self.down(h)), self.training)) * m[..., None]

class AttnBlock(nn.Module):
    def __init__(self, d, heads=4, drop=0.2):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d); self.ff1 = nn.Linear(d, 2 * d); self.ff2 = nn.Linear(2 * d, d)
        self.drop = nn.Dropout(drop)
    def forward(self, x, m):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // self.h)
        att = att + (1.0 - m[:, None, None, :]) * -1e4          # ignore padding frames
        h = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + drop_path(self.drop(self.proj(h)), self.training)
        x = x + drop_path(self.drop(self.ff2(F.silu(self.ff1(self.ln2(x))))), self.training)
        return x * m[..., None]

class SignNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(N_FEATURES, D), nn.LayerNorm(D))
        self.blocks = nn.ModuleList([blk for _ in range(DEPTH) for blk in (ConvBlock(D), ConvBlock(D), ConvBlock(D), AttnBlock(D))])
        self.head = nn.Sequential(nn.LayerNorm(D), nn.Dropout(0.4), nn.Linear(D, n_classes))
    def forward(self, x, mask):
        x = self.stem(x) * mask[..., None]
        for blk in self.blocks:
            x = blk(x, mask)
        pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        return self.head(pooled)

model = SignNet(len(labels)).to(DEVICE)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# %% [markdown]
# ## 6. Training

# %%
EPOCHS = int(os.environ.get('ASL_EPOCHS', 80))
# Training stops when the time budget is used up; the learning rate follows the
# time used, so the run always ends with a properly finished schedule.
BUDGET_S = float(os.environ.get('ASL_BUDGET_HOURS', 10 if DEVICE == 'cuda' else 7)) * 3600
if DEVICE == 'cpu':
    torch.set_num_threads(max(1, os.cpu_count() - 1))
BATCH = int(os.environ.get('ASL_BATCH', 128))
workers = min(8, os.cpu_count()) if DEVICE == 'cuda' else 1
tr_loader = torch.utils.data.DataLoader(SignData(tr_idx, True), BATCH, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=True)
va_loader = torch.utils.data.DataLoader(SignData(val_idx, False), 256, num_workers=workers)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
steps = EPOCHS * len(tr_loader)
MAX_LR = 1e-3
def set_lr(progress):
    # 5% warm-up, then cosine decay to zero.
    lr = MAX_LR * progress / 0.05 if progress < 0.05 else MAX_LR * 0.5 * (1 + math.cos(math.pi * (progress - 0.05) / 0.95))
    for g in opt.param_groups: g['lr'] = lr
scaler = torch.amp.GradScaler(enabled=DEVICE == 'cuda')
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)

# A slowly moving average of the weights (EMA) is evaluated and kept: it is
# smoother than the weights at any single step and usually generalizes better.
import copy
ema = copy.deepcopy(model).eval()
for p_ in ema.parameters(): p_.requires_grad_(False)
EMA_DECAY = float(os.environ.get('ASL_EMA', 0.999))
ema_steps = 0
def update_ema():
    # Early on the average follows faster, so it does not drag the random
    # starting weights along.
    global ema_steps
    ema_steps += 1
    decay = min(EMA_DECAY, (1 + ema_steps) / (10 + ema_steps))
    with torch.no_grad():
        for e, p_ in zip(ema.parameters(), model.parameters()):
            e.mul_(decay).add_(p_.detach(), alpha=1 - decay)

def evaluate():
    model.eval(); correct = total = top5 = 0
    with torch.no_grad():
        for x, m, y in va_loader:
            x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
            logits = ema(x, m)
            correct += (logits.argmax(1) == y).sum().item()
            top5 += (logits.topk(min(5, logits.shape[1]), 1).indices == y[:, None]).any(1).sum().item()
            total += len(y)
    model.train()
    return correct / total, top5 / total

best = -1.0
start = time.time(); step = 0; done = False
for epoch in range(EPOCHS):
    t0 = time.time(); running = 0.0; n = 0
    for x, m, y in tr_loader:
        progress = max(step / steps, (time.time() - start) / BUDGET_S)
        if progress >= 1: done = True; break
        set_lr(progress); step += 1; n += 1
        x, m, y = x.to(DEVICE), m.to(DEVICE), y.to(DEVICE)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=DEVICE == 'cuda'):
            loss = loss_fn(model(x, m), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        update_ema()
        running += loss.item()
    if n == 0: break
    acc, acc5 = evaluate()
    if acc > best:
        best = acc; torch.save(ema.state_dict(), OUT / 'best.pt')
    print(f'epoch {epoch + 1:3d}  loss {running / n:.3f}  val top-1 {acc:.3f}  top-5 {acc5:.3f}  ({time.time() - t0:.0f} s, {100 * min(progress, 1):.0f}% of schedule)', flush=True)
    if done: break

print(f'Best validation accuracy on unseen signers: {best:.3f}')

# %% [markdown]
# ## 7. Export for the browser

# %%
model.load_state_dict(torch.load(OUT / 'best.pt', map_location=DEVICE))
model = model.float().cpu().eval()
dummy_x = torch.zeros(1, MAX_LEN, N_FEATURES); dummy_m = torch.ones(1, MAX_LEN)
export_kwargs = dict(input_names=['x', 'mask'], output_names=['logits'], opset_version=17)
try:
    torch.onnx.export(model, (dummy_x, dummy_m), OUT / 'asl_model.onnx', dynamo=False, **export_kwargs)
except TypeError:
    torch.onnx.export(model, (dummy_x, dummy_m), OUT / 'asl_model.onnx', **export_kwargs)
print('ONNX size:', round(os.path.getsize(OUT / 'asl_model.onnx') / 1e6, 2), 'MB')

json.dump(labels, open(OUT / 'asl_labels.json', 'w'))
json.dump({'maxLen': MAX_LEN, 'features': N_FEATURES, 'valAccuracy': round(best, 4)}, open(OUT / 'asl_config.json', 'w'))

# Parity data: raw input, expected features and expected scores for a few
# validation recordings. The app's tests check that it computes the same.
parity = []
for j in val_idx[:5]:
    raw = RAW[j].astype(np.float32)
    feats = features(drop_empty_frames(raw))
    x, m = fit_length(feats)
    with torch.no_grad():
        logits = model(torch.from_numpy(x)[None], torch.from_numpy(m)[None])[0].numpy()
    parity.append({
        'raw': np.where(np.isnan(raw), None, np.round(raw.astype(float), 6)).tolist(),
        'features': np.round(feats.astype(float), 5).tolist(),
        'logits': np.round(logits.astype(float), 4).tolist(),
        'label': labels[int(Y[j])],
    })
json.dump(parity, open(OUT / 'asl_parity.json', 'w'))

# Check the exported file gives the same scores as PyTorch.
try:
    import onnxruntime as ort
except ImportError:
    os.system('pip install -q onnxruntime')
    import onnxruntime as ort
sess = ort.InferenceSession(str(OUT / 'asl_model.onnx'))
x, m = prepare(RAW[val_idx[0]].astype(np.float32))
onnx_out = sess.run(None, {'x': x[None], 'mask': m[None]})[0][0]
torch_out = model(torch.from_numpy(x)[None], torch.from_numpy(m)[None])[0].detach().numpy()
print('ONNX vs PyTorch max difference:', float(np.abs(onnx_out - torch_out).max()))
print('Done. Download asl_model.onnx, asl_labels.json, asl_config.json and asl_parity.json from the Output panel.')
