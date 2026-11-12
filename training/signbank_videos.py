# %% [markdown]
# # Sign videos for Learn and Phrase to signs
#
# ASL Signbank (Haskins Laboratories, Yale University; CC BY-NC-SA 4.0,
# https://aslsignbank.haskins.yale.edu) has a short video of a native signer
# for about 3,500 ASL signs, each with its English translations.
#
# This script
# 1. picks one video for each of the app's 250 signs, each letter A-Z, and
#    each common English word that has a clear entry,
# 2. cuts the still frames before and after the sign (so a sentence plays
#    without long pauses), crops to 4:3 and makes the file small (silent
#    H.264, 360 pixels high),
# 3. writes the index the app reads (public/models/signs/index.json).
#
# Inputs:
#   catalog.json  the Signbank list: id, gloss "g", lemma "l", translations "t",
#                 video path "v" (saved from the Signbank search page)
#   raw/          the downloaded videos, named <id>.mp4
# Usage:
#   python training/signbank_videos.py catalog.json raw public/models/signs
#   (first run with --list to print the ids to download)
#
# Needs ffmpeg, NumPy and wordfreq (pip install wordfreq).

# %%
import json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from wordfreq import top_n_list, zipf_frequency

CATALOG, RAW, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
LABELS = Path('public/models/asl/asl_labels.json')
COMMON_WORDS = 2000  # how many of the most frequent English words to look up

catalog = [r for r in json.loads(CATALOG.read_text(encoding='utf-8')) if r.get('v')]
by_gloss = {r['g']: r for r in catalog}

# %% [markdown]
# ## 1. Choosing a video for each word
#
# A word matches an entry, from best to worst, when it is the entry's lemma
# ("MOTHER"), its gloss without the variant letters ("CATf" -> "CAT"), or the
# first of its translations. Entries that are annotation conventions (gestures,
# classifiers) are left out.

# %%
def translations(r):
    return [x.strip().lower() for x in (r.get('t') or '').split(',') if x.strip()]

CONVENTION = re.compile(r'^(i\(|E\(|G\(|[DLSBIP]CL|FS\(|NS\(|#)')

def match_rank(r, word):
    if CONVENTION.match(r['g'] or ''):
        return None
    w = word.upper().replace(' ', '-')
    if (r.get('l') or '').upper() == w:
        return 0
    if re.sub(r'(?<=[A-Z])[a-z0-9]+$', '', r['g']).upper() == w:
        return 1
    t = translations(r)
    if t and t[0] == word:
        return 2
    if word in t and len(t) <= 6:
        return 3
    return None

def best_entry(words, worst=3):
    """The best entry for the first word that has one (lowest rank, then lowest id)."""
    found = None
    for k, w in enumerate(words):
        ranked = [(rank + 10 * k, r['id'], r) for r in catalog if (rank := match_rank(r, w)) is not None and rank <= worst]
        if ranked:
            m = min(ranked, key=lambda x: x[:2])
            if found is None or m[0] < found[0]:
                found = m
    return found[2] if found else None

# %% [markdown]
# The 250 built-in signs come from PopSign, a children's vocabulary. Their
# label is not always the Signbank word, and some labels have a specific
# meaning, so these were checked by hand: the Signbank sign that means the same.
# Labels with no Signbank video (backyard, cheek, chin, cowboy, feet, finger,
# goose, pajamas, pool) are left out of Learn.

# %%
CHOSEN = {
    'food': 'EAT', 'cute': 'CUTEnoth', 'snack': 'SNACK-FOOD', 'minemy': 'POSS_1', 'hesheit': 'IX', 'there': 'IX',
    'nuts': 'NUT', 'yucky': 'GROSS', 'beside': 'NEXT-TO', 'noisy': 'LOUD', 'owie': 'PAIN', 'potty': 'TOILET-ROOM',
    'kitty': 'KITTEN', 'wake': 'AWAKE', 'awake': 'AWAKE', 'callonphone': 'TELEPHONE', 'TV': 'TVstr', 'glasswindow': 'WINDOW',
    'frenchfries': 'FRENCH-FRIES', 'thankyou': 'THANK-YOU', 'grandma': 'GRANDMOTHER', 'grandpa': 'GRANDFATHER',
    'fireman': 'FIREFIGHTERb', 'refrigerator': 'REFRIGERATOR', 'mitten': 'MITTENS', 'underwear': 'UNDERWEAR',
    'haveto': 'MUST', 'cut': 'SCISSORS-CUT', 'hear': 'SOUND', 'lips': 'MOUTH', 'gift': 'PRESENT', 'bye': 'BYE-BYE',
    'another': 'OTHER', 'hungry': 'DESIRE', 'nap': 'SLEEPo', 'penny': 'CENT', 'puzzle': 'JIGSAW-PUZZLE', 'yourself': 'SELFneut',
}
NO_VIDEO = {'backyard', 'cheek', 'chin', 'cowboy', 'feet', 'finger', 'goose', 'pajamas', 'pool'}
SEARCH = {'mom': ['mother', 'mom'], 'dad': ['father', 'dad'], 'weus': ['we', 'us'], 'icecream': ['ice cream'],
          'puppy': ['puppy', 'dog'], 'shhh': ['quiet', 'shh'], 'shoe': ['shoe', 'shoes'], 'tooth': ['tooth', 'teeth']}

learn = {}
for label in json.loads(LABELS.read_text()):
    if label in NO_VIDEO:
        continue
    r = by_gloss[CHOSEN[label]] if label in CHOSEN else best_entry(SEARCH.get(label, [label.lower()]))
    if r:
        learn[label] = r['id']

letters = {c: by_gloss[f'LETTER-{c}']['id'] for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}

# English words for Phrase to signs: pronouns are pointing signs; then the most
# common English words, and common two- or three-word phrases ("thank you").
words = {w: by_gloss[g]['id'] for w, g in {
    'you': 'IX', 'me': 'IX_1', 'your': 'POSS', 'yours': 'POSS', 'he': 'IX', 'she': 'IX',
    'him': 'IX', 'her': 'IX', 'they': 'IX', 'them': 'IX'}.items()}
LEFT_OUT = {'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'of', 'to', 'and'}
everyday = ['deaf', 'hearing', 'interpreter', 'sorry', 'please', 'nice', 'meet', 'understand', 'slow', 'again',
            'sign', 'language', 'learn', 'teacher', 'student', 'family', 'sister', 'brother', 'baby', 'hungry',
            'thirsty', 'tired', 'sick', 'hospital', 'doctor', 'emergency', 'bathroom', 'water', 'coffee', 'tea',
            'bread', 'restaurant', 'birthday', 'party', 'weekend', 'today', 'tonight', 'week', 'month', 'year']
# (zipf 3.8 = about 6 times per million words: "thank you", "ice cream", "good morning")
phrases = sorted({x for r in catalog for x in translations(r) if 1 < len(x.split()) <= 3 and zipf_frequency(x, 'en') >= 3.8})
# A word is used only when the entry is named after it: matching a word to the
# first translation of another sign gave wrong pairs ("season" -> SPRING).
for w in [w for w in top_n_list('en', COMMON_WORDS) if re.fullmatch(r'[a-z]{2,}', w)] + everyday + phrases:
    if w in LEFT_OUT or w in words:
        continue
    r = best_entry([w], worst=1)
    if r:
        words[w] = r['id']
# Words whose usual ASL sign has another name, checked by hand.
SAME_SIGN = {'already': 'FINISH', 'actually': 'TRUE-BIZtap', 'ago': 'PASTbk', 'begin': 'START', 'below': 'UNDER',
             'beautiful': 'PRETTY', 'glad': 'HAPPY', 'during': 'WHILE', 'earth': 'WORLD', 'cost': 'PRICE',
             'difficult': 'HARD', 'in': 'INSIDE', 'has': 'HAVE', 'myself': 'SELF_1', 'another': 'OTHER',
             'cash': 'MONEYrub', 'choice': 'PICK', 'view': 'WATCH', 'medical': 'MEDICINE', 'food': 'EAT'}
for w, g in SAME_SIGN.items():
    words[w] = by_gloss[g]['id']

ids = sorted(set(learn.values()) | set(letters.values()) | set(words.values()))
print(f'{len(learn)} built-in signs, 26 letters, {len(words)} words: {len(ids)} videos')
if '--list' in sys.argv:
    print(json.dumps(ids))
    sys.exit()

# %% [markdown]
# ## 2. Cutting and compressing
#
# The signer stands still at the start and the end of each video. To find the
# sign itself, every frame (small and grey) is compared with the first and the
# last frame: the sign starts when the picture first differs from the first
# frame, and ends when it stops differing from the last one. A few frames are
# kept on each side.

# %%
W, H = 64, 36  # size of the frames used to measure movement
PAD_S = 0.2
MIN_S = 1.0

def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries',
                          'stream=width,height,avg_frame_rate', '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)['streams'][0]
    num, den = map(int, s['avg_frame_rate'].split('/'))
    return s['width'], s['height'], num / den

def movement_span(path, fps):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-vf', f'scale={W}:{H},format=gray',
                          '-f', 'rawvideo', '-'], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W).astype(np.float32)
    from_start = np.abs(frames - frames[0]).mean(axis=(1, 2))
    from_end = np.abs(frames - frames[-1]).mean(axis=(1, 2))
    threshold = 2.5  # grey levels (0-255) of average change: more than camera noise
    moving_start = np.nonzero(from_start > threshold)[0]
    moving_end = np.nonzero(from_end > threshold)[0]
    if not len(moving_start) or not len(moving_end):
        return 0.0, len(frames) / fps
    total = len(frames) / fps
    start = max(0.0, moving_start[0] / fps - PAD_S)
    end = min(total, (moving_end[-1] + 1) / fps + PAD_S)
    # Small signs (MY: a hand on the chest) still last at least a second.
    grow = max(0.0, MIN_S - (end - start)) / 2
    return max(0.0, start - grow), min(total, end + grow)

def convert(i):
    src, dst = RAW / f'{i}.mp4', OUT / f'{i}.mp4'
    if dst.exists():
        return
    w, h, fps = probe(src)
    start, end = movement_span(src, fps)
    crop_w = min(w, round(h * 4 / 3 / 2) * 2)  # centre 4:3 part: the signer stands in the middle
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f'{start:.3f}', '-to', f'{end:.3f}', '-i', str(src),
                    '-vf', f'crop={crop_w}:{h},scale=-2:360', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                    '-crf', '27', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst)], check=True)

OUT.mkdir(parents=True, exist_ok=True)
for old in OUT.glob('*.mp4'):  # clips no longer used
    if int(old.stem) not in ids:
        old.unlink()
by_id = {r['id']: r for r in catalog}
# Signbank names pointing signs with annotation codes; show plain names instead.
PLAIN = {'IX': 'POINT', 'IX_1': 'POINT TO SELF', 'POSS': 'YOUR (FLAT HAND)', 'POSS_1': 'MY (FLAT HAND)'}
# Elsewhere the gloss without the variant letters: CATf -> CAT, SLEEPo -> SLEEP.
def display_name(g):
    return PLAIN.get(g) or re.sub(r'(?<=[A-Z])[a-z0-9]+$', '', g).split('_')[0]
glosses = {str(i): display_name(by_id[i]['g']) for i in ids}
with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:  # each ffmpeg runs as its own process
    for n, _ in enumerate(pool.map(convert, ids)):
        if n % 100 == 0:
            print(n, '/', len(ids), flush=True)

# %% [markdown]
# ## 3. The index

# %%
index = {
    'source': 'ASL Signbank, Haskins Laboratories, Yale University (https://aslsignbank.haskins.yale.edu)',
    'license': 'CC BY-NC-SA 4.0',
    'glosses': glosses,
    'learn': {k: str(v) for k, v in learn.items()},
    'letters': {k: str(v) for k, v in letters.items()},
    'words': {k: str(v) for k, v in sorted(words.items())},
}
(OUT / 'index.json').write_text(json.dumps(index, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print('done:', len(ids), 'videos')
