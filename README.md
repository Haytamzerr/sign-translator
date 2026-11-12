# Signal: live sign language translator

Signal watches your hands through your webcam and turns signs into text on
screen, in real time. It recognizes 250 common American Sign Language (ASL)
signs out of the box, and you can teach it any other sign: perform a gesture a
few times, give it a meaning, and the app recognizes it from then on.

Everything runs on your own computer. No video is uploaded anywhere.

## Features

- Built-in ASL vocabulary: a neural network trained on about 94,000 recordings
  recognizes 250 everyday ASL signs without any teaching, shown in English,
  French or Arabic
- Fingerspelling: the 26 letters of the ASL alphabet are recognized, so names
  and other words can be spelled; the letters build up a word in the caption
- Learn: 14 lessons (13 topics and the alphabet) where a native signer shows
  every sign on video (241 of the 250 built-in signs and the 26 letters, from
  ASL Signbank), at 0.5x to 1x speed, mirrored or not; live practice with the
  camera scored with 1 to 3 stars, saved progress, and a quiz mode
- Phrase to signs: type a sentence in English, French or Arabic and watch it
  signed by the same signer, with about 1,450 words and phrases; names and
  unknown words are fingerspelled, and words ASL leaves out ("the", "is") are
  marked as such
- Robust tracking: short tracking losses are bridged, a low-light boost helps
  in dark rooms, on-screen advice explains framing problems, and the tracker
  restarts itself after a graphics or camera failure
- Live hand tracking (21 points per hand, two hands) with a skeleton overlay
- Two kinds of signs, detected automatically:
  - **Still signs**: a held handshape (for example most fingerspelled letters)
  - **Moving signs**: a movement over time, recognized even when performed
    faster or slower than when it was taught
- Teach mode with a countdown, several takes, a preview of each take and a
  warning when a new sign looks too much like an existing one
- Learning from corrections: click a wrong word in the caption, pick the right
  sign, and that performance becomes a new example
- Rejection of random movements (scratching your nose does not write a word)
- Sentences: words build up live, a pause ends the sentence, and finished
  sentences go to a transcript
- Text-to-speech with the voices installed on your system
- Interface in English, French and Arabic (right-to-left layout)
- One library per sign language (ASL, LSF, Moroccan Sign Language, Arabic Sign
  Language, BSL, or your own gestures), with export and import
- Settings for camera, resolution, GPU or CPU processing, sensitivity,
  confidence, hold time, movement threshold, jitter reduction, captions, voice
  and theme
- Speed indicator (frames per second and detection time)

## Getting started

Requirements: [Node.js](https://nodejs.org) 20 or newer, a webcam, and a recent
Chrome or Edge.

```bash
npm install     # also downloads the hand tracking model for offline use
npm run dev     # opens the app at http://localhost:5173
```

Other commands:

```bash
npm test          # unit tests of the recognition engine
npm run build     # production build in dist/
npm run preview   # serve the production build
```

## How it works

```
camera frame
  -> MediaPipe Hand Landmarker (GPU): 21 points per hand
  -> features: points measured from the wrist, divided by hand size
  -> One Euro filter: removes jitter without adding lag
  -> decoder: no hands / holding / moving
  -> recognizer:
       still signs  -> nearest examples (k-nearest neighbors)
       moving signs -> Dynamic Time Warping against each example
       ASL          -> pre-trained neural network (hands + body points)
  -> confidence check against an "unknown" option
  -> word written to the caption
```

Each sign gets its own acceptance threshold, calibrated automatically from how
consistent its examples are.

## Understanding the code

Start with [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md): a reading path from the camera to the words,
with links to the exact lines. In VS Code, the CodeTour extension plays the same
path as a guided tour (`.tours/signal.tour`).

## Project structure

```
src/
  core/                 recognition engine (no interface code)
    types.ts            shared data types
    features.ts         landmark normalization, distances, motion
    smoothing.ts        One Euro filter
    sequence.ts         resampling and Dynamic Time Warping
    recognizer.ts       nearest-neighbor matching, calibration, confidence
    decoder.ts          when a sign starts and ends, when to write a word
    teach.ts            cleaning takes, sign type detection, conflict check
    tracker.ts          MediaPipe hand tracking
    storage.ts          sign library in IndexedDB, export and import
    speech.ts           text-to-speech
    asl/                built-in ASL model: input preparation, inference, labels
    learn/              lesson topics, sign videos, recorded signs, phrase to signs
    letters.ts          fingerspelling recognizer and letter decoder
    continuity.ts       bridging short tracking gaps
    lighting.ts         light measurement and low-light boost
    framing.ts          framing advice
    core.test.ts        unit tests
  state/
    engine.ts           real-time loop: camera to words
    settings.ts         settings and their defaults
  components/           interface (React)
  i18n/strings.ts       English, French and Arabic text
```

## Training the built-in ASL model

The model in `public/models/asl/` is produced by `training/train_asl.ipynb`.
The recorded sign movements and the alphabet recognizer (`public/models/learn/`
and `public/models/letters/`) are produced by `training/prepare_learning_data.py`,
which averages up to 20 recordings of each sign (after aligning them in time)
and trains the letter classifier. The recorded movements serve as examples of
"other signs" when you teach your own.
The sign videos (`public/models/signs/`) are chosen, trimmed and compressed by
`training/signbank_videos.py`.
To retrain it, open the notebook on Kaggle, add the
[Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs)
competition data, and run it (a GPU makes it much faster). It writes
`asl_model.onnx`, `asl_labels.json`, `asl_config.json` and `asl_parity.json`.
The parity file lets `npm test` check that the app prepares the network's input
exactly like the notebook.

## Credits

The built-in ASL vocabulary is trained on the PopSign ASL dataset, released by
Google, the Georgia Institute of Technology, the National Technical Institute
for the Deaf and the Rochester Institute of Technology through the Kaggle
competition "Google - Isolated Sign Language Recognition", under the
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. Hand and body
tracking use [MediaPipe](https://ai.google.dev/edge/mediapipe) by Google.
The sign videos come from [ASL Signbank](https://aslsignbank.haskins.yale.edu)
(Haskins Laboratories, Yale University), under the
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license:
they may be shared and adapted for non-commercial use, with credit, under the
same license. They are shortened (the still frames before and after each sign
are cut), cropped and compressed by `training/signbank_videos.py`.
The fingerspelling recognizer is trained on the
[ASLNow! fingerspelling dataset](https://huggingface.co/datasets/sid220/asl-now-fingerspelling)
(MIT license). "Watch a video" links open [SignASL.org](https://www.signasl.org).

## Privacy

Video frames are processed in the browser and never stored. The library keeps
only hand point coordinates (no images), in the browser's local database.
