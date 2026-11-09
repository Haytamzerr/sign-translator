# Signal: live sign language translator

Signal watches your hands through your webcam and turns signs into text on
screen, in real time. You teach it your own signs: perform a gesture a few
times, give it a meaning, and the app recognizes it from then on.

Everything runs on your own computer. No video is uploaded anywhere.

## Features

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
  -> confidence check against an "unknown" option
  -> word written to the caption
```

Each sign gets its own acceptance threshold, calibrated automatically from how
consistent its examples are.

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
    core.test.ts        unit tests
  state/
    engine.ts           real-time loop: camera to words
    settings.ts         settings and their defaults
  components/           interface (React)
  i18n/strings.ts       English, French and Arabic text
```

## Privacy

Video frames are processed in the browser and never stored. The library keeps
only hand point coordinates (no images), in the browser's local database.
