# Code guide

A reading path through the project, in the order data flows: from camera pixels to words on screen. Click any link to jump to the exact line in VS Code (open this file with **Ctrl+Shift+V** for the preview).

## The project in one picture

```
camera frame
  -> MediaPipe (GPU)      hand + body keypoints          src/core/tracker.ts
  -> normalization        position/size/aspect removed   src/core/features.ts
  -> One Euro filter      jitter removed, no lag         src/core/smoothing.ts
  -> decoder              idle / holding / moving        src/core/decoder.ts
  -> recognizers          k-NN, DTW, neural network      src/core/recognizer.ts, sequence.ts, asl/
  -> decision             confidence + "unknown"         src/core/recognizer.ts
  -> captions and speech                                 src/App.tsx, src/components/
```

## Folder map

| Folder | Contains |
|---|---|
| `src/core/` | the computer vision and recognition engine, with no interface code |
| `src/core/asl/` | the built-in ASL neural network: input preparation, inference, translated labels |
| `src/state/` | the real-time loop (`engine.ts`) and the settings |
| `src/components/` | the interface panels (React) |
| `src/i18n/` | English, French and Arabic text |
| `training/` | the notebook that trained the ASL model on Kaggle |
| `public/models/asl/` | the trained model and its test data |

## 1. Entry point and app state

### Where the app starts

[`src/main.tsx`, line 6](../src/main.tsx#L6)

The browser loads index.html, which runs this file. It mounts the React component App into the page.

### App: the owner of shared state

[`src/App.tsx`, line 53](../src/App.tsx#L53)

App holds the settings, the sign library, the caption words and the transcript, and passes them to the panels. It creates ONE SignEngine, which does all the real-time work outside React.

> **In class:** Separation of concerns: the interface (React) never runs the vision code itself. That keeps the video loop fast.

### Receiving recognized words

[`src/App.tsx`, line 143](../src/App.tsx#L143)

Every time the engine decides a sign was made ("commit"), this callback appends the word to the caption.

### Ending a sentence

[`src/App.tsx`, line 165](../src/App.tsx#L165)

When the hands have been down for the chosen pause, the current words become a sentence: shown above the caption, added to the transcript, and read aloud if enabled.

## 2. The real-time loop

### Starting the camera

[`src/state/engine.ts`, line 278](../src/state/engine.ts#L278)

Asks the browser for the webcam (getUserMedia) at 640x480 or 1280x720, 30 fps, then loads the hand tracker and starts the loop. Camera errors are translated into clear messages (blocked, missing, busy).

### One step per camera frame

[`src/state/engine.ts`, line 387](../src/state/engine.ts#L387)

requestVideoFrameCallback fires exactly once per NEW camera frame. Using it instead of a timer means no frame is processed twice and none is waited for.

> **In class:** Synchronizing processing with frame arrival minimizes latency: this is the "no delay" requirement.

### The whole pipeline for one frame

[`src/state/engine.ts`, line 444](../src/state/engine.ts#L444)

Read top to bottom: detect hands (and body) -> build normalized features -> smooth -> decoder decides if a sign happened -> words are committed -> skeleton drawn -> speed measured. React only receives a summary 10 times per second.

> **In class:** This function is the pipeline diagram from the slides, in code.

## 3. Perception: from pixels to points

### Loading the hand tracker

[`src/core/tracker.ts`, line 60](../src/core/tracker.ts#L60)

Loads MediaPipe Hand Landmarker (palm detector + landmark model) on the GPU. If the GPU is not available, it falls back to the CPU. Files load from the app itself first (offline), then from the internet.

> **In class:** Two-stage detector: palm detection (SSD-style CNN) then landmark regression (21 keypoints).

### Detecting hands in a frame

[`src/core/tracker.ts`, line 126](../src/core/tracker.ts#L126)

Returns 21 points (x, y, z) per hand plus which hand it is (left/right). Timestamps must strictly increase because MediaPipe tracks hands between frames.

### Body tracking

[`src/core/tracker.ts`, line 95](../src/core/tracker.ts#L95)

Pose Landmarker gives 33 body points. Only enabled for the built-in ASL model, which needs to know where the hands are relative to the face and shoulders.

### Low-light boost

[`src/core/lighting.ts`, line 28](../src/core/lighting.ts#L28)

In a dark room the tracker receives a brightened copy of the frame (brightness gain up to 2.4x plus a little contrast). Brightness is measured twice per second on a 32x24 thumbnail, and the boost switches on below 75/255 and off above 95/255.

> **In class:** Image enhancement before detection: a classic preprocessing step. The two thresholds are hysteresis again.

### Bridging tracking gaps

[`src/core/continuity.ts`, line 27](../src/core/continuity.ts#L27)

A hand lost for up to 150 ms (about 4 frames, typically motion blur) is kept at its last position, drawn faded. A hand that reappears under the other left/right label is not duplicated.

> **In class:** Temporal coherence: objects do not vanish between two frames, so short detection failures are filled in from the past.

### Self-repair

[`src/state/engine.ts`, line 420](../src/state/engine.ts#L420)

If 8 frames fail in a row (graphics card reset after sleep, driver problem), the tracker is rebuilt. A second failure within a minute switches to the processor.

### Framing advice

[`src/core/framing.ts`, line 21](../src/core/framing.ts#L21)

Checks the landmarks already found: hand touching the border, hand too small (far) or too large (close), shoulders missing for the ASL model. A hint appears only after 0.7 s, so it does not flicker.

## 4. Representation: from points to features

### Two hand slots

[`src/core/features.ts`, line 26](../src/core/features.ts#L26)

Each frame has slot 0 (signer's right hand) and slot 1 (left hand). If both hands claim the same side, the more confident one keeps it.

### Normalization: the key idea

[`src/core/features.ts`, line 41](../src/core/features.ts#L41)

x is multiplied by width/height (aspect correction), every point is measured from the wrist (translation invariance) and divided by the hand size, wrist to middle knuckle (scale invariance).

> **In class:** Invariance by design, like SIFT: the same handshape gives the same numbers anywhere in the image, near or far.

### Comparing two handshapes

[`src/core/features.ts`, line 62](../src/core/features.ts#L62)

Root mean square distance between matching landmarks, in hand sizes. 0 = identical; about 0.3 and above = clearly different shapes.

### Measuring motion

[`src/core/features.ts`, line 94](../src/core/features.ts#L94)

Speed in hand sizes per second: wrist travel plus a bit of handshape change. This drives the decoder (still vs moving).

### Mirroring

[`src/core/features.ts`, line 114](../src/core/features.ts#L114)

Swaps left and right hands and flips x. Used so a sign taught with one hand is also recognized with the other (data augmentation at recognition time).

### One Euro filter

[`src/core/smoothing.ts`, line 28](../src/core/smoothing.ts#L28)

Estimates how fast the value changes; fast change -> high cutoff frequency -> little smoothing (no lag); slow change -> strong smoothing (no jitter).

> **In class:** An adaptive low-pass filter (Casiez et al., 2012): the classic answer to the jitter vs latency trade-off.

## 5. Time: when does a sign happen?

### The decoder state machine

[`src/core/decoder.ts`, line 101](../src/core/decoder.ts#L101)

Called for every frame. States: idle (no hands), hold (hands still), moving. Each state triggers a different recognizer.

> **In class:** Temporal segmentation of a continuous stream: there is no start button in real signing.

### Hysteresis

[`src/core/decoder.ts`, line 129](../src/core/decoder.ts#L129)

More motion is needed to START moving than to KEEP moving (threshold vs 0.6 x threshold). This prevents flickering between states when the speed hovers around the threshold.

> **In class:** Same idea as a Schmitt trigger in electronics, or the two thresholds of the Canny edge detector.

### Still signs need a stable answer

[`src/core/decoder.ts`, line 162](../src/core/decoder.ts#L162)

A word is only written if the same confident answer is held for the hold time (400 ms by default), and not repeated until the hand changes.

### Moving signs: classify the whole movement

[`src/core/decoder.ts`, line 184](../src/core/decoder.ts#L184)

When the hands stop or leave, the collected frames form a segment, classified as one sign. The segment is also reported to the built-in ASL model.

## 6. Recognition from your own examples

### Choosing example frames

[`src/core/recognizer.ts`, line 53](../src/core/recognizer.ts#L53)

From each recorded take of a still sign, keep evenly spaced frames from the middle 60%, skipping the moments the hand arrives and leaves.

### Automatic thresholds

[`src/core/recognizer.ts`, line 133](../src/core/recognizer.ts#L133)

Leave-one-out: compare each example with the other examples of the same sign. Consistent signs get a strict threshold, looser signs a wider one.

> **In class:** Per-class calibration from intra-class variance, computed instantly, no training needed.

### Checking against real other signs

[`src/core/recognizer.ts`, line 185](../src/core/recognizer.ts#L185)

Three takes only show how much YOUR sign varies, not what other signs look like. The 250 real recordings of Learn mode serve as negative examples: each taught sign's threshold is kept at 80% of its distance to the 5th closest real sign, so real other signs stay outside it. A test teaches signs and performs 124 unseen real signs: about 0.1% wrong words.

> **In class:** Open-set calibration with negative examples (a background set), as used in speaker and face verification.

### k-nearest neighbors

[`src/core/recognizer.ts`, line 196](../src/core/recognizer.ts#L196)

Distance to every stored frame; each sign is scored by the average of its 3 closest examples.

> **In class:** k-NN is the simplest classifier and perfect for few-shot learning: adding a sign is just storing examples.

### Saying "I don't know"

[`src/core/recognizer.ts`, line 234](../src/core/recognizer.ts#L234)

Distances become probabilities with a softmax that includes an extra "unknown" option at ratio 1. If no sign is clearly closer than its threshold, "unknown" wins.

> **In class:** Open-set recognition: a classifier must be able to reject inputs that belong to no class.

### Preparing a movement

[`src/core/sequence.ts`, line 30](../src/core/sequence.ts#L30)

Positions become relative to where each hand started (in hand sizes), and the recording is resampled to 24 steps.

### Dynamic Time Warping

[`src/core/sequence.ts`, line 95](../src/core/sequence.ts#L95)

Dynamic programming: cost[i][j] = distance(a_i, b_j) + min(three neighbors). The best alignment lets a slow and a fast version of the same sign match. The band limits how far time can be stretched, which also makes it faster.

> **In class:** DTW was the standard in speech recognition before deep learning, and it needs no training.

## 7. The built-in ASL neural network

### Training features (Python)

[`training/train_asl.py`, line 91](../training/train_asl.py#L91)

Every point relative to the middle of the shoulders and divided by shoulder width, plus each hand relative to its own wrist. 182 numbers per frame.

### Data augmentation

[`training/train_asl.py`, line 158](../training/train_asl.py#L158)

Each training example is randomly mirrored, rotated, stretched, shifted, sped up or slowed down. The network learns the sign, not one exact recording.

> **In class:** Augmentation fights overfitting: with 21 signers, the model must generalize to new people.

### Convolution block

[`training/train_asl.py`, line 242](../training/train_asl.py#L242)

Depthwise 1D convolution over time (kernel 17 frames): detects local motion patterns. Residual connection and LayerNorm make deep stacks trainable.

### Attention block

[`training/train_asl.py`, line 255](../training/train_asl.py#L255)

Self-attention: every frame looks at every other frame, so the network relates the beginning and end of a sign. Padding frames are masked out.

> **In class:** Convolution = local context, attention = global context. Same mix as in modern speech models (Conformer).

### Training loop

[`training/train_asl.py`, line 344](../training/train_asl.py#L344)

AdamW optimizer, mixed precision on GPU, label smoothing, warm-up then cosine learning rate. Each hand is rotated and scaled on its own, short stretches of time are hidden, and the saved weights are a running average (EMA) of the training weights. Validation uses 2 signers never seen in training: 76.6% top-1, about 92% top-5.

### Same features in TypeScript

[`src/core/asl/features.ts`, line 93](../src/core/asl/features.ts#L93)

Line-by-line copy of the Python function. The parity test (asl.test.ts) proves both give the same numbers on real recordings.

> **In class:** Training/serving skew is a classic production bug: we test against it.

### Even rounding matters

[`src/core/asl/features.ts`, line 149](../src/core/asl/features.ts#L149)

numpy rounds 2.5 to 2 (half to even); JavaScript rounds it to 3. Without this function, a few frames would be picked differently from training.

### Fixing domain shift

[`src/core/asl/model.ts`, line 98](../src/core/asl/model.ts#L98)

The training videos were filmed on vertical phones, so their x axis is stretched about 1.7x compared with a webcam. Comparing body proportions (nose to shoulders vs shoulder width) in the training data and live on camera gives the correction factor.

> **In class:** Domain shift: training and real data differ. Here it is measured from the data and corrected geometrically.

### Inference with test-time augmentation

[`src/core/asl/model.ts`, line 110](../src/core/asl/model.ts#L110)

The sign and its mirror image are both scored and averaged, so left- and right-handed signing work. Runs with ONNX Runtime (WebAssembly) in 20-50 ms.

### Connecting the network to the live stream

[`src/state/engine.ts`, line 581](../src/state/engine.ts#L581)

When the decoder reports a finished movement that your personal library did not recognize, the matching frames are sent to the network. The word is written if its probability passes the threshold.

## 8. Fingerspelling

### A tiny neural network in plain code

[`src/core/letters.ts`, line 47](../src/core/letters.ts#L47)

Three dense layers (63 -> 128 -> 64 -> 26) with ReLU, then softmax. The input is the same normalized hand shape as everywhere else. It runs in well under a millisecond, so no library is needed.

> **In class:** A multi-layer perceptron: the simplest neural network, enough because the hard part (finding the hand) is already done.

### From frames to letters

[`src/core/letters.ts`, line 112](../src/core/letters.ts#L112)

A letter is written when the same confident guess is held for 350 ms. It can be written again only after the hand moves or changes: that is how double letters (LL in HELLO) are spelled.

### Letters in the live loop

[`src/state/engine.ts`, line 529](../src/state/engine.ts#L529)

Only the largest really detected hand, held still and raised (wrist in the upper 85% of the image), is classified. After a letter, movements are not read as whole signs for 1.5 s.

## 9. Learn and Phrase to signs

### Real signer videos, trimmed

[`training/signbank_videos.py`, line 167](../training/signbank_videos.py#L167)

Each sign is shown by a native signer (ASL Signbank, CC BY-NC-SA 4.0). The still frames before and after the sign are found by comparing every frame with the first and the last one (average grey-level change above 2.5 out of 255), and cut, so a sentence plays without pauses. Videos are cropped to 4:3, 360 pixels high, silent: about 40 KB each.

> **In class:** Frame differencing: the simplest motion detector, comparing pixels between frames.

### Which video to show

[`src/core/learn/signVideos.ts`, line 28](../src/core/learn/signVideos.ts#L28)

One index file maps each built-in sign, each letter and about 1,400 English words and phrases to a clip. Built-in sign labels were matched to Signbank signs by hand where the names differ (for example "food" is EAT, "kitty" is KITTEN).

### Two players taking turns

[`src/components/SignVideoPlayer.tsx`, line 89](../src/components/SignVideoPlayer.tsx#L89)

Two video elements take turns: while one plays a sign, the other already loads the next one, so the next sign starts without waiting. Speed is the video playback rate (letters 1.6 times faster, as signers spell quickly); mirroring flips the picture horizontally.

> **In class:** Double buffering: preparing the next frame or clip off screen while the current one is shown.

### Recorded sign movements

[`src/core/learn/demos.ts`, line 73](../src/core/learn/demos.ts#L73)

Each of the 250 signs is also kept as the average of up to 20 real recordings, aligned in time with DTW first (DTW Barycenter Averaging), stored as 159 points per frame in 16-bit integers. They are not shown: they serve as examples of other signs when you teach your own (section 10).

> **In class:** Averaging removes each signer's tracking noise and missing frames; aligning in time first keeps a fast and a slow signer from blurring each other.

### Scoring practice

[`src/state/progress.ts`, line 36](../src/state/progress.ts#L36)

3 stars if the built-in model ranks the target sign first with at least 50% confidence, 2 if first, 1 if in the top three.

### Text to signs

[`src/core/learn/phrase.ts`, line 107](../src/core/learn/phrase.ts#L107)

Greedy longest match against every way of writing each sign (English, French, Arabic, aliases) and every English word with a video, so "thank you" and "ice cream" become one sign. Articles and "to be" are left out, as in ASL; other words are fingerspelled.

> **In class:** Glossing: ASL has its own grammar, so a word-by-word mapping is only an approximation.

## 10. Teaching new signs

### Recording a take

[`src/components/TeachPanel.tsx`, line 75](../src/components/TeachPanel.tsx#L75)

Countdown, then frames are collected for the chosen duration. Takes without visible hands are flagged so they can be redone.

### Still or moving?

[`src/core/teach.ts`, line 38](../src/core/teach.ts#L38)

A take counts as moving if the wrist travels more than 0.6 hand sizes or the handshape changes noticeably. The majority of takes decides.

### Warning about similar signs

[`src/core/teach.ts`, line 67](../src/core/teach.ts#L67)

Runs the new examples through a recognizer built from the existing library. If they would be recognized as another sign, the user is warned before saving.

## Tests

- [`src/core/core.test.ts`](../src/core/core.test.ts): invariance of features, DTW at different speeds, k-NN, rejection of unknown signs, the decoder writing words, sign type detection and conflict warnings, all on synthetic hands.
- [`src/core/asl/asl.test.ts`](../src/core/asl/asl.test.ts): the parity test against real recordings exported by the training notebook.

Run them with `npm test`.
