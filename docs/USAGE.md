# Using the model

How to run predictions, read the output honestly, and avoid the two failure
modes that silently produce confident nonsense.

---

## 1. Which model should I use?

Training writes one bundle per configuration to `models/`:

| File | Classes | Trained on | Reported accuracy |
|---|---|---|---:|
| `mlp_observed_speaker.joblib` | 4 | 18 actors, 6 held out | **59.4%** |
| `mlp_observed_random.joblib` | 4 | all 24 actors | 77.6% ⚠️ |
| `mlp_all_speaker.joblib` | 8 | 18 actors, 6 held out | 40.3% |
| `mlp_all_random.joblib` | 8 | all 24 actors | 66.9% ⚠️ |

**Use a `_speaker` model.** The `_random` ones have seen every actor in the
dataset, so their accuracy is inflated by speaker leakage and tells you nothing
about behaviour on a new voice. They exist only to document the size of that
gap ([RESULTS.md](RESULTS.md)).

Nothing is pre-trained in a fresh clone, build one first:

```bash
python -m speech_emotion.train --split speaker
```

Models trained with `--pooling meanstd` (360 features) record that in the saved
bundle, and `predict` extracts matching features automatically. You do not need
to pass anything extra at prediction time.

## 2. The one-liner

```bash
python -m speech_emotion.predict clip.wav --model models/mlp_observed_speaker.joblib
```

```
clip.wav
  predicted: calm
    calm        90.6% ###########################
    disgust      8.9% ###
    happy        0.4%
    fearful      0.0%
```

Multiple files at once:

```bash
python -m speech_emotion.predict data/ravdess/Actor_04/*.wav --model models/mlp_observed_speaker.joblib
```

After `pip install -e .` the same thing is available as `ser-predict`.

### The app

For clicking rather than typing:

```bash
pip install -e ".[app]"              # adds Streamlit
streamlit run app/streamlit_app.py   # opens http://localhost:8501
```

It lists every bundle in `models/` in the sidebar, the speaker-independent
4-class model first, so train at least one model before launching it. The
dataset is only needed for the game and the side-by-side comparison.

What it adds over the command line:

- **Recording straight from the microphone at 48 kHz.** Streamlit's recorder
  defaults to 16 kHz, which would put every recording into failure mode 2 below
  (29.7% accuracy), so the app asks for 48 kHz explicitly.
- **Warnings before you trust a prediction:** sample rate below 44.1 kHz (quoting
  the accuracy measured at that rate), under a second of voice, near-silence, and
  clipping. The thresholds live in `speech_emotion/demo.py` and are unit-tested.
- **The whole probability distribution against the chance line**, so a confident
  answer from a model that has no word for the emotion looks the way it should.
- **A game against the model** on the six held-out actors, and a view that plays
  one actor saying the same sentence in every emotion.

## 3. From Python

```python
import joblib
from speech_emotion import extract_feature

bundle = joblib.load("models/mlp_observed_speaker.joblib")
model = bundle["model"]          # sklearn Pipeline: StandardScaler -> MLPClassifier

features = extract_feature("clip.wav").reshape(1, -1)   # (1, 180)
label = model.predict(features)[0]
probs = dict(zip(model.classes_, model.predict_proba(features)[0]))

print(label, probs)
```

`.reshape(1, -1)` matters: `extract_feature` returns a flat `(180,)` vector and
sklearn expects a 2-D `(n_samples, n_features)` array.

The bundle carries its own provenance, so a saved model is self-describing:

```python
bundle["emotions"]   # ['calm', 'happy', 'fearful', 'disgust']
bundle["split"]      # 'speaker'
bundle["accuracy"]   # 0.59375
bundle.get("pooling", "mean")   # 'mean' or 'meanstd'; bundles saved before the option existed lack the key
```

### Batch

```python
import numpy as np
from pathlib import Path

paths = sorted(Path("my_clips").glob("*.wav"))
X = np.array([extract_feature(p) for p in paths])
for p, label in zip(paths, model.predict(X)):
    print(p.name, label)
```

---

## 4. Failure mode 1: it can only ever say four things

The 4-class model knows `calm`, `happy`, `fearful`, `disgust`. That is the
entire universe of answers available to it. It is a **closed-set classifier**,
every input is forced into one of those four boxes, including inputs that
belong in none of them.

Measured on real RAVDESS clips of emotions the model was never trained on:

| True emotion | Model says | Confidence |
|---|---|---:|
| angry | happy | 90% |
| surprised | happy | 74% |
| sad | happy | 37% |

**It does not know it is wrong, and confidence will not tell you.** Nothing in
the output distinguishes "this is clearly happy" from "this is angry, which I
have no word for." If your audio might contain emotions outside the four, use
the 8-class model, or treat a low max-probability as *abstain*:

```python
probs = model.predict_proba(features)[0]
label = model.classes_[probs.argmax()] if probs.max() >= 0.6 else "uncertain"
```

A threshold is a blunt instrument here (the table above shows a 90%-confident
error) but it beats nothing.

## 5. Failure mode 2: sample rate (the silent one)

MFCC, chroma, and mel bins are defined **relative to the sample rate**. The same
coefficient index describes a different frequency band at 16 kHz than at 48 kHz.
Hand a model trained on 48 kHz audio a 16 kHz clip and its feature vector is
silently shifted into a space it has never seen.

Measured on the 192 held-out clips, resampled and re-predicted using the
original native-rate extraction:

| Input rate | Accuracy | Mean confidence |
|---|---:|---:|
| 48 kHz | 59.4% | 89.2% |
| 22.05 kHz | **25.0%** (chance) | **100.0%** |
| 16 kHz | **25.0%** (chance) | **100.0%** |
| 8 kHz | 24.5% | 100.0% |

Accuracy falls to the random-guess floor **and confidence rises to 100%**. This
is the worst shape a bug can take: the model gets louder as it gets worse, and
no error is ever raised.

**This is now handled for you.** `extract_feature` resamples every clip to
`TARGET_SR = 48_000` before extracting anything, restoring the bin alignment:

| Input rate | Native-rate extraction | With resampling (current default) |
|---|---:|---:|
| 48 kHz | 59.4% | 59.4% |
| 22.05 kHz | 25.0% | **41.7%** |
| 16 kHz | 25.0% | **29.7%** |
| 8 kHz | 24.5% | 26.6% |

Read that table carefully, because it draws a real distinction:

- Resampling fixes **bin misalignment**, and at 22 kHz that alone recovers
  17 points.
- It cannot fix **lost bandwidth**. A 16 kHz recording physically contains no
  content above 8 kHz (Nyquist). Upsampling to 48 kHz restores the axis but not
  the information, so the upper mel bands arrive near-empty, still a
  distribution the model never saw.

**Practical rule: the closer your audio is to 48 kHz, the better this works.**
If you must deploy on 16 kHz audio (telephony, most speech APIs), the right fix
is not resampling at inference, it is to **downsample the training data and
retrain**, so train and deploy match:

```python
# in src/speech_emotion/features.py
TARGET_SR = 16_000
```
```bash
rm data/features_cache.npz    # the cache is rate-specific; force re-extraction
python -m speech_emotion.train --split speaker
```

## 6. Input formats

`extract_feature` reads through `soundfile` (libsndfile): **WAV, FLAC, OGG,
AIFF** work directly. **MP3 and M4A generally do not.** Convert first:

```bash
ffmpeg -i input.mp3 -ar 48000 -ac 1 output.wav     # 48 kHz, mono
```

Other input notes:

- **Stereo** is downmixed to mono automatically.
- **Clip length** does not matter, features are averaged over time, so any
  duration yields 180 numbers. But very short clips (< 0.5 s) give a noisy
  average, and very long ones blur multiple emotions into one vector. RAVDESS
  clips are 3–5 s; stay near that.
- **Silence and noise** are not handled. There is no voice-activity detection,
  so leading silence and background hiss both enter the average and drag the
  features toward whatever the room sounds like.

## 7. Reading the output honestly

The model is **59.4% accurate on unseen speakers across four classes**, against
a 25% floor. Well above chance, well below trustworthy: two in five predictions
are wrong.

Further limits worth stating plainly before building on it:

- **It works much better for some voices than others.** On the six held-out
  actors, accuracy ranges from 41% to 75% (`reports/verification.json`). The
  59.4% average describes no individual speaker well.
- **It was trained on acted emotion.** Professional actors performing "fearful"
  on cue produce cleaner, more exaggerated cues than a genuinely frightened
  person. Expect worse results on spontaneous speech.
- **It heard 24 North-American English speakers.** Nothing here supports claims
  about other languages, accents, or cultures.
- **It reads arousal better than valence.** Decent at loud-and-activated vs
  quiet-and-calm; weak at positive vs negative at the same energy, which is
  why `angry` comes back as `happy`.
- **It does not measure how someone feels.** It scores acoustic similarity to
  how 24 actors performed an emotion label. Do not use it for anything
  consequential about a real person, hiring, assessment, monitoring.

## 8. Retraining on your own audio

The loader expects RAVDESS filenames (7 hyphen-separated fields, emotion in
field 3, actor in field 7) and raises a clear error otherwise:

```
ValueError: 'my_recording.wav' is not a RAVDESS filename: expected 7
hyphen-separated fields (e.g. '03-01-06-01-02-01-12.wav'), found 1.
```

Two options: rename your files to that scheme, or write your own loader
returning the same three arrays.

```python
from sklearn.model_selection import GroupShuffleSplit
from speech_emotion import extract_feature
from speech_emotion.train import build_model

X = np.array([extract_feature(p) for p in my_paths])   # (n, 180)
y = np.array(my_labels)                                # emotion per clip
groups = np.array(my_speaker_ids)                      # speaker per clip -- required

train, test = next(GroupShuffleSplit(1, test_size=0.25, random_state=9)
                   .split(X, y, groups))
model = build_model().fit(X[train], y[train])
```

**Track a speaker ID for every clip even if you think you will not need it.**
Without `groups` you cannot build a proper split, and a random split on grouped
data will quietly overstate your accuracy, the whole lesson of this repo
([LEARNING_NOTES.md](LEARNING_NOTES.md#3-the-split-that-changes-the-answer)).

Delete `data/features_cache.npz` whenever you change the feature code, the
sample rate, or the audio. The cache is keyed by nothing and will happily serve
stale vectors.
