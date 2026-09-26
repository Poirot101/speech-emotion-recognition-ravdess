# 02: Data augmentation

**Goal:** enlarge the effective training set with transformed copies of training
clips (noise, gain, pitch shift, time stretch) and measure the effect on
speaker-independent accuracy, one transform at a time.

**Effort:** about 1 week. **Prerequisites:** `00_foundations.md`, `01_delta_features.md`.

---

## Why this matters here

- The speaker split trains on only **576 clips from 18 voices**. The MLP has
  ~55k parameters, far more capacity than data.
- The model's main failure is not generalising to **new voices** (77.6% → 59.4%).
  Augmentation can push a model to ignore things that vary between recordings
  and voices, if you choose transforms that vary those things.
- Project 03 (the CNN) will overfit badly without augmentation. Learn it here,
  where experiments take seconds, not hours.

---

## Concepts to understand first

1. **Label-preserving transforms.** An augmentation is valid only if a human
   would still give the clip the same label. That has to be *argued for this
   task*, not assumed:
   - **Additive noise** at a sensible signal-to-noise ratio, preserves emotion.
   - **Gain** (louder or quieter), mostly preserves it, but loudness is itself
     an arousal cue.
   - **Pitch shift** (±1–2 semitones), mostly preserves it; large shifts start
     changing perceived speaker and emotion.
   - **Time stretch**, **suspicious.** Speaking rate is an emotional cue: fast
     speech reads as high arousal. Stretching an angry clip may make it less
     angry. Test this; do not assume it.
2. **Invariance.** Each transform teaches "this change should not change the
   answer." Ask which invariances actually help on *new speakers*.
3. **Signal-to-noise ratio.** `SNR_dB = 20·log10(rms_signal / rms_noise)`. Derive
   the noise RMS for a target SNR yourself before you write any code.
4. **Where augmentation sits in the pipeline.** After splitting, and on training
   data only. See the pitfalls, this is where most augmentation projects go wrong.
5. **Spectrogram-domain augmentation** (SpecAugment: masking frequency bands and
   time spans), read about it now; you will use it in project 03.

---

## Build milestones

### M1: Listen and look
Apply each transform to one clip. **Listen** to every result (`IPython.display.Audio`)
and plot spectrograms. Decide, by ear, which parameter ranges still sound like
the original emotion.
**Checkpoint:** a table in your lab notebook of transform → safe range → why.

### M2: A leakage-safe pipeline
Design the data flow on paper **before** coding:

```
all clips ──► split by actor ──► training clips ──► augment ──► features ──► fit
                   │
                   └────────────► test clips (never augmented) ──► features ──► evaluate
```

Every augmented copy keeps its **source actor's ID** as its group.
**Checkpoint:** a test asserting no augmented copy of a test-actor clip exists.

### M3: Validation on clean clips only
In inner CV, augmented copies go into the *training* folds, but the *validation*
fold must contain **only original clips**, otherwise you are validating on
conditions that never occur at test time.
**Checkpoint:** you can show, in code, where originals and copies are separated.

### M4: One transform at a time
Using inner `GroupKFold(6)` on training actors and ≥ 5 seeds:

| Variant | Training clips |
|---|---:|
| no augmentation (reference) | 576 |
| + noise | 1,152 |
| + gain | 1,152 |
| + pitch shift | 1,152 |
| + time stretch | 1,152 |
| best two combined | 1,728 |

Cache augmented features separately, keyed by `(clip, transform, parameters, random seed)`.

### M5: Held-out once, then nested over all 24 actors
Evaluate the winner on the held-out actors, then confirm with nested grouped CV
over all 24 actors (augmenting only inside each outer fold's training actors).
Report the speaker-level bootstrap and Wilcoxon on per-speaker gains against the
best configuration from project 01. Treat McNemar on clips as optimistic: clips
from one speaker are not independent.

---

## Implementation blueprint

### Module and signatures

```python
# src/speech_emotion/augment.py
@dataclass(frozen=True)
class Transform:
    name: str                    # "noise" | "gain" | "pitch" | "stretch"
    params: tuple[tuple[str, float], ...]   # sorted, hashable: (("snr_db", 20.0),)
    seed: int

def add_noise(y: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray: ...
def apply_gain(y: np.ndarray, gain_db: float) -> np.ndarray: ...
def pitch_shift(y: np.ndarray, sr: int, n_steps: float) -> np.ndarray: ...
def time_stretch(y: np.ndarray, rate: float) -> np.ndarray: ...
def apply(y: np.ndarray, sr: int, t: Transform) -> np.ndarray: ...
```

The SNR arithmetic you should derive yourself and then check in a test:
`noise_rms = signal_rms / 10 ** (snr_db / 20)`. Test: generate noise at 20 dB,
measure the ratio back, assert within 0.5 dB.

### Feature extraction needs an array entry point

`extract_feature` reads a file. Split it:

```python
def features_from_array(y: np.ndarray, sr: int, pooling: str = "mean") -> np.ndarray: ...
def extract_feature(file_name, ..., pooling="mean"):   # reads, then calls the above
```

Keep the existing tests passing, and add one proving both paths give identical
vectors for the same audio.

### Cache key per augmented copy

```python
def aug_key(clip: Path, t: Transform, pooling: str) -> str:
    raw = f"{clip.name}|{t.name}|{t.params}|{t.seed}|{pooling}|{TARGET_SR}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]      # data/aug/<key>.npy
```

### The table you build

One row per clip *or* augmented copy:

| column | meaning |
|---|---|
| `X` | feature vector |
| `y` | label of the source clip |
| `group` | **actor of the source clip** |
| `source_index` | row of the original clip it came from |
| `is_original` | `True` for untransformed clips |

### Grouped CV that validates on originals only

`cross_val_score` cannot express "train on copies, validate on originals", so
write the loop:

```python
actors = np.unique(groups)
for tr_a, va_a in GroupKFold(6).split(actors, groups=actors):
    train_rows = np.isin(groups, actors[tr_a])                   # originals + copies
    val_rows   = np.isin(groups, actors[va_a]) & is_original     # originals only
    assert not set(groups[train_rows]) & set(groups[val_rows])
    assert is_original[val_rows].all()
    ...
```

### Assertions that catch the classic mistakes

```python
def test_no_copy_of_a_test_actor_in_training(): ...
def test_every_copy_keeps_its_source_actor(): ...
def test_same_seed_same_audio(): ...          # reproducibility
def test_augment_off_reproduces_baseline(): ...
```

---

## Pitfalls

- **Augmenting before splitting.** A pitch-shifted copy of a test clip in the
  training set is near-duplicate leakage. The score will look great and mean nothing.
- **Validating on augmented clips.** See M3.
- **Randomness you cannot reproduce.** Seed every random draw and record the
  seeds in the cache key.
- **Extraction time.** Each copy costs a full feature extraction. Cache.
- **Assuming more is better.** Too much augmentation, or unrealistic
  augmentation, can shift the training distribution away from the test data.

---

## How you will know you succeeded

- At least one transform improves inner-CV accuracy by more than the seed spread.
- You can explain *why* each transform helped or hurt, especially time stretch.
- The final held-out result is reported with its speaker-level interval.

---

## Stretch goals

- **mixup:** train on convex combinations of pairs of examples and their labels.
- **Room simulation:** convolve with impulse responses to vary recording conditions.
- **Speaker-style augmentation:** vocal tract length perturbation, look it up and
  judge whether it is label-preserving for emotion.

---

## Resources

- librosa `pitch_shift`, https://librosa.org/doc/0.11.0/generated/librosa.effects.pitch_shift.html
- librosa `time_stretch`, https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html
- audiomentations (audio augmentation library; read its transform list for ideas), https://github.com/iver56/audiomentations
- Park et al. (2019), *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition*, https://arxiv.org/abs/1904.08779
- Zhang et al. (2017), *mixup: Beyond Empirical Risk Minimization*, https://arxiv.org/abs/1710.09412
- scikit-learn common pitfalls (leakage), https://scikit-learn.org/stable/common_pitfalls.html
