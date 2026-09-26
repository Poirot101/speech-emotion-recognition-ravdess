# 06: Accuracy playbook: every lever worth pulling, ranked

**Goal:** a menu of ways to raise **speaker-independent** accuracy, ordered by
how much evidence stands behind each one and what it costs, with enough
implementation detail that you can start any of them today.

**Effort:** each item lists its own. **Prerequisites:** `00_foundations.md` and
`05_verification_and_accuracy.md`. Every idea here is judged with the same
protocol, so learn that first.

---

## The rule that makes this list useful

Every idea below gets evaluated the same way, or its number means nothing:

1. **Selection inside training actors only.** Inner `GroupKFold`.
2. **Final comparison with nested grouped CV over all 24 actors**, reusing
   `scripts/nested_cv.py`. Six held-out speakers could not resolve a 3-point gap;
   24 resolved an 8.7-point one. Plan for 24.
3. **Paired, speaker-level statistics.** Per-speaker gains → speaker bootstrap
   interval and Wilcoxon signed-rank. McNemar on clips is optimistic (clips from
   one speaker are not independent).
4. **Compare numbers from the same protocol only.** Nested 60.55% vs 69.27%.
   Held-out 59.38% vs 63.54%. Never across.

A reusable harness is the first thing to build. Suggested shape:

```python
# scripts/compare.py  (yours to write)
def nested_compare(
    candidates: dict[str, Callable[[], Estimator]],   # name -> factory
    features: dict[str, np.ndarray],                  # name -> X (rows aligned with y, g)
    y: np.ndarray, g: np.ndarray,
    outer=GroupKFold(6), inner=GroupKFold(5), seeds=range(5),
) -> dict:
    """Returns per-fold choices, per-speaker accuracy per procedure, and the
    bootstrap interval + Wilcoxon p for (procedure - baseline)."""
```

Everything in `nested_cv.py` already does this for one fixed candidate grid;
generalise it once and every experiment below becomes a dictionary entry.

---

## Summary table

| # | Lever | Evidence in this repo | Effort | Why it might help |
|---|---|---|---|---|
| 1 | Speaker enrolment (neutral-clip normalisation) in `predict.py` and the app | **Measured:** +8.7 pts nested, CI [+4.1, +13.7] (with mean+std) | 2–4 days | Removes the voice, keeps the emotion |
| 2 | Trim silence before pooling | **Measured cause, untested fix:** median clip is 54% silence | ½ day | Every mean is diluted by a different amount of silence |
| 3 | Pretrained speech embeddings (wav2vec 2.0 / HuBERT / WavLM) + linear classifier | None yet; strongest in the literature | 1–2 weeks | Representations learned from thousands of hours of speech |
| 4 | Prosodic / eGeMAPS features (pitch, jitter, shimmer, loudness) | None yet | 2–4 days | Built by emotion researchers for exactly this task |
| 5 | Ensembles: soft voting or stacking | Indirect: classifiers disagreed across folds | 2–3 days | Different models make different mistakes |
| 6 | Hyperparameter search, nested | Exploratory SVM grid: +0.4 pts, inside the noise | 1–2 days | Mostly a lesson in what *not* to expect |
| 7 | Calibration and abstention | None yet | 2 days | Raises accuracy **on the clips you answer** |
| 8 | Class weighting for the 8-class task | Diagnostic: neutral recall 21% | 1 day | Neutral has half the clips of other classes |
| 9 | Test-time augmentation | None yet | 1–2 days | Averages away a bad frame alignment or pitch |
| 10 | Gender-aware normalisation or models | None yet | 1–2 days | Pitch ranges differ; RAVDESS is 12 + 12 |

Items 01–04 in this folder (deltas, augmentation, CNN, cross-corpus) are the
bigger projects; this list is everything else.

---

## 1. Speaker enrolment: ship the one proven gain

**What.** Before predicting for a new speaker, record 2–4 calm, neutral
sentences. Subtract the mean of their feature vectors from every later clip.
Train on features normalised the same way.

**Evidence.** Every outer fold of `nested_cv.py` selected mean+std pooling with
neutral-clip normalisation. The gain is uneven: 8 of 24 speakers got worse.

### Implementation blueprint

```python
# src/speech_emotion/enrolment.py
@dataclass(frozen=True)
class SpeakerProfile:
    mean: np.ndarray          # shape (n_features,)
    n_clips: int
    pooling: str

def enrol(paths: Sequence[Path], pooling: str = "meanstd") -> SpeakerProfile: ...
def normalise(features: np.ndarray, profile: SpeakerProfile) -> np.ndarray: ...
```

Training side (`train.py`):

- New flag `--normalize {none,neutral}`. With `neutral`, for each actor subtract
  the mean of **that actor's `neutral` clips** (load all 8 emotions, then filter to
  the training classes *after* normalising, exactly as `improve_accuracy.prepare`
  does).
- Save `"normalization": "neutral"` in the bundle. `predict_file` must refuse a
  `neutral` bundle without a profile, a silent unnormalised prediction is the
  sample-rate bug all over again.
- Default output unchanged: a test that `--normalize none` reproduces 59.38%.

Prediction side:

```python
label, probs = predict_file(clip, bundle, profile=enrol(neutral_clips, bundle["pooling"]))
```

App side: a fifth tab, **"Calibrate to your voice"**, recording up to four
sentences into `st.session_state["profile"]`, with a warning if fewer than two.

### Tests you should write

- `normalise(enrol([a, a]), ...)` of clip `a` is all zeros.
- A `neutral` bundle with `profile=None` raises `ValueError`.
- The training normalisation for actor *k* uses only actor *k*'s neutral clips
  (build a tiny fake `X, y, groups` and check the arithmetic by hand).

### Questions to answer with experiments

- **How many enrolment clips are needed?** Re-run nested CV using 1, 2 and 4
  neutral clips per speaker. Plot gain against count.
- **Does enrolment still help if the "neutral" clips are not neutral?** Use calm
  clips instead, a realistic user mistake. Then use a random emotion mix (the
  earlier exploratory run suggests this hurts).
- **Who gets worse, and why?** The 8 speakers with negative gain: are they the
  ones whose neutral clips sound least like their calm clips?

---

## 2. Trim silence before pooling

**What.** The app's own readout for a RAVDESS clip said *3.8 s long with 1.8 s
of voice*. That clip is typical. Measured over all 1,440 clips with the same rule
the app uses (`librosa.effects.split`, anything within 30 dB of the clip's peak
counts as voice):

| Voiced fraction of each clip | Value |
|---|---:|
| Median | 46% |
| 10th–90th percentile | 40%–52% |
| Lowest | 31% |

So **more than half of what the mean pools over is silence**, and the share varies
from clip to clip. Features get pulled toward "quiet room" by a different amount
each time. Whether removing it improves accuracy is untested, that is the
experiment.

### Implementation blueprint

```python
def extract_feature(..., trim_db: float | None = None) -> np.ndarray:
    ...
    if trim_db is not None:
        X, _ = librosa.effects.trim(X, top_db=trim_db)   # leading/trailing only
```

- Default `None`, so the existing cache and every published number stay identical;
  add a bit-identical test like the `pooling` one.
- Cache per setting: `data/features_{pooling}_trim{db}.npz`. **Put every
  parameter that changes the features in the cache filename.**
- Grid for inner CV: `trim_db ∈ {None, 20, 30, 40}` × `pooling ∈ {mean, meanstd}`.

**Predict first:** trimming should matter more for `mean` than `meanstd` (std
over silence is small and roughly constant). Check.

**Pitfall.** `librosa.effects.trim` only removes the ends. Pauses mid-sentence
stay. `librosa.effects.split` returns all non-silent intervals if you want to
concatenate them, decide whether removing pauses removes information (hesitation
is an emotional cue).

---

## 3. Pretrained speech embeddings

**What.** Models such as wav2vec 2.0, HuBERT and WavLM learned general speech
representations from large amounts of unlabelled audio. Freeze one, pool its
hidden states over time, and train logistic regression on top. The SUPERB
benchmark includes emotion recognition among its tasks, which is a good place to
see how such representations are evaluated.

### Concepts first

- **Self-supervised learning:** training on a task built from the data itself
  (predict masked audio), no labels.
- **Frozen feature extraction vs fine-tuning.** Start frozen. With 576 training
  clips per fold, fine-tuning a 95M-parameter model is how you learn about
  overfitting.
- **Layer choice.** Different layers encode different things (lower: acoustic,
  higher: phonetic/linguistic). Emotion often peaks in the middle. Treat the layer
  as a hyperparameter chosen in inner CV.

### Implementation blueprint

```python
# scripts/extract_ssl.py  (needs: pip install torch transformers)
from transformers import AutoFeatureExtractor, AutoModel

MODEL_ID = "facebook/wav2vec2-base"      # check the model card for its sample rate

def embed(path, extractor, model, sr=16_000) -> np.ndarray:
    """Returns (n_layers + 1, hidden_size): the time-mean of every hidden layer."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    inputs = extractor(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    return torch.stack(out.hidden_states)[:, 0].mean(dim=1).numpy()
```

- **Sample rate flips.** These models expect **16 kHz**. That is not the failure
  mode from `docs/USAGE.md`, the model was *trained* at 16 kHz, so train and
  deploy match. Write down why the two situations differ.
- **Cache once:** `data/ssl_{model}_{layer_pooling}.npz` with shape
  `(1440, n_layers + 1, hidden)`. Extraction is the slow step; classifying is seconds.
- **Selection:** inner CV over `layer ∈ range(n_layers + 1)` × `C ∈ {0.1, 1, 10}`
  with `make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=C))`.
- **Then combine with item 1:** normalise embeddings by the speaker's neutral
  clips. Does enrolment still help on top of a strong representation?

### Pitfalls

- `model.eval()` and `torch.no_grad()`, otherwise dropout is on and memory grows.
- Clips longer than the model's positional limits are not an issue at 3–5 s, but
  cross-corpus audio might be.
- **Licences.** Check each model card's licence before redistributing anything.
- **Leakage through the pretrained model.** If a checkpoint was fine-tuned on
  RAVDESS, your test set is in its training data. Use base checkpoints and read
  the model card.

---

## 4. Prosodic and eGeMAPS features

**What.** eGeMAPS is a small, standardised acoustic feature set (pitch, jitter,
shimmer, loudness, formants, spectral slope…) proposed for affective computing.
The `opensmile` Python package extracts it in one call.

```python
import opensmile
smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                        feature_level=opensmile.FeatureLevel.Functionals)
row = smile.process_file(path)          # a one-row pandas DataFrame
```

Or build the prosodic part yourself with librosa, which teaches more:

| Feature | librosa | Summarise with |
|---|---|---|
| Pitch contour | `librosa.pyin(y, fmin=65, fmax=500, sr=sr)` | mean, std, range, slope over voiced frames |
| Voicing ratio | `voiced_flag` from `pyin` | fraction voiced |
| Energy | `librosa.feature.rms` | mean, std, max |
| Speaking rate proxy | `librosa.onset.onset_detect` | onsets per voiced second |

**Experiments:** eGeMAPS alone; MFCC-meanstd alone; concatenated. Then each with
neutral-clip normalisation, pitch is exactly the kind of feature that differs by
speaker, so predict whether enrolment helps prosody more than MFCCs.

**Pitfall.** `pyin` returns `NaN` for unvoiced frames. `np.nanmean`, and decide
what a clip with no voiced frames gets.

---

## 5. Ensembles

Nested CV chose logistic regression in 4 folds and gradient boosting in 2, with
similar inner scores. When models are close but different, averaging their
probabilities often helps.

```python
from sklearn.ensemble import VotingClassifier
vote = VotingClassifier(
    [("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),
     ("hgb", HistGradientBoostingClassifier()),
     ("mlp", build_model())],
    voting="soft")
```

**Stacking without leakage.** `StackingClassifier(cv=...)` builds out-of-fold
predictions internally, but it does not know your clips are grouped by speaker
unless the groups reach the splitter. The transparent route is manual:

```python
oof = np.zeros((len(y_tr), n_models * n_classes))
for tr, va in GroupKFold(5).split(X_tr, y_tr, g_tr):
    for m, make in enumerate(factories):
        oof[va, m * n_classes:(m + 1) * n_classes] = make().fit(X_tr[tr], y_tr[tr]).predict_proba(X_tr[va])
meta = LogisticRegression(max_iter=5000).fit(oof, y_tr)
```

**Check before trusting it:** compute pairwise disagreement between the base
models on out-of-fold predictions. If they agree on 95% of clips, the ensemble
cannot add much.

---

## 6. Hyperparameter search: do it nested, expect little

The exploratory SVM grid found 68.1% against 67.7% for the defaults: noise.
Search is still worth learning, mostly to learn what it cannot do.

```python
search = GridSearchCV(pipe, {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 1e-3, 1e-4]},
                      cv=GroupKFold(5))
for tr, te in GroupKFold(6).split(X, y, g):              # outer
    search.fit(X[tr], y[tr], groups=g[tr])               # inner uses groups
    outer_scores.append(search.score(X[te], y[te]))
```

`search.fit(..., groups=...)` passes groups to the inner `GroupKFold`. Writing the
outer loop by hand keeps it obvious that groups reach both levels.

**Report** the spread of `search.best_params_` across outer folds. If it changes
every fold, the "best" setting is not a property of the problem.

**Optional:** random search (`RandomizedSearchCV`) and Optuna. Same nesting rule.

---

## 7. Calibration and abstention

A model that says "91% disgust" for a calm clip (the app shows exactly this) is
**overconfident**. Two linked skills:

**Calibration.** Does "90% sure" mean right 90% of the time?

```python
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
# CalibratedClassifierCV.fit does not accept groups=, so precompute grouped splits.
splits = list(GroupKFold(5).split(X_tr, y_tr, g_tr))
cal = CalibratedClassifierCV(model, method="sigmoid", cv=splits).fit(X_tr, y_tr)
```

(Checked against scikit-learn 1.9.1: passing `cv=GroupKFold(5)` and
`fit(..., groups=g)` raises *"The 'groups' parameter should not be None"*;
precomputed splits work. `GridSearchCV.fit(..., groups=g)` does accept groups.)

Plot a reliability diagram (top-class confidence vs accuracy in bins) before and
after. Compute the expected calibration error yourself: it is ten lines of NumPy.
Note: scikit-learn 1.9 deprecates `SVC(probability=True)` in favour of
`CalibratedClassifierCV`, this repo's `build_model("svm")` will need that change.

**Abstention.** Only answer when confidence ≥ τ. Plot **accuracy on answered
clips** against **coverage** (fraction answered) as τ varies. That curve is a far
more realistic summary of a deployable system than one accuracy number, and the app
already has a threshold (`UNCERTAIN_BELOW = 0.6`) you could choose from it.

**Pitfall.** Choose τ on inner folds. Picking it on the test speakers is selection
leakage again.

---

## 8. Class weighting for 8 classes

`neutral` has 96 clips; every other emotion has 192. Its recall on unseen
speakers was 21%.

- `LogisticRegression(class_weight="balanced")`, `SVC(class_weight="balanced")`.
- `MLPClassifier` has no `class_weight`. Oversample `neutral` **inside each
  training fold** (duplicate indices), never before splitting.
- Judge by **macro-F1** and per-class recall, not accuracy, weighting usually
  trades a little accuracy for much better minority recall.

---

## 9. Test-time augmentation

Predict on the original clip plus a few mild transforms (±1 semitone, small gain
changes), average the probabilities.

```python
def predict_tta(path, bundle, shifts=(-1, 0, 1)) -> np.ndarray:
    y, sr = librosa.load(path, sr=TARGET_SR)
    probs = [bundle["model"].predict_proba(features_from_array(
        librosa.effects.pitch_shift(y, sr=sr, n_steps=s), sr, bundle["pooling"]))
        for s in shifts]
    return np.mean(probs, axis=0)
```

You will need `features_from_array`, refactor `extract_feature` so the file
reading and the feature maths are separate functions. That refactor is worth
doing anyway (the app and augmentation both want it).

---

## 10. Gender-aware normalisation

RAVDESS actor IDs encode gender (odd male, even female), and pitch ranges differ.
Two cheap experiments:

- **Gender-level normalisation:** standardise with the mean/std of training clips
  of the same gender. Deployable only if you know or predict gender, and
  predicting it is itself a model with errors. Note the ethical cost of building
  that in.
- **Per-gender accuracy:** before changing anything, report baseline accuracy
  separately for the 12 male and 12 female actors in nested CV. If there is a gap,
  that is a finding on its own.

---

## Resources

**Protocol and statistics**
- scikit-learn, nested vs non-nested cross-validation, https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- scikit-learn, probability calibration, https://scikit-learn.org/stable/modules/calibration.html
- `scipy.stats.wilcoxon`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html

**Features**
- librosa `effects.trim`, https://librosa.org/doc/0.11.0/generated/librosa.effects.trim.html
- librosa `pyin`, https://librosa.org/doc/0.11.0/generated/librosa.pyin.html
- openSMILE Python, https://audeering.github.io/opensmile-python/
- Eyben et al. (2016), *The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing*, IEEE Transactions on Affective Computing, https://doi.org/10.1109/TAFFC.2015.2457417

**Pretrained speech models**
- Baevski et al. (2020), *wav2vec 2.0*, https://arxiv.org/abs/2006.11477
- Hsu et al. (2021), *HuBERT*, https://arxiv.org/abs/2106.07447
- Chen et al. (2021), *WavLM*, https://arxiv.org/abs/2110.13900
- Yang et al. (2021), *SUPERB: Speech processing Universal PERformance Benchmark*, https://arxiv.org/abs/2105.01051
- Hugging Face audio course, https://huggingface.co/learn/audio-course/chapter0/introduction

**Ensembles**
- scikit-learn ensemble guide (voting and stacking), https://scikit-learn.org/stable/modules/ensemble.html
