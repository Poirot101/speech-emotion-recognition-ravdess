# Learning notes

What this project actually taught, written as I went. Each section is a thing I
either did not know at the start or got wrong the first time.

---

## 1. Audio is variable-length; classical ML is not

A 3-second clip and a 5-second clip are different-sized arrays, but
`MLPClassifier.fit` needs a rectangular `(n_samples, n_features)` matrix. Every
design decision in `features.py` follows from resolving that mismatch.

The standard fix is **time-averaging**: compute a time–frequency representation
(a matrix of `coefficients × frames`), then take the mean of each coefficient
across frames. A clip of any length collapses to one fixed vector.

```
clip (48000 × 4 samples)
  → STFT / mel filterbank → (coefficients × ~400 frames)
  → mean over frames      → (coefficients,)        ← fixed length
```

**The cost is real and worth naming.** Averaging over time discards *ordering*.
A pitch contour that rises at the end and one that falls at the end have the
same mean and become the same vector, yet that contour is exactly what
distinguishes a question from a statement, or surprise from anger. This single
limitation is the main reason a CNN or LSTM over the raw spectrogram beats this
approach by a wide margin: they can see the time axis that this model throws
away.

## 2. The three feature families, and what each one hears

| Feature | Dims | What it captures | Intuition |
|---|---:|---|---|
| **MFCC** | 40 | Short-term power spectrum on a mel scale, decorrelated by a DCT | *Timbre*, the shape of the vocal tract. The workhorse of speech processing since the 1980s. |
| **Chroma** | 12 | Energy folded into the 12 pitch classes, octave-independent | *Intonation*, borrowed from music IR, where it identifies chords |
| **Mel spectrogram** | 128 | Energy per mel-spaced frequency band | *Loudness across frequency*, weighted the way human hearing is |

**180 features total.** They overlap heavily, MFCCs are literally computed
*from* the mel spectrogram (mel → log → DCT). I assumed the redundancy was
harmless, because the MLP can learn to ignore what it does not need.

**The ablation proved that assumption half-wrong, and it is the most useful
thing I measured.** 40 MFCC features *alone* score 76.04% / 59.38%, matching
the full 180-feature model, and *identical* on the speaker-independent split:

| Variant | Features | Random | Speaker-independent |
|---|---:|---:|---:|
| MFCC + chroma + mel | 180 | 77.60% | 59.38% |
| **MFCC only** | **40** | 76.04% | **59.38%** |
| Mel only | 128 | 61.98% | 51.04% |
| Chroma only | 12 | 37.50% | 32.81% |
| MFCC + mel | 168 | 76.56% | 55.21% |

The DCT inside the MFCC computation is a decorrelating compression: those 40
coefficients are a more compact, better-conditioned encoding of what the 128 mel
bands already contain. Adding the raw mel bands back mostly adds correlated
noise, and on the speaker split, `MFCC + mel` is **4 points worse** than MFCC
alone, because the extra capacity gets spent on speaker-specific spectral detail
that does not transfer to a new voice.

Chroma is close to dead weight here (37.50% alone, barely above the 25% floor).
That makes sense in hindsight: chroma folds energy into 12 pitch classes for
*musical key detection*, and emotional speech is not tonal in that sense.

So the realistic version of "use MFCC + chroma + mel" is: **use MFCC.** The other
140 columns cost 4.5× the extraction time to buy about 1.5 points on a split
that was already lying to me, and nothing at all on the one that wasn't. I kept
all three in the default so the baseline stays comparable to the published
version, but `scripts/run_ablations.py` is the part that says what to actually
ship.

Why the mel scale at all: human pitch perception is roughly logarithmic. We
hear 200 Hz vs. 400 Hz as a large jump and 5000 Hz vs. 5200 Hz as nearly
identical, though both are 200 Hz apart. Mel spacing gives fine resolution where
the ear is sensitive and coarse resolution where it is not, so the model does
not waste capacity on distinctions nobody can hear.

## 3. The split that changes the answer

**This was the most important thing I got wrong.**

The reference tutorial does `train_test_split(X, y, test_size=0.25)`, a random
shuffle over all clips. Since each of the 24 actors contributes 60 clips, that
puts *the same voice* in both the training and test sets. Actor 14's "happy"
clips train the model; actor 14's other "happy" clips then test it.

The model can score well by recognising **voices**, not emotions: "this timbre
belongs to actor 14, and actor 14's happy clips sound like *this*." That is not
the task. On a new speaker the shortcut is worthless.

The better alternative is a **speaker-independent** split: hold out entire
actors, so the test set is voices the model has never heard. `sklearn` provides
exactly this via `GroupShuffleSplit` with `groups=actor_id`.

Both are implemented, and the gap between them is the lesson:

```bash
python -m speech_emotion.train --split random    # optimistic
python -m speech_emotion.train --split speaker   # held-out actors
```

See [RESULTS.md](RESULTS.md) for the measured gap. The generalisable rule: **if
your data has a grouping structure (speakers, patients, users, devices,
sessions), a random split leaks it and inflates your score.** Split on the
group. This is the same error as leaking a patient across folds in medical
imaging, and it is very easy to ship without noticing because nothing crashes
and the number just looks good.

## 4. Scaling is not optional for a neural net

The 180 raw features live on wildly different scales: MFCC coefficient 0 runs to
several hundred, mel energies are small positive numbers spanning orders of
magnitude, chroma sits in [0, 1]. Gradient descent on unscaled inputs takes a
badly-conditioned path, the large-magnitude features dominate the early
gradient and the small ones barely move.

`StandardScaler` (zero mean, unit variance per column) fixes it, and it is the
cheapest accuracy gain in the whole project, **6.25 points on the random split
and 8.9 on the speaker split**, a bigger swing than removing 140 of the 180
features. Toggle it off with `--no-scaler` to reproduce.

A related tell: the unscaled variant is also the one that trips sklearn's
`ConvergenceWarning`, hitting `max_iter=500` without the loss plateauing. Bad
conditioning shows up twice, once as lower accuracy, once as slower
optimisation.

The subtle part: the scaler must be fit on **training data only**. If you
standardise the full matrix before splitting, the test set's mean and variance
have leaked into training. Wrapping it in a `Pipeline` makes this structurally
impossible to get wrong, `pipeline.fit(X_train)` fits the scaler on the
training fold alone, and cross-validation refits it per fold automatically.

## 5. Why an MLP, and what the hyper-parameters do

`MLPClassifier` is a feed-forward network: 180 inputs → 300 hidden units (ReLU)
→ 4 softmax outputs. Unlike SVM or Naive Bayes it learns its own intermediate
representation of the features.

| Parameter | Value | Role |
|---|---|---|
| `hidden_layer_sizes` | `(300,)` | One hidden layer, 300 units |
| `alpha` | `0.01` | L2 penalty. The main brake on overfitting, with ~180 training clips per class and 180 features, there is plenty of room to memorise |
| `batch_size` | `256` | Mini-batch size for the adam optimiser |
| `learning_rate` | `'adaptive'` | Cut the step size when the loss stops improving |
| `max_iter` | `500` | Epoch cap |

Roughly 55,000 parameters against ~576 training clips. That ratio should alarm
you, and it is why `alpha` matters so much and why the speaker-independent
number is the one to trust.

`random_state` is pinned so the run is reproducible. Without it, weight
initialisation and shuffling vary and accuracy wanders by a couple of points
between runs, enough to fool you into thinking a change helped.

## 6. Why only four emotions?

The baseline keeps `calm`, `happy`, `fearful`, `disgust`. This is not arbitrary:

- **It balances perfectly.** 192 clips each, so accuracy is a fair metric and
  the random-guess floor is a clean 25%.
- **It avoids the hardest confusions.** `neutral` vs. `calm` is genuinely
  near-identical acoustically, even human raters disagree on it. Including
  both mostly measures how well the model guesses a coin flip.

`--emotions all` runs all eight. Accuracy drops substantially, and the
confusion matrix shows *where*: neutral↔calm, and the high-arousal cluster
(angry/fearful/surprised) blurring together, because time-averaged energy
features capture **arousal** (how activated) far better than **valence**
(positive or negative). Angry and happy are both loud and high-energy; what
separates them is largely contour and context, the very things step 1 averaged
away.

## 7. Things that bit me

- **`librosa.feature.melspectrogram(X, sr=...)`**, the positional signature in
  most tutorials was removed in librosa 0.10. It is keyword-only now:
  `melspectrogram(y=X, sr=...)`. Same for `stft`.
- **`pip install sklearn`**, that PyPI stub is deprecated and now errors. The
  package is `scikit-learn`.
- **`pyaudio`** is listed as a prerequisite in most write-ups but is never
  imported. It needs the PortAudio system library and fails to build on a clean
  macOS box. It is only needed for live microphone capture; it is deliberately
  **not** in `requirements.txt`.
- **`glob.glob` returns filesystem order**, which differs between machines. The
  loader sorts, so the feature matrix (and therefore every split) is identical
  on any checkout.
- **`soundfile` returns `(n_samples, n_channels)` for stereo**, and librosa
  expects mono. RAVDESS is mono so it never triggers, but the loader downmixes
  defensively; the first custom stereo file you feed it would otherwise crash.
- **Sample rate silently changes what a feature *means*.** This was the worst
  bug in the project and the last one I found. MFCC/chroma/mel bins are defined
  relative to the sample rate, so coefficient 7 at 16 kHz describes a different
  frequency band than coefficient 7 at 48 kHz. My extractor originally used
  whatever rate the file happened to have, which is fine for RAVDESS (uniformly
  48 kHz) and catastrophic for anything else: on the held-out actors, feeding
  the same clips at 16 kHz dropped accuracy from 59.4% to **25.0% (exactly
  chance) while mean confidence rose to 100%**. Nothing raised an error. The
  model just got louder as it got worse. `extract_feature` now pins
  `TARGET_SR = 48_000`. The deeper lesson: resampling fixes *bin misalignment*
  but cannot restore *lost bandwidth*, so 16 kHz audio still only reaches 29.7%.
  If train and deploy rates differ, fix it in training, not at inference.
- **Feature extraction dominates runtime** (~1–2 min for 1,440 clips) while
  fitting the MLP takes seconds. Caching features to an `.npz` turned the
  tune-the-classifier loop from minutes into seconds. Cache the expensive
  deterministic step, not the cheap one.

## 8. What I would do next

1. **Delta features.** Add first and second derivatives of the MFCCs
   (`librosa.feature.delta`). They reintroduce a little of the temporal
   information step 1 discards, for 80 extra columns and no architecture change.
2. **Data augmentation.** Pitch-shift, time-stretch, and additive noise are
   label-preserving for emotion and would roughly triple an effectively tiny
   training set.
3. **A CNN on the mel spectrogram.** Treat the spectrogram as an image and stop
   averaging over time. This is the step that actually closes the gap to
   published RAVDESS numbers.
4. **Report more than accuracy.** Macro-F1 and the per-class recall already in
   the classification report say more than one headline number, especially on
   the 8-class run.

## 9. Trying to beat the baseline: and needing more speakers to prove it

The last thing I did was try to improve accuracy, with the rule that every
choice had to be made on the training actors and the held-out actors used only
once. The best configuration I could actually deploy averaged 61.7% over seeds
against the baseline's 58.3%. Then I tested whether that difference was real,
and it wasn't demonstrable: McNemar p = 0.36, and the speaker-level interval for
the gain ran from −3.1 to +10.4 points.

Three things I did not appreciate before doing this:

- **A single seed is a single sample.** The baseline over 10 seeds is
  58.28% ± 1.18%. The 59.38% I had been quoting all along is near the top of that
  range, not cherry-picked, since seed 9 was fixed at the start, but one draw.
- **Picking a winner inflates its score, measurably.** The best of 20 candidates
  dropped 5.8 points from cross-validation to the held-out actors. The baseline,
  chosen in advance, dropped 1.5. I had read about selection bias; I had not seen
  it this plainly in my own numbers.
- **The test set, not the model, was the bottleneck.** Six held-out speakers give
  a speaker-level interval about 19 points wide, and per-speaker accuracy ranges
  from 41% to 75%. No improvement of a few points could have been confirmed.
  The next step is not a cleverer model; it is nested grouped cross-validation
  over all 24 actors, or more speakers.

I also nearly shipped speaker normalisation as the win. Normalising each speaker
with all their clips scored best, until I noticed every RAVDESS speaker has a
perfectly emotion-balanced set of clips, which no real user would. Normalising
with a few random clips instead scored *worse* than doing nothing. The dataset's
design had been making the technique look better than it is.

So I did the next step. `scripts/nested_cv.py` wraps the whole selection in an
outer `GroupKFold(6)` over all 24 actors: each fold picks its configuration on
20 actors and tests it on the other 4. The search scored 69.27% against the
baseline's 60.55% on the same 768 clips, and the speaker-level interval for the
gain, [+4.1, +13.7] points, excludes zero. What I took from it:

- **The question had an answer; my test set couldn't hear it.** Six speakers gave
  p = 0.36 for a 3-point gap. Twenty-four gave Wilcoxon p = 0.006 for an
  8.7-point one. Nothing about the model changed between those two sentences.
- **Numbers from different protocols don't mix.** 60.55% is not a corrected
  59.38%; it is a different test set. I only compare nested with nested and
  held-out with held-out.
- **The winning ingredients were stable; the classifier wasn't.** Every fold chose
  mean+std pooling with neutral-clip normalisation, but four picked logistic
  regression and two gradient boosting. Representation over model, again.
- **A significant average can still hurt people.** 8 of 24 speakers got worse.
  The sign test (15 up, 8 down) is not significant even though the mean gain is.

Full numbers: [RESULTS.md](RESULTS.md#trying-to-beat-the-baseline).

