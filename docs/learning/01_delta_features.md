# 01: Delta features

**Goal:** add first- and second-order time derivatives of the MFCCs (Δ and ΔΔ)
and measure whether they improve **speaker-independent** accuracy beyond
mean+std pooling.

**Effort:** 2–4 days. **Prerequisite:** `00_foundations.md`.

---

## Why this matters here

- Averaging over time throws away *how the voice changes* (`docs/LEARNING_NOTES.md`, §1).
- There is already evidence that change matters. Adding the **standard deviation**
  over time raised inner-CV MLP accuracy from 60.9% to 66.0%
  (`reports/improvements.md`).
- But std only says *how much* a coefficient moved. Δ says *how fast and in which
  direction* it moves from frame to frame; ΔΔ says whether that change is
  speeding up. A voice rising quickly and a voice rising slowly have similar std
  and very different Δ, and rate of change is an arousal cue.

---

## Concepts to understand first

1. **Finite differences.** The simplest derivative is `Δc[t] ≈ c[t+1] − c[t−1]`.
   Understand why raw frame differences are noisy.
2. **Smoothed derivatives.** librosa's `delta` estimates the derivative over a
   window of frames (`width`, odd, ≥ 3). Larger width: smoother but less
   responsive. Read the docs and find out *how* it smooths.
3. **Order of operations.** Compute deltas on the `(coefficients × frames)`
   matrix, *then* pool over time. A delta of the already-pooled vector would be a
   derivative across coefficient index, meaningless.
4. **Predict before you measure.** The mean of Δ over a clip is roughly
   `(value at the end − value at the start) / duration`: small, and mostly about
   the endpoints. So **mean(Δ) is probably close to useless, and std(Δ) is
   probably the informative summary.** Write your prediction in your lab notebook
   before running anything, then check it.
5. **Bookkeeping.** 40 MFCCs × {Δ, ΔΔ} × {mean, std} = 160 extra columns. With
   ~576 training clips, every column you add has a cost.

---

## Build milestones

### M1: Explore one clip
Compute MFCC, Δ, ΔΔ for a single clip and plot all three as heatmaps
(`librosa.display.specshow`).
**Checkpoint:** all three have shape `(40, T)`; Δ is near zero in silent regions.

### M2: Extend the feature extractor
Add an option to `extract_feature` (for example `deltas=0|1|2`) that appends
pooled Δ / ΔΔ blocks. Follow the pattern `pooling` already uses.
**Checkpoint:** a new test proves the default output is unchanged, copy the idea
from `test_meanstd_pooling_doubles_length_and_keeps_means`. All existing tests pass.

### M3: A new cache
Write to a **new** cache file. Never overwrite `features_cache.npz`.
**Checkpoint:** shape `(1440, 360 + extra)`. `load_dataset` refuses a cache with
the wrong column count, see how, and extend it if your option changes the count.

### M4: The experiment grid (inner CV only)
Reuse the protocol in `scripts/improve_accuracy.py`: `GroupKFold(6)` over the 18
training actors, the held-out actors untouched, and at least 5 seeds for the MLP.

| Variant | Columns |
|---|---:|
| meanstd (reference) | 360 |
| + std(Δ) | 400 |
| + mean(Δ) + std(Δ) | 440 |
| + std(Δ) + std(ΔΔ) | 440 |
| everything | 520 |

Then try `width` of 5, 9, and 15 on the best variant.

### M5: Held-out once, then nested over all 24 actors
Evaluate the single winner on the held-out actors first, quick, and a sanity
check. Then run the real comparison with nested grouped CV (`scripts/nested_cv.py`
with deltas added to the candidate features), against the nested 69.27%
procedure, reporting the speaker-level bootstrap and Wilcoxon on per-speaker
gains. Six held-out speakers could not resolve a 3-point gap (McNemar p = 0.36);
expect the same for deltas.

### M6: Write it up
A lab-notebook entry: hypothesis, the numbers with spread, keep or drop. If you
keep it, add a row to `docs/RESULTS.md` and a test.

---

## Implementation blueprint

Signatures and data flow only, the bodies are yours.

### Data flow

```
wav ─► resample 48 kHz ─► MFCC (40 × T)
                              ├─► Δ  = delta(MFCC, width=w, order=1)   (40 × T)
                              └─► ΔΔ = delta(MFCC, width=w, order=2)   (40 × T)
      pool each over time with the chosen statistics ─► hstack ─► vector
```

Deltas are computed on the frame matrix, **then** pooled. Never the reverse.

### Signatures

```python
# src/speech_emotion/features.py
DELTA_STATS = ("mean", "std")

def extract_feature(file_name, mfcc=True, chroma=True, mel=True,
                    target_sr=TARGET_SR, pooling="mean",
                    deltas: int = 0,                       # 0, 1 (Δ) or 2 (Δ and ΔΔ)
                    delta_width: int = 9,
                    delta_stats: tuple[str, ...] = ("std",)) -> np.ndarray: ...

def feature_dimension(..., deltas=0, delta_stats=("std",)) -> int: ...
def feature_names(..., deltas=0, delta_stats=("std",)) -> list[str]:
    """e.g. 'mfcc_d1_03_std', 'mfcc_d2_17_mean' appended after the existing names."""

def _safe_width(n_frames: int, width: int) -> int:
    """Largest odd width <= min(width, n_frames), at least 3; raise if the clip is too short."""
```

### Cache key

Put every parameter that changes the numbers into the filename:

```python
def cache_path(pooling: str, deltas: int, width: int, stats: tuple[str, ...]) -> Path:
    return REPO_ROOT / "data" / f"features_{pooling}_d{deltas}_w{width}_{'-'.join(stats)}.npz"
```

`load_dataset` checks the column count; also store the settings **inside** the
`.npz` (`np.savez(..., settings=json.dumps(settings))`) and refuse a mismatch.
That closes the "same shape, different contents" hole.

### Assertions to write before the experiment

```python
def test_default_output_unchanged(tone):        # deltas=0 is bit-identical to today
def test_names_match_dimension():                # for every (deltas, stats) combination
def test_delta_of_a_steady_tone_is_near_zero(tone):   # mean |Δ| ≪ mean |MFCC|
def test_short_clip_gets_a_smaller_width(tmp_path):   # 0.1 s clip does not crash
```

### Experiment loop

Reuse the harness idea from `06_accuracy_playbook.md`: build a dict of feature
matrices keyed by variant name, fix the classifier (logistic regression is fast
and deterministic, and it won 4 of 6 nested folds), then run nested CV once.
Add neutral-clip normalisation as a second axis, deltas may help less once the
speaker is normalised away, or more.

### Lab-notebook table to fill in

| Variant | Columns | Inner CV (mean ± sd) | Nested outer | Δ vs meanstd | Speakers improved |
|---|---:|---:|---:|---:|---:|

---

## Pitfalls

- **Width longer than the clip.** librosa's default edge handling needs at least
  `width` frames; very short clips error out. Guard, or choose a smaller width.
- **Stale caches.** The column-count check catches a wrong shape, but a same-shape
  cache with different contents slips through. Name cache files by configuration.
- **Single-seed comparisons.** One MLP run against another tells you nothing if
  the seed spread is 2–3 points.
- **Tuning `width` on the held-out actors.** That is selection leakage.
- **Deltas on everything.** Mel deltas add 256+ columns fast. Start with MFCCs.

---

## How you will know you succeeded

- The inner-CV gain over meanstd is larger than the seed spread.
- The held-out gain is positive, reported *with* its speaker-level interval,
  which may include zero with only six test speakers. Say so if it does.
- **Diagnostic:** if Δ helps a lot on the random split but not on the speaker
  split, the new columns are encoding *speaker identity*. That is a finding worth
  writing up, not a failure.

---

## Stretch goals

- Other summaries: `mean(|Δ|)`, percentiles (10th/50th/90th), skewness.
- Prosodic features, the classic emotion features: pitch contour statistics
  (`librosa.pyin`), energy (`librosa.feature.rms`), zero-crossing rate
  (`librosa.feature.zero_crossing_rate`).

---

## Resources

- librosa `delta`, https://librosa.org/doc/0.11.0/generated/librosa.feature.delta.html
- librosa `mfcc`, https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html
- Fayek's filter bank / MFCC post (background for the MFCC pipeline), https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
- Davis & Mermelstein (1980), the MFCC paper, https://doi.org/10.1109/TASSP.1980.1163420
- scikit-learn common pitfalls, https://scikit-learn.org/stable/common_pitfalls.html
