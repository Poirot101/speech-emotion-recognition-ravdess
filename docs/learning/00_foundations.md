# 00: Foundations

Everything the follow-up projects in this folder assume. Skim what you already know.
**Do not skip section C**, every project in this roadmap can produce a
convincing, wrong number if the evaluation is off, and this repo has already
caught three such errors in itself.

---

## A. Audio and signal processing

### Concepts to be able to explain in your own words

| Concept | What you should be able to explain | Where it appears in this repo |
|---|---|---|
| Sampling rate, Nyquist | A signal sampled at *fs* only contains frequencies up to *fs/2* | `docs/USAGE.md`: 16 kHz audio physically lacks content above 8 kHz, so accuracy stays at 29.7% even after resampling |
| Framing and windowing | Why speech is analysed in short (20–40 ms) overlapping windows; what a Hann window prevents (spectral leakage) | librosa's STFT defaults: `n_fft=2048`, `hop_length=512` |
| STFT | `n_fft` sets frequency resolution, `hop_length` sets time resolution; you cannot maximise both | Every feature starts here |
| Magnitude, power, decibels | power = \|STFT\|²; log/dB compression mirrors loudness perception and tames dynamic range | Mel energies span orders of magnitude (why the feature-profile plot uses a log axis) |
| Mel scale, filterbank | Triangular filters, dense at low frequencies where hearing is sensitive | The 128 mel bands |
| MFCC | log-mel energies → DCT → keep the first *k*; the DCT decorrelates and compresses | Ablation: 40 MFCCs alone match all 180 features |
| Chroma | Energy folded into 12 pitch classes, built for music | 37.5% alone: nearly useless for speech |
| Frames vs clips | A clip of *N* samples gives roughly `1 + N / hop_length` frames | Why time pooling is needed at all |

### Exercises: do them, they are the fastest route to real understanding

1. **Count frames.** For a 3.5 s RAVDESS clip at 48 kHz with `hop_length=512`,
   predict the number of STFT frames on paper, then check with librosa.
2. **Feel the trade-off.** Plot one clip's spectrogram with `n_fft=256` and with
   `n_fft=4096`. Write two sentences on what sharpens and what blurs.
3. **Rebuild MFCCs by hand.** `librosa.feature.melspectrogram` →
   `librosa.power_to_db` → `scipy.fft.dct(type=2, norm="ortho", axis=0)` → keep
   the first 40 rows. Compare against `librosa.feature.mfcc` on the same clip.
   If they differ, find out which default you missed.
4. **See lost bandwidth.** Resample a clip 48 kHz → 16 kHz → 48 kHz and plot both
   spectrograms. Point at the missing energy and connect it to the sample-rate
   table in `docs/USAGE.md`.
5. **Prove the pooling line.** `features.py` computes `np.mean(M.T, axis=0)`.
   Show on paper why that equals `M.mean(axis=1)`, then confirm in code.

---

## B. Machine-learning fundamentals

| Concept | What to understand | In this repo |
|---|---|---|
| Bias–variance, overfitting | Why a flexible model fits training data and fails on new data | A 55k-parameter MLP trained on 576 clips |
| Regularisation | L2 penalty, dropout, early stopping, data augmentation, all trade training fit for generalisation | `alpha=0.01` |
| Feature scaling | Why gradient methods and distance-based models need comparable scales | −6.25 pp without `StandardScaler` |
| Optimisation | Loss, gradients, learning rate, Adam, convergence | sklearn's `ConvergenceWarning` on unscaled data |
| Model families | Linear, kernel (SVM), trees and boosting, neural networks, each has a different inductive bias | `reports/improvements.md` compares five |
| Pipelines | Every fitted transform learns from training data only | `build_model()` in `train.py` |

---

## C. Evaluation methodology: the section that matters most

### C1. Leakage: five kinds, all with examples from this repo

| Kind | What leaks | Example here | Effect |
|---|---|---|---|
| Group leakage | The same speaker in train and test | Random split | 77.6% → 59.4% once removed |
| Preprocessing leakage | Test statistics used to fit a transform | Scaler fit before splitting | Prevented by `Pipeline` |
| Selection leakage | Test set used to choose between models | Picking the best of 30 configs on held-out actors | Why `improve_accuracy.py` selects on inner CV |
| Augmentation leakage | Copies of a clip on both sides of a split | Augmenting before splitting (project 02) | Inflated, meaningless scores |
| Default-parameter traps | A library default quietly changes the experiment | `cv=5` does not shuffle → speaker-grouped folds | Notebook reported 58.86% as "random folds" |

### C2. Cross-validation variants and when each fits

- `KFold` / `StratifiedKFold(shuffle=True)`, independent samples only.
- `GroupKFold`, samples cluster by speaker, patient, session: **the default for this dataset**.
- `LeaveOneGroupOut`, one speaker per fold; 24 folds; good for per-speaker analysis.
- `GroupShuffleSplit`, one grouped train/test split (how the held-out actors are chosen).

### C3. Nested evaluation and selection bias

If you try many configurations and report the best score *on the data used to
choose it*, that score is optimistic, you selected partly on noise. The fix is
two levels: choose inside the training actors (inner CV), then evaluate the
chosen configuration once on held-out actors (outer test).

**Exercise:** in `reports/improvements.md`, compare each reported
configuration's inner-CV score with its held-out score. Explain the gap.

### C4. Metrics

Accuracy, macro-F1, per-class recall, confusion matrices. Know when accuracy
lies: `neutral` has half the clips of every other class, so on 8 classes a
model can ignore it and lose little accuracy (its recall was 21%).

### C5. Uncertainty and significance

- **Seed spread**, rerun with different random seeds; a gain smaller than
  the spread is not evidence.
- **Bootstrap intervals**, resample the test set. Resampling *clips* assumes
  they are independent; they are not (one speaker contributes 32). Resampling
  *speakers* is the realistic one, and with six test speakers it is wide.
- **McNemar's test**, compares two classifiers on the *same* test clips using
  only the clips where they disagree.
- **Permutation test**, trains on shuffled labels many times to show what
  chance looks like.

**Exercise:** run `python scripts/verify_results.py`, read
`reports/verification.json`, and explain why the speaker-level interval is
wider than the clip-level one.

---

## D. Tools and habits

- **NumPy axis semantics**, nearly every bug in feature code is an axis bug.
- **The sklearn estimator API**, `fit` / `transform` / `predict`, `Pipeline`,
  `GridSearchCV(..., groups=...)`.
- **Git**, one experiment per commit, with the result in the message.
- **A lab notebook.** Keep one. Suggested entry format:

```
## <date>: <experiment name>
Hypothesis: what you expect and why
Change:     exactly what differs from the reference
Protocol:   split, folds, seeds
Result:     numbers, with spread
Conclusion: keep / drop / inconclusive, and why
Next:       the one thing to try after this
```

Record failures too. "Delta width 15 did nothing" saves you a week later.

---

## Checklist before starting project 01

- [ ] Exercises A1–A5 done
- [ ] Can explain all five leakage kinds, with an example of each
- [ ] Ran `scripts/verify_results.py` and can interpret every number it prints
- [ ] Read `docs/LEARNING_NOTES.md` end to end

---

## Resources

**Audio and signal processing**
- Haytham Fayek's 2016 blog post on filter banks and MFCCs, https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
- librosa `mfcc`, https://librosa.org/doc/0.11.0/generated/librosa.feature.mfcc.html
- librosa `melspectrogram`, https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html
- librosa `power_to_db`, https://librosa.org/doc/0.11.0/generated/librosa.power_to_db.html
- Davis & Mermelstein (1980), *Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences*, IEEE Trans. ASSP, the paper that established MFCCs. https://doi.org/10.1109/TASSP.1980.1163420

> The librosa links point at the 0.11 docs, which were the version-pinned pages
> reachable when this was written. You have librosa 1.0 installed, if a
> signature looks different, trust `help(librosa.feature.mfcc)` over the page.

**Machine learning**
- 3Blue1Brown, neural networks series, https://www.3blue1brown.com/topics/neural-networks
- Goodfellow, Bengio & Courville, *Deep Learning* (free online), https://www.deeplearningbook.org/
- *Dive into Deep Learning* (free, with code), https://d2l.ai/
- NumPy for absolute beginners, https://numpy.org/doc/stable/user/absolute_beginners.html
- pandas in 10 minutes, https://pandas.pydata.org/docs/user_guide/10min.html

**Evaluation**
- scikit-learn, cross-validation guide, https://scikit-learn.org/stable/modules/cross_validation.html
- `GroupKFold`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html
- `LeaveOneGroupOut`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
- Nested vs non-nested CV (worked example), https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Hyper-parameter search, https://scikit-learn.org/stable/modules/grid_search.html
- `permutation_test_score`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
- Learning curves, https://scikit-learn.org/stable/modules/learning_curve.html
- **Common pitfalls and recommended practices** (read this one twice), https://scikit-learn.org/stable/common_pitfalls.html
