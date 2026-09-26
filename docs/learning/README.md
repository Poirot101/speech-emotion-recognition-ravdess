# Learning roadmap: build the follow-up projects yourself

A study plan for the roadmap items in the main README, plus the concepts behind
the verification, accuracy and app work already in the repo. Each file explains
why the item matters here, the concepts to learn first, milestones with
checkpoints, an implementation blueprint (signatures and data flow, not
solutions), pitfalls, and resources.

---

## The files

| # | File | What you learn | Effort | Needs |
|---|---|---|---|---|
| 00 | [`00_foundations.md`](00_foundations.md) | Audio signal processing, ML fundamentals, **evaluation methodology** | 1–2 weeks |, |
| 01 | [`01_delta_features.md`](01_delta_features.md) | Frame-level feature engineering, disciplined experiment grids | 2–4 days | 00 |
| 02 | [`02_augmentation.md`](02_augmentation.md) | Label-preserving transforms, leakage-safe pipelines | ~1 week | 01 |
| 03 | [`03_cnn_spectrogram.md`](03_cnn_spectrogram.md) | PyTorch, CNNs, training and regularisation on tiny data | 3–5 weeks | 00, 02 |
| 04 | [`04_cross_corpus.md`](04_cross_corpus.md) | Domain shift, multi-dataset evaluation | 1–2 weeks | 01 |
| 05 | [`05_verification_and_accuracy.md`](05_verification_and_accuracy.md) | What the current repo does to verify and improve itself, and why, including the nested CV result | 2–3 days | 00 |
| 06 | [`06_accuracy_playbook.md`](06_accuracy_playbook.md) | Ten more ways to raise accuracy, ranked by evidence, with implementation blueprints | pick per item | 00, 05 |
| 07 | [`07_app_walkthrough.md`](07_app_walkthrough.md) | How the Streamlit app works; building speaker enrolment into it | 1 week | Python; 06 §1 |

Effort assumes you are comfortable with Python and scikit-learn and can spend a
few focused hours a day. Treat them as rough.

**Suggested order:** 00 → 05 → 07 → 06 §1–2 → 01 → 02 → 04 → 06 §3 → 03.
Read 05 early, it explains the tools every later project reuses. Then 07 and
the first item of 06: speaker enrolment is the one improvement already proven,
and building it teaches the codebase end to end. Do cross-corpus (04) before the
CNN (03): it is cheaper, and it tells you whether a CNN is worth the effort for
the data you actually care about. Try pretrained embeddings (06 §3) before the
CNN too, if they win, the CNN becomes an exercise rather than the path to accuracy.

---

## The numbers to beat

4-class, speaker-independent, held-out actors 4, 5, 6, 8, 10, 19.

| Configuration | Inner CV (training actors) | Held-out, seed 9 | Held-out, 10 seeds |
|---|---:|---:|---:|
| Baseline, MLP, mean pooling | 60.9% | 59.38% | 58.28% ± 1.18% |
| Best deployable, MLP, mean+std pooling, neutral-clip speaker normalisation | 69.3% | 63.54% | 61.72% ± 1.40% |
| Upper bound, **not deployable**, logistic regression, mean+std, oracle speaker normalisation | 73.4% | 69.79% | deterministic |

And the same question asked of all 24 actors with nested grouped CV
(`reports/nested_cv.md`), **a different protocol, so compare these two rows only
with each other**:

| Procedure | Nested CV, all 24 actors, 768 clips |
|---|---:|
| Baseline, MLP, mean pooling | 60.55% ± 0.73% |
| Search over 20 deployable configurations | **69.27%** |

Read these honestly before you set yourself a target:

- On the six held-out speakers the best deployable gain (+3.4 points across seeds)
  was **not significant**: McNemar p = 0.36, interval [−3.1, +10.4].
- Over all 24 speakers the search's gain is **+8.7 points**, interval
  [+4.1, +13.7], Wilcoxon p = 0.006. Every fold chose mean+std pooling with
  neutral-clip normalisation. **Use nested CV for your own final comparisons**,
  six speakers could not hear this.
- 8 of 24 speakers got *worse* under the winning setup.
- Per-speaker accuracy on the baseline ranges from **41% to 75%**. An average
  hides that the model works for some voices and not others.

Sources: `reports/improvements.md`, `reports/verification.json`, `reports/nested_cv.md`.

---

## Ground rules for every project

1. **The speaker-independent number is the one that counts.** Report the random
   split only as a comparison.
2. **Choose on training actors, test once.** Every setting, feature and model
   choice is made with `GroupKFold` inside the training actors. The held-out
   actors are evaluated once per configuration you report, and for the final
   comparison, nest the whole procedure over all 24 actors (`scripts/nested_cv.py`).
3. **A gain smaller than the seed spread is not a gain.** Run at least 5 seeds for
   anything stochastic.
4. **Change one thing at a time**, and keep a lab notebook, failures included.
5. **Predict before you measure.** Write down what you expect and why. Being
   wrong in writing is how the intuition gets built.
6. **Leave the baseline reproducible.** New options must not change default
   outputs; write a test that proves it, as the repo does for `pooling`.

---

## Before you start: can you answer these?

If not, begin with `00_foundations.md`.

- Why does a random split inflate accuracy on this dataset, and by how much?
- Why does `StandardScaler` belong *inside* the `Pipeline`?
- What does `hop_length` control, and what does changing it trade away?
- Why did the notebook's `cv=5` produce speaker-grouped folds?
- Why did the selected configuration lose 5.8 points between inner CV and the
  held-out actors, while the baseline lost only 1.5?
- Why can the baseline be 59.38%, 58.28% and 60.55% at once without any of them
  being wrong?
- Why does Wilcoxon on 24 speakers say p = 0.006 while a sign test on the same
  speakers says p = 0.21?
