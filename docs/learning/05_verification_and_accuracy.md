# 05: Verification and accuracy: what the repo does, and why

Learning notes on the two scripts added at the end of the project:
`scripts/verify_results.py` and `scripts/improve_accuracy.py`. Each section is a
concept, the number it produced here, and an exercise.

---

## 1. Reproduction checks

**Concept.** A result you cannot regenerate is an anecdote. `verify_results.py`
retrains each published configuration from cached features and asserts the
accuracy *and* the full confusion matrix match `reports/metrics_*.json` exactly.

**Here.** All four published numbers reproduce exactly (77.60%, 59.38%, 66.94%, 40.28%).

**Why exact equality is possible.** Fixed seeds, sorted file order, a
deterministic split, and features cached once. Change any of these and the check
tells you immediately.

**Exercise.** Remove the `sorted()` in `find_audio_files`, clear the cache, and
rerun the check. Explain what breaks, and why it might pass on one machine and
fail on another.

---

## 2. Leakage checks as assertions

**Concept.** Do not *believe* a split is clean; assert it. The script checks that
no actor appears on both sides of the speaker split.

**Exercise.** Write the equivalent assertion for project 02: no augmented copy
of a held-out actor's clip is in the training set.

---

## 3. Seed spread: one run is one draw

**Concept.** An MLP's weight initialisation and shuffling depend on a random
seed. Different seeds give different models and different scores.

**Here.** The baseline over 10 seeds: **58.28% ± 1.18%** (range 56.25%–59.90%).
The published 59.38% uses seed 9, near the lucky end. Nothing was
cherry-picked (seed 9 was fixed at the start), but the lesson stands: a single
seed is a single sample.

**Rule of thumb.** Compare distributions over seeds, not single runs. A 1-point
gain between two single-seed runs is inside the noise.

**Exercise.** Plot accuracy against seed for the baseline and the best deployable
configuration on one chart. Do the ranges overlap?

---

## 4. Bootstrap confidence intervals: and why clips are not independent

**Concept.** Resample the test set with replacement thousands of times,
recompute accuracy each time, and read off the middle 95%.

**Here.**
- Clip-level bootstrap: **[52.6%, 66.1%]**
- Speaker-level bootstrap: **[49.5%, 68.2%]**

The clip-level interval treats all 192 test clips as independent. They are not:
each of the 6 test speakers contributes 32 clips, and clips from one speaker
succeed or fail together. Resampling *speakers* respects that structure and gives
the wider (and more realistic) interval.

**Per-speaker accuracy:** actor 4: 47%, actor 5: 69%, actor 6: 59%, actor 8: 75%,
actor 10: 41%, actor 19: 66%. The spread between voices is larger than any model
improvement in this repo.

**Exercise.** Explain in two sentences why the speaker-level interval must be
wider. Then work out roughly how many test speakers you would need to halve its width.

---

## 5. Permutation tests: what does chance look like?

**Concept.** Shuffle the labels, retrain, score; repeat. That gives the
distribution of scores a model gets from structure-free labels. The p-value is
the fraction of shuffles that score at least as well as the real labels.

**Here.** Real **62.11%** (grouped CV) against a shuffled-label mean of **25.10%**
(best shuffle 29.69%); p = 0.032.

**The p-value floor.** With *n* permutations, the smallest possible p is
1 / (n + 1). With 30 shuffles that is 0.032, so p = 0.032 means "better than
every shuffle", not "barely significant". Use more permutations for a smaller floor.

**Exercise.** Why does `permutation_test_score` shuffle labels *within* groups
when you pass `groups`? What would shuffling across speakers change?

---

## 6. Selection bias: the winner's curse

**Concept.** Try many configurations, pick the best by cross-validation, and that
best score is optimistic: part of why it won is favourable noise on those folds.

**Here, measured, not hypothetical.** `improve_accuracy.py` scored 30
configurations on the training actors, then evaluated the winners once on held-out actors:

| Configuration | Inner CV | Held-out | Drop |
|---|---:|---:|---:|
| Baseline (fixed in advance) | 60.9% | 59.38% | −1.5 pp |
| Best deployable (winner of 20) | 69.3% | 63.54% | **−5.8 pp** |
| Oracle upper bound (winner of 10) | 73.4% | 69.79% | −3.6 pp |

The configuration selected from the most candidates dropped the most. That is
selection bias, visible in this repo's own numbers.

**The fix at full rigour: nested cross-validation, now done.** Wrap the entire
selection procedure in an outer grouped CV loop. Each outer fold runs its own
inner selection. The outer scores then estimate the performance of *the
selection procedure*, no single test set is reused, and all 24 speakers
contribute. That is `scripts/nested_cv.py`:

```
for each of 6 outer folds (4 test actors, 20 training actors):
    for each of 20 deployable configurations:
        inner GroupKFold(5) score on the 20 training actors
    refit the best configuration on the 20 actors, predict the 4 test actors
baseline (fixed in advance): fit on the same 20, predict the same 4
```

| Fold | Test actors | Chosen | Baseline outer | Nested outer |
|---:|---|---|---:|---:|
| 1 | 6, 12, 18, 24 | hist_gb / meanstd / neutral | 68.4% | 71.1% |
| 2 | 5, 11, 17, 23 | logreg / meanstd / neutral | 54.8% | 73.4% |
| 3 | 4, 10, 16, 22 | logreg / meanstd / neutral | 44.5% | 63.3% |
| 4 | 3, 9, 15, 21 | hist_gb / meanstd / neutral | 55.6% | 63.3% |
| 5 | 2, 8, 14, 20 | logreg / meanstd / neutral | 75.9% | 79.7% |
| 6 | 1, 7, 13, 19 | logreg / meanstd / neutral | 63.9% | 64.8% |

Over all 768 clips: **baseline 60.55% ± 0.73%** (5 seeds), **nested 69.27%**.

**What to notice.**

- **Does the same configuration win every fold?** Partly. The *features* won every
  fold (mean+std pooling, neutral-clip normalisation); the *classifier* did not
  (logistic regression ×4, gradient boosting ×2). The stable part is the finding.
- **Fold-to-fold baseline accuracy ranges from 44.5% to 75.9%.** Which four
  speakers you test on moves the score more than any model change in this repo.
- **Three baseline numbers now exist**, 59.38% (seed 9, 6 actors), 58.28% ± 1.18%
  (10 seeds, same 6 actors), 60.55% ± 0.73% (nested, 24 actors). They are
  different test sets, not corrections of each other. Compare within a protocol.

**Exercise (new).** Rewrite `nested_cv.py` so the inner loop also chooses among
*feature sets only*, with the classifier fixed to logistic regression. Does the
nested score drop? That tells you how much of the gain came from choosing the
classifier.

---

## 7. McNemar's test: comparing two models on the same clips

**Concept.** When two classifiers are tested on the same clips, only the clips
where they *disagree* carry information about which is better. McNemar's test
counts them:
- *b* = clips only the baseline gets right
- *c* = clips only the new model gets right

Under "no difference", b and c are equally likely, so the test is an exact
binomial test on min(b, c) out of b + c (`scipy.stats.binomtest`).

**Here.** b = 25, c = 33, p = 0.36. The new model wins 8 more clips than it loses
not enough to rule out chance.

**Exercise.** How lopsided would b and c need to be, with b + c = 58, for p < 0.05?

---

## 8. Mean+std pooling

**Concept.** Keep the standard deviation over time as well as the mean. The mean
says what a coefficient's typical value was; the std says how much it *moved*.

**Here.** MLP inner CV 60.9% → **66.0%**, with nothing needed at prediction
time. It is available as `--pooling meanstd`.

**Limitation.** Still order-blind: std knows *how much* a coefficient varied,
not *when* or *in which direction*. That is what project 01 (deltas) and project
03 (CNN) add.

---

## 9. Speaker normalisation: a technique the dataset flatters

**Concept.** Remove each speaker's typical feature profile so the model sees
deviations from "how this person normally sounds" rather than the voice itself.
It attacks the exact failure this repo measured.

**The catch.** The normalisation statistics must come from *somewhere*:

| Where the per-speaker statistics come from | Deployable? | Best inner CV |
|---|:---:|---:|
| Nothing (no normalisation) | yes | 66.1% |
| That speaker's 4 neutral clips (mean only) | yes, ask a new user to read a few sentences calmly | 69.3% |
| All of that speaker's clips (mean and std), "oracle" | **no** | 73.4% |

The oracle is invalid for deployment because every RAVDESS speaker recorded a
**perfectly emotion-balanced** set of clips, so their average is emotion-free by
construction. A real user's recordings are not balanced.

An early exploratory run (not part of the committed scripts) made the point
sharply: normalising with **a few random clips** per speaker, in an arbitrary
emotion mix, scored **52–59%**, *worse* than no normalisation at all.

**Exercise.** Explain why normalising with random clips could hurt. Hint: what
happens to an angry clip if that speaker's "average" was computed mostly from
angry clips?

---

## 10. Model families: features mattered more than the classifier

With the same features, the five classifiers (MLP, RBF-SVM, logistic regression,
random forest, histogram gradient boosting) were usually within about 3–4 points
of each other. Switching from mean to mean+std pooling moved every classifier by
more than that. A grid search over the SVM's `C` and `gamma` (exploratory) found
68.1% against 67.7% for the defaults, inside the fold-to-fold spread.

**Lesson.** On small data, representation beats model choice, and tuning mostly fits noise.

---

## 11. Statistical power: why six test speakers limited everything

Every uncertainty measure on the single split said the same thing: **six
held-out speakers cannot reliably detect a gain of a few points.** The
speaker-level interval was about 19 points wide.

Nested CV over all 24 actors (§6) is what resolved it:

| Test | 6 held-out speakers | 24 speakers, nested |
|---|---|---|
| Gain being tested | +3.4 pts (seed mean) | +8.7 pts |
| Speaker-level 95% interval | [−3.1, +10.4] | **[+4.1, +13.7]** |
| Paired test | McNemar p = 0.36 | **Wilcoxon p = 0.006** (on per-speaker gains) |

Two lessons hide in that table:

- **Which test to use depends on the unit of independence.** McNemar on 768
  clips gives p < 0.001 in `reports/nested_cv.md`, but clips from one speaker
  are not independent, so that p-value is optimistic. Wilcoxon on 24 per-speaker
  gains treats the speaker as the unit.
- **The sign test disagrees** (15 up, 8 down, p = 0.21). It ignores *how much*
  each speaker changed. A large average gain with a third of speakers worse off
  is a real result and a real warning at the same time.

Ways forward from here, in increasing order of effort:

1. ~~Nested grouped CV over all 24 actors~~, done (`scripts/nested_cv.py`)
2. Leave-one-speaker-out evaluation for final comparisons (24 outer folds)
3. More speakers, CREMA-D has 91 (project 04)

**Exercise.** Add `scipy.stats.wilcoxon` to `improve_accuracy.py`'s six-speaker
comparison. With only six paired values, what is the smallest p-value the
two-sided test can return? (Work it out, then check.)

---

## Resources

- scikit-learn, nested vs non-nested cross-validation, https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- scikit-learn, `permutation_test_score`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html
- scikit-learn, cross-validation guide, https://scikit-learn.org/stable/modules/cross_validation.html
- scikit-learn, `LeaveOneGroupOut`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
- scikit-learn, common pitfalls, https://scikit-learn.org/stable/common_pitfalls.html
- `scipy.stats.binomtest`, see `help(scipy.stats.binomtest)` in your environment
- `scipy.stats.wilcoxon`, https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html
