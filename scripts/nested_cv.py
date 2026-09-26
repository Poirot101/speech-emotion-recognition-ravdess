"""Nested grouped cross-validation over all 24 actors.

    python scripts/nested_cv.py

improve_accuracy.py only had 6 test speakers, which wasn't enough to tell if
the winner really beat the baseline (McNemar p = 0.36). This uses all 24:

    outer loop   GroupKFold(6) over all 24 actors -> 4 test actors per fold
    inner loop   GroupKFold(5) over the other 20 actors -> choose a configuration
    outer test   fit the chosen configuration on the 20 actors, predict the 4

Each actor is tested exactly once and never affects the choice for its own
fold, so the outer score measures the whole "search then pick the best"
procedure rather than one config chosen after the fact.

The baseline (MLP / mean / no norm) is fixed, so it has no inner loop and just
gets scored on the same outer folds. The oracle normalisation is left out since
it can't be used in practice (see improve_accuracy.py).

Writes reports/nested_cv.json and reports/nested_cv.md.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from scipy.stats import binomtest, wilcoxon
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_score

from improve_accuracy import CLASSIFIERS, POOLINGS, STOCHASTIC, make_classifier, prepare
from speech_emotion.dataset import OBSERVED_EMOTIONS
from speech_emotion.train import RANDOM_STATE

REPORTS = REPO_ROOT / "reports"
OUTER = GroupKFold(n_splits=6)
INNER = GroupKFold(n_splits=5)
DEPLOYABLE_NORMS = ("none", "neutral")
BASELINE = ("mlp", "mean", "none")
N_SEEDS = 5
N_BOOT = 5000


def main() -> int:
    start = time.time()
    data = {p: prepare(p) for p in POOLINGS}
    y, g = data["mean"][1], data["mean"][2]
    assert np.array_equal(y, data["meanstd"][1]) and np.array_equal(g, data["meanstd"][2])
    candidates = [(c, p, n) for p in POOLINGS for n in DEPLOYABLE_NORMS for c in CLASSIFIERS]

    def features(cfg):
        _, pooling, norm = cfg
        return data[pooling][0][norm]

    folds = list(OUTER.split(features(BASELINE), y, g))
    base_pred = {s: np.empty(len(y), dtype=object) for s in range(N_SEEDS)}
    nest_pred = {s: np.empty(len(y), dtype=object) for s in range(N_SEEDS)}
    fold_rows = []

    for k, (tr, te) in enumerate(folds):
        test_actors = sorted(set(g[te].tolist()))
        print(f"\nOuter fold {k + 1}/6  test actors {test_actors}")
        inner = []
        for cfg in candidates:
            X = features(cfg)
            sc = cross_val_score(make_classifier(cfg[0]), X[tr], y[tr], groups=g[tr],
                                 cv=INNER, n_jobs=-1)
            inner.append((float(sc.mean()), cfg))
        inner.sort(key=lambda t: -t[0])
        best_score, best = inner[0]
        base_inner = next(s for s, c in inner if c == BASELINE)
        print(f"  chosen {'/'.join(best)} (inner {best_score:.1%}); "
              f"baseline inner {base_inner:.1%}")

        for seed in range(N_SEEDS):
            for cfg, store in ((BASELINE, base_pred), (best, nest_pred)):
                if seed and cfg[0] not in STOCHASTIC:  # deterministic: reuse seed 0
                    store[seed][te] = store[0][te]
                    continue
                X = features(cfg)
                model = make_classifier(cfg[0], seed)
                store[seed][te] = model.fit(X[tr], y[tr]).predict(X[te])

        fold_rows.append({
            "fold": k + 1, "test_actors": test_actors,
            "chosen": "/".join(best), "chosen_inner_cv": best_score,
            "baseline_inner_cv": base_inner,
            "baseline_outer": float(np.mean([accuracy_score(y[te], base_pred[s][te])
                                             for s in range(N_SEEDS)])),
            "nested_outer": float(np.mean([accuracy_score(y[te], nest_pred[s][te])
                                           for s in range(N_SEEDS)])),
            "top5_inner": [{"config": "/".join(c), "inner_cv": s} for s, c in inner[:5]],
        })
        r = fold_rows[-1]
        print(f"  outer (mean of {N_SEEDS} seeds): baseline {r['baseline_outer']:.1%}  "
              f"nested {r['nested_outer']:.1%}")

    # accuracy over all 768 clips, per seed
    base_acc = [float(np.mean(base_pred[s] == y)) for s in range(N_SEEDS)]
    nest_acc = [float(np.mean(nest_pred[s] == y)) for s in range(N_SEEDS)]

    # paired tests for each seed so one lucky seed can't decide it
    base_ok = np.array([base_pred[s] == y for s in range(N_SEEDS)])   # seeds x clips
    nest_ok = np.array([nest_pred[s] == y for s in range(N_SEEDS)])
    mcnemar = []
    for s in range(N_SEEDS):
        b = int(np.sum(base_ok[s] & ~nest_ok[s])); c = int(np.sum(~base_ok[s] & nest_ok[s]))
        mcnemar.append({"seed": s, "baseline_only": b, "nested_only": c,
                        "p_value": float(binomtest(min(b, c), b + c, 0.5).pvalue)})

    # bootstrap over speakers (averaged over seeds)
    speakers = np.unique(g)
    per_speaker_gain = np.array([nest_ok[:, g == s].mean() - base_ok[:, g == s].mean()
                                 for s in speakers])
    rng = np.random.default_rng(RANDOM_STATE)
    boot = [per_speaker_gain[rng.integers(0, len(speakers), len(speakers))].mean()
            for _ in range(N_BOOT)]
    gain_ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    speakers_improved = int(np.sum(per_speaker_gain > 0))
    speakers_worse = int(np.sum(per_speaker_gain < 0))
    speakers_unchanged = len(speakers) - speakers_improved - speakers_worse
    sign_p = float(binomtest(speakers_improved, speakers_improved + speakers_worse, 0.5).pvalue)
    # wilcoxon also uses the size of each gain, not just the sign
    wilcoxon_p = float(wilcoxon(per_speaker_gain[per_speaker_gain != 0]).pvalue)

    chosen_counts = Counter(r["chosen"] for r in fold_rows)
    report = {
        "protocol": {"outer": "GroupKFold(6) over 24 actors", "inner": "GroupKFold(5)",
                     "candidates": ["/".join(c) for c in candidates],
                     "baseline": "/".join(BASELINE), "seeds": list(range(N_SEEDS)),
                     "emotions": OBSERVED_EMOTIONS, "n_clips": int(len(y))},
        "folds": fold_rows,
        "chosen_configuration_counts": dict(chosen_counts),
        "overall": {
            "baseline_accuracy_per_seed": base_acc,
            "nested_accuracy_per_seed": nest_acc,
            "baseline_mean": float(np.mean(base_acc)), "baseline_std": float(np.std(base_acc)),
            "nested_mean": float(np.mean(nest_acc)), "nested_std": float(np.std(nest_acc)),
        },
        "mcnemar_per_seed": mcnemar,
        "speaker_bootstrap_gain_95": gain_ci,
        "per_speaker_gain": {int(s): float(v) for s, v in zip(speakers, per_speaker_gain)},
        "speakers_improved": speakers_improved, "speakers_worse": speakers_worse,
        "speakers_unchanged": speakers_unchanged,
        "speaker_sign_test_p": sign_p,
        "speaker_wilcoxon_p": wilcoxon_p,
        "runtime_seconds": round(time.time() - start),
    }
    (REPORTS / "nested_cv.json").write_text(json.dumps(report, indent=2) + "\n")

    o = report["overall"]
    L = ["# Nested grouped cross-validation", "",
         "Generated by `python scripts/nested_cv.py`. 4 classes "
         f"({', '.join(OBSERVED_EMOTIONS)}), all 24 actors, {len(y)} clips.", "",
         "**Protocol.** Outer `GroupKFold(6)`: each fold tests 4 actors. Inside each "
         "fold, `GroupKFold(5)` over the remaining 20 actors chooses among "
         f"{len(candidates)} deployable configurations (2 poolings × 5 classifiers × "
         "{none, neutral-clip} speaker normalisation). The chosen configuration is "
         "refit on the 20 actors and scored on the 4. The baseline is fixed in "
         f"advance and scored on the same folds. Stochastic models use {N_SEEDS} seeds; "
         "logistic regression and histogram gradient boosting (early stopping is off "
         "below 10,000 samples) give identical results for every seed.",
         "", "## Overall (every actor tested once)", "",
         "| Procedure | Accuracy, mean ± std over seeds |", "|---|---:|",
         f"| Baseline (MLP / mean / no normalisation) | **{o['baseline_mean']:.2%}** ± {o['baseline_std']:.2%} |",
         f"| Nested selection (search, then test) | **{o['nested_mean']:.2%}** ± {o['nested_std']:.2%} |",
         "", "## Is the gain real?", "",
         f"- **Speaker-level bootstrap, 95% interval for the gain** (24 speakers, "
         f"{N_BOOT} resamples, averaged over seeds): "
         f"[{gain_ci[0]*100:+.1f}, {gain_ci[1]*100:+.1f}] points.",
         f"- **Speakers improved / worse / unchanged:** {speakers_improved} / "
         f"{speakers_worse} / {speakers_unchanged} "
         f"(sign test p = {sign_p:.4f}; Wilcoxon signed-rank on per-speaker gains "
         f"p = {wilcoxon_p:.4f}).",
         "- **McNemar per seed** (treats the 768 clips as independent, which they are "
         "not (each speaker contributes 32), so these p-values are optimistic): " + "; ".join(
             f"seed {m['seed']}: b={m['baseline_only']}, c={m['nested_only']}, "
             f"p={m['p_value']:.4f}" for m in mcnemar) + ".",
         "", "## Per outer fold", "",
         "| Fold | Test actors | Chosen configuration | Chosen inner CV | Baseline outer | Nested outer |",
         "|---:|---|---|---:|---:|---:|"]
    for r in fold_rows:
        L.append(f"| {r['fold']} | {', '.join(map(str, r['test_actors']))} | {r['chosen']} | "
                 f"{r['chosen_inner_cv']:.1%} | {r['baseline_outer']:.1%} | {r['nested_outer']:.1%} |")
    L += ["", "**How often each configuration was chosen:** " + ", ".join(
        f"{c} ×{n}" for c, n in chosen_counts.most_common()) + ".", "",
        "## Per speaker (gain in accuracy points, averaged over seeds)", "",
        "| Actor | Gain |", "|---:|---:|"]
    for s, v in report["per_speaker_gain"].items():
        L.append(f"| {s} | {v*100:+.1f} |")
    (REPORTS / "nested_cv.md").write_text("\n".join(L) + "\n")

    print(f"\nBaseline {o['baseline_mean']:.2%} ± {o['baseline_std']:.2%}   "
          f"nested {o['nested_mean']:.2%} ± {o['nested_std']:.2%}")
    print(f"speaker-bootstrap gain [{gain_ci[0]*100:+.1f}, {gain_ci[1]*100:+.1f}] pp; "
          f"speakers improved {speakers_improved}/{len(speakers)}; sign p = {sign_p:.4f}; "
          f"Wilcoxon p = {wilcoxon_p:.4f}")
    print(f"chosen: {dict(chosen_counts)}")
    print(f"Wrote reports/nested_cv.md and reports/nested_cv.json "
          f"({report['runtime_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
