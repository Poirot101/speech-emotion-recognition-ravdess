"""Try to beat the baseline without cheating on the test actors.

    python scripts/improve_accuracy.py

The test actors are the 6 from the speaker split (4, 5, 6, 8, 10, 19). All the
model selection happens with GroupKFold(6) on the other 18 actors, and the test
actors are only used once per configuration at the end. (If I scored everything
on the test actors and reported the best one, that number would be biased.)

Search space: pooling x classifier x speaker normalisation, where the
normalisation is one of
    none     nothing
    neutral  subtract the mean of the speaker's 4 neutral clips. This is usable
             in practice: a new user records a few calm sentences first.
    oracle   standardise using the speaker's own 4-class clips. Not usable in
             practice, because RAVDESS has exactly balanced emotions per speaker
             and real users don't. Only here as an upper bound.

Writes reports/improvements.json and reports/improvements.md.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTHONWARNINGS", "ignore")  # silence ConvergenceWarning in workers

import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from speech_emotion.dataset import OBSERVED_EMOTIONS, load_dataset
from speech_emotion.train import DEFAULT_CACHES, RANDOM_STATE, build_model

DATA_DIR = REPO_ROOT / "data" / "ravdess"
REPORTS = REPO_ROOT / "reports"
HELD_OUT = [4, 5, 6, 8, 10, 19]
INNER = GroupKFold(n_splits=6)
POOLINGS = ("mean", "meanstd")
NORMS = ("none", "neutral", "oracle")
STOCHASTIC = {"mlp", "random_forest", "hist_gb"}


def make_classifier(name: str, seed: int = RANDOM_STATE):
    if name == "mlp":
        return build_model(classifier="mlp").set_params(mlp__random_state=seed)
    if name == "svm":  # plain SVC: same predictions as train.py's, without Platt scaling cost
        return make_pipeline(StandardScaler(), SVC(C=10, gamma="scale"))
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=1)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(name)


CLASSIFIERS = ("mlp", "svm", "logreg", "random_forest", "hist_gb")


def speaker_normalize(X, g, ref_X, ref_g, use_std):
    """Normalise each speaker's rows using only that speaker's reference clips."""
    out = np.empty_like(X)
    for s in np.unique(g):
        ref = ref_X[ref_g == s]
        if len(ref) == 0:
            raise ValueError(f"no reference clips for speaker {s}")
        mu = ref.mean(axis=0)
        sd = ref.std(axis=0) + 1e-8 if use_std else 1.0
        out[g == s] = (X[g == s] - mu) / sd
    return out


def prepare(pooling):
    X8, y8, g8 = load_dataset(DATA_DIR, None, cache_path=DEFAULT_CACHES[pooling],
                              verbose=False, pooling=pooling)
    obs = np.isin(y8, OBSERVED_EMOTIONS)
    X, y, g = X8[obs], y8[obs], g8[obs]
    neutral = y8 == "neutral"
    variants = {
        "none": X,
        "neutral": speaker_normalize(X, g, X8[neutral], g8[neutral], use_std=False),
        "oracle": speaker_normalize(X, g, X, g, use_std=True),
    }
    return variants, y, g


def label(r):
    return f"{r['classifier']} / {r['pooling']} / norm={r['normalization']}"


def main() -> int:
    data = {p: prepare(p) for p in POOLINGS}
    y, g = data["mean"][1], data["mean"][2]
    assert np.array_equal(y, data["meanstd"][1]) and np.array_equal(g, data["meanstd"][2])

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=RANDOM_STATE)
                  .split(data["mean"][0]["none"], y, g))
    assert sorted(set(g[te].tolist())) == HELD_OUT, "held-out actors changed"
    y_te, g_te = y[te], g[te]

    # model selection, training actors only
    print("Selection: GroupKFold(6) over the 18 training actors\n")
    print(f"{'classifier':<14} {'pooling':<8} {'norm':<8} {'inner CV':>9} {'+/-':>6}")
    rows = []
    for pooling in POOLINGS:
        for norm in NORMS:
            X = data[pooling][0][norm]
            for clf in CLASSIFIERS:
                sc = cross_val_score(make_classifier(clf), X[tr], y[tr], groups=g[tr],
                                     cv=INNER, n_jobs=-1)
                rows.append({"classifier": clf, "pooling": pooling, "normalization": norm,
                             "deployable": norm != "oracle",
                             "inner_cv_mean": float(sc.mean()), "inner_cv_std": float(sc.std()),
                             "fold_scores": sc.tolist()})
                print(f"{clf:<14} {pooling:<8} {norm:<8} {sc.mean():>8.1%} {sc.std():>6.1%}")

    def find(clf, pooling, norm):
        return next(r for r in rows if (r["classifier"], r["pooling"], r["normalization"])
                    == (clf, pooling, norm))

    baseline = find("mlp", "mean", "none")
    best_dep = max((r for r in rows if r["deployable"]), key=lambda r: r["inner_cv_mean"])
    best_oracle = max((r for r in rows if not r["deployable"]), key=lambda r: r["inner_cv_mean"])

    # evaluate the chosen configs on the test actors (once each)
    def fit_predict(r, seed=RANDOM_STATE):
        X = data[r["pooling"]][0][r["normalization"]]
        return make_classifier(r["classifier"], seed).fit(X[tr], y[tr]).predict(X[te])

    chosen = {"baseline": baseline, "best_deployable": best_dep, "oracle_upper_bound": best_oracle}
    preds = {k: fit_predict(r) for k, r in chosen.items()}
    held = {}
    for k, r in chosen.items():
        held[k] = {"config": label(r), "inner_cv": r["inner_cv_mean"],
                   "held_out_accuracy": float(accuracy_score(y_te, preds[k])),
                   "held_out_macro_f1": float(f1_score(y_te, preds[k], average="macro"))}

    # is the gain significant?
    base_ok = preds["baseline"] == y_te
    new_ok = preds["best_deployable"] == y_te
    b = int(np.sum(base_ok & ~new_ok)); c = int(np.sum(~base_ok & new_ok))
    mcnemar_p = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0

    rng = np.random.default_rng(RANDOM_STATE)
    speakers = np.unique(g_te)
    diffs = []
    for _ in range(2000):
        idx = np.concatenate([np.flatnonzero(g_te == s) for s in rng.choice(speakers, len(speakers))])
        diffs.append(new_ok[idx].mean() - base_ok[idx].mean())
    diff_ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]

    seed_spread = None
    if best_dep["classifier"] in STOCHASTIC:
        accs = [accuracy_score(y_te, fit_predict(best_dep, seed)) for seed in range(10)]
        seed_spread = {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                       "min": float(np.min(accs)), "max": float(np.max(accs))}

    per_speaker = {int(s): {"baseline": float(base_ok[g_te == s].mean()),
                            "best_deployable": float(new_ok[g_te == s].mean())}
                   for s in speakers}

    report = {"held_out_actors": HELD_OUT, "selection": "GroupKFold(6) on training actors",
              "candidates": rows, "held_out": held,
              "mcnemar_baseline_vs_best_deployable": {
                  "baseline_only_correct": b, "best_only_correct": c, "p_value": mcnemar_p},
              "speaker_bootstrap_accuracy_gain_95": diff_ci,
              "best_deployable_seed_spread": seed_spread,
              "per_speaker_accuracy": per_speaker}
    (REPORTS / "improvements.json").write_text(json.dumps(report, indent=2) + "\n")

    # write the markdown report
    L = ["# Improving accuracy", "",
         "Generated by `python scripts/improve_accuracy.py`. 4 classes "
         f"({', '.join(OBSERVED_EMOTIONS)}).", "",
         "**Protocol.** Every candidate is scored with `GroupKFold(6)` over the 18 "
         "training actors. The held-out actors (4, 5, 6, 8, 10, 19) are touched only "
         "for the configurations reported in the second table.", "",
         "## Selection (inner cross-validation, training actors only)", "",
         "| Classifier | Pooling | Speaker norm | Deployable | Inner CV | ± |",
         "|---|---|---|:---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["inner_cv_mean"]):
        L.append(f"| {r['classifier']} | {r['pooling']} | {r['normalization']} | "
                 f"{'yes' if r['deployable'] else 'no'} | {r['inner_cv_mean']:.1%} | "
                 f"{r['inner_cv_std']:.1%} |")
    L += ["", "## Held-out actors (evaluated once)", "",
          "| Role | Configuration | Inner CV | Held-out accuracy | Held-out macro-F1 |",
          "|---|---|---:|---:|---:|"]
    for k, v in held.items():
        L.append(f"| {k.replace('_', ' ')} | {v['config']} | {v['inner_cv']:.1%} | "
                 f"**{v['held_out_accuracy']:.2%}** | {v['held_out_macro_f1']:.3f} |")
    L += ["", "## Is the gain real?", "",
          f"- **McNemar (exact), baseline vs best deployable:** baseline-only correct = {b}, "
          f"best-only correct = {c}, p = {mcnemar_p:.4f}.",
          f"- **Accuracy gain, 95% speaker-level bootstrap:** "
          f"[{diff_ci[0]*100:+.1f}, {diff_ci[1]*100:+.1f}] pp. Only six test speakers, "
          "so this interval is wide by nature."]
    if seed_spread:
        L.append(f"- **Best deployable over 10 seeds:** {seed_spread['mean']:.2%} ± "
                 f"{seed_spread['std']:.2%} (min {seed_spread['min']:.2%}, max {seed_spread['max']:.2%}).")
    L += ["", "### Per test speaker", "", "| Actor | Baseline | Best deployable |", "|---:|---:|---:|"]
    for s, v in per_speaker.items():
        L.append(f"| {s} | {v['baseline']:.0%} | {v['best_deployable']:.0%} |")
    (REPORTS / "improvements.md").write_text("\n".join(L) + "\n")

    print("\nHeld-out (once each):")
    for k, v in held.items():
        print(f"  {k:<20} {v['config']:<40} inner {v['inner_cv']:.1%}  held-out {v['held_out_accuracy']:.2%}")
    print(f"McNemar p = {mcnemar_p:.4f} (b={b}, c={c}); speaker-bootstrap gain "
          f"[{diff_ci[0]*100:+.1f}, {diff_ci[1]*100:+.1f}] pp")
    if seed_spread:
        print(f"best deployable seeds: {seed_spread['mean']:.2%} +/- {seed_spread['std']:.2%}")
    print("Wrote reports/improvements.md and reports/improvements.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
