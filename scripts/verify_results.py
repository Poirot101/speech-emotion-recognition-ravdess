"""Re-check the numbers in docs/RESULTS.md and how much to trust them.

    python scripts/verify_results.py           # a few minutes
    python scripts/verify_results.py --quick   # skip the permutation test

Needs the dataset / feature cache, so it doesn't run in CI.

1. retrain and check the accuracies match reports/metrics_*.json exactly
2. check no actor is in both train and test for the speaker split
3. seed spread: how much accuracy moves just from the MLP's random init
4. 95% bootstrap intervals, resampling clips and resampling speakers. Clips
   from the same speaker aren't independent so the speaker one is the one to
   believe (and it's wide, there are only 6 test speakers)
5. permutation test against shuffled labels

Writes reports/verification.json. Exit code 1 if check 1 or 2 fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold, permutation_test_score

from speech_emotion.dataset import ALL_EMOTIONS, OBSERVED_EMOTIONS, load_dataset
from speech_emotion.train import DEFAULT_CACHE, RANDOM_STATE, build_model, split_data

DATA_DIR = REPO_ROOT / "data" / "ravdess"
REPORTS = REPO_ROOT / "reports"


def quiet(fn, *args, **kwargs):
    """Run fn with stdout suppressed (split_data prints the held-out actors)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def load(emotions):
    return load_dataset(DATA_DIR, emotions, cache_path=DEFAULT_CACHE, verbose=False)


def check_reproduction() -> tuple[bool, list[dict]]:
    print("1. Reproduction of published metrics")
    ok, rows = True, []
    for emotions_key, emotions in [("observed", OBSERVED_EMOTIONS), ("all", ALL_EMOTIONS)]:
        X, y, g = load(emotions)
        for split in ["random", "speaker"]:
            path = REPORTS / f"metrics_{emotions_key}_{split}.json"
            published = json.loads(path.read_text())
            X_tr, X_te, y_tr, y_te = quiet(split_data, X, y, g, strategy=split)
            y_pred = build_model().fit(X_tr, y_tr).predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            labels = published["confusion_matrix"]["labels"]
            cm_match = (confusion_matrix(y_te, y_pred, labels=labels).tolist()
                        == published["confusion_matrix"]["matrix"])
            match = abs(acc - published["accuracy"]) < 1e-12 and cm_match
            ok &= match
            rows.append({"run": path.stem, "published": published["accuracy"],
                         "reproduced": acc, "confusion_matrix_match": cm_match,
                         "match": match})
            print(f"   {path.stem:<28} published {published['accuracy']:.4%}  "
                  f"reproduced {acc:.4%}  {'OK' if match else 'MISMATCH'}")
    return ok, rows


def check_leakage() -> bool:
    print("\n2. Speaker leakage in the speaker-independent split")
    X, y, g = load(OBSERVED_EMOTIONS)
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=RANDOM_STATE)
                  .split(X, y, g))
    overlap = set(g[tr].tolist()) & set(g[te].tolist())
    print(f"   test actors {sorted(set(g[te].tolist()))}; overlap with train: "
          f"{sorted(overlap) or 'none'}  {'OK' if not overlap else 'LEAK'}")
    return not overlap


def check_seed_spread(n_seeds: int = 10) -> dict:
    print(f"\n3. Initialisation-seed spread (4-class, speaker split, {n_seeds} seeds)")
    X, y, g = load(OBSERVED_EMOTIONS)
    X_tr, X_te, y_tr, y_te = quiet(split_data, X, y, g, strategy="speaker")
    accs = []
    for seed in range(n_seeds):
        model = build_model().set_params(mlp__random_state=seed)
        accs.append(accuracy_score(y_te, model.fit(X_tr, y_tr).predict(X_te)))
    accs = np.array(accs)
    print(f"   accuracy {accs.mean():.2%} +/- {accs.std():.2%}  "
          f"(min {accs.min():.2%}, max {accs.max():.2%})")
    print("   -> improvements smaller than this are probably just noise.")
    return {"seeds": list(range(n_seeds)), "accuracies": accs.tolist(),
            "mean": float(accs.mean()), "std": float(accs.std()),
            "min": float(accs.min()), "max": float(accs.max())}


def bootstrap_intervals(n_boot: int = 2000) -> dict:
    print("\n4. 95% bootstrap intervals (4-class, speaker split, seed 9)")
    X, y, g = load(OBSERVED_EMOTIONS)
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=RANDOM_STATE)
                  .split(X, y, g))
    y_pred = build_model().fit(X[tr], y[tr]).predict(X[te])
    correct = (y_pred == y[te]).astype(float)
    g_te = g[te]
    rng = np.random.default_rng(RANDOM_STATE)

    clip = [correct[rng.integers(0, len(correct), len(correct))].mean()
            for _ in range(n_boot)]
    speakers = np.unique(g_te)
    per_speaker = {s: correct[g_te == s] for s in speakers}
    spk = [np.concatenate([per_speaker[s] for s in rng.choice(speakers, len(speakers))]).mean()
           for _ in range(n_boot)]

    def ci(a):
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    out = {"accuracy": float(correct.mean()), "clip_bootstrap_95": ci(clip),
           "speaker_bootstrap_95": ci(spk),
           "per_speaker_accuracy": {int(s): float(per_speaker[s].mean()) for s in speakers}}
    print(f"   accuracy {out['accuracy']:.2%}")
    print(f"   clip-level bootstrap    [{out['clip_bootstrap_95'][0]:.1%}, "
          f"{out['clip_bootstrap_95'][1]:.1%}]  (treats clips as independent, they aren't)")
    print(f"   speaker-level bootstrap [{out['speaker_bootstrap_95'][0]:.1%}, "
          f"{out['speaker_bootstrap_95'][1]:.1%}]  (only 6 test speakers)")
    print("   per speaker: " + ", ".join(f"actor {s}: {a:.0%}"
                                        for s, a in out["per_speaker_accuracy"].items()))
    return out


def permutation_check(n_permutations: int) -> dict:
    print(f"\n5. Permutation test ({n_permutations} label shuffles, GroupKFold by actor)")
    X, y, g = load(OBSERVED_EMOTIONS)
    score, null, p = permutation_test_score(
        build_model(), X, y, groups=g, cv=GroupKFold(n_splits=6),
        n_permutations=n_permutations, random_state=RANDOM_STATE, n_jobs=-1)
    print(f"   real {score:.2%} vs shuffled-label mean {null.mean():.2%} "
          f"(max {null.max():.2%}); p = {p:.3f} "
          f"(smallest possible p = {1 / (n_permutations + 1):.3f})")
    return {"score": float(score), "null_mean": float(null.mean()),
            "null_max": float(null.max()), "p_value": float(p),
            "n_permutations": n_permutations}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="skip the permutation test")
    parser.add_argument("--permutations", type=int, default=30)
    args = parser.parse_args(argv)

    repro_ok, repro_rows = check_reproduction()
    leak_ok = check_leakage()
    report = {"reproduction": repro_rows, "reproduction_ok": repro_ok,
              "no_speaker_leakage": leak_ok,
              "seed_spread": check_seed_spread(),
              "bootstrap": bootstrap_intervals()}
    if not args.quick:
        report["permutation_test"] = permutation_check(args.permutations)

    (REPORTS / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote reports/verification.json")
    status = repro_ok and leak_ok
    print("VERIFIED" if status else "VERIFICATION FAILED")
    return 0 if status else 1


if __name__ == "__main__":
    raise SystemExit(main())
