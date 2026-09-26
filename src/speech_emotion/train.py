"""Train and evaluate the emotion classifier.

Run from the repo root:

    python -m speech_emotion.train --split random   # same as the tutorial
    python -m speech_emotion.train --split speaker  # test actors not in training
    python -m speech_emotion.train --emotions all --split speaker

The random split puts the same actors in train and test, so its accuracy is
inflated (the model partly learns voices). The speaker split is the number
that actually matters. See docs/LEARNING_NOTES.md.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .dataset import ALL_EMOTIONS, OBSERVED_EMOTIONS, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "ravdess"
DEFAULT_CACHES = {
    "mean": REPO_ROOT / "data" / "features_cache.npz",
    "meanstd": REPO_ROOT / "data" / "features_meanstd_cache.npz",
}
DEFAULT_CACHE = DEFAULT_CACHES["mean"]
DEFAULT_MODEL_DIR = REPO_ROOT / "models"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"

RANDOM_STATE = 9


def build_model(scale: bool = True, classifier: str = "mlp") -> Pipeline:
    """Scaler + classifier pipeline.

    MLP settings are copied from the tutorial so the baseline is comparable.
    alpha=0.01 is the L2 penalty, which matters here since there are about as
    many features (180) as clips per class.

    I added StandardScaler: MFCC 0 is in the hundreds while chroma is in [0, 1],
    and the MLP trains a lot better once they're on the same scale. It's inside
    the pipeline so the scaler only ever sees training data.

    classifier="svm" uses an RBF SVM instead (C=10, gamma="scale"). A grid
    search over C/gamma didn't find anything clearly better
    (reports/improvements.md).
    """
    if classifier == "mlp":
        estimator = ("mlp", MLPClassifier(
            alpha=0.01,
            batch_size=256,
            epsilon=1e-08,
            hidden_layer_sizes=(300,),
            learning_rate="adaptive",
            max_iter=500,
            random_state=RANDOM_STATE,
        ))
    elif classifier == "svm":
        # probability=True so predict.py can print probabilities
        estimator = ("svm", SVC(C=10, gamma="scale", probability=True,
                                random_state=RANDOM_STATE))
    else:
        raise ValueError(f"Unknown classifier {classifier!r}; use 'mlp' or 'svm'.")
    steps = ([("scaler", StandardScaler())] if scale else []) + [estimator]
    return Pipeline(steps)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    strategy: str = "random",
    test_size: float = 0.25,
):
    """Train/test split.

    "random": plain stratified shuffle, like the tutorial. Every actor ends up
    on both sides.
    "speaker": GroupShuffleSplit on actor id, so test actors are unseen.
    """
    if strategy == "random":
        return train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )
    if strategy == "speaker":
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(X, y, groups))
        held_out = sorted(set(groups[test_idx].tolist()))
        print(f"Held-out actors (unseen voices): {held_out}")
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    raise ValueError(f"Unknown split strategy {strategy!r}; use 'random' or 'speaker'.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help="folder holding Actor_01 ... Actor_24")
    parser.add_argument("--split", choices=["random", "speaker"], default="random",
                        help="random = tutorial baseline; speaker = held-out actors")
    parser.add_argument("--emotions", choices=["observed", "all"], default="observed",
                        help="'observed' = the 4-class subset; 'all' = 8 classes")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--pooling", choices=["mean", "meanstd"], default="mean",
                        help="'mean' = 180 features; 'meanstd' = 360 (adds std over time)")
    parser.add_argument("--classifier", choices=["mlp", "svm"], default="mlp")
    parser.add_argument("--no-scaler", action="store_true",
                        help="drop StandardScaler, matching the raw tutorial model")
    parser.add_argument("--cv", type=int, default=0, metavar="K",
                        help="also run K-fold cross-validation (0 = skip)")
    parser.add_argument("--cache", default="auto",
                        help="feature cache path; 'auto' picks one per pooling mode, "
                             "'none' disables caching")
    parser.add_argument("--model-out", default=None,
                        help="where to write the fitted model (.joblib)")
    args = parser.parse_args(argv)

    emotions = OBSERVED_EMOTIONS if args.emotions == "observed" else ALL_EMOTIONS
    if args.cache.lower() == "none":
        cache = None
    elif args.cache.lower() == "auto":
        cache = str(DEFAULT_CACHES[args.pooling])
    else:
        cache = args.cache
    # keep the old filenames for the default config so links in the docs still work
    is_default = (args.pooling, args.classifier) == ("mean", "mlp")
    stem = "mlp" if is_default else f"{args.classifier}_{args.pooling}"
    metrics_stem = "metrics" if is_default else f"metrics_{args.classifier}_{args.pooling}"

    print("=" * 68)
    print(f"Speech Emotion Recognition  |  split={args.split}  "
          f"classes={len(emotions)}")
    print("=" * 68)

    X, y, groups = load_dataset(args.data_dir, emotions, cache_path=cache,
                                pooling=args.pooling)

    X_train, X_test, y_train, y_test = split_data(
        X, y, groups, strategy=args.split, test_size=args.test_size
    )
    print(f"Train / test clips: {X_train.shape[0]} / {X_test.shape[0]}")
    print(f"Features extracted: {X_train.shape[1]}")

    model = build_model(scale=not args.no_scaler, classifier=args.classifier)
    print(f"Training {model.steps[-1][1].__class__.__name__} "
          f"(pooling={args.pooling}) ...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%\n")

    labels = sorted(set(y_test.tolist()))
    print(classification_report(y_test, y_pred, labels=labels, zero_division=0))
    print("Confusion matrix (rows = true, cols = predicted)")
    print("labels:", labels)
    print(confusion_matrix(y_test, y_pred, labels=labels))

    cv_scores = None
    if args.cv:
        # CV should match the split. Careful: plain cv=5 uses StratifiedKFold
        # without shuffling, and since files are sorted by actor that quietly
        # gives you actor-grouped folds. So both are spelled out here.
        if args.split == "speaker":
            cv = GroupKFold(n_splits=args.cv)
            cv_kwargs = {"groups": groups}
            scheme = f"GroupKFold by actor, {args.cv} folds"
        else:
            cv = StratifiedKFold(n_splits=args.cv, shuffle=True,
                                 random_state=RANDOM_STATE)
            cv_kwargs = {}
            scheme = f"StratifiedKFold (shuffled), {args.cv} folds"
        print(f"\nCross-validating: {scheme} ...")
        cv_scores = cross_val_score(build_model(scale=not args.no_scaler,
                                                classifier=args.classifier),
                                    X, y, cv=cv, n_jobs=-1, **cv_kwargs)
        print(f"CV accuracy: {cv_scores.mean() * 100:.2f}% "
              f"(+/- {cv_scores.std() * 100:.2f}%)")
        print("per fold:", np.round(cv_scores * 100, 2).tolist())

    model_path = Path(args.model_out) if args.model_out else (
        DEFAULT_MODEL_DIR / f"{stem}_{args.emotions}_{args.split}.joblib"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "emotions": emotions, "split": args.split,
                 "accuracy": float(accuracy), "pooling": args.pooling,
                 "classifier": args.classifier}, model_path)
    print(f"\nSaved model -> {model_path.relative_to(REPO_ROOT)}")

    metrics = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "emotions": emotions,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "scaler": not args.no_scaler,
        "pooling": args.pooling,
        "classifier": args.classifier,
        "accuracy": float(accuracy),
        "cv_folds": args.cv or None,
        "cv_scheme": (scheme if cv_scores is not None else None),
        "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
        "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
        "per_class": classification_report(
            y_test, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        },
        "python": platform.python_version(),
        "random_state": RANDOM_STATE,
    }
    DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = DEFAULT_REPORT_DIR / f"{metrics_stem}_{args.emotions}_{args.split}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved metrics -> {metrics_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
