"""Plots for the report (confusion matrices, class balance, feature profiles).

    python -m speech_emotion.visualize

Saves PNGs to reports/figures/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import OBSERVED_EMOTIONS, load_dataset
from .features import N_CHROMA, N_MEL, N_MFCC

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "reports" / "figures"
DEFAULT_CACHE = REPO_ROOT / "data" / "features_cache.npz"


def plot_confusion_matrix(metrics_path: Path, out_path: Path) -> None:
    """Plot the confusion matrix from a metrics_*.json file."""
    metrics = json.loads(metrics_path.read_text())
    labels = metrics["confusion_matrix"]["labels"]
    matrix = np.array(metrics["confusion_matrix"]["matrix"], dtype=float)
    # normalise per row since classes aren't the same size
    normed = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.2 * len(labels) + 2))
    im = ax.imshow(normed, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix - {metrics['split']} split\n"
                 f"accuracy {metrics['accuracy']:.1%}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{int(matrix[i, j])}\n{normed[i, j]:.0%}",
                    ha="center", va="center",
                    color="white" if normed[i, j] > 0.55 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalised")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def plot_class_balance(y: np.ndarray, out_path: Path) -> None:
    """Number of clips per emotion."""
    labels, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, counts, color="#4C72B0")
    ax.set_ylabel("clips")
    ax.set_title("Class balance")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def plot_feature_profiles(X: np.ndarray, y: np.ndarray, out_path: Path) -> None:
    """Average feature vector for each emotion, one subplot per feature type.

    Quick sanity check that the emotions actually look different. The mel
    panel shows it best.
    """
    blocks = [("MFCC", 0, N_MFCC),
              ("Chroma", N_MFCC, N_MFCC + N_CHROMA),
              ("Mel (log scale)", N_MFCC + N_CHROMA, N_MFCC + N_CHROMA + N_MEL)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (title, lo, hi) in zip(axes, blocks):
        for emotion in sorted(set(y.tolist())):
            mean_vec = X[y == emotion, lo:hi].mean(axis=0)
            ax.plot(range(lo, hi), mean_vec, label=emotion, linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("feature index")
        ax.spines[["top", "right"]].set_visible(False)
        if title.startswith("Mel"):
            ax.set_yscale("log")
    axes[0].set_ylabel("mean value")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Average feature profile per emotion")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "ravdess"))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y, _ = load_dataset(args.data_dir, OBSERVED_EMOTIONS,
                           cache_path=args.cache, verbose=False)
    plot_class_balance(y, FIG_DIR / "class_balance.png")
    plot_feature_profiles(X, y, FIG_DIR / "feature_profiles.png")

    for metrics_path in sorted((REPO_ROOT / "reports").glob("metrics_*.json")):
        out = FIG_DIR / f"confusion_{metrics_path.stem.replace('metrics_', '')}.png"
        plot_confusion_matrix(metrics_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
