"""Predict the emotion of one or more audio files.

    python -m speech_emotion.predict path/to/clip.wav
    python -m speech_emotion.predict clip.wav --model models/mlp_observed_speaker.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .features import extract_feature

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "mlp_observed_random.joblib"


def load_model(model_path: str | Path):
    """Load a .joblib bundle saved by train.py."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Train one first:\n"
            "    python -m speech_emotion.train --split random"
        )
    return joblib.load(model_path)


def predict_file(audio_path: str | Path, bundle) -> tuple[str, dict[str, float]]:
    """Returns (label, {emotion: probability})."""
    model = bundle["model"]
    # older bundles don't have a "pooling" key, they were all mean-pooled
    pooling = bundle.get("pooling", "mean")
    features = extract_feature(audio_path, pooling=pooling).reshape(1, -1)
    label = str(model.predict(features)[0])

    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
        probabilities = {str(c): float(p) for c, p in zip(classes, probs)}
    return label, probabilities


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("audio", nargs="+", help="one or more audio files")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args(argv)

    bundle = load_model(args.model)
    print(f"Model: {Path(args.model).name}  "
          f"(classes: {', '.join(bundle['emotions'])})\n")

    for audio in args.audio:
        label, probs = predict_file(audio, bundle)
        print(f"{Path(audio).name}")
        print(f"  predicted: {label}")
        for name, p in sorted(probs.items(), key=lambda kv: -kv[1]):
            bar = "#" * int(round(p * 30))
            print(f"    {name:<10} {p:6.1%} {bar}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
