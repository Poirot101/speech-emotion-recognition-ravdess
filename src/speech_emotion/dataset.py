"""Loading RAVDESS: filename parsing + a feature cache.

There's no labels CSV in RAVDESS, everything is in the filename. For example
03-01-06-01-02-01-12.wav splits into 7 fields:

    01 modality       01=full-AV, 02=video-only, 03=audio-only
    02 vocal channel  01=speech,  02=song
    03 EMOTION        01=neutral 02=calm 03=happy 04=sad
                      05=angry 06=fearful 07=disgust 08=surprised
    04 intensity      01=normal, 02=strong  (neutral has no strong variant)
    05 statement      01="Kids are talking by the door"
                      02="Dogs are sitting by the door"
    06 repetition     01 or 02
    07 ACTOR          01..24  (odd = male, even = female)

Field 3 is the label. Field 7 (actor) gets returned as `groups` so we can do
a split where test actors never appear in training. With a random split the
model can partly cheat by recognising the voice.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import extract_feature, feature_dimension

EMOTIONS: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# 4-class subset, same one the tutorial I started from uses
OBSERVED_EMOTIONS: list[str] = ["calm", "happy", "fearful", "disgust"]

ALL_EMOTIONS: list[str] = list(EMOTIONS.values())

INTENSITY = {"01": "normal", "02": "strong"}
STATEMENT = {"01": "kids_talking", "02": "dogs_sitting"}


@dataclass(frozen=True)
class RavdessFile:
    """Parsed RAVDESS filename."""

    path: Path
    emotion: str
    intensity: str
    statement: str
    repetition: int
    actor: int

    @property
    def gender(self) -> str:
        # odd = male, even = female
        return "male" if self.actor % 2 == 1 else "female"


def parse_ravdess_filename(path: str | Path) -> RavdessFile:
    """Parse a RAVDESS filename.

    Raises ValueError for anything that isn't a RAVDESS name (wrong number of
    fields, unknown emotion code). This mostly happens when you point it at
    your own recordings, and a clear error beats an IndexError later on.
    """
    path = Path(path)
    parts = path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"{path.name!r} is not a RAVDESS filename: expected 7 hyphen-separated "
            f"fields (e.g. '03-01-06-01-02-01-12.wav'), found {len(parts)}."
        )

    emotion_code = parts[2]
    if emotion_code not in EMOTIONS:
        raise ValueError(
            f"{path.name!r} has unknown emotion code {emotion_code!r}; "
            f"expected one of {sorted(EMOTIONS)}."
        )

    return RavdessFile(
        path=path,
        emotion=EMOTIONS[emotion_code],
        intensity=INTENSITY.get(parts[3], "unknown"),
        statement=STATEMENT.get(parts[4], "unknown"),
        repetition=int(parts[5]),
        actor=int(parts[6]),
    )


def find_audio_files(data_dir: str | Path) -> list[Path]:
    """All Actor_*/*.wav files under data_dir, sorted.

    Sorted because glob order depends on the filesystem, and I want the same
    feature matrix (and the same splits) on every machine.
    """
    pattern = os.path.join(str(data_dir), "Actor_*", "*.wav")
    return sorted(Path(p) for p in glob.glob(pattern))


def load_dataset(
    data_dir: str | Path,
    observed_emotions: list[str] | None = None,
    cache_path: str | Path | None = None,
    verbose: bool = True,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for every clip in data_dir (the folder with Actor_01..24).

    Returns (X, y, groups): X is (n_clips, 180) or 360 for meanstd, y is the
    emotion labels, groups is the actor id of each clip.

    observed_emotions filters the labels (None = all 8). Extraction takes a
    minute or two for all 1440 clips, so pass cache_path to save an .npz and
    skip it next time. Use a different cache file for each pooling mode.
    """
    observed = list(observed_emotions) if observed_emotions else ALL_EMOTIONS

    if cache_path and Path(cache_path).exists():
        cached = np.load(cache_path, allow_pickle=True)
        X, y, groups = cached["X"], cached["y"], cached["groups"]
        expected = feature_dimension(pooling=pooling)
        if X.shape[1] != expected:
            raise ValueError(
                f"Cache {cache_path} holds {X.shape[1]}-dim features but pooling="
                f"{pooling!r} needs {expected}. Point --cache at a different file or "
                "delete this one to re-extract."
            )
        keep = np.isin(y, observed)
        if verbose:
            print(f"Loaded cached features from {cache_path} "
                  f"({keep.sum()} of {len(y)} clips match the requested emotions)")
        return X[keep], y[keep], groups[keep]

    files = find_audio_files(data_dir)
    if not files:
        raise FileNotFoundError(
            f"No audio found under {data_dir}/Actor_*/*.wav. "
            "Run `python scripts/download_dataset.py` first."
        )

    # always extract all 8 emotions so the cache works for any subset
    X_list: list[np.ndarray] = []
    y_list: list[str] = []
    group_list: list[int] = []

    for i, path in enumerate(files, start=1):
        meta = parse_ravdess_filename(path)
        X_list.append(extract_feature(path, mfcc=True, chroma=True, mel=True,
                                      pooling=pooling))
        y_list.append(meta.emotion)
        group_list.append(meta.actor)
        if verbose and i % 100 == 0:
            print(f"  extracted {i}/{len(files)} files")

    X = np.asarray(X_list, dtype=np.float64)
    y = np.asarray(y_list)
    groups = np.asarray(group_list, dtype=int)

    expected = feature_dimension(pooling=pooling)
    if X.shape[1] != expected:
        raise RuntimeError(
            f"Feature matrix has {X.shape[1]} columns, expected {expected}."
        )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, X=X, y=y, groups=groups)
        if verbose:
            print(f"Cached features -> {cache_path}")

    keep = np.isin(y, observed)
    if verbose:
        print(f"Extracted {len(y)} clips; keeping {keep.sum()} for {observed}")
    return X[keep], y[keep], groups[keep]
