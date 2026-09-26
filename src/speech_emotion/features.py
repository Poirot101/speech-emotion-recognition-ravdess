"""Audio feature extraction.

Clips have different lengths but the classifiers need fixed-length input, so
each feature is computed per frame and then averaged over time. That gives:

    MFCC   (40)   timbre / vocal tract shape
    Chroma (12)   energy per pitch class
    Mel    (128)  energy per mel band

180 numbers per clip. Averaging loses the ordering of events (a rising vs
falling pitch look the same), which is the main thing a CNN/LSTM on the full
spectrogram would get back. More in docs/LEARNING_NOTES.md.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import soundfile

# Everything gets resampled to this before extraction. The mel/chroma bins
# depend on the sample rate, so a 16 kHz clip fed to a 48 kHz model gives a
# shifted feature vector. I measured it: held-out accuracy dropped from 59.4%
# to 25% (chance) while confidence went *up* to ~100%. RAVDESS is already
# 48 kHz so this doesn't change any of the training numbers.
TARGET_SR = 48_000

# "mean" = average over time (180 features)
# "meanstd" = mean + std over time (360 features)
# The std part captures how much each coefficient moves around, which helps a
# bit for emotion (60.9% -> 66.0% grouped CV on the training actors, see
# docs/RESULTS.md). Still doesn't know the order of frames though.
POOLING_OPTIONS = ("mean", "meanstd")

N_MFCC = 40
N_MEL = 128
N_CHROMA = 12  # one per semitone


def _check_pooling(pooling: str) -> None:
    if pooling not in POOLING_OPTIONS:
        raise ValueError(f"pooling must be one of {POOLING_OPTIONS}, got {pooling!r}.")


def feature_dimension(
    mfcc: bool = True, chroma: bool = True, mel: bool = True, pooling: str = "mean"
) -> int:
    """How long extract_feature's output will be for these options."""
    _check_pooling(pooling)
    base = (N_MFCC if mfcc else 0) + (N_CHROMA if chroma else 0) + (N_MEL if mel else 0)
    return base * (2 if pooling == "meanstd" else 1)


def extract_feature(
    file_name: str | Path,
    mfcc: bool = True,
    chroma: bool = True,
    mel: bool = True,
    target_sr: int | None = TARGET_SR,
    pooling: str = "mean",
) -> np.ndarray:
    """Turn one audio file into a 1-D feature vector.

    mfcc/chroma/mel switch each block on or off (at least one has to be on).
    target_sr=None skips resampling and uses the file's own rate, which is only
    there to reproduce the sample-rate experiment in docs/USAGE.md.
    With pooling="meanstd" the output is all the means followed by all the stds.

    Length is feature_dimension(...): 180 by default, 360 for meanstd.
    Raises ValueError if every block is off or the file is empty.
    """
    if not (mfcc or chroma or mel):
        raise ValueError("At least one of mfcc/chroma/mel must be True.")
    _check_pooling(pooling)

    with soundfile.SoundFile(str(file_name)) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate

    # stereo -> mono (RAVDESS is mono anyway, this is for user uploads)
    if X.ndim > 1:
        X = np.mean(X, axis=1)
    X = np.ascontiguousarray(X, dtype=np.float32)

    if X.size == 0:
        raise ValueError(f"{file_name} contains no audio samples.")

    if target_sr is not None and sample_rate != target_sr:
        X = librosa.resample(y=X, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    blocks: list[np.ndarray] = []

    if mfcc:
        blocks.append(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=N_MFCC))

    if chroma:
        stft = np.abs(librosa.stft(y=X))
        blocks.append(librosa.feature.chroma_stft(S=stft, sr=sample_rate))

    if mel:
        blocks.append(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=N_MEL))

    # average over frames. Written as mean(M.T, axis=0) on purpose so the
    # numbers match the old cached features exactly
    result = np.hstack([np.mean(M.T, axis=0) for M in blocks])
    if pooling == "meanstd":
        result = np.hstack([result] + [np.std(M.T, axis=0) for M in blocks])

    return result


def feature_names(
    mfcc: bool = True, chroma: bool = True, mel: bool = True, pooling: str = "mean"
) -> list[str]:
    """Column names in the same order as extract_feature's output."""
    _check_pooling(pooling)
    names: list[str] = []
    if mfcc:
        names += [f"mfcc_{i:02d}" for i in range(N_MFCC)]
    if chroma:
        names += [f"chroma_{p}" for p in
                  ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]]
    if mel:
        names += [f"mel_{i:03d}" for i in range(N_MEL)]
    if pooling == "meanstd":
        names += [f"{n}_std" for n in names]
    return names
