"""Non-UI code for the Streamlit app (app/streamlit_app.py).

Kept separate from the app so it can be tested without Streamlit.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import librosa
import numpy as np
import soundfile
from sklearn.model_selection import GroupShuffleSplit

from .dataset import ALL_EMOTIONS, EMOTIONS, OBSERVED_EMOTIONS
from .features import N_MEL, TARGET_SR
from .predict import predict_file

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data" / "ravdess"
REPORT_DIR = REPO_ROOT / "reports"

# same colour for an emotion everywhere in the app
EMOTION_COLORS: dict[str, str] = {
    "neutral": "#7F8794",
    "calm": "#2E86AB",
    "happy": "#D9A21B",
    "sad": "#34568B",
    "angry": "#C8402F",
    "fearful": "#7D5BA6",
    "disgust": "#5F8A2E",
    "surprised": "#DD7A2C",
}

STATEMENTS: dict[int, str] = {
    1: "Kids are talking by the door",
    2: "Dogs are sitting by the door",
}

# below this the app says the prediction is uncertain (same cutoff as docs/USAGE.md)
UNCERTAIN_BELOW = 0.6

# held-out accuracy of the 4-class speaker model when the test audio was first
# downsampled to each rate (docs/USAGE.md, section 5)
MEASURED_ACCURACY_BY_RATE = {48_000: 0.594, 22_050: 0.417, 16_000: 0.297, 8_000: 0.266}


# models

@dataclass(frozen=True)
class ModelInfo:
    path: Path
    emotions: tuple[str, ...]
    split: str
    accuracy: float
    pooling: str
    classifier: str

    @property
    def chance(self) -> float:
        return 1 / len(self.emotions)

    @property
    def label(self) -> str:
        n = len(self.emotions)
        voices = "tested on new voices" if self.split == "speaker" else "random split"
        extra = "" if (self.classifier, self.pooling) == ("mlp", "mean") else \
            f", {self.classifier.upper()} + {self.pooling}"
        return f"{n} emotions, {voices}{extra}"

    @property
    def unseen_voices(self) -> bool:
        # accuracy was measured on actors it didn't train on
        return self.split == "speaker"


def describe_bundle(path: Path, bundle: dict) -> ModelInfo:
    return ModelInfo(
        path=Path(path),
        emotions=tuple(bundle["emotions"]),
        split=bundle.get("split", "unknown"),
        accuracy=float(bundle.get("accuracy", float("nan"))),
        pooling=bundle.get("pooling", "mean"),
        classifier=bundle.get("classifier", "mlp"),
    )


def list_models(model_dir: str | Path = MODEL_DIR) -> list[ModelInfo]:
    """Find the trained models in model_dir. Speaker-split 4-class model first.

    Anything that isn't a train.py bundle (e.g. notebook_model.joblib) is skipped.
    """
    infos = []
    for path in sorted(Path(model_dir).glob("*.joblib")):
        try:
            bundle = joblib.load(path)
            infos.append(describe_bundle(path, bundle))
        except Exception:  # noqa: BLE001
            continue

    def rank(info: ModelInfo):
        return (not info.unseen_voices, len(info.emotions),
                (info.classifier, info.pooling) != ("mlp", "mean"), info.path.name)

    return sorted(infos, key=rank)


def model_labels(infos: list[ModelInfo]) -> dict[Path, str]:
    """Display name for each model. Adds the filename if two names would clash."""
    counts: dict[str, int] = {}
    for info in infos:
        counts[info.label] = counts.get(info.label, 0) + 1
    return {info.path: info.label + (f" ({info.path.stem})" if counts[info.label] > 1 else "")
            for info in infos}


def train_command(emotions: str = "observed", split: str = "speaker") -> str:
    """Command to train the default model for these settings."""
    extra = " --emotions all" if emotions == "all" else ""
    return f"python -m speech_emotion.train --split {split}{extra}"


# audio checks

@dataclass(frozen=True)
class Notice:
    level: str   # "warning" or "info"
    title: str
    detail: str


@dataclass
class AudioCheck:
    sample_rate: int
    channels: int
    duration: float
    speech_seconds: float
    peak: float
    rms_dbfs: float
    notices: list[Notice] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(n.level == "warning" for n in self.notices)


def _closest_measured_rate(rate: int) -> int:
    """Closest measured rate that's <= rate (so we don't overstate accuracy)."""
    below = [r for r in MEASURED_ACCURACY_BY_RATE if r <= rate]
    return max(below) if below else min(MEASURED_ACCURACY_BY_RATE)


def inspect_audio(path: str | Path) -> AudioCheck:
    """Check a clip for things the model is known to handle badly.

    - sample rate under 44.1 kHz (see docs/USAGE.md section 5)
    - less than 1 s of speech
    - longer than 8 s (different emotions get averaged together)
    - near silence or clipping
    """
    info = soundfile.info(str(path))
    audio, sr = soundfile.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    duration = len(mono) / sr if sr else 0.0

    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    rms_dbfs = 20 * np.log10(rms) if rms > 0 else -np.inf

    # count anything within 30 dB of the peak as speech
    if mono.size and peak > 0:
        intervals = librosa.effects.split(mono, top_db=30)
        speech_seconds = float(sum(e - s for s, e in intervals) / sr)
    else:
        speech_seconds = 0.0

    notices: list[Notice] = []
    if sr < 44_100:
        measured = _closest_measured_rate(sr)
        notices.append(Notice(
            "warning",
            f"Recorded at {sr / 1000:g} kHz, but the model was trained on 48 kHz audio",
            f"Audio at this rate has no sound above {sr / 2000:g} kHz, so the upper "
            f"frequency bands the model relies on are empty. On voices it had never "
            f"heard, accuracy fell from 59.4% at 48 kHz to "
            f"{MEASURED_ACCURACY_BY_RATE[measured]:.1%} at {measured / 1000:g} kHz. "
            "Record at 44.1 or 48 kHz if you can.",
        ))
    if rms_dbfs < -50 or speech_seconds < 0.3:
        notices.append(Notice(
            "warning", "Almost no voice in this clip",
            "The prediction will describe background noise. Record closer to the "
            "microphone, or check the right input is selected.",
        ))
    elif speech_seconds < 1.0:
        notices.append(Notice(
            "warning", f"Only {speech_seconds:.1f} s of voice",
            "The model averages features over the whole clip; with this little "
            "speech that average is noisy. RAVDESS sentences last 3-5 seconds.",
        ))
    if duration > 8:
        notices.append(Notice(
            "info", f"Long clip ({duration:.0f} s)",
            "Everything is averaged into one prediction, so a clip that changes "
            "emotion partway will blur. One sentence at a time works best.",
        ))
    if mono.size and np.mean(np.abs(mono) >= 0.999) > 0.001:
        notices.append(Notice(
            "info", "The recording is clipping",
            "Parts of the waveform hit full scale. Distortion adds energy the "
            "model may read as intensity. Lower the input gain.",
        ))

    return AudioCheck(sample_rate=int(info.samplerate), channels=int(info.channels),
                      duration=float(duration), speech_seconds=speech_seconds,
                      peak=peak, rms_dbfs=float(rms_dbfs), notices=notices)


# prediction

@dataclass(frozen=True)
class Prediction:
    label: str
    probabilities: dict[str, float]   # in the bundle's emotion order

    @property
    def confidence(self) -> float:
        return self.probabilities.get(self.label, 0.0)

    @property
    def uncertain(self) -> bool:
        return self.confidence < UNCERTAIN_BELOW

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])


def predict_clip(path: str | Path, bundle: dict) -> Prediction:
    label, probs = predict_file(path, bundle)
    ordered = {e: probs.get(e, 0.0) for e in bundle["emotions"]}
    return Prediction(label=label, probabilities=ordered)


def log_mel(path: str | Path, n_mels: int = N_MEL) -> tuple[np.ndarray, int]:
    """Log-mel spectrogram (dB) at 48 kHz, for plotting."""
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max), sr


# RAVDESS helpers

EMOTION_CODES = {label: code for code, label in EMOTIONS.items()}


def ravdess_path(actor: int, emotion: str, statement: int = 1, intensity: int = 1,
                 repetition: int = 1, data_dir: str | Path = DATA_DIR) -> Path:
    """Build a RAVDESS path, e.g. Actor_04/03-01-03-01-01-01-04.wav."""
    name = (f"03-01-{EMOTION_CODES[emotion]}-{intensity:02d}-{statement:02d}-"
            f"{repetition:02d}-{actor:02d}.wav")
    return Path(data_dir) / f"Actor_{actor:02d}" / name


def held_out_actors(test_size: float = 0.25, n_actors: int = 24,
                    random_state: int = 9) -> list[int]:
    """The test actors of the speaker split.

    GroupShuffleSplit only shuffles the unique group ids, so this gives the same
    actors as train.split_data without needing the dataset loaded.
    """
    actors = np.arange(1, n_actors + 1)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    _, test_idx = next(splitter.split(actors, actors, actors))
    return sorted(int(a) for a in actors[test_idx])


def dataset_available(data_dir: str | Path = DATA_DIR) -> bool:
    return any(Path(data_dir).glob("Actor_*/*.wav"))


# game

@dataclass(frozen=True)
class Round:
    path: Path
    actor: int
    emotion: str
    statement: int
    intensity: int


@dataclass
class Game:
    """Guessing game: player vs model on held-out actors."""

    emotions: tuple[str, ...]
    actors: tuple[int, ...]
    data_dir: Path = DATA_DIR
    seed: int | None = None
    played: int = 0
    human_correct: int = 0
    model_correct: int = 0
    history: list[dict] = field(default_factory=list)
    current: Round | None = None
    guess: str | None = None

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def new_round(self) -> Round:
        for _ in range(50):
            actor = self._rng.choice(self.actors)
            emotion = self._rng.choice(self.emotions)
            statement = self._rng.choice((1, 2))
            intensity = 1 if emotion == "neutral" else self._rng.choice((1, 2))
            repetition = self._rng.choice((1, 2))
            path = ravdess_path(actor, emotion, statement, intensity, repetition,
                                self.data_dir)
            if path.exists():
                self.current = Round(path, actor, emotion, statement, intensity)
                self.guess = None
                return self.current
        raise FileNotFoundError(f"No RAVDESS clips found under {self.data_dir}")

    @property
    def revealed(self) -> bool:
        return self.guess is not None

    def submit(self, guess: str, model_label: str) -> dict:
        if self.current is None:
            raise RuntimeError("Start a round before guessing.")
        if self.revealed:
            raise RuntimeError("This round is already scored; start a new one.")
        if guess not in self.emotions:
            raise ValueError(f"{guess!r} is not one of {self.emotions}")
        self.guess = guess
        truth = self.current.emotion
        result = {"actor": self.current.actor, "truth": truth, "you": guess,
                  "model": model_label, "you_right": guess == truth,
                  "model_right": model_label == truth}
        self.played += 1
        self.human_correct += result["you_right"]
        self.model_correct += result["model_right"]
        self.history.append(result)
        return result


# saved reports

def load_report(name: str, report_dir: str | Path = REPORT_DIR) -> dict | None:
    path = Path(report_dir) / name
    return json.loads(path.read_text()) if path.exists() else None


__all__ = [
    "ALL_EMOTIONS", "OBSERVED_EMOTIONS", "EMOTION_COLORS", "STATEMENTS",
    "ModelInfo", "list_models", "model_labels", "describe_bundle", "train_command",
    "Notice", "AudioCheck", "inspect_audio",
    "Prediction", "predict_clip", "log_mel",
    "ravdess_path", "held_out_actors", "dataset_available",
    "Round", "Game", "load_report",
]
