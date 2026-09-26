"""Tests for speech_emotion.demo. Uses generated audio, no dataset or Streamlit."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
import soundfile

from speech_emotion import demo
from speech_emotion.dataset import ALL_EMOTIONS, OBSERVED_EMOTIONS, parse_ravdess_filename
from speech_emotion.train import build_model


def _voice_like(sr: int, seconds: float, amplitude: float = 0.3) -> np.ndarray:
    """A pitch-gliding harmonic tone with a syllable-rate envelope: loud and varied."""
    t = np.arange(int(sr * seconds)) / sr
    f0 = 180 + 40 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    wave = sum(np.sin(k * phase) / k for k in range(1, 6))
    envelope = 0.55 + 0.45 * np.sin(2 * np.pi * 4 * t)
    return (amplitude * wave * envelope / np.max(np.abs(wave))).astype(np.float32)


def _write(tmp_path, name, audio, sr):
    path = tmp_path / name
    soundfile.write(path, audio, sr)
    return path


# inspect_audio

def test_clean_48k_clip_has_no_warnings(tmp_path):
    path = _write(tmp_path, "ok.wav", _voice_like(48_000, 3.0), 48_000)
    check = demo.inspect_audio(path)
    assert check.sample_rate == 48_000
    assert check.duration == pytest.approx(3.0, abs=0.01)
    assert check.ok, check.notices


def test_16k_clip_warns_with_the_measured_accuracy(tmp_path):
    path = _write(tmp_path, "phone.wav", _voice_like(16_000, 3.0), 16_000)
    check = demo.inspect_audio(path)
    assert not check.ok
    warning = check.notices[0]
    assert "16 kHz" in warning.title
    assert "29.7%" in warning.detail          # docs/USAGE.md, section 5


def test_rate_between_measurements_uses_the_lower_one(tmp_path):
    path = _write(tmp_path, "24k.wav", _voice_like(24_000, 3.0), 24_000)
    assert "41.7%" in demo.inspect_audio(path).notices[0].detail   # measured at 22.05 kHz


def test_silence_is_flagged(tmp_path):
    path = _write(tmp_path, "quiet.wav", np.zeros(48_000 * 2, dtype=np.float32), 48_000)
    titles = [n.title for n in demo.inspect_audio(path).notices]
    assert any("no voice" in t for t in titles)


def test_short_clip_is_flagged(tmp_path):
    audio = np.concatenate([_voice_like(48_000, 0.5), np.zeros(48_000, dtype=np.float32)])
    path = _write(tmp_path, "short.wav", audio, 48_000)
    assert any("Only" in n.title for n in demo.inspect_audio(path).notices)


def test_clipping_is_noted(tmp_path):
    audio = np.clip(_voice_like(48_000, 3.0, amplitude=3.0), -1, 1)
    path = _write(tmp_path, "hot.wav", audio, 48_000)
    assert any("clipping" in n.title for n in demo.inspect_audio(path).notices)


# models and prediction

@pytest.fixture
def tiny_bundle(tmp_path):
    """A real pipeline fitted on random 180-dim vectors, saved like train.py saves one."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 180))
    y = np.array(OBSERVED_EMOTIONS * 10)
    model = build_model().set_params(mlp__max_iter=300, mlp__batch_size=16).fit(X, y)
    bundle = {"model": model, "emotions": OBSERVED_EMOTIONS, "split": "speaker",
              "accuracy": 0.5938, "pooling": "mean", "classifier": "mlp"}
    path = tmp_path / "mlp_observed_speaker.joblib"
    joblib.dump(bundle, path)
    return path, bundle


def test_predict_clip_returns_all_classes_in_bundle_order(tmp_path, tiny_bundle):
    _, bundle = tiny_bundle
    clip = _write(tmp_path, "c.wav", _voice_like(48_000, 2.0), 48_000)
    pred = demo.predict_clip(clip, bundle)
    assert list(pred.probabilities) == OBSERVED_EMOTIONS
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)
    assert pred.label in OBSERVED_EMOTIONS
    assert pred.confidence == max(pred.probabilities.values())
    assert pred.ranked()[0][0] == pred.label


def test_uncertain_threshold():
    probs = {"calm": 0.55, "happy": 0.45}
    assert demo.Prediction("calm", probs).uncertain
    assert not demo.Prediction("calm", {"calm": 0.8, "happy": 0.2}).uncertain


def test_list_models_puts_speaker_4_class_first_and_skips_foreign_files(tmp_path, tiny_bundle):
    path, bundle = tiny_bundle
    joblib.dump({**bundle, "split": "random", "accuracy": 0.776},
                tmp_path / "mlp_observed_random.joblib")
    joblib.dump({**bundle, "emotions": ALL_EMOTIONS}, tmp_path / "mlp_all_speaker.joblib")
    joblib.dump(object(), tmp_path / "notebook_model.joblib")
    infos = demo.list_models(tmp_path)
    assert [i.path.name for i in infos] == [
        "mlp_observed_speaker.joblib", "mlp_all_speaker.joblib", "mlp_observed_random.joblib"]
    assert infos[0].label == "4 emotions, tested on new voices"
    assert infos[0].chance == 0.25 and infos[0].unseen_voices
    assert not infos[2].unseen_voices


def test_model_labels_disambiguate_duplicates(tmp_path, tiny_bundle):
    path, bundle = tiny_bundle
    joblib.dump(bundle, tmp_path / "notebook_model.joblib")   # same settings, other file
    labels = demo.model_labels(demo.list_models(tmp_path))
    assert labels[path] == "4 emotions, tested on new voices (mlp_observed_speaker)"
    assert len(set(labels.values())) == 2


def test_list_models_on_missing_dir_is_empty(tmp_path):
    assert demo.list_models(tmp_path / "nope") == []


def test_train_command():
    assert demo.train_command() == "python -m speech_emotion.train --split speaker"
    assert demo.train_command("all").endswith("--emotions all")


# RAVDESS helpers

def test_held_out_actors_match_the_published_speaker_split():
    # Published in docs/RESULTS.md and asserted by scripts/improve_accuracy.py.
    assert demo.held_out_actors() == [4, 5, 6, 8, 10, 19]


def test_ravdess_path_round_trips_through_the_parser(tmp_path):
    path = demo.ravdess_path(7, "fearful", statement=2, intensity=2, repetition=1,
                             data_dir=tmp_path)
    assert path.parent.name == "Actor_07"
    parsed = parse_ravdess_filename(path)
    assert (parsed.emotion, parsed.actor, parsed.statement, parsed.intensity) == \
        ("fearful", 7, "dogs_sitting", "strong")


def test_every_emotion_has_a_colour():
    assert set(demo.EMOTION_COLORS) == set(ALL_EMOTIONS)


# the game

@pytest.fixture
def fake_dataset(tmp_path):
    for actor in (4, 5):
        for emotion in OBSERVED_EMOTIONS:
            for statement in (1, 2):
                for intensity in (1, 2):
                    for repetition in (1, 2):
                        p = demo.ravdess_path(actor, emotion, statement, intensity,
                                              repetition, data_dir=tmp_path)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        p.touch()
    return tmp_path


def test_game_scores_both_players(fake_dataset):
    game = demo.Game(emotions=tuple(OBSERVED_EMOTIONS), actors=(4, 5),
                     data_dir=fake_dataset, seed=1)
    rnd = game.new_round()
    assert rnd.path.exists() and rnd.actor in (4, 5) and not game.revealed

    wrong = next(e for e in OBSERVED_EMOTIONS if e != rnd.emotion)
    result = game.submit(rnd.emotion, model_label=wrong)
    assert result["you_right"] and not result["model_right"]
    assert (game.played, game.human_correct, game.model_correct) == (1, 1, 0)
    assert game.revealed

    with pytest.raises(RuntimeError):
        game.submit(rnd.emotion, model_label=wrong)   # no double scoring

    game.new_round()
    assert not game.revealed
    game.submit(wrong if wrong != game.current.emotion else game.current.emotion,
                model_label=game.current.emotion)
    assert game.played == 2 and game.model_correct == 1


def test_game_rejects_unknown_guess(fake_dataset):
    game = demo.Game(emotions=tuple(OBSERVED_EMOTIONS), actors=(4,), data_dir=fake_dataset)
    game.new_round()
    with pytest.raises(ValueError):
        game.submit("angry", model_label="calm")


def test_game_without_data_raises(tmp_path):
    game = demo.Game(emotions=("calm",), actors=(4,), data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        game.new_round()
