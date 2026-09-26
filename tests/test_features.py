"""Feature/dataset tests. Uses generated tones, so the dataset isn't needed."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile

from speech_emotion.dataset import (
    EMOTIONS,
    OBSERVED_EMOTIONS,
    find_audio_files,
    parse_ravdess_filename,
)
from speech_emotion.features import extract_feature, feature_dimension, feature_names


@pytest.fixture
def tone(tmp_path):
    """A 2-second 440 Hz mono tone at 48 kHz, matching RAVDESS's sample rate."""
    sr = 48_000
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    wav = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    path = tmp_path / "03-01-03-01-01-01-05.wav"
    soundfile.write(path, wav, sr)
    return path


def test_feature_vector_has_expected_length(tone):
    assert extract_feature(tone).shape == (180,)


def test_feature_dimension_matches_names():
    assert feature_dimension() == len(feature_names()) == 180


@pytest.mark.parametrize(
    "flags, expected",
    [
        ({"mfcc": True, "chroma": False, "mel": False}, 40),
        ({"mfcc": False, "chroma": True, "mel": False}, 12),
        ({"mfcc": False, "chroma": False, "mel": True}, 128),
        ({"mfcc": True, "chroma": True, "mel": False}, 52),
    ],
)
def test_feature_flags_control_length(tone, flags, expected):
    assert extract_feature(tone, **flags).shape == (expected,)


def test_all_flags_off_is_an_error(tone):
    with pytest.raises(ValueError):
        extract_feature(tone, mfcc=False, chroma=False, mel=False)


def test_features_are_finite(tone):
    assert np.all(np.isfinite(extract_feature(tone)))


def test_length_invariance(tmp_path):
    """Clips of different durations must still yield 180 features."""
    sr = 48_000
    for seconds in (0.5, 1.0, 3.7):
        t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        path = tmp_path / f"tone_{seconds}.wav"
        soundfile.write(path, (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr)
        assert extract_feature(path).shape == (180,)


def test_stereo_is_downmixed(tmp_path):
    sr = 22_050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    stereo = np.stack([np.sin(2 * np.pi * 300 * t), np.sin(2 * np.pi * 500 * t)], axis=1)
    path = tmp_path / "stereo.wav"
    soundfile.write(path, stereo.astype(np.float32), sr)
    assert extract_feature(path).shape == (180,)


def test_parse_filename_decodes_every_field():
    meta = parse_ravdess_filename("Actor_12/03-01-06-02-02-01-12.wav")
    assert meta.emotion == "fearful"
    assert meta.intensity == "strong"
    assert meta.statement == "dogs_sitting"
    assert meta.repetition == 1
    assert meta.actor == 12
    assert meta.gender == "female"


def test_gender_follows_actor_parity():
    assert parse_ravdess_filename("03-01-01-01-01-01-01.wav").gender == "male"
    assert parse_ravdess_filename("03-01-01-01-01-01-02.wav").gender == "female"


def test_bad_filenames_raise_clear_errors():
    with pytest.raises(ValueError, match="7 hyphen-separated"):
        parse_ravdess_filename("my_recording.wav")
    with pytest.raises(ValueError, match="unknown emotion code"):
        parse_ravdess_filename("03-01-99-01-01-01-01.wav")


def test_observed_emotions_are_real_emotions():
    assert set(OBSERVED_EMOTIONS) <= set(EMOTIONS.values())
    assert len(OBSERVED_EMOTIONS) == 4


def test_find_audio_files_is_sorted_and_empty_when_missing(tmp_path):
    assert find_audio_files(tmp_path) == []
    for name in ["03-01-03-01-01-01-02.wav", "03-01-01-01-01-01-01.wav"]:
        actor = tmp_path / f"Actor_{name.split('-')[-1][:2]}"
        actor.mkdir(exist_ok=True)
        (actor / name).touch()
    found = find_audio_files(tmp_path)
    assert len(found) == 2
    assert found == sorted(found)


# sample rate handling

def _tone(path, seconds=1.0, sr=48_000, freq=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    soundfile.write(path, (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr)
    return path


def test_resampling_makes_features_rate_invariant(tmp_path):
    """The same tone at 48 kHz and 22.05 kHz must land close in feature space."""
    a = extract_feature(_tone(tmp_path / "a.wav", sr=48_000))
    b = extract_feature(_tone(tmp_path / "b.wav", sr=22_050))
    # Compare on the MFCC block, which is what the model mostly relies on.
    cosine = np.dot(a[:40], b[:40]) / (np.linalg.norm(a[:40]) * np.linalg.norm(b[:40]))
    assert cosine > 0.99, f"MFCCs drifted across sample rates (cosine={cosine:.4f})"


def test_native_rate_opt_out_still_works(tmp_path):
    """target_sr=None restores the original native-rate behaviour."""
    path = _tone(tmp_path / "c.wav", sr=22_050)
    assert extract_feature(path, target_sr=None).shape == (180,)


def test_resampling_is_a_noop_at_target_rate(tmp_path):
    path = _tone(tmp_path / "d.wav", sr=48_000)
    assert np.array_equal(extract_feature(path, target_sr=48_000),
                          extract_feature(path, target_sr=None))


# pooling

def test_meanstd_pooling_doubles_length_and_keeps_means(tone):
    mean_vec = extract_feature(tone)
    both = extract_feature(tone, pooling="meanstd")
    assert both.shape == (360,)
    assert np.array_equal(both[:180], mean_vec)   # means first, unchanged
    assert np.all(both[180:] >= 0)                # then standard deviations


def test_feature_names_follow_pooling():
    names = feature_names(pooling="meanstd")
    assert len(names) == feature_dimension(pooling="meanstd") == 360
    assert names[180] == "mfcc_00_std"


def test_unknown_pooling_is_rejected(tone):
    with pytest.raises(ValueError, match="pooling"):
        extract_feature(tone, pooling="median")
