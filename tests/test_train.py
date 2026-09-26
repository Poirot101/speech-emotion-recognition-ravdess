"""Tests for build_model and split_data (synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from speech_emotion.train import build_model, split_data

LABELS = ["calm", "disgust", "fearful", "happy"]


def _toy(n_speakers: int = 8, per_speaker: int = 12, n_features: int = 20, seed: int = 0):
    """Four roughly separable classes, with a speaker ID for every row."""
    rng = np.random.default_rng(seed)
    y = np.array(LABELS * (n_speakers * per_speaker // len(LABELS)))
    class_index = np.searchsorted(LABELS, y)
    X = rng.normal(size=(len(y), n_features)) + class_index[:, None] * 1.5
    groups = np.repeat(np.arange(n_speakers), per_speaker)
    return X, y, groups


@pytest.mark.parametrize("classifier", ["mlp", "svm"])
def test_classifiers_fit_predict_and_give_probabilities(classifier):
    X, y, _ = _toy()
    model = build_model(classifier=classifier).fit(X, y)
    assert set(model.predict(X)) <= set(LABELS)
    assert (model.predict(X) == y).mean() > 0.8
    probs = model.predict_proba(X)
    assert probs.shape == (len(y), len(LABELS))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_scaler_is_first_and_optional():
    assert [name for name, _ in build_model().steps] == ["scaler", "mlp"]
    assert [name for name, _ in build_model(scale=False).steps] == ["mlp"]
    assert [name for name, _ in build_model(classifier="svm").steps] == ["scaler", "svm"]


def test_unknown_classifier_is_rejected():
    with pytest.raises(ValueError, match="classifier"):
        build_model(classifier="xgboost")


def test_speaker_split_never_shares_a_speaker():
    X, y, groups = _toy()
    X = np.column_stack([groups, X])  # carry the speaker ID through the split
    X_train, X_test, _, _ = split_data(X, y, groups, strategy="speaker")
    assert not set(X_train[:, 0]) & set(X_test[:, 0])


def test_unknown_split_is_rejected():
    X, y, groups = _toy()
    with pytest.raises(ValueError, match="split"):
        split_data(X, y, groups, strategy="stratified")
