"""Smoke tests for the Streamlit app. Skipped when Streamlit is not installed.

Streamlit's AppTest runs the script headlessly, so these catch exceptions in
layout code without a browser, a dataset, or the real trained models.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from speech_emotion.dataset import OBSERVED_EMOTIONS  # noqa: E402
from speech_emotion.train import build_model  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


def _run(monkeypatch, model_dir, data_dir, report_dir):
    monkeypatch.setenv("SER_MODEL_DIR", str(model_dir))
    monkeypatch.setenv("SER_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SER_REPORT_DIR", str(report_dir))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_without_a_model_the_app_explains_how_to_train(tmp_path, monkeypatch):
    at = _run(monkeypatch, tmp_path / "models", tmp_path / "data", tmp_path / "reports")
    assert not at.exception
    assert any("Train a model first" in s.value for s in at.subheader)
    assert any("speech_emotion.train --split speaker" in c.value for c in at.code)


def test_with_a_model_but_no_dataset_every_tab_renders(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(40, 180)), np.array(OBSERVED_EMOTIONS * 10)
    model = build_model().set_params(mlp__max_iter=300, mlp__batch_size=16).fit(X, y)
    (tmp_path / "models").mkdir()
    joblib.dump({"model": model, "emotions": OBSERVED_EMOTIONS, "split": "speaker",
                 "accuracy": 0.5938, "pooling": "mean", "classifier": "mlp"},
                tmp_path / "models" / "mlp_observed_speaker.joblib")

    at = _run(monkeypatch, tmp_path / "models", tmp_path / "data", tmp_path / "reports")
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Analyse a clip", "Guess the emotion", "One sentence, many ways", "How good is it?"]
    # No dataset: the game and comparison views point at the download script.
    assert sum("download_dataset.py" in c.value for c in at.code) >= 2
