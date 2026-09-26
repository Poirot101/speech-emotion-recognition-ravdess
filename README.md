# Speech Emotion Recognition

[![tests](https://github.com/Poirot101/speech-emotion-recognition-ravdess/actions/workflows/tests.yml/badge.svg)](https://github.com/Poirot101/speech-emotion-recognition-ravdess/actions/workflows/tests.yml)

Predicting the emotion in a speech clip from MFCC / chroma / mel features with a
scikit-learn `MLPClassifier`, trained on the RAVDESS dataset.

<p align="center">
  <img src="reports/figures/confusion_observed_speaker.png" width="560"
       alt="Confusion matrix on held-out speakers">
</p>

| | |
|---|---|
| **Task** | Classify a `.wav` speech clip into an emotion |
| **Data** | [RAVDESS](https://zenodo.org/record/1188976): 1,440 clips, 24 actors, 8 emotions |
| **Features** | 180 per clip (40 MFCC + 12 chroma + 128 mel), averaged over time |
| **Model** | `MLPClassifier` with one hidden layer of 300 units, after a `StandardScaler` |
| **Result** | **77.6%** on a random split, **59.4%** on unseen speakers (4 classes, chance is 25%) |

---

## Why two numbers?

I started from the usual tutorial for this task, which gets around 72-78% with a
random train/test split. The problem is that every actor has 60 clips, so a
random split puts the same voices in both train and test. The model can do well
partly by recognising the speaker instead of the emotion, and that doesn't help
at all on someone new.

So I evaluated both ways:

| Split | 4-class accuracy | 8-class accuracy |
|---|---:|---:|
| Random (same voices in train and test) | 77.60% | 66.94% |
| **Speaker-independent** (held-out actors) | **59.38%** | **40.28%** |

That's an 18 point drop on 4 classes and 27 points on 8, with the same data,
features and model. The only difference is whether the test speakers were seen
during training.

This applies to any dataset with groups in it (speakers, patients, users,
devices...). A random split leaks the group and the score looks better than it
really is, and nothing warns you about it.

I treat **59.38%** as the actual result. All the numbers, confusion matrices and
ablations are in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Quickstart

```bash
git clone https://github.com/Poirot101/speech-emotion-recognition-ravdess.git
cd speech-emotion-recognition-ravdess

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python scripts/download_dataset.py                  # ~198 MB from Zenodo
python -m speech_emotion.train --split speaker --cv 5
```

Needs Python 3.12+ (librosa 1.0 doesn't support older versions). CI runs on
3.12, 3.13 and 3.14. The results in `docs/` came from Python 3.14.2, librosa
1.0.0 and scikit-learn 1.9.1.

It's a `src/` layout, so install the package (`pip install -e .`) before running
the modules, otherwise `python -m speech_emotion.train` won't find it. Installing
also gives you `ser-train` and `ser-predict` commands. If you'd rather use
`pip install -r requirements.txt`, set `PYTHONPATH=src` yourself.

### Train

```bash
python -m speech_emotion.train --split random              # same setup as the tutorial
python -m speech_emotion.train --split speaker             # test actors not seen in training
python -m speech_emotion.train --emotions all --split speaker --cv 5
python -m speech_emotion.train --split random --no-scaler  # ablation
python -m speech_emotion.train --split speaker --pooling meanstd --classifier svm  # 360 features, SVM
```

Each run saves the model to `models/` and the metrics (per-class
precision/recall/F1 and the confusion matrix) to
`reports/metrics_<emotions>_<split>.json`.

The first run extracts features from all 1,440 clips, which takes a minute or
two, and caches them in `data/features_cache.npz`. After that it's a few seconds.

### Predict

```bash
python -m speech_emotion.predict clip.wav --model models/mlp_observed_speaker.joblib
```

```
03-01-03-01-01-01-02.wav
  predicted: happy
    happy       99.7% ##############################
    disgust      0.3%
    fearful      0.0%
    calm         0.0%
```

Two gotchas: the model can only ever answer one of its four labels, and audio at
a different sample rate drops accuracy to chance while the confidence goes up to
~100%. Both are measured in [`docs/USAGE.md`](docs/USAGE.md).

### App

```bash
pip install -e ".[app]"
streamlit run app/streamlit_app.py
```

You can record or upload a clip and see the prediction, the probabilities
compared to chance, the mel spectrogram and the 180 numbers the model actually
gets. It warns you about clips it's likely to get wrong (under 44.1 kHz, too
little speech, clipping). There's also a game where you guess against the model
on actors it never trained on, and a tab that plays one actor saying the same
sentence in every emotion. It uses whatever models you've trained in `models/`.
More in [`docs/USAGE.md`](docs/USAGE.md#the-app).

### Figures, ablations, tests

```bash
python -m speech_emotion.visualize      # confusion matrices, class balance, feature profiles
python scripts/run_ablations.py         # feature-block and scaler ablations
python scripts/verify_results.py        # re-check published numbers, seed spread, bootstrap, permutation test
python scripts/improve_accuracy.py      # model selection on training actors, then one held-out test
python scripts/nested_cv.py             # same search, nested over all 24 actors
pytest                                  # doesn't need the dataset; runs in CI
```

---

## How it works

Clips are different lengths but `MLPClassifier` needs a fixed number of
features. So each feature is computed per frame and then averaged over time:

```
clip.wav  ->  STFT / mel filterbank  ->  (coefficients x ~400 frames)
          ->  mean over frames        ->  (180,)  ->  StandardScaler  ->  MLP  ->  emotion
```

| Feature | Dims | Roughly captures |
|---|---:|---|
| **MFCC** | 40 | Timbre (vocal tract shape) |
| **Chroma** | 12 | Energy per pitch class |
| **Mel spectrogram** | 128 | Energy across frequency bands |

The downside of averaging is that the order of things is lost. A rising and a
falling pitch contour end up with the same mean, even though the contour is a
big part of how emotion sounds. I think this is the main limit of the approach
and why CNNs on the full spectrogram do better.

RAVDESS has no labels file. Everything is in the filename:
`03-01-06-01-02-01-12.wav` has the emotion in field 3 and the actor in field 7,
and the actor field is what makes the speaker split possible. See
[`docs/DATASET.md`](docs/DATASET.md).

---

## Findings

**MFCCs do most of the work.** The 40 MFCC features on their own get 76.04% /
59.38%, basically the same as all 180, and exactly the same on unseen speakers.
The other 140 columns add ~1.5 points on the random split and nothing on the
speaker split. This makes sense since MFCCs are computed from the mel
spectrogram anyway. If I were to deploy this I'd use the 40-feature model.

**Scaling matters more than which features you pick.** Removing
`StandardScaler` costs 6.25 points, which is more than dropping 140 of the 180
features.

**The features pick up arousal more than valence.** In the 8-class run the
mistakes follow a pattern: `neutral` recall is only 21% (mostly predicted as
`sad`), and angry / fearful / surprised get confused with each other. Those are
all high-energy emotions, and what tells them apart is contour and context,
which averaging throws away.

**Improving on the baseline needed a bigger test to confirm.** I searched 30
configurations, selecting only on the training actors, and the best usable one
was mean+std pooling plus normalising each speaker by a few of their neutral
clips. On the 6 held-out speakers I couldn't show it was better (61.7% ± 1.4% vs
58.3% ± 1.2% over seeds, McNemar p = 0.36). Nested cross-validation over all 24
speakers did show it: **69.27%** vs **60.55%** for the baseline on those 768
clips, with a speaker-level 95% interval of [+4.1, +13.7] points for the gain
(Wilcoxon p = 0.006). These two numbers should only be compared with each other,
not with the 59.38% above. The gain isn't uniform (8 of 24 speakers got worse)
and it needs neutral recordings from each new speaker, which `predict.py`
doesn't ask for yet, so I haven't changed the default model.

Details in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Repository layout

```
├── src/speech_emotion/
│   ├── features.py      # audio -> 180-dim vector (MFCC + chroma + mel)
│   ├── dataset.py       # RAVDESS filename parsing, loading, feature cache
│   ├── train.py         # both splits, CV, metrics, saving the model
│   ├── predict.py       # predict from the command line
│   ├── demo.py          # non-UI code for the app (audio checks, model list, game)
│   └── visualize.py     # confusion matrices and EDA plots
├── app/
│   └── streamlit_app.py # the app (theme in .streamlit/config.toml)
├── notebooks/
│   └── speech_emotion_recognition.ipynb   # step-by-step walkthrough
├── scripts/
│   ├── download_dataset.py
│   ├── run_ablations.py
│   ├── verify_results.py    # reproduction, leakage, seed spread, bootstrap, permutation
│   ├── improve_accuracy.py  # model selection + significance tests on held-out actors
│   └── nested_cv.py         # nested grouped CV over all 24 actors
├── tests/               # generated audio, no dataset needed (app tests skip without Streamlit)
├── docs/
│   ├── USAGE.md            # running the model and where it fails
│   ├── LEARNING_NOTES.md   # what I learned, including mistakes
│   ├── RESULTS.md          # all numbers, confusion matrices, ablations
│   ├── DATASET.md          # RAVDESS format and caveats
│   └── learning/           # notes and next steps I'm working through
└── reports/figures/     # generated PNGs
```

If you want to run it, start with [`docs/USAGE.md`](docs/USAGE.md). The
reasoning is in [`docs/LEARNING_NOTES.md`](docs/LEARNING_NOTES.md), and the
[notebook](notebooks/speech_emotion_recognition.ipynb) goes through everything
step by step.

---

## Limitations

- **RAVDESS is acted.** Actors performing "fearful" on cue sound more
  exaggerated than people who are actually scared, so don't expect these
  numbers on real conversations.
- **Only two sentences.** The words are held fixed on purpose, so the model
  can't use them, but it also never sees varied content.
- **24 speakers, all North American English.** Nothing here says anything
  about other languages or accents.
- **Averaging over time** limits accuracy (see above).
- **Sample rate matters.** Features are extracted at 48 kHz. Audio recorded at
  a lower rate is missing the upper frequencies and accuracy drops a lot: 41.7%
  at 22 kHz, 29.7% at 16 kHz. Better to retrain at the rate you'll actually use.

## TODO

- [ ] Delta / delta-delta MFCCs (`librosa.feature.delta`) to get some timing information back
- [ ] Augmentation: pitch shift, time stretch, noise
- [ ] CNN on the mel spectrogram instead of averaging
- [ ] Cross-corpus test (train on RAVDESS, test on TESS / CREMA-D)
- [ ] Neutral-clip speaker enrolment in `predict.py` and the app, since nested CV says it helps

## License

Code is [MIT](LICENSE).

The dataset isn't included. RAVDESS is © Livingstone & Russo under
**CC BY-NC-SA 4.0** (non-commercial). `scripts/download_dataset.py` downloads it
from Zenodo.

> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of
> Emotional Speech and Song (RAVDESS). *PLoS ONE, 13*(5), e0196391.
> https://doi.org/10.1371/journal.pone.0196391

The baseline model settings come from the
[DataFlair speech emotion recognition tutorial](https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/).
The speaker-independent evaluation, ablations, caching, tests, app and analysis
are my additions.
