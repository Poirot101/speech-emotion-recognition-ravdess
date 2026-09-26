# 07: The app: how it is built, and how to extend it

**Goal:** understand `app/streamlit_app.py` well enough to change it without
breaking it, then add the feature the nested cross-validation result asks for:
speaker enrolment.

**Effort:** 1–2 days to read and run; 3–5 days for the enrolment extension.
**Prerequisites:** comfortable Python. Item 1 of `06_accuracy_playbook.md` for
the extension.

---

## Run it first

```bash
pip install -e ".[app]"
python -m speech_emotion.train --split speaker          # the app needs at least one model
streamlit run app/streamlit_app.py
```

Then, before reading any code, **try to break it**:

1. Upload a clip you recorded on your phone. Which warning appears, and why?
2. Pick a dataset clip whose emotion the model doesn't know (angry, for the
   4-class model). Look at the confidence.
3. Play ten rounds of the game. Write down your score and the model's.
4. In "One sentence, many ways", switch between a held-out actor and one the
   model trained on. What changes?

---

## The architecture in one picture

```
app/streamlit_app.py          layout only: widgets, columns, HTML, figures
        │ calls
        ▼
src/speech_emotion/demo.py    logic: list_models, inspect_audio, predict_clip, Game
        │ calls
        ▼
features.py / predict.py      the same code the CLI uses
```

**Why the split matters.** Logic in `demo.py` is tested by plain `pytest`
(`tests/test_demo.py`) with synthesised audio, and runs in CI without Streamlit.
The app file is tested by headless smoke tests (`tests/test_app.py`) that skip
when Streamlit is not installed. If logic crept into the app file, it would only
be tested by clicking.

**Exercise.** Find one piece of logic still in the app file that belongs in
`demo.py`. (Hint: look at how the "What the model actually receives" block
decides whether to scale features.) Move it and write the test.

---

## Concept 1: Streamlit reruns the whole script

Every interaction (a click, a new upload, a changed selectbox) **re-executes
`streamlit_app.py` from top to bottom.** There are no callbacks wired to
buttons in the usual sense: `st.button(...)` returns `True` on the one rerun
that follows its click.

Consequences you can see in the code:

- `if col.button(emotion, ...): game.submit(...); st.rerun()`, the click is
  handled, then the script reruns immediately so the page shows the new state.
- Expensive work would repeat on every click, hence concept 2.
- Anything that must survive reruns lives in `st.session_state`, hence concept 3.

**Exercise.** Add `print("rerun")` at the top of the file, run the app, and
count how many reruns one round of the game triggers.

---

## Concept 2: caching: `cache_data` vs `cache_resource`

| Decorator | Returns | Used for |
|---|---|---|
| `@st.cache_resource` | the *same object* every time (not copied) | the loaded model bundle |
| `@st.cache_data` | a *copy* of a serialisable value | predictions, spectrograms, audio checks |

**The cache key is the function's arguments.** That is why every cached function
takes a file's `mtime` as well as its path: retrain a model and its modification
time changes, so the stale prediction is not reused.

**Exercise.** Remove the `model_mtime` argument from `cached_predict`, retrain
a model while the app is running, and observe the bug. Put it back.

---

## Concept 3: session state and the game

`demo.Game` is a plain dataclass holding the score, the current round, and
whether the round has been revealed. The app keeps one per model:

```python
key = f"game_{info.path.name}"
if key not in st.session_state:
    st.session_state[key] = demo.Game(emotions=info.emotions, actors=tuple(held_out), ...)
```

Keying by model means switching models in the sidebar does not mix their scores.

**Honesty rules built into the game** (each is a line of code, find them):

- Clips come only from the six held-out actors.
- A model trained on the random split refuses to play: it has heard those voices.
- `Game.submit` raises if a round is scored twice.

**Exercise.** Add a "streak" counter to `Game` (longest run of rounds where you
beat the model). Test it in `tests/test_demo.py` first.

---

## Concept 4: widget keys and styling

Every widget has a `key=`. Keys keep a widget's identity across reruns, and
Streamlit adds a CSS class `st-key-<key>` to the widget's container. The app
uses that to colour the guess buttons:

```css
.st-key-guess_happy button { border-left: 0.4rem solid #D9A21B; }
```

The theme (colours, fonts) is in `.streamlit/config.toml`. Every emotion has one
colour in `demo.EMOTION_COLORS`, reused by bars, chips, buttons and spectrogram
tints, so an emotion looks the same everywhere.

---

## Concept 5: the audio details that matter

- **`st.audio_input(..., sample_rate=48_000)`.** The default is 16 kHz. Recording
  at 16 kHz would put every recording in the failure mode measured in
  `docs/USAGE.md` §5 (29.7% accuracy). One keyword argument is the difference
  between a demo that works and one that quietly doesn't.
- **Uploads are buffers, the feature code wants paths.** `save_upload` writes the
  bytes to a temp file named by a hash of their contents, so the same upload maps
  to the same path and hits the cache.
- **`inspect_audio` runs before trusting a prediction.** Its thresholds are
  documented in its docstring and pinned by tests.

**Exercise.** Record yourself in the app, download the recording from the audio
widget's menu, and check its rate with `soundfile.info`. Then change the
`sample_rate` to 16 000 and record the same sentence. Compare the predictions
and the warning.

---

## Concept 6: testing an app without a browser

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("app/streamlit_app.py", default_timeout=60)
at.run()
assert not at.exception
at.segmented_control[0].set_value("Pick a dataset clip").run()
at.button(key="guess_calm").click().run()
```

`tests/test_app.py` points the app at empty temporary folders through
environment variables (`SER_MODEL_DIR`, `SER_DATA_DIR`, `SER_REPORT_DIR`) to
test the "no model yet" and "no dataset" paths in CI.

**Exercise.** Add a test that, with a tiny fake dataset folder (see the
`fake_dataset` fixture in `tests/test_demo.py`) and a tiny model, clicking a
guess button shows the "Next clip" button.

---

## The extension: "Calibrate to your voice"

Nested cross-validation showed normalising by a speaker's neutral clips is the
one change worth shipping. Build it end to end.

### Milestones

**M1, Library.** `enrol` / `normalise` / `SpeakerProfile` as sketched in
`06_accuracy_playbook.md` §1, with tests.

**M2, Training.** `train.py --normalize neutral --pooling meanstd --classifier …`
writes a bundle with `"normalization": "neutral"`. Checkpoint: the default
command still reproduces 59.38%.

**M3, Prediction.** `predict_file(..., profile=None)` raises for a `neutral`
bundle without a profile. Checkpoint: a test for the error message.

**M4, The tab.** A fifth tab:

```
Calibrate to your voice
  Record 4 calm sentences   [mic] [mic] [mic] [mic]       2 of 4 recorded
  [Use this calibration]   [Clear]
  ─────────────────────────────────────────────────────────
  Now analysed with your calibration: on / off  (toggle)
```

Store the profile in `st.session_state["profile"]`. In "Analyse a clip", if a
`neutral` model is selected and no profile exists, show a notice pointing to the
tab rather than predicting.

**M5, Show the difference.** For the same recording, show predictions with and
without calibration side by side. That comparison is the most educational
screen the app could have.

**M6, Measure the realistic version.** Enrolment in the app will be recorded by
a person in a room, not an actor in a studio. Record yourself (4 neutral + a few
of each emotion), label them honestly, and report how the calibrated and plain
models do on *you*. One speaker proves nothing statistically (say so) but it is
the first test on a voice outside RAVDESS.

### Pitfalls

- **The profile must use the bundle's pooling.** A mean-pooled profile subtracted
  from mean+std features is a shape error at best.
- **Users will not read calmly.** Warn when the enrolment clips' own predicted
  arousal is high, or when fewer than two were recorded.
- **Privacy.** Enrolment recordings are voice data. Keep them in session state
  only; do not write them anywhere persistent.

---

## Deploying (optional)

Streamlit Community Cloud can host the app from a GitHub repository. Before you
do, think about three constraints:

1. **Models and data are gitignored.** A hosted app would need a trained model;
   decide whether committing a small `.joblib` is acceptable.
2. **RAVDESS is CC BY-NC-SA 4.0.** Do not upload the dataset. The game and the
   comparison view would need to be disabled or use your own recordings.
3. **Anyone could upload voices.** The app says it does not measure how people
   feel; a public deployment makes that disclaimer carry more weight.

---

## Resources

- How Streamlit runs your app, https://docs.streamlit.io/develop/concepts/architecture/run-your-app
- Caching, https://docs.streamlit.io/develop/concepts/architecture/caching
- Session state, https://docs.streamlit.io/develop/concepts/architecture/session-state
- App testing (`AppTest`), https://docs.streamlit.io/develop/api-reference/app-testing
- `st.audio_input`, https://docs.streamlit.io/develop/api-reference/widgets/st.audio_input
- Theming, https://docs.streamlit.io/develop/concepts/configuration/theming
