"""Streamlit app for the emotion model.

    pip install -e ".[app]"
    streamlit run app/streamlit_app.py

Tabs: analyse a clip, guessing game, compare one actor across emotions, results.
The non-UI logic is in speech_emotion/demo.py.
"""

from __future__ import annotations

import html
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

from speech_emotion import demo
from speech_emotion.features import extract_feature, feature_names

MODEL_DIR = Path(os.environ.get("SER_MODEL_DIR", demo.MODEL_DIR))
DATA_DIR = Path(os.environ.get("SER_DATA_DIR", demo.DATA_DIR))
REPORT_DIR = Path(os.environ.get("SER_REPORT_DIR", demo.REPORT_DIR))

PAPER, INK, MUTED, RULE = "#F2F4F7", "#1B2230", "#5A6376", "#D3D9E2"

st.set_page_config(page_title="Speech emotion recognition", page_icon="🎙️",
                   layout="wide", initial_sidebar_state="auto")

st.html(f"""
<style>
  .block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
  h1, h2, h3 {{ letter-spacing: -0.015em; }}
  .lede {{ color: {MUTED}; font-size: 1.05rem; max-width: 62ch; margin: -0.4rem 0 0.6rem; }}

  .verdict {{ border-left: 0.45rem solid var(--emo); padding: 0.2rem 0 0.4rem 1.1rem; }}
  .verdict .said {{ color: {MUTED}; font-size: 0.95rem; margin: 0; }}
  .verdict .word {{ font-family: "Bricolage Grotesque", sans-serif; font-weight: 800;
                   font-size: clamp(3rem, 7vw, 5.2rem); line-height: 0.95; color: var(--emo);
                   margin: 0.1rem 0 0.35rem; letter-spacing: -0.03em; }}
  .verdict .how {{ color: {INK}; margin: 0; max-width: 46ch; }}

  .bars {{ margin-top: 1.1rem; display: grid; gap: 0.45rem; }}
  .bar-row {{ display: grid; grid-template-columns: 6.2rem 1fr 3.4rem; gap: 0.7rem;
             align-items: center; font-size: 0.95rem; }}
  .bar-row .name {{ font-weight: 700; }}
  .track {{ position: relative; height: 0.85rem; background: #E1E6EC; border-radius: 2px; }}
  .fill {{ position: absolute; inset: 0 auto 0 0; background: var(--emo); border-radius: 2px; }}
  .chance {{ position: absolute; top: -0.25rem; bottom: -0.25rem; width: 0;
            border-left: 2px dashed {INK}; opacity: 0.45; }}
  .bar-row .pct {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .legend {{ color: {MUTED}; font-size: 0.85rem; margin-top: 0.5rem; }}

  .notice {{ border-left: 0.3rem solid #B7791F; background: #FBF4E6; padding: 0.6rem 0.9rem;
            margin: 0.5rem 0; border-radius: 0 0.3rem 0.3rem 0; }}
  .notice.info {{ border-color: #5A6376; background: #E9EDF2; }}
  .notice b {{ display: block; margin-bottom: 0.15rem; }}
  .fine {{ color: {MUTED}; font-size: 0.88rem; }}

  .score {{ display: flex; gap: 2.5rem; align-items: baseline; flex-wrap: wrap; }}
  .score .n {{ font-family: "Bricolage Grotesque", sans-serif; font-weight: 800;
              font-size: 2.6rem; line-height: 1; }}
  .score .who {{ color: {MUTED}; }}

  .chip {{ display: inline-block; padding: 0.05rem 0.55rem; border-radius: 1rem;
          color: white; background: var(--emo); font-weight: 700; font-size: 0.9rem; }}

  @media (prefers-reduced-motion: no-preference) {{
    .fill {{ transition: width 400ms ease-out; }}
  }}
</style>
""")

# colour the game buttons to match the charts
st.html("<style>" + "".join(
    f""".st-key-guess_{e} button {{ border-left: 0.4rem solid {c}; font-weight: 700; }}
        .st-key-guess_{e} button:hover {{ border-color: {c}; color: {c}; }}"""
    for e, c in demo.EMOTION_COLORS.items()) + "</style>")


# Cached work

@st.cache_resource(show_spinner=False)
def load_bundle(path: str, mtime: float):
    return joblib.load(path)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_predict(audio_path: str, mtime: float, model_path: str, model_mtime: float):
    bundle = load_bundle(model_path, model_mtime)
    return demo.predict_clip(audio_path, bundle)


@st.cache_data(show_spinner=False, max_entries=256)
def cached_log_mel(audio_path: str, mtime: float):
    return demo.log_mel(audio_path)


@st.cache_data(show_spinner=False, max_entries=64)
def cached_check(audio_path: str, mtime: float):
    return demo.inspect_audio(audio_path)


def mtime(path: Path | str) -> float:
    return Path(path).stat().st_mtime


def predict(audio_path: Path, info: demo.ModelInfo) -> demo.Prediction:
    return cached_predict(str(audio_path), mtime(audio_path), str(info.path), mtime(info.path))


# Small renderers

def color(emotion: str) -> str:
    return demo.EMOTION_COLORS.get(emotion, INK)


def chip(emotion: str) -> str:
    return f'<span class="chip" style="--emo:{color(emotion)}">{html.escape(emotion)}</span>'


def verdict_html(pred: demo.Prediction, info: demo.ModelInfo) -> str:
    conf = pred.confidence
    if pred.uncertain:
        how = (f"Only {conf:.0%} sure. Below 60% this model is close to guessing, "
               "so treat this as a hint.")
    else:
        how = (f"{conf:.0%} sure. Confidence is not accuracy: this model is right about "
               f"{info.accuracy:.0%} of the time on {'new' if info.unseen_voices else 'familiar'} voices.")
    return f"""
    <div class="verdict" style="--emo:{color(pred.label)}">
      <p class="said">The model hears</p>
      <p class="word">{html.escape(pred.label)}</p>
      <p class="how">{html.escape(how)}</p>
    </div>"""


def bars_html(pred: demo.Prediction, chance: float) -> str:
    rows = []
    for emotion, p in pred.ranked():
        rows.append(f"""
        <div class="bar-row" style="--emo:{color(emotion)}">
          <span class="name">{html.escape(emotion)}</span>
          <span class="track" role="img" aria-label="{emotion} {p:.0%}">
            <span class="fill" style="width:{p * 100:.1f}%"></span>
            <span class="chance" style="left:{chance * 100:.1f}%"></span>
          </span>
          <span class="pct">{p:.0%}</span>
        </div>""")
    legend = (f'<p class="legend">Dashed line: {chance:.0%}, what guessing at random scores. '
              "The model must pick one of these emotions, even for audio that fits none.</p>")
    return f'<div class="bars">{"".join(rows)}</div>{legend}'


def notices_html(check: demo.AudioCheck) -> str:
    return "".join(
        f'<div class="notice {n.level}"><b>{html.escape(n.title)}</b>{html.escape(n.detail)}</div>'
        for n in check.notices)


def spectrogram_figure(S_db: np.ndarray, sr: int, emotion: str | None, height: float = 2.6):
    tint = color(emotion) if emotion else "#2B3A67"
    cmap = LinearSegmentedColormap.from_list("emo", [PAPER, tint, INK])
    fig, ax = plt.subplots(figsize=(10, height), facecolor=PAPER)
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel", cmap=cmap, ax=ax)
    ax.set_facecolor(PAPER)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("seconds", color=MUTED, fontsize=9)
    ax.set_ylabel("Hz (mel scale)", color=MUTED, fontsize=9)
    fig.tight_layout(pad=0.4)
    return fig


def thumbnail_figure(S_db: np.ndarray, sr: int, emotion: str):
    """Small spectrogram with no axes, for the compare tab."""
    cmap = LinearSegmentedColormap.from_list("emo", [PAPER, color(emotion), INK])
    fig, ax = plt.subplots(figsize=(3.2, 1.9), facecolor=PAPER)
    librosa.display.specshow(S_db, sr=sr, y_axis="mel", cmap=cmap, ax=ax)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    return fig


def feature_strip_figure(values: np.ndarray, names: list[str]):
    """Scaled feature vector drawn as a single coloured strip."""
    fig, ax = plt.subplots(figsize=(10, 1.35), facecolor=PAPER)
    clipped = np.clip(values, -3, 3)
    ax.imshow(clipped[np.newaxis, :], aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
              interpolation="nearest")
    ax.set_yticks([])
    starts = {}
    for i, n in enumerate(names):
        starts.setdefault(n.split("_")[0] + ("_std" if n.endswith("_std") else ""), i)
    for label, i in starts.items():
        ax.axvline(i - 0.5, color=INK, lw=1)
        ax.text(i + 1, -0.62, label.replace("_std", " spread"), color=INK, fontsize=8,
                va="bottom")
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout(pad=0.3)
    return fig


def save_upload(uploaded) -> Path:
    suffix = Path(uploaded.name or "clip.wav").suffix or ".wav"
    tmp = Path(tempfile.gettempdir()) / "ser_app"
    tmp.mkdir(exist_ok=True)
    data = uploaded.getvalue()
    path = tmp / f"{abs(hash(data))}{suffix}"
    if not path.exists():
        path.write_bytes(data)
    return path


# Sidebar: which model

models = demo.list_models(MODEL_DIR)
labels = demo.model_labels(models)

with st.sidebar:
    st.subheader("Model")
    if models:
        info: demo.ModelInfo = st.radio(
            "Which trained model to use", models, format_func=lambda m: labels[m.path],
            label_visibility="collapsed", key="model_choice")
        chance = info.chance
        st.markdown(
            f"**{info.accuracy:.1%}** accurate on its test set, against **{chance:.1%}** "
            "for random guessing.")
        if info.unseen_voices:
            st.caption("Tested on actors it never trained on: the realistic number.")
        else:
            st.caption("Tested on clips from the same actors it trained on, which flatters "
                       "it. On new voices this setup scores far lower.")
        st.markdown("It can only answer: " + " ".join(chip(e) for e in info.emotions),
                    unsafe_allow_html=True)
    else:
        info = None
    st.divider()
    st.caption("Scores how closely a voice matches how 24 actors performed each emotion. "
               "It does not measure how someone feels; don't use it to judge real people.")


st.title("How does this voice sound?")
st.html('<p class="lede">A small neural network, trained on 24 actors reading two '
        "sentences with different feelings, guesses the emotion in speech. Record yourself, "
        "play the guessing game, or look at what it listens for.</p>")

if info is None:
    st.subheader("Train a model first")
    st.markdown(
        f"No trained model found in `{MODEL_DIR}`. Download the dataset and train the "
        "realistic 4-emotion model (about two minutes the first time):")
    st.code("python scripts/download_dataset.py\n" + demo.train_command(), language="bash")
    st.markdown("Then reload this page.")
    st.stop()

bundle = load_bundle(str(info.path), mtime(info.path))
have_data = demo.dataset_available(DATA_DIR)
held_out = demo.held_out_actors()

tab_analyse, tab_game, tab_compare, tab_results = st.tabs(
    ["Analyse a clip", "Guess the emotion", "One sentence, many ways", "How good is it?"])


# Analyse a clip

with tab_analyse:
    sources = ["Record", "Upload a file"] + (["Pick a dataset clip"] if have_data else [])
    source = st.segmented_control("Audio source", sources, default="Record",
                                  required=True, key="source")
    audio_path: Path | None = None
    truth: str | None = None

    if source == "Record":
        # streamlit defaults to 16 kHz, where accuracy drops to 29.7% (docs/USAGE.md)
        recording = st.audio_input("Say one sentence with feeling, about 3 seconds",
                                   sample_rate=48_000, key="recording")
        st.caption("Try the RAVDESS sentence: “Kids are talking by the door.” "
                   "Same words, different feeling each time.")
        if recording is not None:
            audio_path = save_upload(recording)
    elif source == "Upload a file":
        uploaded = st.file_uploader("WAV, FLAC, OGG or AIFF. For MP3, convert first "
                                    "with ffmpeg.", type=["wav", "flac", "ogg", "aiff", "aif"],
                                    key="upload")
        if uploaded is not None:
            audio_path = save_upload(uploaded)
    else:
        c1, c2, c3 = st.columns(3)
        actor = c1.selectbox("Actor", list(range(1, 25)),
                             index=held_out[0] - 1, key="pick_actor",
                             format_func=lambda a: f"Actor {a}" + (
                                 " (never heard)" if info.unseen_voices and a in held_out else ""))
        truth = c2.selectbox("Emotion", list(demo.EMOTION_COLORS), key="pick_emotion",
                             index=list(demo.EMOTION_COLORS).index(info.emotions[0]))
        statement = c3.selectbox("Sentence", [1, 2], format_func=demo.STATEMENTS.get,
                                 key="pick_statement")
        audio_path = demo.ravdess_path(actor, truth, statement, data_dir=DATA_DIR)
        if truth not in info.emotions:
            st.html(f'<div class="notice"><b>This model has no word for {html.escape(truth)}</b>'
                    "Whatever it says will be wrong. Try it anyway: its confidence often "
                    "stays high, which is the point.</div>")

    if audio_path is None:
        st.html('<p class="fine">Nothing to analyse yet. Record a sentence or choose a file above.</p>')
    elif not audio_path.exists():
        st.error(f"Can't find {audio_path}. Run `python scripts/download_dataset.py`.")
    else:
        try:
            check = cached_check(str(audio_path), mtime(audio_path))
            pred = predict(audio_path, info)
            S_db, sr = cached_log_mel(str(audio_path), mtime(audio_path))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Couldn't read that audio: {exc}. Save it as a WAV file and try again.")
            st.stop()

        if source != "Record":
            st.audio(str(audio_path))
        left, right = st.columns([1.05, 1], gap="large")
        with left:
            st.html(verdict_html(pred, info))
            if truth:
                right_or_wrong = "right" if pred.label == truth else "wrong"
                st.html(f'<p class="fine">The actor was performing {chip(truth)}. '
                        f"The model is {right_or_wrong}.</p>")
            if check.notices:
                st.html(notices_html(check))
        with right:
            st.html(bars_html(pred, info.chance))

        st.subheader("What it heard")
        st.pyplot(spectrogram_figure(S_db, sr, pred.label), clear_figure=True)
        st.caption(
            f"Mel spectrogram at {sr // 1000} kHz: time left to right, pitch low to high, "
            "darker means louder. Recorded at "
            f"{check.sample_rate / 1000:g} kHz, {check.duration:.1f} s long with "
            f"{check.speech_seconds:.1f} s of voice.")

        with st.expander("What the model actually receives"):
            features = extract_feature(audio_path, pooling=info.pooling)
            model = bundle["model"]
            scaled = model[0].transform(features[np.newaxis, :])[0] \
                if "scaler" in model.named_steps else features
            names = feature_names(pooling=info.pooling)
            st.pyplot(feature_strip_figure(scaled, names), clear_figure=True)
            st.markdown(
                f"The whole clip becomes these **{len(features)} numbers**, each one a "
                "feature averaged over time" +
                (" plus how much it varied" if info.pooling == "meanstd" else "") +
                ". Red is higher than the typical training clip, blue lower. Averaging is "
                "why the model can't tell a rising voice from a falling one: both give the "
                "same average.")


# Guess the emotion

with tab_game:
    st.subheader("Can you beat the model?")
    if not have_data:
        st.markdown("The game plays clips from the RAVDESS dataset. Download it first:")
        st.code("python scripts/download_dataset.py", language="bash")
    elif not info.unseen_voices:
        st.markdown(
            "This model trained on clips from every actor, so the game would be rigged in "
            "its favour. Pick a model marked **tested on new voices** in the sidebar.")
    else:
        key = f"game_{info.path.name}"
        if key not in st.session_state:
            st.session_state[key] = demo.Game(emotions=info.emotions, actors=tuple(held_out),
                                              data_dir=DATA_DIR)
        game: demo.Game = st.session_state[key]
        if game.current is None:
            game.new_round()

        st.markdown(
            f"Every clip comes from actors {', '.join(map(str, held_out))}, whom the model "
            "never heard during training. Listen, pick an emotion, then see what the model said.")

        rnd = game.current
        st.audio(str(rnd.path))
        model_pred = predict(rnd.path, info)

        cols = st.columns(len(info.emotions))
        for col, emotion in zip(cols, info.emotions):
            if col.button(emotion, key=f"guess_{emotion}", width="stretch",
                          disabled=game.revealed):
                game.submit(emotion, model_pred.label)
                st.rerun()

        if game.revealed:
            last = game.history[-1]
            you = "You got it." if last["you_right"] else f"You said {chip(last['you'])}."
            them = ("The model got it too." if last["model_right"] and last["you_right"] else
                    "The model got it." if last["model_right"] else
                    f"The model said {chip(last['model'])}.")
            st.html(f'<p style="font-size:1.1rem">It was {chip(last["truth"])}. {you} {them}</p>')
            st.html(bars_html(model_pred, info.chance))
            intensity = "strong" if rnd.intensity == 2 else "normal"
            st.caption(f"Actor {rnd.actor}, {intensity} intensity: "
                       f"“{demo.STATEMENTS[rnd.statement]}.”")
            if st.button("Next clip", type="primary", key="next_round"):
                game.new_round()
                st.rerun()

        if game.played:
            st.divider()
            st.html(f"""
            <div class="score">
              <div><div class="n">{game.human_correct}/{game.played}</div><div class="who">you</div></div>
              <div><div class="n">{game.model_correct}/{game.played}</div><div class="who">the model</div></div>
            </div>""")
            st.caption(f"Over the full test set the model scores {info.accuracy:.0%}. "
                       "Five rounds is too few to say who is better; twenty starts to mean something.")
            if st.button("Start over", key="reset_game"):
                del st.session_state[key]
                st.rerun()


# One sentence, many ways

with tab_compare:
    st.subheader("Same actor, same words, different emotion")
    if not have_data:
        st.markdown("This view plays RAVDESS clips. Download the dataset first:")
        st.code("python scripts/download_dataset.py", language="bash")
    else:
        c1, c2, c3 = st.columns(3)
        actor = c1.selectbox("Actor", list(range(1, 25)), index=held_out[0] - 1,
                             key="cmp_actor",
                             format_func=lambda a: f"Actor {a}" + (
                                 " (never heard)" if info.unseen_voices and a in held_out else ""))
        statement = c2.selectbox("Sentence", [1, 2], format_func=demo.STATEMENTS.get,
                                 key="cmp_statement")
        strong = c3.toggle("Strong intensity", key="cmp_strong",
                           help="RAVDESS recorded each emotion at normal and strong "
                                "intensity. Neutral only has normal.")
        seen = (not info.unseen_voices) or actor not in held_out
        st.caption(("The model trained on this actor's voice, so expect it to do well. "
                    if seen else "The model never heard this actor. ") +
                   "Listen for what changes when only the feeling does: loudness, pitch, "
                   "speed, breathiness.")

        per_row = 4
        emotions = list(info.emotions)
        for start in range(0, len(emotions), per_row):
            cols = st.columns(per_row, gap="medium")
            for col, emotion in zip(cols, emotions[start:start + per_row]):
                intensity = 2 if strong and emotion != "neutral" else 1
                path = demo.ravdess_path(actor, emotion, statement, intensity, data_dir=DATA_DIR)
                with col:
                    st.html(f"<h4 style='color:{color(emotion)};margin:0.4rem 0 0.2rem'>"
                            f"{html.escape(emotion)}</h4>")
                    if not path.exists():
                        st.caption("Clip not found.")
                        continue
                    st.audio(str(path))
                    p = predict(path, info)
                    S_db, sr = cached_log_mel(str(path), mtime(path))
                    st.pyplot(thumbnail_figure(S_db, sr, emotion), clear_figure=True)
                    mark = "right" if p.label == emotion else "wrong"
                    st.html(f'<p class="fine">Model says {chip(p.label)} '
                            f"({p.confidence:.0%}), {mark}.</p>")


# How good is it?

with tab_results:
    st.subheader("How good is it?")
    st.markdown(
        "Tested on clips from actors it trained on, the 4-emotion model scores **77.6%**. "
        "Tested on actors it never heard, it scores **59.4%**. The second number is the "
        "one that matters: nobody using this app was in the training data.")

    figure = REPORT_DIR / "figures" / f"confusion_{'all' if len(info.emotions) == 8 else 'observed'}_{info.split}.png"
    left, right = st.columns([1, 1], gap="large")
    with left:
        if figure.exists():
            st.image(str(figure), caption="Rows are the true emotion, columns what the "
                     "model said. The diagonal is where it is right.")
    with right:
        verification = demo.load_report("verification.json", REPORT_DIR)
        if verification and len(info.emotions) == 4:
            per_speaker = verification.get("bootstrap", {}).get("per_speaker_accuracy", {})
            if per_speaker:
                st.markdown("**Accuracy for each new voice**")
                actors_ = sorted(per_speaker, key=lambda a: int(a))
                vals = [per_speaker[a] for a in actors_]
                fig, ax = plt.subplots(figsize=(5, 2.6), facecolor=PAPER)
                ax.bar([f"Actor {a}" for a in actors_], vals, color="#2B3A67")
                ax.axhline(0.25, color=INK, ls="--", lw=1, alpha=0.5)
                ax.set_ylim(0, 1)
                ax.set_facecolor(PAPER)
                ax.tick_params(colors=MUTED, labelsize=8)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                st.caption("The same model ranges from 41% to 75% depending on whose voice "
                           "it hears. An average hides that.")
        nested = demo.load_report("nested_cv.json", REPORT_DIR)
        if nested:
            o = nested["overall"]
            lo, hi = nested["speaker_bootstrap_gain_95"]
            st.markdown("**Can it do better?**")
            st.markdown(
                "Yes, if it can hear a few calm recordings of each new speaker first, which "
                "this app doesn't ask for yet. Testing every one of the 24 actors in turn, "
                f"that approach scored **{o['nested_mean']:.1%}** where this kind of model "
                f"scored **{o['baseline_mean']:.1%}** on the same clips, a gain likely "
                f"between {lo * 100:+.0f} and {hi * 100:+.0f} points. Those two numbers come "
                "from a different test than the accuracy above, so compare them only with "
                "each other.")
    st.caption("Full analysis: docs/RESULTS.md in the repository.")
