# 04: Cross-corpus evaluation

**Goal:** train on RAVDESS, test on a dataset recorded by different people under
different conditions, and measure how much of the model's skill survives. Then
try to shrink the gap, one change at a time.

**Effort:** 1–2 weeks. **Prerequisites:** `00_foundations.md`,
`01_delta_features.md`. Project 03 is optional here, but comparing CNN and
classical models across corpora is a strong extension.

---

## Why this matters here

- The speaker split tests new **voices**, but the same studio, microphones,
  sentences, acting direction and sample rate. Real audio differs in all of those.
- This repo has already measured two symptoms of that kind of shift:
  - **Sample rate:** 59.4% → 25.0% at 22.05 kHz without resampling, and only
    29.7% at 16 kHz even with it (`docs/USAGE.md`).
  - **Speaker normalisation:** it looks strong on RAVDESS only because every
    RAVDESS speaker has a perfectly emotion-balanced set of clips
    (`reports/improvements.md`).
- Cross-corpus testing is the most direct way to find out whether the model
  learned *emotion* or learned *RAVDESS*.

---

## The datasets

These facts were checked against each dataset's official page. Still verify
anything your code depends on (see "Check it yourself" below).

### TESS: Toronto emotional speech set
- 200 target words spoken in the carrier phrase *"Say the word ___"*
- 2 actresses, aged 26 and 64
- 7 emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, neutral
 , so 200 × 7 × 2 = 2,800 recordings
- Licence: **CC BY-NC 4.0**. Hosted on the University of Toronto's Borealis
  dataverse, mirrored on Kaggle.
- **Consequences:** only two speakers, so any TESS result describes two voices.
  And the clips are single words in a fixed phrase, much shorter than RAVDESS sentences.

### CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
- 7,442 clips from 91 actors: 48 male, 43 female, aged 20–74, of varied ethnicities
- 12 sentences, 6 emotions (anger, disgust, fear, happy, neutral, sad) at
  several intensity levels
- Licence: **Open Database License**
- **The audio is stored with Git LFS.** A plain zip download gives you stub
  files, not audio. A full `git lfs clone` needs about 7.55 GB (it includes video).
- **Consequence:** 91 speakers makes this a far stronger test of speaker
  independence than TESS.

### Check it yourself: do this before writing any model code
- **Sample rate and channel count** of every corpus: `soundfile.info(path)`.
  You already know what a mismatch does.
- **Duration distribution** per corpus.
- **Filename → label parsing.** Write a parser per corpus, with tests, the same way
  `parse_ravdess_filename` has them.

---

## Concepts to understand first

1. **Domain shift, three kinds.**
   - *Covariate shift*, the inputs change: microphone, room, sample rate,
     speakers, spoken material.
   - *Label shift*, class frequencies change between corpora.
   - *Concept shift*, what "happy" sounds like differs, because each corpus
     directed its actors differently.
2. **A shared label set.** RAVDESS has 8 emotions, TESS 7, CREMA-D 6. The
   existing 4-class model includes `calm`, which neither of the others has. All
   three share **angry, disgust, fearful, happy, neutral, sad**. Write down every
   mapping decision: `fear` → `fearful` is trivial; TESS's *pleasant surprise*
   versus RAVDESS's *surprised* is a judgement call, and CREMA-D has no surprise at all.
3. **Imbalance.** Class balance differs per corpus (RAVDESS `neutral` has half the
   clips of the other classes). Report macro-F1 and per-class recall, not only accuracy.
4. **Normalisation as adaptation.** Standardising each corpus with its *own*
   statistics is speaker normalisation one level up, and it inherits the same
   caveat about what data you would actually have at deployment time.
5. **Evaluation designs.**
   - Train on A, test on B (pure cross-corpus)
   - An in-corpus reference: a speaker-independent split *within* B, as the ceiling
   - Leave-one-corpus-out: train on two corpora, test on the third
   - Pooled training with `GroupKFold`, where the group is `(corpus, speaker)`

---

## Build milestones

### M1: Acquire and inspect
Start with TESS; it is small. Then CREMA-D.
**Checkpoint:** a table in your lab notebook for each corpus, clips per emotion,
sample rate, channels, mean duration.

### M2: A multi-corpus loader
Generalise loading to return `(X, y, speaker, corpus)`. One filename parser per
corpus, with tests. Map labels to the shared 6-class set.
**Checkpoint:** tests pass, and per-label counts match your M1 table.

### M3: Common preprocessing
Resample every corpus to a single rate. Choose the **lowest** rate among the
corpora you use and retrain RAVDESS at that rate, `docs/USAGE.md` §5 explains
why. Keep separate caches per corpus and rate.
**Checkpoint:** RAVDESS's in-corpus speaker-split accuracy at the new rate is
recorded. It will differ from 59.4%; that is your new in-corpus reference.

### M4: The core measurement

| Train | Test | Accuracy | Macro-F1 |
|---|---|---:|---:|
| RAVDESS (speaker split) | RAVDESS held-out actors | | |
| RAVDESS (all) | TESS | | |
| RAVDESS (all) | CREMA-D | | |
| CREMA-D (speaker split) | CREMA-D held-out actors | | |
| CREMA-D (all) | RAVDESS | | |

Choose hyperparameters with inner `GroupKFold` on the **training** corpus only.
Expect a large drop from in-corpus to cross-corpus; the size of that drop is the finding.

### M5: Mitigations, one at a time
- Per-corpus feature standardisation
- Pooled training, evaluated leave-one-corpus-out
- The features from project 01, do they help across corpora, or only within one?

### M6: Write it up
Include a confusion matrix for each train → test direction. Which emotions
transfer, and which collapse? Relate that to arousal versus valence.

---

## Implementation blueprint

### One record type for every corpus

```python
# src/speech_emotion/corpora.py
SHARED_LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad")

@dataclass(frozen=True)
class Clip:
    path: Path
    corpus: str          # "ravdess" | "tess" | "cremad"
    speaker: str         # corpus-local id, e.g. "12", "OAF", "1001"
    label: str           # already mapped into SHARED_LABELS
    raw_label: str       # what the corpus called it, for the audit trail

    @property
    def group(self) -> str:
        return f"{self.corpus}:{self.speaker}"      # unique across corpora

def parse_ravdess(path: Path) -> Clip | None: ...   # None for calm / surprised
def parse_tess(path: Path) -> Clip | None: ...
def parse_cremad(path: Path) -> Clip | None: ...
PARSERS = {"ravdess": parse_ravdess, "tess": parse_tess, "cremad": parse_cremad}
```

Returning `None` for clips outside the shared label set keeps the dropping
explicit and countable.

### Filenames: confirm on your own download

- **CREMA-D** names look like `1001_DFA_ANG_XX.wav`: actor, sentence code,
  emotion code (`ANG DIS FEA HAP NEU SAD`), intensity level.
- **TESS** is organised by speaker and emotion (the two speakers are usually
  labelled `OAF` and `YAF`), with pleasant surprise abbreviated in some names.

Treat both descriptions as hypotheses: list 20 real filenames from each download
and write the parser tests from those, not from this page.

### Label mapping as data, with a test per entry

```python
CREMAD_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fearful",
              "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
TESS_MAP = {...}            # decide and document what happens to pleasant surprise
RAVDESS_MAP = {"angry": "angry", ..., "calm": None, "surprised": None}
```

### Loading and caching

```python
def load_corpus(name: str, root: Path, sr: int, pooling: str,
                cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Clip]]:
    """X, y, groups ("corpus:speaker"), clips. Cache at cache_dir/f"{name}_{sr}_{pooling}.npz"."""
```

Every corpus uses the **same `sr`**. `extract_feature` already accepts
`target_sr`, but `load_dataset` does not pass one through and the cache filenames
don't record it. Thread it through first, put the rate in the cache name, and add
a test that the default behaviour is unchanged.

### Evaluation shapes

```python
# In-corpus reference for corpus B
cross_val_score(pipe, X_b, y_b, groups=g_b, cv=GroupKFold(5))

# Train A, test B (no B data touches training or selection)
pipe.fit(X_a, y_a); accuracy_score(y_b, pipe.predict(X_b))

# Leave-one-corpus-out on pooled data
corpus = np.array([c.corpus for c in clips])
for tr, te in LeaveOneGroupOut().split(X, y, groups=corpus): ...
```

### A results table generator, not a hand-typed table

Write `scripts/cross_corpus.py` that emits `reports/cross_corpus.md` with the M4
table, per-class recall per direction, and confusion matrices, the same way
`improve_accuracy.py` generates its report. Hand-typed numbers drift.

---

## Pitfalls

- **Sample-rate mismatch**, the silent failure measured in `docs/USAGE.md`.
- **Tuning on the target corpus**, then it is no longer a cross-corpus result.
- **Labels that sound alike but differ**, *pleasant surprise* is not *surprised*.
- **Duration and silence differences.** TESS clips are short single words, so
  leading and trailing silence make up a larger share of each clip. If you trim
  silence (`librosa.effects.trim`), do it identically for every corpus.
- **Accuracy alone**, when class balance differs between corpora.
- **Generalising from TESS.** Two speakers is not a population. CREMA-D's 91
  actors carry much more weight.
- **CREMA-D zip downloads contain no audio** (Git LFS stubs).
- **Licences.** RAVDESS is CC BY-NC-SA 4.0, TESS is CC BY-NC 4.0, CREMA-D is ODbL.
  Never commit the audio. Commit download scripts, following
  `scripts/download_dataset.py`.

---

## How you will know you succeeded

- A table of in-corpus and cross-corpus results with the gap stated for each direction.
- At least one mitigation tested under a clean protocol, with a clear conclusion.
- A per-emotion analysis of what transfers, tied back to arousal and valence.

---

## Stretch goals

- **IEMOCAP**, larger, conversational, partly improvised speech. Check the site
  for access terms before planning around it.
- **Pretrained speech embeddings** (wav2vec 2.0) as a cross-corpus baseline, they
  were trained on far more varied audio than any single emotion corpus.

---

## Resources

- Schuller et al. (2010), *Cross-Corpus Acoustic Emotion Recognition: Variances and Strategies*, IEEE Transactions on Affective Computing, https://doi.org/10.1109/T-AFFC.2010.8
- Cao et al. (2014), *CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset*, IEEE Transactions on Affective Computing, https://doi.org/10.1109/TAFFC.2014.2336244
- CREMA-D repository (read the README before downloading), https://github.com/CheyneyComputerScience/CREMA-D
- TESS on Borealis (official), https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF
- TESS on Kaggle (mirror), https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
- Livingstone & Russo (2018), the RAVDESS paper, https://doi.org/10.1371/journal.pone.0196391
- IEMOCAP, https://sail.usc.edu/iemocap/
- Baevski et al. (2020), *wav2vec 2.0*, https://arxiv.org/abs/2006.11477
- scikit-learn `LeaveOneGroupOut`, https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html
- scikit-learn common pitfalls, https://scikit-learn.org/stable/common_pitfalls.html
