# The RAVDESS dataset

**R**yerson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong.

| | |
|---|---|
| Source | Zenodo record [1188976](https://zenodo.org/record/1188976) (DOI `10.5281/zenodo.1188976`) |
| Subset used here | `Audio_Speech_Actors_01-24.zip`, audio-only **speech** (~198 MB) |
| Clips | 1,440 = 24 actors × 60 clips |
| Actors | 24 professional actors, 12 male / 12 female, North-American accent |
| Format | 16-bit WAV, 48 kHz, mono |
| Clip length | ~3–5 seconds |
| Licence | CC BY-NC-SA 4.0, free for **non-commercial** use with attribution |

The full RAVDESS release (7,356 files, 24.8 GB) also contains song, video, and
audio-visual modalities. This project uses only the audio-speech subset, the
`Audio_Speech_Actors_01-24.zip` archive, which is what `scripts/download_dataset.py`
fetches.

## Labels live in the filename

There is no CSV of labels to join against. Every file is named with seven
two-digit fields, for example `03-01-06-01-02-01-12.wav`:

| Position | Field | Values |
|---:|---|---|
| 1 | Modality | `01` full-AV · `02` video-only · **`03` audio-only** |
| 2 | Vocal channel | **`01` speech** · `02` song |
| 3 | **Emotion** | `01` neutral · `02` calm · `03` happy · `04` sad · `05` angry · `06` fearful · `07` disgust · `08` surprised |
| 4 | Intensity | `01` normal · `02` strong (never present for `neutral`) |
| 5 | Statement | `01` "Kids are talking by the door" · `02` "Dogs are sitting by the door" |
| 6 | Repetition | `01` · `02` |
| 7 | **Actor** | `01`–`24`; **odd = male, even = female** |

So `03-01-06-01-02-01-12.wav` is audio-only speech, *fearful*, normal intensity,
the "dogs" statement, first repetition, actor 12 (female).

Field **3** is the training target. Field **7** is the speaker identity, and this
project treats it as a first-class output of the loader (`groups`) because it is
needed for a proper train/test split, see [LEARNING_NOTES.md](LEARNING_NOTES.md#3-the-split-that-changes-the-answer).

## Class balance

Every emotion has 192 clips except `neutral`, which has 96. The reason is field 4:
the other seven emotions were recorded at both normal and strong intensity, while
neutral has no "strong" variant. Two consequences:

- The 4-class subset used by the baseline (`calm`, `happy`, `fearful`, `disgust`)
  is **perfectly balanced** at 192 clips each, 768 total. Accuracy is a fair
  metric there, and the random-guess floor is 25%.
- The 8-class run is mildly imbalanced (neutral at half weight), so the per-class
  recall in the classification report matters more than the headline accuracy.

## Two things to know before trusting any RAVDESS result

1. **It is acted, not spontaneous.** Actors performing "fearful" on cue produce
   cleaner, more exaggerated emotional cues than a real frightened person does.
   Models trained here transfer poorly to spontaneous speech.
2. **Only two sentences, repeated.** Lexical content is held constant by design,
   which is good, the model cannot cheat by reading the words. But it also means
   the model never learns to be robust to varied content.

## Citation

> Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of
> Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and
> vocal expressions in North American English. *PLoS ONE, 13*(5), e0196391.
> https://doi.org/10.1371/journal.pone.0196391
