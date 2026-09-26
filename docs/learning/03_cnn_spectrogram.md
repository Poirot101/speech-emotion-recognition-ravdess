# 03: A CNN on the log-mel spectrogram

**Goal:** stop averaging over time. Train a small convolutional neural network
directly on log-mel spectrograms and find out whether it beats the best classical
model **on speakers it has never heard**.

**Effort:** 3–5 weeks, most of it learning PyTorch and debugging training.
**Prerequisites:** `00_foundations.md`, `02_augmentation.md`.

---

## Why this matters here

- Every classical model in this repo sees a time-averaged vector. Pitch *contour*
  (rising versus falling) is invisible to it, and that is the main cue
  separating emotions at the same energy level (the 8-class confusions:
  `angry` → `happy`, neutral recall of 21%).
- A CNN treats the spectrogram as an image: its filters can detect a rising
  harmonic or a sudden burst wherever it occurs.
- The real risk: **576 training clips is very little data for a CNN.** It may
  not beat the classical model. That is a legitimate, publishable result if it is
  measured properly.

---

## Concepts to understand first

**PyTorch mechanics**
- Tensors, devices, `autograd`
- `nn.Module`, `forward`, parameters
- `Dataset` and `DataLoader`
- The training loop: forward → loss → `backward()` → `optimizer.step()` → `zero_grad()`
- `model.train()` versus `model.eval()`, and why forgetting it silently breaks
  batch norm and dropout

**CNN concepts**
- Convolution, kernels, channels, stride, padding
- Receptive field, how many time frames and frequency bands a unit can "see"
- Pooling, and global average pooling
- Batch normalisation, dropout, weight decay
- Cross-entropy loss, the Adam optimiser, learning-rate schedules
- Early stopping

**Input representation**
- Log-mel spectrogram as a `[batch, 1, n_mels, frames]` tensor
- Fixed-length input: RAVDESS clips are 3–5 s. Choose a length, then pad or crop.
  Random crops during training act as augmentation; use a fixed centre crop
  for evaluation.
- Per-band normalisation using **training-set** statistics only

---

## Build milestones

### M0: Prove your training loop on a problem you already solved
In PyTorch, train a plain MLP on the existing **180-dim feature vectors** with the
same speaker split. You should land near sklearn's 59%.
**Checkpoint:** within a few points of sklearn. **Do not start M1 until this
works**, if the CNN fails later, you will know the loop is not the reason.

### M1: Input pipeline
Compute the log-mel spectrogram for every clip (`n_mels` of 64 or 128), fix the
length, cache each as `.npy`. Compute per-band mean and std from training actors only.
**Checkpoint:** plot five cached inputs; the shapes are identical; the
normalisation statistics came from training actors only.

### M2: A small model
A reasonable starting point: four blocks of
`Conv2d(3×3) → BatchNorm → ReLU → MaxPool(2)` with 16 → 32 → 64 → 128 channels,
then global average pooling, dropout, and a linear layer to 4 classes. Count the
parameters and compare with the dataset size.
**Checkpoint:** a forward pass on a batch returns `[batch, 4]`; the parameter
count is in your lab notebook.

### M3: Train with a validation split *inside the training actors*
Split the 18 training actors again, by actor, into train and validation. Early
stopping and every hyperparameter choice use this validation split.
**The held-out actors (4, 5, 6, 8, 10, 19) are never used for any decision.**
**Checkpoint:** training and validation loss curves are plotted; you can say
whether the model is overfitting, underfitting, or neither.

### M4: Regularise
Add SpecAugment masking (`torchaudio.transforms.FrequencyMasking` and
`TimeMasking`), plus the transforms that helped in project 02. Tune dropout and
weight decay on the validation split.
**Checkpoint:** the gap between training and validation accuracy narrows.

### M5: Evaluate like the rest of the repo
Train with at least **5 seeds**. Evaluate each on the held-out actors, then under
nested grouped CV over all 24 actors if compute allows (6 outer folds × 5 seeds is
30 trainings). Report mean ± std, a speaker-level bootstrap interval, and Wilcoxon
on per-speaker gains against the best classical procedure (nested 69.27%).

### M6 (stretch): Pretrained speech representations
Extract embeddings from a pretrained speech model (for example wav2vec 2.0) and
train a simple classifier on top. On small datasets this is often the strongest
approach, and a good lesson in transfer learning.

---

## Implementation blueprint

### Input sizing: work it out before writing code

At 48 kHz with `hop_length=512`, a clip of *N* samples gives `1 + N // 512`
frames. A 4-second crop is 192,000 samples → **376 frames**. With `n_mels=128`
the input is `[batch, 1, 128, 376]`. Consider resampling to 16 kHz for the CNN
(fewer samples, same speech band up to 8 kHz), but then retrain the classical
baseline at 16 kHz too, or the comparison is unfair.

### Files and signatures

```python
# src/speech_emotion/cnn/data.py
def cache_log_mels(files: list[Path], out_dir: Path, sr: int, n_mels: int,
                   hop: int) -> None: ...              # one .npy per clip: (n_mels, T)

@dataclass(frozen=True)
class BandStats:
    mean: np.ndarray    # (n_mels,)  computed from TRAINING actors only
    std: np.ndarray

def band_stats(paths: list[Path]) -> BandStats: ...

class LogMelDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels, stats: BandStats, n_frames: int,
                 train: bool, spec_augment: nn.Module | None = None): ...
    def __getitem__(self, i) -> tuple[torch.Tensor, int]:
        """train=True: random crop (+ masking); train=False: centre crop. Pad if short."""

# src/speech_emotion/cnn/model.py
class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, channels=(16, 32, 64, 128), dropout: float = 0.3): ...

# src/speech_emotion/cnn/train.py
@dataclass
class History:
    train_loss: list[float]; val_loss: list[float]; val_acc: list[float]; best_epoch: int

def train_one(model, train_loader, val_loader, *, epochs: int, lr: float,
              weight_decay: float, patience: int, device: str) -> History: ...
def predict_proba(model, loader, device) -> np.ndarray: ...   # model.eval(), no_grad
def seed_everything(seed: int) -> None: ...                   # random, numpy, torch
```

**Parameter count check.** The four conv blocks in M2 with 3×3 kernels, batch
norm and a 4-way linear head come to roughly 98,000 parameters. Compute it by
hand (conv weights are `in × out × 9 + out`), then confirm with
`sum(p.numel() for p in model.parameters())`.

### Split structure

```
24 actors
 ├── outer test fold (nested CV) or the 6 held-out actors
 └── the rest
      ├── validation actors  → early stopping, every hyperparameter
      └── training actors    → gradient updates, BandStats
```

Assertions:

```python
assert not set(train_actors) & set(val_actors)
assert not (set(train_actors) | set(val_actors)) & set(test_actors)
assert stats_were_computed_from(train_actors)      # store the actor list in BandStats
```

### Checkpointing

Save the state dict at the best validation epoch, reload it before predicting on
test actors. Record `{seed, epoch, val_acc, lr, weight_decay, git commit}` next to
each checkpoint.

### Comparison that counts

Five seeds × nested outer folds, per-speaker accuracy, then the same bootstrap
and Wilcoxon test against the **best classical procedure** (nested 69.27%),
not against the MLP baseline.

---

## Pitfalls

- **Normalisation statistics from all clips**, preprocessing leakage.
- **Early stopping on the held-out actors**, selection leakage, and the most
  common mistake in small-data deep learning.
- **Evaluating on random crops**, scores then change from run to run.
- **Forgetting `model.eval()`**, batch norm uses the wrong statistics.
- **Wrong tensor layout**, `[batch, channels, height, width]`; get `n_mels` and
  `frames` in the right order.
- **A learning rate that is too high**, the loss explodes or stays flat. Try a
  learning-rate range test.
- **Reporting one seed.** With 576 clips, run-to-run variance is large.

---

## Practical notes

- **Hardware.** CPU is enough for this dataset size. On Apple silicon, PyTorch can
  also use the `mps` device.
- **Installation.** A dry run in this repo's venv resolved PyTorch and torchaudio
  wheels for Python 3.14. If torchaudio gives you trouble, compute log-mels with
  librosa (which you already know) and use PyTorch only for the model.
- **Reproducibility.** Seed Python, NumPy and PyTorch; record library versions.

---

## How you will know you succeeded

- M0 reproduces the sklearn baseline, your loop is trustworthy.
- The CNN's held-out mean over seeds is reported with its spread and compared
  fairly against the best classical configuration.
- **Either outcome is a result.** "A CNN trained from scratch on 576 clips did
  not beat a tuned SVM on unseen speakers" is useful to know, if measured this carefully.

---

## Resources

**PyTorch**
- Learn the basics (start here), https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- Datasets and DataLoaders, https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
- Building a model, https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- torchaudio transforms, https://docs.pytorch.org/audio/stable/transforms.html
- `MelSpectrogram`, https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.MelSpectrogram.html
- `FrequencyMasking`, https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.FrequencyMasking.html
- `TimeMasking`, https://docs.pytorch.org/audio/stable/generated/torchaudio.transforms.TimeMasking.html

**CNNs and deep learning**
- Stanford CS231n notes on convolutional networks, https://cs231n.github.io/convolutional-networks/
- *Dive into Deep Learning*, https://d2l.ai/
- fast.ai practical deep learning course, https://course.fast.ai/
- Hugging Face audio course, https://huggingface.co/learn/audio-course/chapter0/introduction
- Hugging Face audio classification guide (for M6), https://huggingface.co/docs/transformers/tasks/audio_classification

**Papers**
- Hershey et al. (2016), *CNN Architectures for Large-Scale Audio Classification*, https://arxiv.org/abs/1609.09430
- He et al. (2015), *Deep Residual Learning for Image Recognition*, https://arxiv.org/abs/1512.03385
- Ioffe & Szegedy (2015), *Batch Normalization*, https://arxiv.org/abs/1502.03167
- Kingma & Ba (2014), *Adam: A Method for Stochastic Optimization*, https://arxiv.org/abs/1412.6980
- Hinton et al. (2012), *Improving neural networks by preventing co-adaptation of feature detectors* (dropout), https://arxiv.org/abs/1207.0580
- Park et al. (2019), *SpecAugment*, https://arxiv.org/abs/1904.08779
- Baevski et al. (2020), *wav2vec 2.0*, https://arxiv.org/abs/2006.11477
