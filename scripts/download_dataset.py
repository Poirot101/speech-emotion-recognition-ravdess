"""Download and unpack the RAVDESS speech dataset (~198 MB).

    python scripts/download_dataset.py

The archive is fetched from the authors' Zenodo record (DOI 10.5281/zenodo.1188976)
and unpacked into ``data/ravdess/Actor_01 ... Actor_24``. The download is
skipped if the audio is already in place, so re-running is cheap.

RAVDESS is released under CC BY-NC-SA 4.0 (non-commercial use, with
attribution). Cite: Livingstone & Russo (2018), PLoS ONE 13(5): e0196391.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "ravdess"
ARCHIVE = REPO_ROOT / "data" / "Audio_Speech_Actors_01-24.zip"
URL = ("https://zenodo.org/records/1188976/files/"
       "Audio_Speech_Actors_01-24.zip?download=1")
EXPECTED_FILES = 1440  # 24 actors x 60 clips


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = block_num * block_size
    pct = min(100.0, downloaded * 100 / total_size)
    done = int(pct // 2)
    sys.stdout.write(f"\r  [{'=' * done}{' ' * (50 - done)}] {pct:5.1f}%")
    sys.stdout.flush()


def count_wavs(data_dir: Path) -> int:
    return len(list(data_dir.glob("Actor_*/*.wav")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the data is already present")
    parser.add_argument("--keep-archive", action="store_true",
                        help="keep the .zip after extracting")
    args = parser.parse_args(argv)

    existing = count_wavs(DATA_DIR)
    if existing >= EXPECTED_FILES and not args.force:
        print(f"Dataset already present: {existing} wav files in "
              f"{DATA_DIR.relative_to(REPO_ROOT)}. Nothing to do.")
        return 0

    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists() or args.force:
        print(f"Downloading RAVDESS (~198 MB) from Zenodo ...")
        urllib.request.urlretrieve(URL, ARCHIVE, reporthook=_progress)
        print()
    else:
        print(f"Using existing archive {ARCHIVE.name}")

    print(f"Extracting to {DATA_DIR.relative_to(REPO_ROOT)} ...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        zf.extractall(DATA_DIR)

    found = count_wavs(DATA_DIR)
    if found != EXPECTED_FILES:
        print(f"WARNING: found {found} wav files, expected {EXPECTED_FILES}.")
    else:
        print(f"Done: {found} wav files across "
              f"{len(list(DATA_DIR.glob('Actor_*')))} actors.")

    if not args.keep_archive:
        ARCHIVE.unlink(missing_ok=True)
        print("Removed the archive (pass --keep-archive to keep it).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
