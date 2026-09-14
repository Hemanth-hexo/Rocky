"""Download training data for local (Mac) openWakeWord training — the local
equivalent of the notebook's "Download Data" section, adapted for two things
the notebook couldn't anticipate: newer `datasets` decodes audio via
torchcodec (AudioDecoder objects, not the old dict), and the AudioSet HF repo
switched from per-part .tar files to parquet shards, so the notebook's
`bal_train09.tar` URL 404s now regardless of platform.

Downloads, into training/local/data/:
  - mit_rirs/        room impulse responses (streamed, small)
  - audioset_16k/     ~2000 clips sampled from agkphysics/AudioSet (streamed)
  - fma/              ~1 hour of Free Music Archive clips (streamed)
  - openwakeword_features_ACAV100M_2000_hrs_16bit.npy   (17.3GB, direct download)
  - validation_set_features.npy                          (185MB, direct download)
"""

import os

import librosa
import numpy as np
import requests
import scipy.io.wavfile
from tqdm import tqdm

import datasets

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")

N_AUDIOSET_CLIPS = 2000
N_FMA_HOURS = 1


def download_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"already have {dest}, skipping")
        return
    tmp = dest + ".part"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
    os.rename(tmp, dest)


def decode_row_audio(row: dict, target_sr: int = 16000) -> np.ndarray:
    """AudioDecoder-based decode (newer `datasets`), downmixed to mono and
    resampled to target_sr if needed."""
    samples = row["audio"].get_all_samples()
    audio = samples.data.numpy()
    if audio.ndim == 2:
        audio = audio.mean(axis=0)
    if samples.sample_rate != target_sr:
        audio = librosa.resample(audio, orig_sr=samples.sample_rate, target_sr=target_sr)
    return audio


def download_rirs() -> None:
    output_dir = os.path.join(DATA_DIR, "mit_rirs")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 200:
        print(f"{output_dir} already populated, skipping")
        return
    os.makedirs(output_dir, exist_ok=True)
    ds = datasets.load_dataset("davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True)
    for i, row in enumerate(tqdm(ds, desc="MIT RIRs")):
        audio = decode_row_audio(row)
        scipy.io.wavfile.write(os.path.join(output_dir, f"rir_{i:05d}.wav"), 16000, (audio * 32767).astype(np.int16))


def download_audioset() -> None:
    output_dir = os.path.join(DATA_DIR, "audioset_16k")
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= N_AUDIOSET_CLIPS * 0.95:
        print(f"{output_dir} already populated, skipping")
        return
    os.makedirs(output_dir, exist_ok=True)
    # The notebook's original bal_train09.tar approach 404s — AudioSet on HF moved to
    # parquet shards. Stream the "balanced" config instead and take a subset.
    ds = datasets.load_dataset("agkphysics/AudioSet", "balanced", split="train", streaming=True)
    for i, row in enumerate(tqdm(ds.take(N_AUDIOSET_CLIPS), total=N_AUDIOSET_CLIPS, desc="AudioSet")):
        audio = decode_row_audio(row)
        scipy.io.wavfile.write(os.path.join(output_dir, f"audioset_{i:05d}.wav"), 16000, (audio * 32767).astype(np.int16))


def download_fma() -> None:
    output_dir = os.path.join(DATA_DIR, "fma")
    n_clips = N_FMA_HOURS * 3600 // 30  # FMA clips here are ~30s each
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) >= n_clips * 0.95:
        print(f"{output_dir} already populated, skipping")
        return
    os.makedirs(output_dir, exist_ok=True)
    ds = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    for i, row in enumerate(tqdm(ds.take(n_clips), total=n_clips, desc="FMA")):
        audio = decode_row_audio(row)
        scipy.io.wavfile.write(os.path.join(output_dir, f"fma_{i:05d}.wav"), 16000, (audio * 32767).astype(np.int16))


def download_precomputed_features() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    download_file(
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        os.path.join(DATA_DIR, "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"),
    )
    download_file(
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
        os.path.join(DATA_DIR, "validation_set_features.npy"),
    )


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    download_rirs()
    download_audioset()
    # download_fma() skipped: rudraml/fma uses a legacy HF "dataset script" loader,
    # which modern `datasets` versions removed support for entirely (not a version
    # pin issue). It only ever contributed 1hr of background music next to AudioSet's
    # 2000 clips + the 2000hr precomputed set, so dropping it rather than chasing a
    # replacement source. rocky_config.yml's background_paths no longer lists it.
    download_precomputed_features()
    print("Done.")
