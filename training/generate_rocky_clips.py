"""Generate synthetic wake-word training clips locally with Kokoro TTS.

Stands in for openWakeWord's automated notebook's --generate_clips step,
which needs piper-phonemize (no Python 3.13 wheel, so it can't run on
current Colab). Kokoro is pure ONNX + onnxruntime — no such wall — and is
already set up in this project.

Produces two sets of 16kHz/16-bit mono WAVs, matching the directory layout
train.py's --augment_clips step expects:
  - positive_train/positive_test: the word "Rocky", across every Kokoro
    voice at a few speeds.
  - negative_train/negative_test: a curated set of phonetically-similar
    decoys (rocket, hockey, lucky, ...) plus generic common words/phrases,
    across a handful of voices. --augment_clips requires these dirs to
    exist and be non-empty, same as the positive ones — the original
    --generate_clips step produced both from piper; this produces both
    from Kokoro instead.

Upload the rocky_clips/ folder this produces to Colab and skip straight to
the --augment_clips and --train_model steps.
"""

import os
import random
import sys

import numpy as np
import scipy.signal
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import voice.speak  # noqa: F401,E402 — side effect: patches Kokoro's dtype bug
from kokoro_onnx import Kokoro  # noqa: E402

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "..", "voice", "kokoro-v1.0.fp16.onnx")
VOICES_PATH = os.path.join(HERE, "..", "voice", "voices-v1.0.bin")
OUTPUT_DIR = os.path.join(HERE, "rocky_clips")
TARGET_SR = 16000
TEST_FRACTION = 0.15

POSITIVE_WORD = "Rocky"
POSITIVE_SPEEDS = [0.9, 1.0, 1.1]

# Phonetically-similar decoys (harden against false positives) plus generic
# common words/phrases (general negative coverage). Deliberately excludes
# actual homophones of "Rocky".
NEGATIVE_WORDS = [
    "rocket", "rock", "rocky mountain", "hockey", "lucky", "jockey", "hokey",
    "rocking", "rocks", "rosy", "roxy", "rugby", "walkie", "cookie", "foggy",
    "lobby", "copy", "coffee", "toffee", "trolley", "grocery",
    "hello", "computer", "weather", "music", "stop", "play", "open", "close",
    "turn on", "turn off", "good morning", "thank you", "yes", "no", "maybe",
    "later", "today", "tomorrow", "system", "internet", "battery", "camera",
    "picture", "message", "email", "calendar", "reminder", "timer", "volume",
    "light",
]
NEGATIVE_VOICES = ["af_heart", "am_michael", "bf_emma", "bm_george", "af_bella", "am_adam"]


def resample_to_16k(audio: np.ndarray, orig_sr: int) -> np.ndarray:
    return scipy.signal.resample_poly(audio, TARGET_SR, orig_sr).astype(np.float32)


def synth(kokoro: Kokoro, text: str, voice: str, speed: float) -> np.ndarray:
    samples, sr = kokoro.create(text, voice=voice, speed=speed, lang="en-us")
    return resample_to_16k(samples, sr)


def generate_positive(kokoro: Kokoro) -> None:
    voices = kokoro.get_voices()
    train_dir = os.path.join(OUTPUT_DIR, "positive_train")
    test_dir = os.path.join(OUTPUT_DIR, "positive_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(0)
    test_voices = set(random.sample(voices, max(1, round(len(voices) * TEST_FRACTION))))

    count = 0
    for voice in voices:
        for speed in POSITIVE_SPEEDS:
            clip = synth(kokoro, POSITIVE_WORD, voice, speed)
            out_dir = test_dir if voice in test_voices else train_dir
            sf.write(os.path.join(out_dir, f"{voice}_{speed}.wav"), clip, TARGET_SR, subtype="PCM_16")
            count += 1
    print(f"Positive: {count} clips across {len(voices)} voices "
          f"({len(voices) - len(test_voices)} voices -> train, {len(test_voices)} -> test).")


def generate_negative(kokoro: Kokoro) -> None:
    train_dir = os.path.join(OUTPUT_DIR, "negative_train")
    test_dir = os.path.join(OUTPUT_DIR, "negative_test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    random.seed(1)
    test_words = set(random.sample(NEGATIVE_WORDS, max(1, round(len(NEGATIVE_WORDS) * TEST_FRACTION))))

    count = 0
    for word in NEGATIVE_WORDS:
        for voice in NEGATIVE_VOICES:
            clip = synth(kokoro, word, voice, 1.0)
            out_dir = test_dir if word in test_words else train_dir
            safe_word = word.replace(" ", "_")
            sf.write(os.path.join(out_dir, f"{safe_word}_{voice}.wav"), clip, TARGET_SR, subtype="PCM_16")
            count += 1
    print(f"Negative: {count} clips across {len(NEGATIVE_WORDS)} words "
          f"({len(NEGATIVE_WORDS) - len(test_words)} words -> train, {len(test_words)} -> test).")


def main() -> None:
    kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    generate_positive(kokoro)
    generate_negative(kokoro)


if __name__ == "__main__":
    main()
