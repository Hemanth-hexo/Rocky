"""Generate synthetic "stop" wake-word clips locally with Kokoro TTS.

Same approach as generate_rocky_clips.py, for the "stop" barge-in word
(interrupts Rocky mid-speech, alongside saying "Rocky" itself). See that
file's docstring for why Kokoro instead of piper.
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
OUTPUT_DIR = os.path.join(HERE, "stop_clips")
TARGET_SR = 16000
TEST_FRACTION = 0.15

POSITIVE_WORD = "Stop"
POSITIVE_SPEEDS = [0.9, 1.0, 1.1]

# Phonetically-similar decoys plus generic common words/phrases.
NEGATIVE_WORDS = [
    "shop", "top", "stock", "stomp", "stopped", "drop", "chop", "hop", "cop",
    "stopping", "swap", "strap", "step", "stab", "spot", "stove", "stove top",
    "hello", "computer", "weather", "music", "play", "open", "close",
    "turn on", "turn off", "good morning", "thank you", "yes", "no", "maybe",
    "later", "today", "tomorrow", "system", "internet", "battery", "camera",
    "picture", "message", "email", "calendar", "reminder", "timer", "volume",
    "light", "rocky",
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
