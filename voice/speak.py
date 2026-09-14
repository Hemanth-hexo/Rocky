"""Text-to-speech via kokoro-onnx, running fully local/offline, no API key."""

import os
import time

import numpy as np
import onnxruntime as ort
import sounddevice as sd
from kokoro_onnx import Kokoro
from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE
from kokoro_onnx.log import log

ort.set_default_logger_severity(3)  # silence harmless fp16 constant-folding warnings

_kokoro = None

_HERE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", os.path.join(_HERE, "kokoro-v1.0.fp16.onnx"))
VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", os.path.join(_HERE, "voices-v1.0.bin"))


def _patched_create_audio(self, phonemes, voice, speed):
    """kokoro-onnx 0.4.7 builds the "speed" input as int32 for newer (input_ids)
    model exports, but those models declare it as float32 — ONNXRuntime then
    rejects it with InvalidArgument. This is the same method with that one
    dtype fixed; drop it once upstream ships a fix."""
    phonemes = phonemes[:MAX_PHONEME_LENGTH]
    start_t = time.time()
    tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
    voice = voice[len(tokens)]
    tokens = [[0, *tokens, 0]]
    if "input_ids" in [i.name for i in self.sess.get_inputs()]:
        inputs = {
            "input_ids": tokens,
            "style": np.array(voice, dtype=np.float32),
            "speed": np.array([speed], dtype=np.float32),
        }
    else:
        inputs = {
            "tokens": tokens,
            "style": voice,
            "speed": np.ones(1, dtype=np.float32) * speed,
        }
    audio = self.sess.run(None, inputs)[0]
    audio_duration = len(audio) / SAMPLE_RATE
    log.debug(f"Created audio in {time.time() - start_t:.2f}s for {audio_duration:.2f}s of audio")
    return audio, SAMPLE_RATE


Kokoro._create_audio = _patched_create_audio


def _get_kokoro() -> Kokoro:
    global _kokoro
    if _kokoro is None:
        _kokoro = Kokoro(MODEL_PATH, VOICES_PATH)
    return _kokoro


def speak(text: str, voice: str = "am_adam") -> None:
    samples, sample_rate = _get_kokoro().create(text, voice=voice, speed=1.0, lang="en-us")
    sd.play(samples, sample_rate)
    sd.wait()
