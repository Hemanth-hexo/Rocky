"""Text-to-speech via kokoro-onnx, running fully local/offline, no API key."""

import os
import subprocess
import tempfile
import threading
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from kokoro_onnx import Kokoro
from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE
from kokoro_onnx.log import log

from voice.robot_effect import robotize

ort.set_default_logger_severity(3)  # silence harmless fp16 constant-folding warnings

_kokoro = None

_HERE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get("KOKORO_MODEL_PATH", os.path.join(_HERE, "kokoro-v1.0.fp16.onnx"))
VOICES_PATH = os.environ.get("KOKORO_VOICES_PATH", os.path.join(_HERE, "voices-v1.0.bin"))
DEFAULT_VOICE = "af_nova"
ROBOT_EFFECT_ENABLED = os.environ.get("ROCKY_ROBOT_VOICE", "0") == "1"


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


def speak(text: str, voice: str = DEFAULT_VOICE, interrupt_event: threading.Event | None = None) -> bool:
    """Speaks `text` aloud. Returns True if playback completed normally,
    False if `interrupt_event` was set partway through (barge-in).

    Plays via a temp WAV file + macOS's `afplay` rather than streaming the
    raw array through sounddevice directly — sounddevice underrunning its
    buffer for irregularly-sized TTS output was causing audible crackling;
    afplay's own buffering through CoreAudio is more robust.

    `interrupt_event` decouples "how playback gets interrupted" from this
    function — pass a threading.Event that some other mechanism (the
    push-to-talk hotkey, in main.py) sets to cut playback off immediately.
    """
    samples, sample_rate = _get_kokoro().create(text, voice=voice, speed=1.0, lang="en-us")
    if ROBOT_EFFECT_ENABLED:
        samples = robotize(samples, sample_rate, pitch_shift_semitones=0.0, ring_mod_mix=0.15)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    sf.write(wav_path, samples, sample_rate)

    try:
        proc = subprocess.Popen(["afplay", wav_path])
        if interrupt_event is None:
            proc.wait()
            return True

        while proc.poll() is None:
            if interrupt_event.is_set():
                proc.terminate()
                proc.wait()
                return False
            time.sleep(0.05)
        return not interrupt_event.is_set()
    finally:
        os.remove(wav_path)
