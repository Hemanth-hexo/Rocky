"""Wake word detection via openWakeWord — fully local/offline, no account needed.

"Rocky" and "stop" aren't among openWakeWord's built-in pretrained words
(alexa, hey_jarvis, hey_mycroft, hey_rhasspy), so both expect custom-trained
ONNX models. See the project README/chat history for how to train one.
"""

import os
import threading

import numpy as np
import pyaudio
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openWakeWord's expected frame size
THRESHOLD = float(os.environ.get("ROCKY_WAKEWORD_THRESHOLD", "0.5"))

_HERE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get("ROCKY_WAKEWORD_MODEL_PATH", os.path.join(_HERE, "rocky_wakeword.onnx"))
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]

STOP_MODEL_PATH = os.environ.get("ROCKY_STOP_MODEL_PATH", os.path.join(_HERE, "stop_wakeword.onnx"))
STOP_MODEL_NAME = os.path.splitext(os.path.basename(STOP_MODEL_PATH))[0]


def _listen_until_detected(stop_event: threading.Event, model_paths: list[str]) -> bool:
    """Opens a fresh mic stream and listens with one or more wake-word models
    loaded simultaneously (openWakeWord natively supports this — predict()
    returns one score per model, keyed by each model file's stem name).
    Returns True as soon as ANY of them crosses threshold, False if
    stop_event is set first."""
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No wake word model at {path}. Train a custom openWakeWord model "
                "and set the matching env var, or drop it at that default path."
            )
    model_names = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

    model = Model(wakeword_models=model_paths, inference_framework="onnx")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=SAMPLE_RATE,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )
    try:
        while not stop_event.is_set():
            chunk = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            audio = np.frombuffer(chunk, dtype=np.int16)
            scores = model.predict(audio)
            if any(scores.get(name, 0.0) >= THRESHOLD for name in model_names):
                return True
        return False
    finally:
        stream.close()
        pa.terminate()


def wait_for_wake_word() -> None:
    _listen_until_detected(threading.Event(), [MODEL_PATH])


def listen_for_wake_word_until(stop_event: threading.Event, on_detected) -> None:
    """For barge-in: listens for "Rocky" OR "stop" until one is heard (calls
    on_detected() and returns) or stop_event is set by the caller (returns
    without calling on_detected()). Meant to run on its own thread alongside
    TTS playback. Falls back to "Rocky" only if the stop model isn't trained
    yet, so this doesn't break before that model exists."""
    model_paths = [MODEL_PATH]
    if os.path.exists(STOP_MODEL_PATH):
        model_paths.append(STOP_MODEL_PATH)
    if _listen_until_detected(stop_event, model_paths):
        on_detected()
