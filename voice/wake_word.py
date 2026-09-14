"""Wake word detection via openWakeWord — fully local/offline, no account needed.

"Rocky" isn't one of openWakeWord's built-in pretrained words (alexa, hey_jarvis,
hey_mycroft, hey_rhasspy), so this expects a custom-trained ONNX model at
ROCKY_WAKEWORD_MODEL_PATH. See the project README/chat history for how to train one.
"""

import os

import numpy as np
import pyaudio
from openwakeword.model import Model

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280  # 80ms at 16kHz — openWakeWord's expected frame size
THRESHOLD = float(os.environ.get("ROCKY_WAKEWORD_THRESHOLD", "0.5"))

_HERE = os.path.dirname(__file__)
MODEL_PATH = os.environ.get("ROCKY_WAKEWORD_MODEL_PATH", os.path.join(_HERE, "rocky_wakeword.onnx"))
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]


def wait_for_wake_word() -> None:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No wake word model at {MODEL_PATH}. Train a custom openWakeWord model for "
            "\"Rocky\" and set ROCKY_WAKEWORD_MODEL_PATH, or drop it at that default path."
        )

    model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=SAMPLE_RATE,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES,
    )
    try:
        while True:
            chunk = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
            audio = np.frombuffer(chunk, dtype=np.int16)
            scores = model.predict(audio)
            if scores.get(MODEL_NAME, 0.0) >= THRESHOLD:
                return
    finally:
        stream.close()
        pa.terminate()
