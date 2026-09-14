"""Live mic test for the trained Rocky wake-word model — prints the running
score continuously instead of silently waiting, so you can see how it
responds even to near-miss attempts. Ctrl+C to stop."""

import os
import sys

import numpy as np
import pyaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from openwakeword.model import Model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "voice", "rocky_wakeword.onnx")
MODEL_NAME = os.path.splitext(os.path.basename(MODEL_PATH))[0]
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 1280
THRESHOLD = 0.5

model = Model(wakeword_models=[MODEL_PATH], inference_framework="onnx")
pa = pyaudio.PyAudio()
stream = pa.open(rate=SAMPLE_RATE, channels=1, format=pyaudio.paInt16, input=True, frames_per_buffer=CHUNK_SAMPLES)

print("Listening — say 'Rocky'. Ctrl+C to stop.\n")
max_score = 0.0
was_above_threshold = False
try:
    while True:
        chunk = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)
        audio = np.frombuffer(chunk, dtype=np.int16)
        score = model.predict(audio)[MODEL_NAME]
        max_score = max(max_score, score)

        is_above = score >= THRESHOLD
        if is_above and not was_above_threshold:
            print(f"\n*** TRIGGER at {score:.3f} ***")
        was_above_threshold = is_above

        bar = "#" * int(score * 40)
        marker = "<-- TRIGGER" if is_above else ""
        line = f"{score:.3f} {bar:<40}{marker}"
        print(f"\r{line:<70}", end="", flush=True)
except KeyboardInterrupt:
    print(f"\nStopped. Session max score: {max_score:.3f}")
finally:
    stream.close()
    pa.terminate()
