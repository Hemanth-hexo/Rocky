"""Wake word -> listen -> agent loop -> speak."""

import tempfile

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

from agent.loop import new_conversation, run_turn
from agent.rocky_transform import rocky_transform
from voice.listen import transcribe
from voice.speak import speak
from voice.wake_word import wait_for_wake_word

RECORD_SECONDS = 5
SAMPLE_RATE = 16000


def record_command() -> str:
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        return f.name


def main() -> None:
    messages = new_conversation()
    print("Rocky is listening for the wake word...")
    while True:
        wait_for_wake_word()
        print("Wake word heard — recording...")
        audio_path = record_command()
        text = transcribe(audio_path)
        print(f"You: {text}")

        messages.append({"role": "user", "content": text})
        run_turn(messages)
        reply = rocky_transform(messages[-1]["content"])
        print(f"Rocky: {reply}")
        speak(reply)


if __name__ == "__main__":
    main()
