"""Wake word -> listen -> agent loop -> speak."""

import tempfile
import threading
from typing import Callable, Optional

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

from agent.loop import new_conversation, run_turn
from agent.obsidian import append_daily_log
from agent.rocky_transform import rocky_transform
from voice.listen import transcribe
from voice.speak import speak
from voice.wake_word import wait_for_wake_word

RECORD_SECONDS = 5
FOLLOWUP_RECORD_SECONDS = 5
SAMPLE_RATE = 16000

StatusCallback = Callable[[str, str], None]


def record_command(seconds: float = RECORD_SECONDS) -> str:
    audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        return f.name


def handle_turn(messages: list, text: str, on_status: StatusCallback, lock: Optional[threading.Lock] = None) -> bool:
    """Runs one turn of the conversation (agent loop + speak) for `text`.
    Takes `lock` when given, so this can be called safely from a typed-input
    handler running alongside the voice loop on `messages` shared between them.

    Returns True if the reply was cut off partway through because the wake
    word was heard again mid-speech (barge-in) — the caller should treat
    that as "the user wants to talk now" rather than waiting for silence."""
    def _run() -> bool:
        on_status("heard_command", text)
        messages.append({"role": "user", "content": text})
        on_status("thinking", "")
        run_turn(messages)
        reply = rocky_transform(messages[-1]["content"])
        append_daily_log(text, reply)
        on_status("speaking", reply)
        completed = speak(reply, interruptible=True)
        return not completed

    if lock is None:
        return _run()
    with lock:
        return _run()


def run_rocky(on_status: StatusCallback, messages: Optional[list] = None, lock: Optional[threading.Lock] = None) -> None:
    """Runs the wake word -> listen -> agent loop -> speak loop forever,
    reporting each state transition via on_status(state, detail).

    After answering, listens again immediately for a follow-up without
    needing the wake word repeated — like a smart speaker's "follow-up mode".
    A follow-up recording that transcribes to nothing (silence) ends the
    conversation and drops back to waiting for the wake word. Saying the
    wake word again while Rocky is still speaking cuts it off (barge-in)
    and jumps straight to recording the new command.

    `messages`/`lock` can be supplied to share this conversation with a
    typed-input path (e.g. a GUI text box) running on another thread;
    otherwise a private conversation is created."""
    if messages is None:
        messages = new_conversation()
    on_status("listening", "")
    while True:
        wait_for_wake_word()
        on_status("heard_wake_word", "")
        audio_path = record_command()
        on_status("transcribing", "")
        text = transcribe(audio_path)
        interrupted = handle_turn(messages, text, on_status, lock)

        # Follow-up mode: keep listening without the wake word until silence,
        # unless a barge-in interrupt means the user is already talking again.
        while True:
            if interrupted:
                on_status("heard_wake_word", "")
                audio_path = record_command()
            else:
                on_status("listening_followup", "")
                audio_path = record_command(FOLLOWUP_RECORD_SECONDS)

            on_status("transcribing", "")
            text = transcribe(audio_path)
            if not text.strip():
                break
            interrupted = handle_turn(messages, text, on_status, lock)

        on_status("listening", "")


def _print_status(state: str, detail: str) -> None:
    labels = {
        "listening": "Rocky is listening for the wake word...",
        "heard_wake_word": "Wake word heard — recording...",
        "listening_followup": "Listening for a follow-up (no wake word needed)...",
        "transcribing": "Transcribing...",
        "heard_command": f"You: {detail}",
        "thinking": "Rocky is thinking...",
        "speaking": f"Rocky: {detail}",
    }
    print(labels.get(state, state))


if __name__ == "__main__":
    try:
        run_rocky(_print_status)
    except KeyboardInterrupt:
        print("\nRocky signing off.")
