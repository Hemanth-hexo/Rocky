"""Push-to-talk (hold Option+Control) -> listen -> agent loop -> speak."""

import tempfile
import threading
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

from agent.conversation_store import load_conversation, save_conversation
from agent.loop import run_turn
from agent.obsidian import append_daily_log
from agent.rocky_transform import rocky_transform
from voice.listen import transcribe
from voice.push_to_talk import PushToTalkListener
from voice.speak import speak

SAMPLE_RATE = 16000

# Option+Control are common modifier keys — briefly, accidentally holding
# both together (e.g. while typing another shortcut) can trigger a
# recording nobody meant to start. Combined with faster-whisper's tendency
# to hallucinate plausible-sounding text from near-silent audio, that
# produced Rocky responding to "messages" the user never said. Both guards
# below exist specifically to kill that.
MIN_HOLD_SECONDS = 0.25
SILENCE_RMS_THRESHOLD = 150  # int16 scale; real speech sits well above this

StatusCallback = Callable[[str, str], None]


class _Recorder:
    """Records audio via a sounddevice InputStream until stop() is called —
    unlike a fixed-duration recording, this runs for exactly as long as the
    hotkey is held."""

    def __init__(self):
        self._frames: list[np.ndarray] = []
        self._stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16", callback=self._callback)
        self._started_at = 0.0

    def _callback(self, indata, frame_count, time_info, status) -> None:
        self._frames.append(indata.copy())

    def start(self) -> None:
        self._started_at = time.monotonic()
        self._stream.start()

    def stop(self) -> np.ndarray:
        self._stream.stop()
        self._stream.close()
        return np.concatenate(self._frames, axis=0) if self._frames else np.zeros((0, 1), dtype=np.int16)

    def held_seconds(self) -> float:
        return time.monotonic() - self._started_at


def _save_wav(audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, SAMPLE_RATE)
        return f.name


def _is_silent(audio: np.ndarray) -> bool:
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return rms < SILENCE_RMS_THRESHOLD


def handle_turn(
    messages: list,
    text: str,
    on_status: StatusCallback,
    lock: Optional[threading.Lock] = None,
    interrupt_event: Optional[threading.Event] = None,
) -> bool:
    """Runs one turn of the conversation (agent loop + speak) for `text`.
    Takes `lock` when given, so this can be called safely from a typed-input
    handler running alongside the push-to-talk loop on `messages` shared
    between them.

    Returns True if the reply was cut off partway through because
    `interrupt_event` was set (barge-in via the hotkey)."""
    def _run() -> bool:
        on_status("heard_command", text)
        messages.append({"role": "user", "content": text})
        on_status("thinking", "")
        run_turn(messages)
        reply = rocky_transform(messages[-1]["content"])
        append_daily_log(text, reply)
        save_conversation(messages)
        on_status("speaking", reply)
        completed = speak(reply, interrupt_event=interrupt_event)
        return not completed

    if lock is None:
        return _run()
    with lock:
        return _run()


def run_rocky(on_status: StatusCallback, messages: Optional[list] = None, lock: Optional[threading.Lock] = None) -> None:
    """Push-to-talk loop: hold Option+Control to record, release to send.
    Pressing it again while Rocky is speaking interrupts it and immediately
    starts recording your next command — one gesture does both jobs.

    Replaces the earlier always-on wake-word listener: that kept the mic
    continuously "in use" (a real macOS privacy indicator, unavoidable for
    any third-party software wake-word system — see chat history) and voice
    -based barge-in risked Rocky hearing itself say its own name. Push-to
    -talk has neither problem, at the cost of needing a held hotkey instead
    of just saying "Rocky".

    `messages`/`lock` can be supplied to share this conversation with a
    typed-input path (e.g. a GUI text box) running on another thread;
    otherwise the last saved conversation is resumed (or a fresh one
    started, the first time there's nothing to resume)."""
    if messages is None:
        messages = load_conversation()

    state_lock = threading.Lock()
    recorder: Optional[_Recorder] = None
    speaking_interrupt: Optional[threading.Event] = None

    def on_hotkey_press() -> None:
        nonlocal recorder, speaking_interrupt
        with state_lock:
            if speaking_interrupt is not None:
                speaking_interrupt.set()
            if recorder is None:
                recorder = _Recorder()
                recorder.start()
                on_status("recording", "")

    def on_hotkey_release() -> None:
        nonlocal recorder
        with state_lock:
            active_recorder, recorder = recorder, None
        if active_recorder is None:
            return
        # A very short hold is almost always accidental modifier-key overlap
        # from typing/other shortcuts, not an intentional talk gesture —
        # discard it before even touching the audio.
        if active_recorder.held_seconds() < MIN_HOLD_SECONDS:
            active_recorder.stop()
            on_status("idle", "")
            return
        audio = active_recorder.stop()
        # transcribe() alone can take a second or more — must not run on the
        # hotkey callback thread. macOS silently disables a global key-event
        # tap if its callback doesn't return quickly (no error, no crash —
        # just stops delivering events), which is exactly what "worked for a
        # while, then stopped working" looks like.
        threading.Thread(target=_transcribe_and_process, args=(audio,), daemon=True).start()

    def _transcribe_and_process(audio: np.ndarray) -> None:
        # faster-whisper reliably hallucinates plausible-sounding text from
        # near-silent audio — reject it here before it ever reaches Whisper,
        # rather than trying to filter its output after the fact.
        if _is_silent(audio):
            on_status("idle", "")
            return
        on_status("transcribing", "")
        audio_path = _save_wav(audio)
        text = transcribe(audio_path)
        if not text.strip():
            on_status("idle", "")
            return
        _process(text)

    def _process(text: str) -> None:
        nonlocal speaking_interrupt
        interrupt_event = threading.Event()
        with state_lock:
            speaking_interrupt = interrupt_event
        handle_turn(messages, text, on_status, lock, interrupt_event=interrupt_event)
        with state_lock:
            speaking_interrupt = None
        on_status("idle", "")

    listener = PushToTalkListener(on_hotkey_press, on_hotkey_release)
    listener.start()
    on_status("idle", "")
    threading.Event().wait()  # block forever; hotkey callbacks drive everything


def _print_status(state: str, detail: str) -> None:
    labels = {
        "idle": "Hold Option+Control to talk...",
        "recording": "Recording — release to send...",
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
