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

from agent.claude_memory import claude_history_context, mentions_claude_history
from agent.conversation_store import load_conversation, save_conversation
from agent.loop import run_turn
from agent.mood import MOOD_SPEECH_SPEED, extract_mood
from agent.obsidian import append_daily_log
from agent.profile import mentions_profile_update, remember_from_text
from agent.rocky_transform import rocky_transform
from agent.scheduling import (
    create_calendar_event_from_text,
    create_reminder_from_text,
    mentions_calendar_event,
    mentions_reminder,
)
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

StatusCallback = Callable[..., None]  # (state, detail, mood="neutral") — mood is only ever real for "speaking"

# Deterministic trigger-phrase bypasses for requests where letting the chat
# model decide whether to call a tool has proven unreliable (see
# agent/profile.py's module docstring for the case — profile-saving — that
# started this pattern). Each entry: (trigger check, handler that performs
# the action and returns a result string or None, ok-label, fail-label).
# Independent checks run in order, not elif — a message can reasonably ask
# for more than one of these at once ("add it to my calendar and remind me
# too" used to only create the reminder because of a stray elif).
_SYNC_TRIGGERS = (
    (mentions_reminder, create_reminder_from_text, "reminder created", "couldn't parse a clear reminder from that message"),
    (mentions_calendar_event, create_calendar_event_from_text, "calendar event created", "couldn't parse a clear event from that message"),
)


def _run_sync_triggers(text: str, messages: list) -> None:
    for mentions, handler, ok_label, fail_label in _SYNC_TRIGGERS:
        if mentions(text):
            result = handler(text)
            note = f"[{ok_label}] {result}" if result else f"[{fail_label}]"
            messages.append({"role": "tool", "content": note})


def _safe_remember(text: str) -> None:
    # Runs on its own daemon thread (fire-and-forget) — this only stops it
    # from dying silently on a transient ollama/network error; a failure
    # here was already best-effort (see agent/profile.py) and doesn't need
    # to surface to the user, just not vanish without a trace in the log.
    try:
        remember_from_text(text)
    except Exception as e:
        print(f"[handle_turn] remember_from_text failed: {e!r}")


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
        if mentions_profile_update(text):
            # Runs in the background — extraction is its own model call and
            # shouldn't delay the spoken reply — but it's fired unconditionally
            # rather than left to the chat model to decide whether to call the
            # remember() tool, which has already proven unreliable.
            threading.Thread(target=_safe_remember, args=(text,), daemon=True).start()
        on_status("thinking", "")
        try:
            # These run synchronously and BEFORE run_turn — their outcome is
            # fed in as a synthetic tool message so the model's actual reply
            # can reference what really happened ("set for 5pm tomorrow")
            # instead of guessing, and since create_reminder/
            # create_calendar_event are no longer tools the model can call
            # itself (see agent/tools.py), there's no risk of it also trying
            # and creating a duplicate.
            _run_sync_triggers(text, messages)
            if mentions_claude_history(text):
                # Deterministic — tested directly: the model doesn't
                # reliably call list_notes/read_note on its own for this,
                # even with explicit phrasing naming the tool and folder.
                # Pure local keyword matching against note filenames, no
                # model call, so it can't fail the way relying on the
                # model's judgment did.
                context = claude_history_context(text)
                if context:
                    messages.append({"role": "tool", "content": f"[from imported Claude history]\n{context}"})
            run_turn(messages)
            # .get() instead of [] — messages[-1] here can be a raw ollama
            # Message object (not yet normalized to a plain dict), which
            # raises KeyError on [] access if a pure-tool-call turn ever
            # left content unset rather than "". `or ""` covers .get()
            # returning None for a field that exists-but-is-unset, which a
            # bare default wouldn't catch.
            raw_content = messages[-1].get("content") or ""
            # Mood extraction happens on the RAW model output, before
            # rocky_transform — the tag is "[mood: excited]"-shaped and
            # rocky_transform's word-mangling (article-dropping etc.) would
            # corrupt it if it ran first.
            mood, stripped_content = extract_mood(raw_content)
            reply = rocky_transform(stripped_content)
            append_daily_log(text, reply)
            save_conversation(messages)
            on_status("speaking", reply, mood)
            completed = speak(reply, interrupt_event=interrupt_event, speed=MOOD_SPEECH_SPEED.get(mood, 1.0))
            return not completed
        except Exception as e:
            # Every step above (extraction, run_turn's ollama call, speak's
            # TTS) was previously unguarded — a transient failure in any of
            # them killed this thread silently: the UI froze on "Thinking…"
            # forever, and the user's message above was left dangling in
            # `messages` with no reply, corrupting the next turn's context.
            # This still surfaces the failure (spoken + logged) instead of
            # papering over it, but it can no longer take the whole turn
            # down silently.
            print(f"[handle_turn] turn failed: {e!r}")
            fallback = "Sorry, something went wrong there. Try again?"
            messages.append({"role": "assistant", "content": fallback})
            save_conversation(messages)
            # "sympathetic" — gentler pulse/pacing actually suits an apology.
            on_status("speaking", fallback, "sympathetic")
            try:
                speak(fallback, interrupt_event=interrupt_event, speed=MOOD_SPEECH_SPEED["sympathetic"])
            except Exception as speak_err:
                print(f"[handle_turn] fallback speak() also failed: {speak_err!r}")
            return False

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
        try:
            audio_path = _save_wav(audio)
            text = transcribe(audio_path)
        except Exception as e:
            # Previously unguarded: a transcription failure (a malformed
            # WAV edge case, faster-whisper raising on unusual input) killed
            # this thread before on_status("idle", ...) ran, leaving the UI
            # stuck showing "Transcribing…" until the next successful turn
            # happened to overwrite it — with the recording silently lost.
            print(f"[run_rocky] transcription failed: {e!r}")
            on_status("idle", "")
            return
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

    # REVERTED (2026-09-15): this used to tear down and recreate the
    # PushToTalkListener every HOTKEY_REFRESH_SECONDS to work around the
    # hotkey silently going dead after the app sat idle a while. It was
    # wrong — it crashed the whole app instead. Confirmed via three macOS
    # crash reports landing at exact 10-minute intervals: SIGTRAP /
    # EXC_BREAKPOINT from a `dispatch_assert_queue_fail` inside HIToolbox's
    # TSM input-source lookup (islGetInputSourceListWithAdditions /
    # TSMGetInputSourceProperty), reached via pynput's macOS keyboard
    # backend querying the keyboard layout through ctypes — that HIToolbox
    # call asserts it must run on a specific (main) dispatch queue, and
    # constructing a new Listener from this background thread on a timer
    # violated that repeatedly until it trapped. Recreating a Listener ONCE
    # at startup is fine (that's what's below); doing it periodically from
    # here is not safe. If the original idle-hotkey-death bug recurs, don't
    # reach for this fix again — a manual quit+reopen is the known-working
    # workaround, and any real fix needs the listener recreated on the Qt
    # main thread, not this background one.
    listener = PushToTalkListener(on_hotkey_press, on_hotkey_release)
    listener.start()
    on_status("idle", "")
    threading.Event().wait()  # block forever; hotkey callbacks drive everything


def _print_status(state: str, detail: str, mood: str = "neutral") -> None:
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
