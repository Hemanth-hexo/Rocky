"""Global push-to-talk hotkey: hold Option+Control to talk to Rocky, instead
of an always-on wake-word listener. Replaces continuous mic listening (and
the self-feedback risk of listening through the same mic Rocky speaks
through) with an explicit hold-to-record / release-to-send / press-again-
to-interrupt gesture.

Needs macOS's "Input Monitoring" permission granted to this process the
first time it runs — System Settings > Privacy & Security > Input
Monitoring. Without it, pynput's listener silently receives no events."""

import threading

from pynput import keyboard

_MODIFIER_ALIASES = {
    keyboard.Key.alt: "alt",
    keyboard.Key.alt_l: "alt",
    keyboard.Key.alt_r: "alt",
    keyboard.Key.alt_gr: "alt",
    keyboard.Key.ctrl: "ctrl",
    keyboard.Key.ctrl_l: "ctrl",
    keyboard.Key.ctrl_r: "ctrl",
}
_HOTKEY = {"alt", "ctrl"}


class PushToTalkListener:
    """on_press() fires once when both Option and Control are held together;
    on_release() fires once when that combo breaks (either key released).
    Both run on the listener's own background thread — callers must not do
    slow work directly in them, hand off to another thread instead."""

    def __init__(self, on_press, on_release):
        self._on_press = on_press
        self._on_release = on_release
        self._held: set[str] = set()
        self._active = False
        self._lock = threading.Lock()
        self._listener = keyboard.Listener(on_press=self._handle_press, on_release=self._handle_release)

    def _handle_press(self, key) -> None:
        name = _MODIFIER_ALIASES.get(key)
        if name is None:
            return
        with self._lock:
            self._held.add(name)
            if _HOTKEY.issubset(self._held) and not self._active:
                self._active = True
                self._on_press()

    def _handle_release(self, key) -> None:
        name = _MODIFIER_ALIASES.get(key)
        if name is None:
            return
        with self._lock:
            self._held.discard(name)
            if self._active and not _HOTKEY.issubset(self._held):
                self._active = False
                self._on_release()

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
