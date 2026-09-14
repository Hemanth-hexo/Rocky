"""Rocky as a desktop app with a real window: chat history, live status,
and a text box to type instead of speaking.

Voice input still runs continuously on a background thread (main.run_rocky);
this window shares its conversation state with typed input via a lock, and
uses Qt's thread-safe signal/slot mechanism to update the UI (touching Qt
widgets directly from a background thread isn't safe, the same way AppKit
menu updates weren't safe in the earlier menu bar version)."""

import html
import sys
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from main import handle_turn, new_conversation, run_rocky

STATUS_LABELS = {
    "listening": "🪨 Listening for the wake word...",
    "heard_wake_word": "👂 Heard you! Recording...",
    "listening_followup": "👂 Anything else? (no wake word needed)",
    "transcribing": "Transcribing...",
    "heard_command": "🤔 Thinking...",
    "thinking": "🤔 Thinking...",
    "speaking": "💬 Speaking...",
}


class Bridge(QObject):
    """Qt signals are thread-safe: emitting one from the background voice
    thread queues its connected slot to run on the GUI thread instead of
    executing it inline on whichever thread emitted it."""

    status_changed = Signal(str, str)


class RockyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rocky")
        self.resize(480, 640)

        self.messages = new_conversation()
        self.lock = threading.Lock()

        central = QWidget()
        layout = QVBoxLayout(central)

        self.status_label = QLabel(STATUS_LABELS["listening"])
        self.status_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self.status_label)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        layout.addWidget(self.chat_log)

        input_row = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Type instead of speaking...")
        self.input_box.returnPressed.connect(self.send_typed_text)
        send_button = QPushButton("Send")
        send_button.clicked.connect(self.send_typed_text)
        input_row.addWidget(self.input_box)
        input_row.addWidget(send_button)
        layout.addLayout(input_row)

        self.setCentralWidget(central)

        self.bridge = Bridge()
        self.bridge.status_changed.connect(self.on_status_changed)

        self.voice_thread = threading.Thread(
            target=run_rocky,
            args=(self._emit_status, self.messages, self.lock),
            daemon=True,
        )
        self.voice_thread.start()

    def _emit_status(self, state: str, detail: str) -> None:
        self.bridge.status_changed.emit(state, detail)

    def on_status_changed(self, state: str, detail: str) -> None:
        self.status_label.setText(STATUS_LABELS.get(state, state))
        if state == "heard_command":
            self.append_chat("You", detail)
        elif state == "speaking":
            self.append_chat("Rocky", detail)

    def append_chat(self, speaker: str, text: str) -> None:
        self.chat_log.append(f"<b>{speaker}:</b> {html.escape(text)}")

    def send_typed_text(self) -> None:
        text = self.input_box.text().strip()
        if not text:
            return
        self.input_box.clear()
        threading.Thread(
            target=handle_turn,
            args=(self.messages, text, self._emit_status, self.lock),
            daemon=True,
        ).start()

    def closeEvent(self, event: QCloseEvent) -> None:
        # Keep listening in the background when the window is closed —
        # only Cmd+Q (or the app actually quitting) stops Rocky.
        event.ignore()
        self.hide()


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    window = RockyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
