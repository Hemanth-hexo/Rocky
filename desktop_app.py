"""Rocky as a desktop app with a real window: an animated status orb, chat
history, and a text box to type instead of speaking.

The push-to-talk loop (main.run_rocky) runs on a background thread; this
window shares its conversation state with typed input via a lock, and uses
Qt's thread-safe signal/slot mechanism to update the UI (touching Qt widgets
directly from a background thread isn't safe, the same way AppKit menu
updates weren't safe in the earlier menu bar version)."""

import html
import sys
import threading

from PySide6.QtCore import QObject, Qt, Signal
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
from orb_widget import OrbWidget

STATUS_LABELS = {
    "idle": "Hold ⌥ Option + ⌃ Control to talk",
    "recording": "🎙️ Recording — release to send",
    "transcribing": "👂 Heard you! Transcribing...",
    "heard_command": "🤔 Thinking...",
    "thinking": "🤔 Thinking...",
    "speaking": "💬 Speaking...",
}

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #16161f;
    color: #e4e4f2;
}
QLabel#statusLabel {
    font-size: 14px;
    font-weight: 600;
    color: #9d9dc4;
    padding: 6px;
}
QTextEdit {
    background-color: #1e1e2c;
    border: 1px solid #2f2f45;
    border-radius: 12px;
    padding: 10px;
    font-size: 13px;
    selection-background-color: #5865f2;
}
QLineEdit {
    background-color: #1e1e2c;
    border: 1px solid #2f2f45;
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 13px;
}
QLineEdit:focus {
    border: 1px solid #5865f2;
}
QPushButton {
    background-color: #5865f2;
    border: none;
    border-radius: 10px;
    padding: 9px 20px;
    color: white;
    font-weight: 600;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #6b76ff;
}
QPushButton:pressed {
    background-color: #4952d1;
}
"""


class Bridge(QObject):
    """Qt signals are thread-safe: emitting one from the background voice
    thread queues its connected slot to run on the GUI thread instead of
    executing it inline on whichever thread emitted it."""

    status_changed = Signal(str, str)


class RockyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rocky")
        self.resize(480, 700)
        self.setStyleSheet(DARK_STYLESHEET)

        self.messages = new_conversation()
        self.lock = threading.Lock()

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self.orb = OrbWidget()
        orb_row = QHBoxLayout()
        orb_row.addStretch()
        orb_row.addWidget(self.orb)
        orb_row.addStretch()
        layout.addLayout(orb_row)

        self.status_label = QLabel(STATUS_LABELS["idle"])
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
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
        self.orb.set_state(state)
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
