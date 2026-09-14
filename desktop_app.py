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

from PySide6.QtCore import QObject, QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QCloseEvent, QColor, QPainter, QRadialGradient
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
from orb_widget import STATE_COLORS, OrbWidget

_BASE_BG = QColor(15, 15, 22)


class GradientBackground(QWidget):
    """Paints one bold ambient glow behind everything, anchored near where
    the orb sits, matching the reference look — no per-frame animation here
    (only repaints on a state/color change), so it costs almost nothing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._color = QColor(STATE_COLORS["idle"])

    def set_color(self, color: QColor) -> None:
        if self._color != color:
            self._color = color
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), _BASE_BG)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h * 0.22
        glow_radius = h * 1.05

        # A 4-stop falloff (hot core -> vivid -> dim -> gone) reads much
        # richer than a plain 2-stop linear alpha fade, which looked washed
        # out and barely-there against the dark base.
        gradient = QRadialGradient(QPointF(cx, cy), glow_radius)
        hot = QColor(self._color)
        hot.setAlpha(235)
        vivid = QColor(self._color)
        vivid.setAlpha(170)
        dim = QColor(self._color)
        dim.setAlpha(60)
        gone = QColor(self._color)
        gone.setAlpha(0)
        gradient.setColorAt(0.0, hot)
        gradient.setColorAt(0.22, vivid)
        gradient.setColorAt(0.55, dim)
        gradient.setColorAt(1.0, gone)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)

STATUS_LABELS = {
    "idle": "Hold ⌥ Option + ⌃ Control to talk",
    "recording": "🎙️ Recording — release to send",
    "transcribing": "👂 Heard you! Transcribing...",
    "heard_command": "🤔 Thinking...",
    "thinking": "🤔 Thinking...",
    "speaking": "💬 Speaking...",
}

DARK_STYLESHEET = """
QMainWindow {
    background-color: #0f0f16;
}
QWidget {
    color: #e4e4f2;
}
QLabel#statusLabel {
    font-size: 14px;
    font-weight: 600;
    color: #9d9dc4;
    padding: 6px;
}
QTextEdit {
    background: transparent;
    border: none;
    padding: 10px;
    font-size: 13px;
    selection-background-color: #5865f2;
}
QLineEdit {
    background-color: rgba(30, 30, 44, 190);
    border: 1px solid rgba(255, 255, 255, 30);
    border-radius: 20px;
    padding: 11px 18px;
    font-size: 13px;
}
QLineEdit:focus {
    border: 1px solid #5865f2;
}
QPushButton {
    background-color: #5865f2;
    border: none;
    border-radius: 20px;
    padding: 11px 22px;
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

        central = GradientBackground()
        self.background = central
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
        self.background.set_color(STATE_COLORS.get(state, STATE_COLORS["idle"]))
        if state == "heard_command":
            self.append_chat("You", detail)
        elif state == "speaking":
            self.append_chat("Rocky", detail)

    def append_chat(self, speaker: str, text: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        if speaker == "You":
            align, bg, fg, label = "right", "#5865f2", "#ffffff", ""
        else:
            align, bg, fg, label = "left", "#22283a", "#e4e4f2", '<b style="color:#38bdf8;">Rocky</b><br>'
        bubble = f'''
        <table width="100%" cellspacing="0" style="margin-bottom:8px;"><tr>
            <td align="{align}">
                <table cellpadding="9" style="background-color:{bg}; border-radius:14px;">
                    <tr><td style="color:{fg}; font-size:13px;">{label}{escaped}</td></tr>
                </table>
            </td>
        </tr></table>
        '''
        self.chat_log.append(bubble)

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
