"""Rocky as a desktop app: a Raycast-style sidebar (Chat / Projects / Tools /
Memory / Settings), a flat ChatGPT-style conversation view, and a compact
composer to type instead of speaking.

The push-to-talk loop (main.run_rocky) runs on a background thread; this
window shares its conversation state with typed input via a lock, and uses
Qt's thread-safe signal/slot mechanism to update the UI (touching Qt widgets
directly from a background thread isn't safe, the same way AppKit menu
updates weren't safe in the earlier menu bar version)."""

import html
import os
import sys
import threading

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from agent.greeting import generate_greeting, greeting_due, mark_greeted
from agent.loop import ALL_SCHEMAS, MODEL
from agent.obsidian import LOG_DIR, VAULT_DIR, append_daily_log, list_notes, read_note
from agent.rocky_transform import rocky_transform
from main import handle_turn, new_conversation, run_rocky
from orb_widget import STATE_COLORS, OrbWidget, ParticleOrbWidget
from voice.push_to_talk import HOTKEY_LABEL
from voice.speak import DEFAULT_VOICE, speak

_ROCKY_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "rocky_figure.png")

# Raycast-inspired dark palette: flat surfaces, thin low-contrast borders,
# a single restrained accent — no big gradients or glass panels.
BG = "#18181b"
BG_SIDEBAR = "#141416"
BG_ELEVATED = "#1e1e22"
BG_COMPOSER = "#202024"
BORDER = "rgba(255, 255, 255, 0.08)"
BORDER_STRONG = "rgba(255, 255, 255, 0.14)"
TEXT = "#e6e6eb"
TEXT_MUTED = "#8b8b95"
TEXT_FAINT = "#5c5c66"
ACCENT = "#5b6eff"
ACCENT_SOFT = "rgba(91, 110, 255, 0.14)"

NAV_ITEMS = ["Chat", "Projects", "Tools", "Memory", "Settings"]

STATUS_LABELS = {
    "idle": f"Hold {HOTKEY_LABEL} to talk",
    "recording": "Listening…",
    "transcribing": "Transcribing…",
    "heard_command": "Thinking…",
    "thinking": "Thinking…",
    "speaking": "Speaking…",
}

DARK_STYLESHEET = f"""
QMainWindow {{
    background-color: {BG};
}}
QWidget {{
    color: {TEXT};
    font-size: 13px;
}}
#sidebar {{
    background-color: {BG_SIDEBAR};
    border-right: 1px solid {BORDER};
}}
#sidebarHeader {{
    border-bottom: 1px solid {BORDER};
}}
#brandLabel {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT};
}}
QPushButton#navButton {{
    background-color: transparent;
    border: none;
    border-left: 2px solid transparent;
    border-radius: 0px;
    text-align: left;
    padding: 8px 16px;
    color: {TEXT_MUTED};
    font-size: 13px;
    font-weight: 500;
}}
QPushButton#navButton:hover {{
    background-color: rgba(255, 255, 255, 0.04);
    color: {TEXT};
}}
QPushButton#navButton:checked {{
    background-color: {ACCENT_SOFT};
    border-left: 2px solid {ACCENT};
    color: {TEXT};
}}
#statusPill {{
    color: {TEXT_MUTED};
    font-size: 12px;
    padding: 10px 16px;
    border-top: 1px solid {BORDER};
}}
QTextEdit {{
    background-color: {BG};
    border: none;
    padding: 4px 16px;
    font-size: 13px;
    selection-background-color: {ACCENT};
}}
QTextEdit#infoPane {{
    background-color: {BG_ELEVATED};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 14px;
}}
QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
    font-size: 13px;
}}
QListWidget::item {{
    padding: 7px 10px;
    border-radius: 6px;
    color: {TEXT_MUTED};
}}
QListWidget::item:selected {{
    background-color: {ACCENT_SOFT};
    color: {TEXT};
}}
QListWidget::item:hover {{
    background-color: rgba(255, 255, 255, 0.04);
}}
#composerBar {{
    background-color: {BG};
    border-top: 1px solid {BORDER};
}}
QLineEdit#composerInput {{
    background-color: {BG_COMPOSER};
    border: 1px solid {BORDER_STRONG};
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 13px;
    color: {TEXT};
}}
QLineEdit#composerInput:focus {{
    border: 1px solid {ACCENT};
}}
QPushButton#sendButton {{
    background-color: {ACCENT};
    border: none;
    border-radius: 8px;
    padding: 9px 16px;
    color: white;
    font-weight: 600;
    font-size: 13px;
}}
QPushButton#sendButton:hover {{
    background-color: #6b7cff;
}}
QPushButton#sendButton:pressed {{
    background-color: #4a5adf;
}}
#pageTitle {{
    font-size: 13px;
    font-weight: 600;
    color: {TEXT};
    padding: 14px 16px 6px 16px;
}}
#emptyState {{
    color: {TEXT_FAINT};
    font-size: 13px;
}}
"""


class Bridge(QObject):
    """Qt signals are thread-safe: emitting one from a background thread
    queues its connected slot to run on the GUI thread instead of executing
    it inline on whichever thread emitted it."""

    status_changed = Signal(str, str)


def _nav_button(label: str) -> QPushButton:
    btn = QPushButton(label)
    btn.setObjectName("navButton")
    btn.setCheckable(True)
    btn.setCursor(Qt.PointingHandCursor)
    return btn


def _separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(f"color: {BORDER}; background-color: {BORDER}; max-height: 1px; border: none;")
    return line


class ChatPage(QWidget):
    """The default view: a flat, ChatGPT-style message list plus a compact
    composer. The dot-particle orb sits inline next to the status text,
    small, and only while actively recording/transcribing."""

    def __init__(self, on_send, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.chat_log = QTextEdit()
        self.chat_log.setReadOnly(True)
        layout.addWidget(self.chat_log, 1)

        composer_bar = QWidget()
        composer_bar.setObjectName("composerBar")
        composer_layout = QVBoxLayout(composer_bar)
        composer_layout.setContentsMargins(16, 10, 16, 14)
        composer_layout.setSpacing(6)

        # Sits right next to the status text, only while actively
        # recording/transcribing — small and inline instead of a large
        # overlay above the chat log.
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.voice_orb = ParticleOrbWidget(size=32, particle_count=70)
        status_row.addWidget(self.voice_orb)
        self.status_pill = QLabel(STATUS_LABELS["idle"])
        self.status_pill.setObjectName("statusPill")
        self.status_pill.setStyleSheet("border-top: none; padding: 0px;")
        status_row.addWidget(self.status_pill)
        status_row.addStretch()
        composer_layout.addLayout(status_row)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.input_box = QLineEdit()
        self.input_box.setObjectName("composerInput")
        self.input_box.setPlaceholderText("Message Rocky, or hold the hotkey to talk…")
        self.input_box.returnPressed.connect(on_send)
        send_button = QPushButton("Send")
        send_button.setObjectName("sendButton")
        send_button.setCursor(Qt.PointingHandCursor)
        send_button.clicked.connect(on_send)
        input_row.addWidget(self.input_box, 1)
        input_row.addWidget(send_button)
        composer_layout.addLayout(input_row)

        layout.addWidget(composer_bar)

    def set_state(self, state: str, color) -> None:
        # Always visible now — it sits still while idle and spins for
        # everything else (ParticleOrbWidget.set_state decides that itself
        # from the state name).
        self.voice_orb.set_state(state)
        dot = f'<span style="color:{color.name()};">●</span>'
        self.status_pill.setText(f"{dot} {STATUS_LABELS.get(state, state)}")

    def append_chat(self, speaker: str, text: str) -> None:
        escaped = html.escape(text).replace("\n", "<br>")
        if speaker == "You":
            label_color, align = ACCENT, "right"
        else:
            label_color, align = "#38bdf8", "left"
        block = f'''
        <div style="margin: 10px 0; text-align: {align};">
            <div style="font-size: 11px; font-weight: 600; color: {label_color}; letter-spacing: 0.4px; margin-bottom: 2px;">
                {speaker.upper()}
            </div>
            <div style="font-size: 13px; color: {TEXT}; line-height: 1.5;">{escaped}</div>
        </div>
        '''
        self.chat_log.append(block)


class ToolsPage(QWidget):
    """Read-only list of every tool Rocky can currently call — pulled
    straight from the live tool schemas, not a hand-maintained list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = QLabel("Tools")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        body = QTextEdit()
        body.setReadOnly(True)
        rows = []
        for schema in ALL_SCHEMAS:
            fn = schema["function"]
            rows.append(
                f'<div style="margin-bottom:12px;">'
                f'<div style="font-size:13px; font-weight:600; color:{TEXT};">{html.escape(fn["name"])}</div>'
                f'<div style="font-size:12px; color:{TEXT_MUTED}; margin-top:2px;">{html.escape(fn.get("description", ""))}</div>'
                f'</div>'
            )
        body.setHtml("".join(rows))
        layout.addWidget(body, 1)


class MemoryPage(QWidget):
    """Browses the Obsidian daily-log notes Rocky writes after every turn —
    list on the left, note preview on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = QLabel("Memory")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(16, 4, 16, 16)
        body_layout.setSpacing(12)

        self.note_list = QListWidget()
        self.note_list.setFixedWidth(180)
        self.note_list.itemClicked.connect(self._on_select)
        body_layout.addWidget(self.note_list)

        self.preview = QTextEdit()
        self.preview.setObjectName("infoPane")
        self.preview.setReadOnly(True)
        body_layout.addWidget(self.preview, 1)

        layout.addWidget(body, 1)
        self._loaded = False

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if not self._loaded:
            self._reload()
            self._loaded = True

    def _reload(self) -> None:
        try:
            listing = list_notes(LOG_DIR)
        except Exception as e:
            self.preview.setPlainText(f"Couldn't reach the Obsidian vault:\n{e}")
            return
        notes = [n for n in listing.splitlines() if n.strip()]
        if not notes or notes == ["(no notes found)"]:
            self.preview.setPlainText("No conversations logged yet.")
            return
        for note in sorted(notes, reverse=True):
            name = os.path.basename(note).removesuffix(".md")
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, note)
            self.note_list.addItem(item)
        self.note_list.setCurrentRow(0)
        self._on_select(self.note_list.item(0))

    def _on_select(self, item: QListWidgetItem) -> None:
        note_path = item.data(Qt.UserRole)
        try:
            content = read_note(note_path)
        except Exception as e:
            content = f"Couldn't read this note:\n{e}"
        self.preview.setPlainText(content)


class SettingsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        info = QTextEdit()
        info.setObjectName("infoPane")
        info.setReadOnly(True)
        rows = [
            ("Model", MODEL),
            ("Voice", DEFAULT_VOICE),
            ("Push-to-talk", HOTKEY_LABEL),
            ("Obsidian vault", VAULT_DIR),
        ]
        html_rows = "".join(
            f'<div style="margin-bottom:10px;">'
            f'<div style="font-size:11px; color:{TEXT_MUTED}; letter-spacing:0.4px;">{k.upper()}</div>'
            f'<div style="font-size:13px; color:{TEXT}; margin-top:1px;">{html.escape(v)}</div>'
            f'</div>'
            for k, v in rows
        )
        info.setHtml(html_rows)
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(16, 4, 16, 16)
        wrapper_layout.addWidget(info)
        layout.addWidget(wrapper, 1)


class ProjectsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        label = QLabel("Projects — coming soon")
        label.setObjectName("emptyState")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)


class RockyWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Rocky")
        self.resize(760, 620)
        self.setStyleSheet(DARK_STYLESHEET)

        if os.path.exists(_ROCKY_IMAGE_PATH):
            self.setWindowIcon(QIcon(_ROCKY_IMAGE_PATH))

        self.messages = new_conversation()
        self.lock = threading.Lock()

        central = QWidget()
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_sidebar())

        self.pages = QStackedWidget()
        self.chat_page = ChatPage(on_send=self.send_typed_text)
        self.tools_page = ToolsPage()
        self.memory_page = MemoryPage()
        self.settings_page = SettingsPage()
        self.projects_page = ProjectsPage()
        for page in (self.chat_page, self.projects_page, self.tools_page, self.memory_page, self.settings_page):
            self.pages.addWidget(page)
        root_layout.addWidget(self.pages, 1)

        self.setCentralWidget(central)

        self.bridge = Bridge()
        self.bridge.status_changed.connect(self.on_status_changed)

        self._setup_shortcuts()

        self.voice_thread = threading.Thread(
            target=run_rocky,
            args=(self._emit_status, self.messages, self.lock),
            daemon=True,
        )
        self.voice_thread.start()

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(190)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setObjectName("sidebarHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 14, 14, 14)
        header_layout.setSpacing(8)
        brand_orb = OrbWidget(size=22)
        header_layout.addWidget(brand_orb)
        brand_label = QLabel("Rocky")
        brand_label.setObjectName("brandLabel")
        header_layout.addWidget(brand_label)
        header_layout.addStretch()
        layout.addWidget(header)

        nav_group = QButtonGroup(self)
        nav_group.setExclusive(True)
        for i, name in enumerate(NAV_ITEMS):
            btn = _nav_button(name)
            nav_group.addButton(btn, i)
            layout.addWidget(btn)
        nav_group.button(0).setChecked(True)
        nav_group.idClicked.connect(self._on_nav_clicked)
        self._nav_group = nav_group

        layout.addStretch()
        return sidebar

    def _on_nav_clicked(self, index: int) -> None:
        self.pages.setCurrentIndex(index)

    def _setup_shortcuts(self) -> None:
        for i in range(len(NAV_ITEMS)):
            shortcut = QShortcut(QKeySequence(f"Meta+{i + 1}"), self)
            shortcut.activated.connect(lambda idx=i: self._nav_group.button(idx).click())

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._maybe_greet()

    def _maybe_greet(self) -> None:
        if not greeting_due():
            return
        mark_greeted()
        threading.Thread(target=self._run_greeting, daemon=True).start()

    def _run_greeting(self) -> None:
        with self.lock:
            try:
                reply = rocky_transform(generate_greeting())
            except Exception:
                return
            self.messages.append({"role": "assistant", "content": reply})
            append_daily_log("(no message — Rocky greeted first)", reply)
        self._emit_status("speaking", reply)
        speak(reply)
        self._emit_status("idle", "")

    def _emit_status(self, state: str, detail: str) -> None:
        self.bridge.status_changed.emit(state, detail)

    def on_status_changed(self, state: str, detail: str) -> None:
        color = STATE_COLORS.get(state, STATE_COLORS["idle"])
        self.chat_page.set_state(state, color)
        if state == "heard_command":
            self.chat_page.append_chat("You", detail)
        elif state == "speaking":
            self.chat_page.append_chat("Rocky", detail)

    def send_typed_text(self) -> None:
        text = self.chat_page.input_box.text().strip()
        if not text:
            return
        self.chat_page.input_box.clear()
        threading.Thread(
            target=handle_turn,
            args=(self.messages, text, self._emit_status, self.lock),
            daemon=True,
        ).start()

    def closeEvent(self, event) -> None:
        # Keep listening in the background when the window is closed —
        # only Cmd+Q (or the app actually quitting) stops Rocky.
        event.ignore()
        self.hide()


def main() -> None:
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    if os.path.exists(_ROCKY_IMAGE_PATH):
        app.setWindowIcon(QIcon(_ROCKY_IMAGE_PATH))
    window = RockyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
