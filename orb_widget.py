"""Animated glowing orb for Rocky's desktop app — a glossy glass-sphere
look (radial gradient body + a crisp specular highlight + a thin rim),
matching a reference "assistant orb" style rather than a particle-heavy
effect. Kept to a handful of draw calls per frame on purpose: the earlier
particle-based version visibly lagged."""

import math

from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

# All-blue family — different states get a different shade, not a
# different hue, so the palette stays cohesive.
STATE_COLORS = {
    "idle": QColor(59, 130, 246),  # blue
    "recording": QColor(37, 99, 235),  # deeper blue
    "transcribing": QColor(56, 189, 248),  # sky
    "heard_command": QColor(56, 189, 248),
    "thinking": QColor(99, 102, 241),  # blue-violet
    "speaking": QColor(34, 211, 238),  # cyan
}
_STATE_SPEED = {
    "idle": 0.45,
    "recording": 1.8,
    "transcribing": 2.0,
    "heard_command": 1.4,
    "thinking": 1.3,
    "speaking": 1.7,
}
_FRAME_MS = 40  # 25fps — the pulse is subtle, doesn't need more
_COLOR_EASE = 0.10


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    return QColor(
        int(c1.red() + (c2.red() - c1.red()) * t),
        int(c1.green() + (c2.green() - c1.green()) * t),
        int(c1.blue() + (c2.blue() - c1.blue()) * t),
    )


class OrbWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(190, 190)
        self._state = "idle"
        self._t = 0.0
        self._color = QColor(STATE_COLORS["idle"])
        self._target_color = QColor(STATE_COLORS["idle"])
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_FRAME_MS)

    def set_state(self, state: str) -> None:
        self._state = state if state in STATE_COLORS else "idle"
        self._target_color = QColor(STATE_COLORS[self._state])

    def current_color(self) -> QColor:
        return QColor(self._color)

    def _tick(self) -> None:
        dt = _FRAME_MS / 1000.0
        speed = _STATE_SPEED.get(self._state, 1.0)
        self._t += dt * speed
        if self._color != self._target_color:
            self._color = _lerp_color(self._color, self._target_color, _COLOR_EASE)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        base_radius = min(w, h) * 0.26
        color = self._color

        # Gentle breathing pulse — subtle on purpose, this is a light touch not a bounce.
        pulse = 0.06 * math.sin(self._t * math.pi)
        radius = base_radius * (1.0 + pulse)

        painter.setPen(Qt.NoPen)

        # Sphere body: one gradient, light source implied top-left.
        body = QRadialGradient(QPointF(cx - radius * 0.35, cy - radius * 0.4), radius * 1.6)
        body.setColorAt(0.0, color.lighter(180))
        body.setColorAt(0.4, color.lighter(110))
        body.setColorAt(0.75, color)
        body.setColorAt(1.0, color.darker(230))
        painter.setBrush(QBrush(body))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Crisp specular highlight — the single detail that sells "glossy glass".
        hl_cx, hl_cy = cx - radius * 0.32, cy - radius * 0.38
        highlight = QRadialGradient(QPointF(hl_cx, hl_cy), radius * 0.5)
        highlight.setColorAt(0.0, QColor(255, 255, 255, 210))
        highlight.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.setBrush(QBrush(highlight))
        painter.drawEllipse(QPointF(hl_cx, hl_cy), radius * 0.42, radius * 0.34)

        # Thin dark rim for definition against the ambient glow behind it.
        painter.setPen(QPen(color.darker(260), 1.4))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)
