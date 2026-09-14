"""Animated glowing orb for Rocky's desktop app — a Siri-style visual that
changes color and motion with Rocky's state. Purely 2D (QPainter radial
gradients + a timer-driven animation loop), but the light/dark gradient
shading and layered glow halos read as dimensional without needing an
actual 3D rendering pipeline (Qt3D/OpenGL)."""

import math

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

# (core color, background glow tint) per state
_STATE_COLORS = {
    "idle": QColor(88, 101, 242),  # soft indigo
    "recording": QColor(255, 90, 90),  # red
    "transcribing": QColor(255, 200, 80),  # amber
    "heard_command": QColor(255, 200, 80),
    "thinking": QColor(255, 190, 60),  # gold
    "speaking": QColor(80, 220, 180),  # teal
}
# animation speed multiplier per state — idle breathes slowly, active states pulse faster
_STATE_SPEED = {
    "idle": 0.5,
    "recording": 2.4,
    "transcribing": 2.6,
    "heard_command": 1.8,
    "thinking": 1.6,
    "speaking": 2.2,
}
_SPINNING_STATES = {"thinking", "transcribing", "recording"}
_FRAME_MS = 33  # ~30fps


class OrbWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self._state = "idle"
        self._t = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_FRAME_MS)

    def set_state(self, state: str) -> None:
        self._state = state if state in _STATE_COLORS else "idle"

    def _tick(self) -> None:
        speed = _STATE_SPEED.get(self._state, 1.0)
        self._t += (_FRAME_MS / 1000.0) * speed
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        base_radius = min(w, h) * 0.26

        color = _STATE_COLORS.get(self._state, _STATE_COLORS["idle"])

        # Slow breathing pulse — the orb's size gently rises and falls.
        pulse = 0.12 * math.sin(self._t * math.pi)
        radius = base_radius * (1.0 + pulse)

        # Layered glow halo: several translucent circles of increasing size
        # and decreasing opacity, giving a soft light-bloom effect around the core.
        for i in range(6, 0, -1):
            glow = QColor(color)
            glow.setAlpha(int(16 * (1 - i / 7)))
            glow_radius = radius * (1 + i * 0.22)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)

        # Core sphere: a radial gradient offset toward the upper-left gives
        # the illusion of a light source and dimensional shading.
        gradient = QRadialGradient(QPointF(cx - radius * 0.3, cy - radius * 0.35), radius * 1.5)
        gradient.setColorAt(0.0, color.lighter(170))
        gradient.setColorAt(0.55, color)
        gradient.setColorAt(1.0, color.darker(200))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # A rotating highlight arc on active states, suggesting motion/work.
        if self._state in _SPINNING_STATES:
            painter.setPen(QPen(QColor(255, 255, 255, 70), 3))
            painter.setBrush(Qt.NoBrush)
            angle = (self._t * 140) % 360
            arc_rect = QRectF(cx - radius * 1.35, cy - radius * 1.35, radius * 2.7, radius * 2.7)
            painter.drawArc(arc_rect, int(angle * 16), int(70 * 16))
