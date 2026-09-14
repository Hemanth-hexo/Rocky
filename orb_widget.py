"""Animated glowing orb for Rocky's desktop app — a Siri-style visual that
changes color and motion with Rocky's state. Purely 2D (QPainter radial
gradients + a timer-driven animation loop), but layered glow, orbiting
particles, smooth color easing, and organic idle drift give it a lot more
life than a static gradient circle would."""

import math
import random

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QWidget

_STATE_COLORS = {
    "idle": QColor(99, 102, 241),  # indigo
    "recording": QColor(248, 81, 81),  # red
    "transcribing": QColor(250, 204, 21),  # amber
    "heard_command": QColor(250, 204, 21),
    "thinking": QColor(251, 191, 36),  # gold
    "speaking": QColor(45, 212, 191),  # teal
}
_STATE_SPEED = {
    "idle": 0.45,
    "recording": 2.6,
    "transcribing": 2.8,
    "heard_command": 2.0,
    "thinking": 1.8,
    "speaking": 2.4,
}
_SPINNING_STATES = {"thinking", "transcribing", "recording"}
_PARTICLE_COUNT = 16
_FRAME_MS = 33
_COLOR_EASE = 0.10  # fraction of the remaining color gap closed each frame


def _lerp_channel(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    return QColor(
        _lerp_channel(c1.red(), c2.red(), t),
        _lerp_channel(c1.green(), c2.green(), t),
        _lerp_channel(c1.blue(), c2.blue(), t),
    )


class _Particle:
    __slots__ = ("angle", "radius_factor", "speed", "size", "phase")

    def __init__(self):
        self.angle = random.uniform(0, 2 * math.pi)
        self.radius_factor = random.uniform(1.5, 2.3)
        self.speed = random.uniform(0.25, 0.9) * random.choice([-1, 1])
        self.size = random.uniform(1.5, 3.8)
        self.phase = random.uniform(0, 2 * math.pi)


class OrbWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(190, 190)
        self._state = "idle"
        self._t = 0.0
        self._color = QColor(_STATE_COLORS["idle"])
        self._target_color = QColor(_STATE_COLORS["idle"])
        self._particles = [_Particle() for _ in range(_PARTICLE_COUNT)]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_FRAME_MS)

    def set_state(self, state: str) -> None:
        self._state = state if state in _STATE_COLORS else "idle"
        self._target_color = QColor(_STATE_COLORS[self._state])

    def _tick(self) -> None:
        dt = _FRAME_MS / 1000.0
        speed = _STATE_SPEED.get(self._state, 1.0)
        self._t += dt * speed
        self._color = _lerp_color(self._color, self._target_color, _COLOR_EASE)
        for p in self._particles:
            p.angle += p.speed * dt * (1.7 if self._state != "idle" else 0.5)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        base_radius = min(w, h) * 0.22
        color = self._color

        # Organic idle drift — the orb wanders gently around center when
        # idle, and settles closer to center (more "focused") once active.
        drift_scale = 1.0 if self._state == "idle" else 0.25
        drift_x = (math.sin(self._t * 0.7) * 7 + math.sin(self._t * 1.3) * 3) * drift_scale
        drift_y = (math.cos(self._t * 0.5) * 7 + math.sin(self._t * 1.9) * 3) * drift_scale
        cx, cy = w / 2 + drift_x, h / 2 + drift_y

        # Breathing pulse.
        pulse = 0.14 * math.sin(self._t * math.pi)
        radius = base_radius * (1.0 + pulse)

        # Wide, soft ambient bloom behind everything.
        bloom = QRadialGradient(QPointF(cx, cy), radius * 4.2)
        bloom_color = QColor(color)
        bloom_color.setAlpha(70)
        bloom.setColorAt(0.0, bloom_color)
        transparent = QColor(color)
        transparent.setAlpha(0)
        bloom.setColorAt(1.0, transparent)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(bloom))
        painter.drawEllipse(QPointF(cx, cy), radius * 4.2, radius * 4.2)

        # Orbiting particles, twinkling and drawn behind the core so the
        # core reads as solid/in-front.
        for p in self._particles:
            orbit_radius = radius * p.radius_factor
            px = cx + orbit_radius * math.cos(p.angle)
            py = cy + orbit_radius * math.sin(p.angle) * 0.88
            twinkle = 0.5 + 0.5 * math.sin(self._t * 3.2 + p.phase)
            particle_color = QColor(color)
            particle_color.setAlpha(int(130 * twinkle))
            painter.setBrush(QBrush(particle_color))
            psize = p.size * (0.6 + 0.6 * twinkle)
            painter.drawEllipse(QPointF(px, py), psize, psize)

        # Tighter glow halo right around the core.
        for i in range(6, 0, -1):
            glow = QColor(color)
            glow.setAlpha(int(20 * (1 - i / 7)))
            glow_radius = radius * (1 + i * 0.22)
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)

        # Core sphere: gradient offset toward the upper-left simulates a
        # light source and gives the flat circle dimensional shading.
        gradient = QRadialGradient(QPointF(cx - radius * 0.3, cy - radius * 0.35), radius * 1.5)
        gradient.setColorAt(0.0, color.lighter(175))
        gradient.setColorAt(0.55, color)
        gradient.setColorAt(1.0, color.darker(210))
        painter.setBrush(QBrush(gradient))
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

        # Small bright specular highlight, drifting slightly, for extra "wet glass" feel.
        highlight = QColor(255, 255, 255, 90)
        hl_offset = radius * 0.22
        hl_x = cx - radius * 0.32 + math.sin(self._t * 0.9) * 2
        hl_y = cy - radius * 0.38 + math.cos(self._t * 0.7) * 2
        painter.setBrush(QBrush(highlight))
        painter.drawEllipse(QPointF(hl_x, hl_y), hl_offset, hl_offset * 0.8)

        # Rotating highlight arc on active states, suggesting motion/work.
        if self._state in _SPINNING_STATES:
            painter.setPen(QPen(QColor(255, 255, 255, 80), 3))
            painter.setBrush(Qt.NoBrush)
            angle = (self._t * 150) % 360
            arc_rect = QRectF(cx - radius * 1.4, cy - radius * 1.4, radius * 2.8, radius * 2.8)
            painter.drawArc(arc_rect, int(angle * 16), int(65 * 16))
