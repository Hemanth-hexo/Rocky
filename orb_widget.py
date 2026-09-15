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
    def __init__(self, parent=None, size: int = 220):
        super().__init__(parent)
        self.setFixedSize(size, size)
        # Paints its own full background every frame (below) instead of
        # leaving transparent corners around the circle — without this,
        # Qt has to ask the parent to re-render the (expensive, large)
        # ambient gradient underneath on every single animation tick.
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
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

        # WA_OpaquePaintEvent above means Qt trusts us to cover the whole
        # rect ourselves — fill it first so the corners aren't garbage.
        painter.fillRect(self.rect(), QColor(15, 15, 22))

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        base_radius = min(w, h) * 0.34
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

        # Thin dark rim for definition against the ambient glow behind it —
        # scaled down for the small sidebar-brand size, or it reads as a bezel.
        painter.setPen(QPen(color.darker(260), max(1.0, radius * 0.03)))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), radius, radius)


class ParticleOrbWidget(QWidget):
    """"Thinking orb" style prototype — a sparse dot-particle sphere,
    inspired by libraries.dev's thinking-orbs component. Always visible;
    sits still while idle and spins while anything is actively happening
    (listening/transcribing/thinking/speaking) — set_state() decides which
    from the state name alone ("idle" pauses, anything else spins).

    Kept cheap on purpose (unlike the earlier particle-based orb that got
    dropped for lag): a fixed set of points computed ONCE at construction
    (Fibonacci sphere — even coverage, no per-frame random sampling), then
    just rotated and redrawn as flat dots every frame it's actually
    spinning. No per-particle gradients, no dynamic point count.

    Transparent by design — no background fill, just the dots — which only
    works cheaply because it sits on the new flat, statically-painted page
    background (desktop_app.py's old GradientBackground, which WAS
    expensive to redraw and needed the opaque-fill trick, is gone)."""

    _PARTICLE_COUNT = 180
    _ROTATE_SPEED = 0.55  # radians/sec
    _DOT_RADIUS = 1.15  # small and uniform — pixel-like dots, not tiny balls

    def __init__(self, parent=None, size: int = 120, particle_count: int | None = None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._state = "idle"
        self._color = QColor(STATE_COLORS["idle"])
        self._spinning = False
        self._angle = 0.0
        # Dot count doesn't scale down with area at small sizes — a truly
        # proportional count would leave almost nothing visible at icon
        # size, so callers pick a count that still reads as a dot cluster.
        self._points = self._make_sphere_points(particle_count or self._PARTICLE_COUNT)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_FRAME_MS)

    @staticmethod
    def _make_sphere_points(n: int) -> list[tuple[float, float, float]]:
        # Fibonacci sphere: evenly spread points on a unit sphere with a
        # single closed-form pass — no rejection sampling, no per-point trig
        # beyond one sin/cos pair.
        points = []
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        for i in range(n):
            y = 1 - (i / (n - 1)) * 2
            radius_at_y = math.sqrt(max(0.0, 1 - y * y))
            theta = golden_angle * i
            points.append((math.cos(theta) * radius_at_y, y, math.sin(theta) * radius_at_y))
        return points

    def set_state(self, state: str) -> None:
        self._state = state if state in STATE_COLORS else "idle"
        self._color = QColor(STATE_COLORS[self._state])
        self._spinning = self._state != "idle"

    def _tick(self) -> None:
        # Paused (no repaint at all, not just a frozen angle) while idle —
        # the orb is visible at all times now, so this is what keeps it
        # from burning a frame budget for the majority of the app's runtime.
        if not self._spinning:
            return
        self._angle += self._ROTATE_SPEED * (_FRAME_MS / 1000.0)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        # No background fill — this widget only ever draws the dots
        # themselves, so it blends straight into whatever page it sits on.

        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.42
        cos_a, sin_a = math.cos(self._angle), math.sin(self._angle)
        color = self._color

        # Rotate around Y, then sort back-to-front so near dots draw over
        # far ones. Only opacity carries the depth cue now — dot size stays
        # fixed, so this reads as a flat scatter of pixel-dots rather than
        # a cluster of little spheres.
        rotated = [(x * cos_a - z * sin_a, y, x * sin_a + z * cos_a) for x, y, z in self._points]
        rotated.sort(key=lambda p: p[2])

        for x, y, z in rotated:
            depth = (z + 1) / 2  # 0 (far) .. 1 (near)
            c = QColor(color)
            c.setAlpha(int(50 + depth * 150))
            painter.setBrush(QBrush(c))
            painter.drawEllipse(QPointF(cx + x * r, cy + y * r), self._DOT_RADIUS, self._DOT_RADIUS)
