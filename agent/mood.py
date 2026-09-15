"""Parses the mood tag rocky_prompt.txt instructs the model to prefix every
reply with, e.g. "[mood: excited]\nThat's awesome!" — drives the sidebar
orb's visual reaction (desktop_app.py/orb_widget.py) and a small TTS pacing
variation (voice/speak.py). Never shown or spoken; stripped before
rocky_transform/display/speak ever see the rest of the text.

Kept deliberately small and forgiving: a 7B model won't always follow the
tag format exactly, and a missing/malformed tag must never break the
pipeline — it just means "neutral", the same as before this existed."""

import re

MOODS = ("neutral", "excited", "playful", "sympathetic", "unimpressed")
DEFAULT_MOOD = "neutral"

# Subtle on purpose — enough to feel like a real reaction, not enough to
# hurt intelligibility or read as a gimmick.
MOOD_SPEECH_SPEED = {
    "neutral": 1.0,
    "excited": 1.12,
    "playful": 1.05,
    "sympathetic": 0.92,
    "unimpressed": 0.95,
}

_MOOD_TAG_RE = re.compile(r"^\s*\[mood:\s*(\w+)\]\s*", re.IGNORECASE)


def extract_mood(text: str) -> tuple[str, str]:
    """Returns (mood, remaining_text). Falls back to DEFAULT_MOOD with the
    text unchanged if the tag is missing or names something outside the
    known set."""
    if not text:
        return DEFAULT_MOOD, text
    match = _MOOD_TAG_RE.match(text)
    if not match:
        return DEFAULT_MOOD, text
    mood = match.group(1).lower()
    remaining = text[match.end():]
    if mood not in MOODS:
        mood = DEFAULT_MOOD
    return mood, remaining
