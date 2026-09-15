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

# Brackets, the colon, AND the word "mood" itself are all optional now —
# tested against real model output across several turns and none of
# "[mood: excited]", "Mood:sympathetic", or "[neutral]" (no "mood" word at
# all, observed live) are reliable — each earlier, stricter version of this
# regex missed one of these and leaked the raw tag into the spoken/
# displayed reply. The downstream MOODS membership check is what keeps
# this safe despite being permissive: a reply that genuinely starts with
# an unrelated word ("Mood tracking...") only gets stripped if that word
# happens to BE one of the five known moods, which is checked below before
# anything is ever cut from the text.
_MOOD_TAG_RE = re.compile(r"^\s*\[?\s*(?:mood\s*:?\s*)?(\w+)\s*\]?\s*", re.IGNORECASE)


def extract_mood(text: str) -> tuple[str, str]:
    """Returns (mood, remaining_text). Falls back to DEFAULT_MOOD with the
    text UNCHANGED if the tag is missing — or if something tag-shaped
    matched but the captured word isn't an actual known mood, e.g. a reply
    that genuinely starts with "Mood tracking..." shouldn't get its first
    two words silently eaten just because "mood" appears up front."""
    if not text:
        return DEFAULT_MOOD, text
    match = _MOOD_TAG_RE.match(text)
    if not match:
        return DEFAULT_MOOD, text
    mood = match.group(1).lower()
    if mood not in MOODS:
        return DEFAULT_MOOD, text
    return mood, text[match.end():]
