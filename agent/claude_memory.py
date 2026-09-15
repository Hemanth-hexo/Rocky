"""Deterministic surfacing of imported Claude.ai history (agent/obsidian.py
already exposes it via list_notes/read_note, and rocky_prompt.txt tells
the model to check it — but tested directly: the model doesn't reliably
call those tools on its own, even with explicit phrasing like "check your
claude memories notes about X"). Same reasoning as every other
deterministic-bypass this session: don't trust tool-calling judgment for
something that should reliably happen.

Pure local keyword matching against note filenames — no model call, so
this can't fail the way an LLM-driven search could."""

import os
import re

from .obsidian import list_notes, read_note

_CLAUDE_TRIGGER = "claude"
_STOPWORDS = {
    "claude", "what", "did", "know", "about", "my", "the", "a", "an", "is", "was",
    "check", "your", "you", "memories", "memory", "notes", "note", "project", "tell",
    "me", "do", "does", "have", "has", "from", "with", "and", "of", "to", "in", "on",
}


def mentions_claude_history(text: str) -> bool:
    return _CLAUDE_TRIGGER in text.lower()


def _keywords(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


def find_relevant_claude_notes(text: str, max_notes: int = 2) -> list[str]:
    """Returns up to `max_notes` note paths (relative to the vault root)
    ranked by keyword overlap between `text` and each note's filename."""
    query_words = _keywords(text)
    if not query_words:
        return []
    candidates: list[tuple[int, str]] = []
    for folder in ("Claude Conversations", "Claude Memories"):
        try:
            listing = list_notes(folder)
        except Exception:
            continue
        for note_path in listing.splitlines():
            note_path = note_path.strip()
            if not note_path or note_path == "(no notes found)":
                continue
            name_words = _keywords(os.path.basename(note_path))
            overlap = len(query_words & name_words)
            if overlap:
                candidates.append((overlap, note_path))
    candidates.sort(key=lambda x: -x[0])
    return [path for _, path in candidates[:max_notes]]


def claude_history_context(text: str) -> str:
    """Reads the best-matching note(s) and formats them as context to feed
    into the conversation. Empty string if nothing matched."""
    paths = find_relevant_claude_notes(text)
    if not paths:
        return ""
    blocks = []
    for path in paths:
        try:
            content = read_note(path)
        except Exception:
            continue
        if len(content) > 4000:
            content = content[:4000] + "\n\n[truncated]"
        blocks.append(f"--- {path} ---\n{content}")
    return "\n\n".join(blocks)
