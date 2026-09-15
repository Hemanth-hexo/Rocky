"""Persists the live conversation across app restarts, so Rocky doesn't
start from a blank slate every time it's reopened. A companion to the
Obsidian daily logs (agent/obsidian.py) — those are the durable long-term
archive Rocky reads on request, but aren't loaded back in automatically;
this is what keeps the actual session continuous."""

import json
import os

from .loop import new_conversation

_STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "state")
_CONVERSATION_FILE = os.path.join(_STATE_DIR, "conversation.json")

# Keeps the live session bounded — a 7B q4_K_M model's context window isn't
# infinite, and the Obsidian daily logs already cover long-term memory, so
# this only needs to carry recent back-and-forth forward.
_MAX_SAVED_MESSAGES = 40


def _to_plain(msg) -> dict:
    # ollama's chat() returns assistant turns as pydantic Message objects,
    # not plain dicts (unlike the user/tool messages this app constructs
    # itself) — those need converting before they're JSON-serializable.
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return msg


def load_conversation() -> list[dict]:
    """Always starts from the CURRENT system prompt (so rocky_prompt.txt
    edits take effect immediately, never frozen into an old save) plus
    whatever recent history was saved."""
    fresh = new_conversation()
    if not os.path.exists(_CONVERSATION_FILE):
        return fresh
    try:
        with open(_CONVERSATION_FILE, "r") as f:
            saved = json.load(f)
    except (json.JSONDecodeError, OSError):
        return fresh
    if not isinstance(saved, list):
        return fresh
    return fresh + saved[-_MAX_SAVED_MESSAGES:]


def save_conversation(messages: list[dict]) -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    body = [plain for m in messages if (plain := _to_plain(m)).get("role") != "system"]
    with open(_CONVERSATION_FILE, "w") as f:
        json.dump(body[-_MAX_SAVED_MESSAGES:], f)
