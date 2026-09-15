"""Deterministic path for reminder/calendar requests — same reasoning as
agent/profile.py's remember_from_text(): letting the chat model notice an
explicit request and decide on its own to call a tool has already proven
unreliable once (a message flagged "for my profile" got no tool call at
all). Trigger-phrase detection bypasses that judgment call entirely.

Unlike remember() though, create_reminder/create_calendar_event are NOT
also left available to the chat model (see agent/tools.py) — a duplicate
saved fact is harmless, but a duplicate real Reminders.app/Calendar.app
entry isn't, so there's exactly one path to creating either.

Date/time handling is split in two on purpose: the model extracts WHAT the
user said ("next monday at 10am"), and dateparser resolves that into an
actual date. Letting the model compute the date itself was tested and
found to be unreliable — it resolved "next monday" three days wrong."""

import datetime
import re

import dateparser
import ollama

from .config import MODEL
from .text_utils import extract_json_object
from .tools import create_calendar_event, create_reminder

_REMINDER_TRIGGERS = (
    "remind me", "set a reminder", "add a reminder", "create a reminder", "reminder to",
)
_CALENDAR_TRIGGERS = (
    "add to my calendar", "add to calendar", "schedule a", "calendar event",
    "book a meeting", "add an event", "create an event",
)

# dateparser resolves a bare weekday name to its next upcoming occurrence
# already (exactly what "next <weekday>" means) but fails to parse the
# "next "/"this " prefix itself — strip it before handing off.
_WEEKDAY_PREFIX_RE = re.compile(
    r"\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE
)


def mentions_reminder(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in _REMINDER_TRIGGERS)


def mentions_calendar_event(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in _CALENDAR_TRIGGERS)


def _now_str() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def _resolve_when(phrase: str) -> str:
    """Turns a natural-language phrase into 'MM/DD/YYYY HH:MM', or '' if
    dateparser can't make sense of it."""
    if not phrase:
        return ""
    normalized = _WEEKDAY_PREFIX_RE.sub(lambda m: m.group(2), phrase)
    dt = dateparser.parse(
        normalized, settings={"PREFER_DATES_FROM": "future", "RELATIVE_BASE": datetime.datetime.now()}
    )
    return dt.strftime("%m/%d/%Y %H:%M") if dt else ""


def extract_reminder(text: str) -> dict | None:
    prompt = (
        f"The current date and time is {_now_str()}. The message below asks to be reminded of "
        'something. Reply with ONLY a JSON object, no other text: {"text": "<the reminder '
        'text>", "when": "<the date/time phrase exactly as the user said it, e.g. \'tomorrow at '
        "5pm' or 'friday', or empty string if no date/time was mentioned>\"}.\n\nMessage:\n" + text
    )
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    obj = extract_json_object(response["message"]["content"])
    if not obj or not obj.get("text"):
        return None
    return {"text": str(obj["text"]), "due_date": _resolve_when(str(obj.get("when") or ""))}


def extract_calendar_event(text: str) -> dict | None:
    prompt = (
        f"The current date and time is {_now_str()}. The message below asks to schedule a "
        'calendar event. Reply with ONLY a JSON object, no other text: {"title": "<event '
        "title>\", \"when\": \"<the date/time phrase exactly as the user said it>\"}.\n\nMessage:\n"
        + text
    )
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    obj = extract_json_object(response["message"]["content"])
    if not obj or not obj.get("title") or not obj.get("when"):
        return None
    due = _resolve_when(str(obj["when"]))
    if not due:
        return None
    return {"title": str(obj["title"]), "start_date": due, "end_date": ""}


def create_reminder_from_text(text: str) -> str | None:
    parsed = extract_reminder(text)
    if not parsed:
        return None
    return create_reminder(parsed["text"], parsed["due_date"])


def create_calendar_event_from_text(text: str) -> str | None:
    parsed = extract_calendar_event(text)
    if not parsed:
        return None
    return create_calendar_event(parsed["title"], parsed["start_date"], parsed["end_date"])
