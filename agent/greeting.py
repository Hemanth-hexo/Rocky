"""Proactive time-of-day greeting — spoken once per calendar day, the first
time the window is shown that day, without the user needing to say anything
first. A one-off model call outside the main conversation loop, so it works
without a preceding user turn."""

import datetime
import os

import ollama

from .loop import MODEL, load_system_prompt

_STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "state")
_STATE_FILE = os.path.join(_STATE_DIR, "last_greeting.txt")


def _time_of_day() -> str:
    hour = datetime.datetime.now().hour
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def greeting_due() -> bool:
    today = datetime.date.today().isoformat()
    if not os.path.exists(_STATE_FILE):
        return True
    with open(_STATE_FILE, "r") as f:
        return f.read().strip() != today


def mark_greeted() -> None:
    os.makedirs(_STATE_DIR, exist_ok=True)
    with open(_STATE_FILE, "w") as f:
        f.write(datetime.date.today().isoformat())


def generate_greeting() -> str:
    prompt = (
        f"It's {_time_of_day()} and this is the first time today you're greeting the user — "
        "they haven't said anything yet, you're speaking first. Greet them naturally, one or two "
        "short sentences. Vary the wording every time so it never feels canned or repeated. Don't "
        "default to asking if they need a problem solved or need help with something — just a "
        "normal, warm, in-character hello."
    )
    messages = [
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": prompt},
    ]
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"]
