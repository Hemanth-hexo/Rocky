"""Small text-parsing helpers shared across agent/* modules that pull
structured data out of a local model's free-text replies."""

import json


def extract_json_object(text: str) -> dict | None:
    """Finds and parses the first {...} JSON object substring in `text`,
    stripping common wrapping noise (code fences, tool-call tags) a small
    local model tends to add around structured output even when told to
    reply with ONLY JSON. Returns None if nothing valid is found."""
    if not text:
        return None
    cleaned = text.strip()
    for tag in ("<tool_call>", "</tool_call>", "```json", "```"):
        cleaned = cleaned.replace(tag, "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
