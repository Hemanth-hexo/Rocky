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
        # strict=False — this exact model has been observed emitting a
        # multi-line "content" string value with literal, unescaped
        # newlines instead of "\n" (invalid per strict JSON, which requires
        # control characters inside strings to be escaped). That's not
        # academic: it silently broke write_note calls with multi-line
        # Markdown content — the JSON failed to parse, so neither this nor
        # the structured tool_calls path recognized it as a tool call, and
        # the broken JSON blob got shown/spoken to the user as if it were
        # the final reply. strict=False accepts raw control characters in
        # strings (treating them as the character they obviously meant)
        # without loosening anything else about JSON validity.
        obj = json.loads(cleaned[start : end + 1], strict=False)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
