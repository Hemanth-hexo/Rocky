"""The agent loop: send conversation + tool menu to the model, run any tool call it
asks for, feed the result back in, repeat until it gives a plain answer."""

import json
import os

import ollama

from .config import MODEL
from .github_mcp import GITHUB_FUNCTIONS, GITHUB_SCHEMAS
from .obsidian import OBSIDIAN_FUNCTIONS, OBSIDIAN_SCHEMAS
from .profile import PROFILE_FUNCTIONS, PROFILE_SCHEMAS, profile_block
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "rocky_prompt.txt")

ALL_FUNCTIONS = {**TOOL_FUNCTIONS, **OBSIDIAN_FUNCTIONS, **GITHUB_FUNCTIONS, **PROFILE_FUNCTIONS}
ALL_SCHEMAS = TOOL_SCHEMAS + OBSIDIAN_SCHEMAS + GITHUB_SCHEMAS + PROFILE_SCHEMAS


def load_system_prompt() -> str:
    with open(PROMPT_PATH, "r") as f:
        base = f.read()
    # The profile ("central brain") is folded straight into the system
    # prompt rather than kept as its own message — the system message is
    # never trimmed by conversation_store's rolling history cap, so this is
    # what keeps it "always known" regardless of how old the conversation
    # that mentioned it is.
    return base + profile_block()


def _parse_loose_tool_call(content: str) -> dict | None:
    """qwen2.5-coder:7b-q4_K_M sometimes emits a bare {"name":..,"arguments":..}
    JSON object as plain content instead of wrapping it in <tool_call> tags, so
    ollama's parser never populates tool_calls. Catch that case manually.

    It also sometimes wraps the JSON in conversational text ("Sure thing!
    Let's do this.\n\n{...}") rather than emitting pure JSON, so this scans
    for a {...} substring anywhere in the content rather than requiring the
    whole message to be just the JSON object."""
    if not content:
        return None
    text = content.strip()
    for tag in ("<tool_call>", "</tool_call>", "```json", "```"):
        text = text.replace(tag, "")
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and "name" in obj and "arguments" in obj and obj["name"] in ALL_FUNCTIONS:
        return obj
    return None


def run_turn(messages: list[dict]) -> list[dict]:
    """Run the loop for one user turn. Mutates and returns `messages` with the
    assistant's reply (and any tool exchanges) appended."""
    while True:
        response = ollama.chat(model=MODEL, messages=messages, tools=ALL_SCHEMAS)
        msg = response["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            loose = _parse_loose_tool_call(msg.get("content", ""))
            if not loose:
                return messages
            tool_calls = [{"function": {"name": loose["name"], "arguments": loose["arguments"]}}]

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)

            fn = ALL_FUNCTIONS.get(name)
            try:
                result = fn(**args) if fn else f"unknown tool: {name}"
            except Exception as e:
                result = f"error: {e}"

            messages.append({"role": "tool", "content": str(result)})


def new_conversation() -> list[dict]:
    return [{"role": "system", "content": load_system_prompt()}]
