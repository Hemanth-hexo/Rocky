"""The agent loop: send conversation + tool menu to the model, run any tool call it
asks for, feed the result back in, repeat until it gives a plain answer."""

import json
import os

import ollama

from .config import MODEL
from .github_mcp import GITHUB_FUNCTIONS, GITHUB_SCHEMAS
from .obsidian import OBSIDIAN_FUNCTIONS, OBSIDIAN_SCHEMAS
from .profile import PROFILE_FUNCTIONS, PROFILE_SCHEMAS, profile_block
from .project_files import PROJECT_FILE_FUNCTIONS, PROJECT_FILE_SCHEMAS
from .projects import PROJECT_FUNCTIONS, PROJECT_SCHEMAS
from .text_utils import extract_json_object
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "rocky_prompt.txt")

ALL_FUNCTIONS = {
    **TOOL_FUNCTIONS,
    **OBSIDIAN_FUNCTIONS,
    **GITHUB_FUNCTIONS,
    **PROFILE_FUNCTIONS,
    **PROJECT_FUNCTIONS,
    **PROJECT_FILE_FUNCTIONS,
}
ALL_SCHEMAS = (
    TOOL_SCHEMAS + OBSIDIAN_SCHEMAS + GITHUB_SCHEMAS + PROFILE_SCHEMAS + PROJECT_SCHEMAS + PROJECT_FILE_SCHEMAS
)

# GitHub's 10 tool schemas alone are ~1,472 of ~3,100 total tool-schema
# tokens — real, measured overhead that's irrelevant to typical usage (a
# health question, checking reminders, personal notes) but was being sent
# on literally every single turn regardless. Confirmed via live testing
# this tips total prompt+schema size close to/over the model's actual
# 4096-token context window, and behavior degrades noticeably under that
# pressure — not just slower, genuinely incoherent (grabbing random
# unrelated tools — GitHub commit lookups on unrelated repos — instead of
# just answering a plain question). Only sent when the message that
# triggered this turn plausibly needs them.
_NON_GITHUB_SCHEMAS = TOOL_SCHEMAS + OBSIDIAN_SCHEMAS + PROFILE_SCHEMAS + PROJECT_SCHEMAS + PROJECT_FILE_SCHEMAS
_GITHUB_KEYWORDS = ("github", "repo", "pull request", "open source", "git clone")


def _mentions_github(text: str) -> bool:
    lower = text.lower()
    return any(k in lower for k in _GITHUB_KEYWORDS)


def _schemas_for(messages: list[dict]) -> list[dict]:
    last_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
    return ALL_SCHEMAS if _mentions_github(last_user) else _NON_GITHUB_SCHEMAS


def load_system_prompt() -> str:
    with open(PROMPT_PATH, "r") as f:
        base = f.read()
    # The profile ("central brain") is folded straight into the system
    # prompt rather than kept as its own message — the system message is
    # never trimmed by conversation_store's rolling history cap, so this is
    # what keeps it "always known" regardless of how old the conversation
    # that mentioned it is.
    return base + profile_block()


def _parse_loose_tool_call(content: str) -> list[dict] | None:
    """qwen2.5-coder:7b-q4_K_M sometimes emits a bare {"name":..,"arguments":..}
    JSON object as plain content instead of wrapping it in <tool_call> tags, so
    ollama's parser never populates tool_calls. Catch that case manually.

    It also sometimes wraps the JSON in conversational text ("Sure thing!
    Let's do this.\n\n{...}") rather than emitting pure JSON, so this scans
    for a {...} substring anywhere in the content rather than requiring the
    whole message to be just the JSON object.

    Returns the already-unified tool_calls shape (a list of {"function":
    {...}} dicts, matching what msg.get("tool_calls") returns for a
    properly-tagged call) so the caller doesn't have to know or care which
    path a tool call arrived by."""
    obj = extract_json_object(content)
    if obj and "name" in obj and "arguments" in obj and obj["name"] in ALL_FUNCTIONS:
        return [{"function": {"name": obj["name"], "arguments": obj["arguments"]}}]
    return None


# Real bug found 2026-09-15: this loop used to be `while True` with no cap
# at all. The model got stuck re-calling remember() with the same fact
# hundreds of times in a single turn (674 duplicate appends observed) —
# each round trip grows `messages`, which grows the next ollama.chat()
# call's context, which slows the next call down, which gave it more time
# to loop again before anyone noticed. A capped, bounded loop can't do that.
MAX_TOOL_ITERATIONS = 8


def run_turn(messages: list[dict]) -> list[dict]:
    """Run the loop for one user turn. Mutates and returns `messages` with the
    assistant's reply (and any tool exchanges) appended."""
    # Real bug found 2026-09-15: for a plain question ("what should I take
    # before a workout"), the model hallucinated a note path, got a "no
    # such file" error, and retried the EXACT same call 8 times — burning
    # the whole iteration budget on a doomed repeat instead of adapting,
    # so the forced final answer had nothing useful to work with. Track
    # (name, args) pairs already tried this turn; a repeat skips the real
    # call and tells the model directly to stop retrying it.
    seen_calls: set[tuple[str, str]] = set()
    schemas = _schemas_for(messages)

    for _ in range(MAX_TOOL_ITERATIONS):
        response = ollama.chat(model=MODEL, messages=messages, tools=schemas)
        msg = response["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or _parse_loose_tool_call(msg.get("content", ""))
        if not tool_calls:
            return messages

        for call in tool_calls:
            name = call["function"]["name"]
            fn = ALL_FUNCTIONS.get(name)
            try:
                # json.loads used to sit outside this try — malformed
                # arguments (this exact model is documented above as
                # sometimes emitting loose/imperfect JSON) raised
                # uncaught, killing the whole turn's thread silently
                # instead of reporting a tool error the model could see
                # and recover from.
                args = call["function"]["arguments"]
                if isinstance(args, str):
                    # strict=False — same literal-unescaped-newline issue
                    # documented in text_utils.extract_json_object; this
                    # call site parses the same kind of model-generated
                    # JSON and is exposed to the identical failure mode.
                    args = json.loads(args, strict=False)

                call_key = (name, json.dumps(args, sort_keys=True))
                if call_key in seen_calls:
                    result = (
                        f"you already called {name} with these exact arguments this turn and it "
                        "didn't work — retrying it again won't help. Either try meaningfully "
                        "different arguments, a different tool, or just answer from what you "
                        "already know."
                    )
                else:
                    seen_calls.add(call_key)
                    result = fn(**args) if fn else f"unknown tool: {name}"
            except Exception as e:
                result = f"error: {e}"

            messages.append({"role": "tool", "content": str(result)})

    # Hit the cap — force a plain-text final answer by dropping `tools`
    # from this last call, so the model physically cannot request another
    # tool call no matter how insistent it is.
    messages.append({
        "role": "tool",
        "content": "(tool-call limit reached for this turn — give your final answer now, in plain text, no more tool calls)",
    })
    response = ollama.chat(model=MODEL, messages=messages)
    messages.append(response["message"])
    return messages


def new_conversation() -> list[dict]:
    return [{"role": "system", "content": load_system_prompt()}]
