"""Rocky's "central brain" — a small, steady set of durable facts about the
user that's baked into the system prompt every single conversation,
regardless of how the rolling chat history gets trimmed (see
agent/conversation_store.py, capped at 40 messages) or how far back an
Obsidian daily log sits. Chat memory feeds it: the user (or Rocky, when it
notices something durable) calls remember() to fold a fact in.

Lives as a plain Markdown note in the Obsidian vault, not hidden state —
the user can open it directly and see or edit exactly what Rocky "knows"
about them."""

import os

import ollama

from .config import MODEL
from .obsidian import VAULT_DIR

PROFILE_PATH = os.path.join(VAULT_DIR, "Rocky Profile.md")

# Explicit "save this permanently" signals. Relying on the chat model to
# notice these mid-conversation and decide to call the remember() tool
# turned out to be unreliable on this hardware-constrained 7B model — a
# long, clearly-flagged message ("...it's for my profile") went completely
# unsaved because the model just replied in prose instead of calling the
# tool. These phrases trigger a deterministic extract-and-save pass
# instead of leaving it to the model's tool-calling judgment.
_PROFILE_TRIGGER_PHRASES = (
    "for my profile", "to my profile", "update profile", "update my profile",
    "add to profile", "add to my profile", "save to profile", "save this to profile",
    "remember that", "remember this", "please remember", "keep this in memory",
)


def mentions_profile_update(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _PROFILE_TRIGGER_PHRASES)


def read_profile() -> str:
    if not os.path.exists(PROFILE_PATH):
        return ""
    with open(PROFILE_PATH, "r") as f:
        return f.read().strip()


def remember(fact: str) -> str:
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    exists = os.path.exists(PROFILE_PATH)
    with open(PROFILE_PATH, "a") as f:
        if not exists:
            f.write("# Rocky Profile\n\nDurable facts about the user — always loaded into context.\n\n")
        f.write(f"- {fact.strip()}\n")
    return f"remembered: {fact}"


def extract_profile_facts(text: str) -> list[str]:
    """One-off model call (same pattern as agent/greeting.py's
    generate_greeting) that pulls durable, standalone facts out of a
    message worth keeping permanently."""
    prompt = (
        "Extract 1-6 short, standalone, durable facts about the user from the message "
        "below, suitable for a permanent profile note that's loaded into every future "
        "conversation. Only include things that stay true over time — identity, "
        "preferences, relationships or situations worth long-term context, ongoing "
        "projects, habits, values. Leave out anything only relevant to this one exchange. "
        "Write each fact as one plain line, no numbering, no bullets, no extra commentary "
        "— just the facts, one per line.\n\nMessage:\n" + text
    )
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    content = response["message"]["content"].strip()
    return [line.strip("-*• \t") for line in content.splitlines() if line.strip()]


def remember_from_text(text: str) -> list[str]:
    """Deterministic fallback for explicit remember/profile requests —
    extracts facts and saves every one, instead of hoping the chat model
    calls the remember() tool itself mid-conversation."""
    facts = extract_profile_facts(text)
    for fact in facts:
        remember(fact)
    return facts


def profile_block() -> str:
    """Appended to the system prompt so this is always in context — not
    something the model has to go looking for."""
    profile = read_profile()
    if not profile:
        return ""
    return (
        "\n\nWhat you already know about the user — always true, not just "
        "this conversation, so don't ask about it again unless something's "
        f"clearly changed:\n{profile}"
    )


PROFILE_FUNCTIONS = {
    "remember": remember,
}

PROFILE_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Save one durable fact about the user to long-term memory — something true "
                "across conversations (name, job, preferences, ongoing projects, recurring "
                "habits), not a one-off detail from this exchange. This is separate from the "
                "daily conversation logs and always stays in context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string", "description": "The fact to remember, written as a short standalone statement"}
                },
                "required": ["fact"],
            },
        },
    },
]
