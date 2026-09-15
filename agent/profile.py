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

from .obsidian import VAULT_DIR

PROFILE_PATH = os.path.join(VAULT_DIR, "Rocky Profile.md")


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
