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
    fact = fact.strip()
    # Defense in depth against exact-duplicate spam — the main protection
    # is the tool-call iteration cap in agent/loop.py (this file's profile
    # note once grew to 700+ copies of the same fact from a single runaway
    # turn before that cap existed), but even a bounded loop still appends
    # once per call, and the same fact can legitimately come up again in a
    # later, separate conversation. Exact-match only; not trying to catch
    # near-duplicates worded differently.
    existing = read_profile()
    if f"- {fact}" in existing.splitlines():
        return f"already known: {fact}"
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    exists = os.path.exists(PROFILE_PATH)
    with open(PROFILE_PATH, "a") as f:
        if not exists:
            f.write("# Rocky Profile\n\nDurable facts about the user — always loaded into context.\n\n")
        f.write(f"- {fact}\n")
    return f"remembered: {fact}"


def extract_profile_facts(text: str) -> list[str]:
    """One-off model call (same pattern as agent/greeting.py's
    generate_greeting) that pulls durable, standalone facts out of a
    message worth keeping permanently.

    First version of this prompt asked for "1-6" facts and just said
    "extract durable facts" — tested against a genuinely long, detailed
    message and it collapsed specific, concrete details into vague
    LinkedIn-bio-style lines ("Romantic Interests: I have a special
    someone..."), losing most of what actually mattered. This version asks
    for as many facts as genuinely apply, explicitly asks to keep concrete
    detail instead of paraphrasing it away, and calls out that a standing
    instruction for how Rocky should behave (e.g. "don't assume without
    evidence") is itself a fact worth saving, not just information."""
    prompt = (
        "Extract durable, standalone facts about the user from the message below, for a "
        "permanent profile note loaded into every future conversation. Extract as many as "
        "genuinely apply — don't force it into a small fixed number, and for a long or detailed "
        "message don't over-compress into vague generic phrases. Keep specific, concrete details "
        "(names, what happened, why it matters) rather than paraphrasing them into a generic "
        "bio-style line.\n\n"
        "If the message gives you an instruction for how to talk about or handle a topic in "
        "future conversations (e.g. \"don't assume X without evidence\", \"always ask me before "
        "Y\"), extract that as its own fact too — it's a standing rule for you to follow, not "
        "just information about the user.\n\n"
        "Only include things that stay true over time — identity, preferences, relationships or "
        "situations worth long-term context, ongoing projects, habits, values, standing "
        "instructions to you. Leave out anything only relevant to this one exchange.\n\n"
        "Write each fact as one plain line, no numbering, no bullets, no extra commentary — just "
        "the facts, one per line.\n\n"
        "Example message: \"I'm Sam, a nurse, I work night shifts so don't message me before "
        "2pm, and my dog Bailey just had surgery so I'm worried about her this week.\"\n"
        "Example facts:\n"
        "Name is Sam, works as a nurse\n"
        "Works night shifts — don't expect a response before 2pm\n"
        "Has a dog named Bailey who had surgery recently\n\n"
        "Message:\n" + text
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
