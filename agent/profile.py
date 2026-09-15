"""Rocky's "central brain" — a small, steady set of durable facts about the
user that's baked into the system prompt every single conversation,
regardless of how the rolling chat history gets trimmed (see
agent/conversation_store.py, capped at 40 messages) or how far back an
Obsidian daily log sits. Fed exclusively through remember_from_text()'s
deterministic trigger-phrase path (main.py calls it when the user's
message matches mentions_profile_update()) — NOT left to the chat model's
own discretion to call a remember() tool mid-conversation. That was tried
and failed: see remember()'s docstring for the actual incident.

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
    """NOT exposed to the chat model as a callable tool (see the empty
    PROFILE_FUNCTIONS/PROFILE_SCHEMAS below) — only reachable via
    remember_from_text()'s deterministic trigger-phrase path. Real
    incident 2026-09-15: this used to be model-callable, on the theory
    that "a duplicate fact is harmless" (unlike create_reminder/
    create_calendar_event, which were pulled from model access for the
    same reason earlier that day). That theory was wrong — the model
    called this repeatedly within single turns, rewording the same fact
    slightly each time (exact-match dedup below doesn't catch "I got a
    promotion" vs "I received a promotion"), and the profile note grew to
    700+ near-duplicate lines, which then bloated every subsequent
    system prompt past the model's context window. Duplicates aren't
    harmless at scale — removing model discretion entirely was the fix,
    matching create_reminder/create_calendar_event's precedent."""
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


# Deliberately empty — remember() is not model-callable, see its
# docstring above. Kept as dicts (not removed outright) so agent/loop.py's
# `**PROFILE_FUNCTIONS`/`+ PROFILE_SCHEMAS` merges don't need special-casing
# if this module's shape changes again later.
PROFILE_FUNCTIONS = {}

PROFILE_SCHEMAS = []
