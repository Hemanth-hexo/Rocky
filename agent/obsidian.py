"""Obsidian note tools, scoped to a single vault — kept separate from the
general-purpose sandbox tools in tools.py."""

import datetime
import os

VAULT_DIR = os.environ.get(
    "ROCKY_VAULT_PATH",
    "/Users/hemanthsarode/Hexo's vault/Hexo's Vault",
)

LOG_DIR = "Rocky Conversations"


def _resolve(note_path: str) -> str:
    if not note_path.endswith(".md"):
        note_path += ".md"
    full = os.path.abspath(os.path.join(VAULT_DIR, note_path))
    if os.path.commonpath([full, os.path.abspath(VAULT_DIR)]) != os.path.abspath(VAULT_DIR):
        raise ValueError(f"note path '{note_path}' escapes the vault")
    return full


def read_note(note_path: str) -> str:
    full = _resolve(note_path)
    with open(full, "r") as f:
        return f.read()


def write_note(note_path: str, content: str) -> str:
    full = _resolve(note_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {note_path}"


def append_daily_log(user_text: str, rocky_text: str) -> None:
    """Auto-logs one conversation turn to today's note — called directly by
    main.py after every turn, not exposed as an LLM tool, so it happens
    unconditionally rather than only when the model decides to write a note."""
    today = datetime.date.today().isoformat()
    note_path = f"{LOG_DIR}/{today}.md"
    full = _resolve(note_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%H:%M")
    entry = f"## {timestamp}\n**You:** {user_text}\n**Rocky:** {rocky_text}\n\n"

    if not os.path.exists(full):
        entry = f"# {today}\n\n" + entry
    with open(full, "a") as f:
        f.write(entry)


def list_notes(subdir: str = "") -> str:
    full = _resolve(subdir).removesuffix(".md") if subdir else VAULT_DIR
    notes = []
    for root, dirs, files in os.walk(full):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in files:
            if name.endswith(".md"):
                rel = os.path.relpath(os.path.join(root, name), VAULT_DIR)
                notes.append(rel)
    return "\n".join(sorted(notes)) or "(no notes found)"


OBSIDIAN_FUNCTIONS = {
    "read_note": read_note,
    "write_note": write_note,
    "list_notes": list_notes,
}

OBSIDIAN_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_note",
            "description": "Read a note from the Obsidian vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_path": {"type": "string", "description": "Note path relative to the vault root, e.g. 'Projects/Rocky.md'"}
                },
                "required": ["note_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_note",
            "description": "Create or overwrite a note in the Obsidian vault.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note_path": {"type": "string", "description": "Note path relative to the vault root"},
                    "content": {"type": "string", "description": "Note contents (Markdown)"},
                },
                "required": ["note_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List note paths in the vault, optionally under a subfolder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {"type": "string", "description": "Subfolder relative to the vault root (optional)"}
                },
            },
        },
    },
]
