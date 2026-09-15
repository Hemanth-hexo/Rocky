"""One-off importer: converts a Claude.ai data export (conversations.json
+ memories/*.json, as unzipped from claude.ai's account-settings export)
into Markdown notes in the Obsidian vault, so Rocky's existing
list_notes/read_note tools can reference them like any other note.

NOT a live Rocky tool — this is a manual data-migration utility, run once
after generating a fresh export from claude.ai. Skips light_metadata/
(users.json, login_history.json) deliberately — that's account/login
bookkeeping, not conversation content worth Rocky's memory.

Usage: python3 scripts/import_claude_export.py /path/to/export/dir
(the dir should directly contain conversations.json and a memories/
folder — that's exactly what claude.ai's export download unzips to)
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agent.obsidian import VAULT_DIR  # noqa: E402

CONV_DIR_NAME = "Claude Conversations"
MEM_DIR_NAME = "Claude Memories"


def _safe_filename(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "", name).strip()
    return name[:max_len] or "untitled"


def _extract_text(content_blocks: list) -> str:
    # Deliberately keeps only "text" blocks (what a human actually saw) —
    # skips thinking/tool_use/tool_result/token_budget/injected_prompt_block
    # blocks, which are internal mechanics that would bloat these notes
    # without adding much a future reader (or Rocky) needs. The
    # conversation's own pre-written "summary" field already captures the
    # tool-usage narrative at a readable level.
    parts = []
    for block in content_blocks or []:
        if block.get("type") == "text" and block.get("text"):
            parts.append(block["text"])
    return "\n".join(parts).strip()


def import_conversations(export_dir: str) -> int:
    path = os.path.join(export_dir, "conversations.json")
    if not os.path.exists(path):
        print(f"no conversations.json found at {path}, skipping")
        return 0
    with open(path) as f:
        conversations = json.load(f)

    out_dir = os.path.join(VAULT_DIR, CONV_DIR_NAME)
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for convo in conversations:
        name = convo.get("name") or "Untitled conversation"
        created = (convo.get("created_at") or "")[:10]
        summary = (convo.get("summary") or "").strip()

        lines = [f"# {name}", ""]
        if created:
            lines.append(f"*{created}*")
            lines.append("")
        if summary:
            lines.append("## Summary")
            lines.append("")
            lines.append(summary)
            lines.append("")
        lines.append("## Transcript")
        lines.append("")
        for msg in convo.get("chat_messages", []):
            text = _extract_text(msg.get("content"))
            if not text:
                continue
            speaker = "You" if msg.get("sender") == "human" else "Claude"
            lines.append(f"**{speaker}:** {text}")
            lines.append("")

        filename = f"{created} - {_safe_filename(name)}.md" if created else f"{_safe_filename(name)}.md"
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write("\n".join(lines))
        written += 1
    return written


def import_memories(export_dir: str) -> int:
    mem_dir = os.path.join(export_dir, "memories")
    if not os.path.isdir(mem_dir):
        print(f"no memories/ folder found at {mem_dir}, skipping")
        return 0

    out_root = os.path.join(VAULT_DIR, MEM_DIR_NAME)
    written = 0
    for fname in os.listdir(mem_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(mem_dir, fname)) as f:
            data = json.load(f)
        for entry in data.get("memory_files", []):
            rel_path = entry["path"].lstrip("/")
            content = entry.get("content", "")
            full_path = os.path.join(out_root, rel_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as out:
                out.write(content)
            written += 1
    return written


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 scripts/import_claude_export.py /path/to/export/dir")
        sys.exit(1)
    export_dir = sys.argv[1]
    n_conv = import_conversations(export_dir)
    n_mem = import_memories(export_dir)
    print(f"imported {n_conv} conversations, {n_mem} memory notes")
