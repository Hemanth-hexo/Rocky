"""Lightweight project tracking — one Markdown note per project in the
Obsidian vault, in the same spirit as the daily conversation logs and the
central-brain profile (agent/profile.py). A "project" here is just a name,
a description, and a running log of updates — nothing more structured
than that until there's a real reason to want more."""

import datetime
import os

from .obsidian import VAULT_DIR

PROJECTS_DIR = os.path.join(VAULT_DIR, "Rocky Projects")


def _project_path(name: str) -> str:
    safe = name.strip().replace("/", "-")
    return os.path.join(PROJECTS_DIR, f"{safe}.md")


def list_projects() -> str:
    if not os.path.isdir(PROJECTS_DIR):
        return "(no projects yet)"
    names = sorted(f.removesuffix(".md") for f in os.listdir(PROJECTS_DIR) if f.endswith(".md"))
    return "\n".join(names) if names else "(no projects yet)"


def create_project(name: str, description: str = "") -> str:
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    path = _project_path(name)
    if os.path.exists(path):
        return f"a project named '{name}' already exists — use add_project_update instead"
    with open(path, "w") as f:
        f.write(f"# {name}\n\n{description.strip()}\n\n## Updates\n\n")
    return f"created project: {name}"


def add_project_update(name: str, note: str) -> str:
    os.makedirs(PROJECTS_DIR, exist_ok=True)
    path = _project_path(name)
    if not os.path.exists(path):
        create_project(name)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(path, "a") as f:
        f.write(f"- **{timestamp}**: {note.strip()}\n")
    return f"logged update for {name}"


def read_project(name: str) -> str:
    path = _project_path(name)
    if not os.path.exists(path):
        return f"no project named '{name}' found — check list_projects for the exact name"
    with open(path, "r") as f:
        return f.read()


PROJECT_FUNCTIONS = {
    "create_project": create_project,
    "add_project_update": add_project_update,
    "list_projects": list_projects,
    "read_project": read_project,
}

PROJECT_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "create_project",
            "description": "Start tracking a new project the user is working on — a named, ongoing effort with a short description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short project name, e.g. 'Rocky' or 'Portfolio site'"},
                    "description": {"type": "string", "description": "One or two sentences on what it is"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_project_update",
            "description": "Log a dated update/note against an existing project — progress, a decision, a blocker, anything worth remembering later. Creates the project first if it doesn't exist yet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The project's name"},
                    "note": {"type": "string", "description": "What happened or what to remember"},
                },
                "required": ["name", "note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_projects",
            "description": "List all tracked project names.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_project",
            "description": "Read a project's full description and update history.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "The project's name"}},
                "required": ["name"],
            },
        },
    },
]
