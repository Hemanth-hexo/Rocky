"""Read-only access to the user's real project directories on disk
(~/Desktop/PROJECTS), separate from the sandboxed scratch tools in
tools.py and the lightweight note-based tracker in projects.py.

Read-only on purpose: write_file/run_command already exist for the
sandbox, and letting a 7B model overwrite files or run shell commands in
real, active codebases is a different risk level than in a disposable
scratch folder — a hallucinated `rm` or a bad overwrite here can't be
shrugged off the way it can in sandbox/.

EXCLUDED entirely, at the user's explicit request: "Self Healing" (a
banking-related project) — checked before any path resolves, at any
depth under it."""

import os

PROJECTS_ROOT = os.path.expanduser("~/Desktop/PROJECTS")
EXCLUDED_PROJECTS = {"Self Healing"}

# Filtered out of directory listings (not blocked from direct reads) —
# real dev projects are full of this, and it drowns out anything useful.
_NOISY_DIRS = {"node_modules", ".git", "venv", ".venv", "__pycache__", ".next", "dist", "build"}

_MAX_FILE_CHARS = 20000


def _resolve(project: str, rel_path: str = "") -> str:
    full = os.path.abspath(os.path.join(PROJECTS_ROOT, project, rel_path))
    if os.path.commonpath([full, PROJECTS_ROOT]) != PROJECTS_ROOT:
        raise ValueError(f"path escapes {PROJECTS_ROOT}")
    top = os.path.relpath(full, PROJECTS_ROOT).split(os.sep)[0]
    if top in EXCLUDED_PROJECTS:
        raise ValueError(f"'{top}' is off-limits")
    return full


def list_user_projects() -> str:
    if not os.path.isdir(PROJECTS_ROOT):
        return "(projects folder not found)"
    entries = sorted(
        d for d in os.listdir(PROJECTS_ROOT)
        if os.path.isdir(os.path.join(PROJECTS_ROOT, d)) and not d.startswith(".") and d not in EXCLUDED_PROJECTS
    )
    return "\n".join(entries) if entries else "(no projects found)"


def list_project_files(project: str, subpath: str = "") -> str:
    try:
        full = _resolve(project, subpath)
    except ValueError as e:
        return str(e)
    if not os.path.isdir(full):
        return f"'{project}/{subpath}' is not a directory"
    entries = sorted(e for e in os.listdir(full) if not e.startswith(".") and e not in _NOISY_DIRS)
    if not entries:
        return "(empty directory)"
    lines = []
    for e in entries:
        full_entry = os.path.join(full, e)
        lines.append(f"{e}/" if os.path.isdir(full_entry) else e)
    return "\n".join(lines)


def read_project_file(project: str, path: str) -> str:
    try:
        full = _resolve(project, path)
    except ValueError as e:
        return str(e)
    if not os.path.isfile(full):
        return f"'{project}/{path}' is not a file"
    try:
        with open(full, "r", errors="replace") as f:
            content = f.read()
    except Exception as e:
        return f"couldn't read '{project}/{path}': {e}"
    if len(content) > _MAX_FILE_CHARS:
        return content[:_MAX_FILE_CHARS] + f"\n\n[truncated — file is longer than {_MAX_FILE_CHARS:,} characters]"
    return content


PROJECT_FILE_FUNCTIONS = {
    "list_user_projects": list_user_projects,
    "list_project_files": list_project_files,
    "read_project_file": read_project_file,
}

PROJECT_FILE_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_user_projects",
            "description": "List the user's real project folders on disk (their actual codebases, not the lightweight tracked-project notes).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_project_files",
            "description": "List files/folders inside one of the user's real project folders, one level at a time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project folder name, from list_user_projects"},
                    "subpath": {"type": "string", "description": "Subdirectory inside the project, e.g. 'src'. Omit for the project root."},
                },
                "required": ["project"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_project_file",
            "description": "Read one file's contents from inside a real project folder. Use list_project_files first if unsure of the path. Read-only.",
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {"type": "string", "description": "Project folder name, from list_user_projects"},
                    "path": {"type": "string", "description": "File path inside the project, e.g. 'src/main.py'"},
                },
                "required": ["project", "path"],
            },
        },
    },
]
