"""Rocky's tools — sandboxed to SANDBOX_DIR so a bad command can't touch anything outside it."""

import datetime
import os
import subprocess

SANDBOX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sandbox"))
os.makedirs(SANDBOX_DIR, exist_ok=True)


def _resolve(path: str) -> str:
    full = os.path.abspath(os.path.join(SANDBOX_DIR, path))
    if os.path.commonpath([full, SANDBOX_DIR]) != SANDBOX_DIR:
        raise ValueError(f"path '{path}' escapes the sandbox")
    return full


def write_file(path: str, content: str) -> str:
    full = _resolve(path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


def read_file(path: str) -> str:
    full = _resolve(path)
    with open(full, "r") as f:
        return f.read()


def run_command(command: str) -> str:
    result = subprocess.run(
        command, shell=True, cwd=SANDBOX_DIR,
        capture_output=True, text=True, timeout=60,
    )
    out = result.stdout + result.stderr
    return out.strip() or f"(exit {result.returncode}, no output)"


def open_app(name: str) -> str:
    subprocess.run(["osascript", "-e", f'tell application "{name}" to activate'], check=False)
    return f"opened {name}"


def run_applescript(script: str) -> str:
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    return (result.stdout + result.stderr).strip() or f"(exit {result.returncode})"


def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def play_music(query: str) -> str:
    """Searches the local Music.app library for `query` and plays the first
    match. Only finds content actually in the user's library (downloaded/
    synced tracks) — not the full Apple Music streaming catalog."""
    safe_query = query.replace("\\", "\\\\").replace('"', '\\"')
    script = f'''
    tell application "Music"
        activate
        set searchResults to (search playlist "Library" for "{safe_query}")
        if (count of searchResults) > 0 then
            play (item 1 of searchResults)
            return "playing " & (name of item 1 of searchResults) & " by " & (artist of item 1 of searchResults)
        else
            return "no match found in the local Music library for \\"{safe_query}\\""
        end if
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15)
    output = (result.stdout + result.stderr).strip()
    return output or f"(exit {result.returncode})"


TOOL_FUNCTIONS = {
    "write_file": write_file,
    "read_file": read_file,
    "run_command": run_command,
    "open_app": open_app,
    "run_applescript": run_applescript,
    "get_current_datetime": get_current_datetime,
    "play_music": play_music,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file inside the sandbox folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the sandbox folder"},
                    "content": {"type": "string", "description": "File contents"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file back from the sandbox folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the sandbox folder"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command with cwd locked to the sandbox folder.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open/activate a macOS application by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Application name, e.g. 'Safari'"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_applescript",
            "description": "Run an AppleScript snippet via osascript.",
            "parameters": {
                "type": "object",
                "properties": {"script": {"type": "string", "description": "AppleScript source"}},
                "required": ["script"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current real-world date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": "Search the local Music.app library and play the first matching song. Only finds tracks actually in the user's library, not the full streaming catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Song title and/or artist to search for, e.g. 'Prisoner The Weeknd'"}
                },
                "required": ["query"],
            },
        },
    },
]
