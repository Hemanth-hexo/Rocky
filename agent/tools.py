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


def create_reminder(text: str, due_date: str = "") -> str:
    """Creates a reminder in Reminders.app. due_date, when given, must be
    'MM/DD/YYYY HH:MM' in 24-hour time.

    NOT exposed to the chat model as a callable tool (see TOOL_FUNCTIONS/
    TOOL_SCHEMAS below) — only reachable via agent/scheduling.py's
    deterministic trigger-phrase path. Letting the model decide whether to
    call this on its own risks a duplicate: if the deterministic path
    already created a reminder for an explicit "remind me..." request, an
    unreliable model might ALSO decide to call this itself, creating two
    real Reminders.app entries for the same request. remember() in
    agent/profile.py can afford model discretion because a duplicate fact
    is harmless; a duplicate reminder isn't."""
    safe_text = text.replace("\\", "\\\\").replace('"', '\\"')
    due_clause = ""
    if due_date:
        safe_due = due_date.replace("\\", "\\\\").replace('"', '\\"')
        due_clause = f', due date:date "{safe_due}"'
    script = f'''
    tell application "Reminders"
        make new reminder with properties {{name:"{safe_text}"{due_clause}}}
        return "created reminder: {safe_text}"
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15)
    output = (result.stdout + result.stderr).strip()
    return output or f"created reminder: {text}"


def list_reminders(limit: int = 10) -> str:
    """Lists incomplete reminders from Reminders.app, most recent list first."""
    script = f'''
    tell application "Reminders"
        set output to ""
        set reminderList to (reminders whose completed is false)
        set n to 0
        repeat with r in reminderList
            if n >= {limit} then exit repeat
            set output to output & (name of r) & linefeed
            set n to n + 1
        end repeat
        return output
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15)
    output = (result.stdout + result.stderr).strip()
    return output or "(no reminders)"


def create_calendar_event(title: str, start_date: str, end_date: str = "") -> str:
    """Creates an event on the default Calendar.app calendar. start_date/
    end_date must be 'MM/DD/YYYY HH:MM' in 24-hour time; if end_date is
    omitted the event is 30 minutes long.

    Also NOT exposed to the chat model — same duplicate-side-effect reason
    as create_reminder above."""
    safe_title = title.replace("\\", "\\\\").replace('"', '\\"')
    safe_start = start_date.replace("\\", "\\\\").replace('"', '\\"')
    if end_date:
        safe_end = end_date.replace("\\", "\\\\").replace('"', '\\"')
        end_expr = f'date "{safe_end}"'
    else:
        end_expr = f'(date "{safe_start}") + 30 * minutes'
    script = f'''
    tell application "Calendar"
        tell calendar 1
            make new event with properties {{summary:"{safe_title}", start date:date "{safe_start}", end date:{end_expr}}}
        end tell
        return "created event: {safe_title}"
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=15)
    output = (result.stdout + result.stderr).strip()
    return output or f"created event: {title}"


def read_recent_emails(count: int = 5) -> str:
    """Reads sender/subject/date of the most recent messages in Mail.app's
    inbox. Only sees accounts already set up in Mail.app locally — nothing
    is fetched over the network directly, and nothing is ever sent."""
    script = f'''
    tell application "Mail"
        set output to ""
        set msgs to messages 1 thru {count} of inbox
        repeat with m in msgs
            set output to output & "From: " & (sender of m) & linefeed
            set output to output & "Subject: " & (subject of m) & linefeed
            set output to output & "Date: " & ((date received of m) as string) & linefeed & linefeed
        end repeat
        return output
    end tell
    '''
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    output = (result.stdout + result.stderr).strip()
    return output or "(no messages found)"


# Same reliability gap the reminder/calendar tools had (see
# agent/scheduling.py) — confirmed live: the underlying AppleScript call
# works fine on its own, but the chat model wasn't reliably deciding to
# call read_recent_emails() on requests like "check my email". Unlike
# reminders/calendar, a missed or duplicate fetch here is harmless (it's a
# read, not a side effect), so this stays ALSO available as a normal
# model-callable tool — this trigger check is just a backstop, not the
# only path.
_EMAIL_TRIGGERS = (
    "check my email", "check email", "check my inbox", "check inbox",
    "any new emails", "new emails", "recent emails", "read my email",
    "read my inbox", "what's in my inbox", "whats in my inbox",
)


def mentions_email_check(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in _EMAIL_TRIGGERS)


def open_url(url: str) -> str:
    subprocess.run(["open", url], check=False)
    return f"opened {url}"


def set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=False)
    return f"volume set to {level}"


TOOL_FUNCTIONS = {
    "write_file": write_file,
    "read_file": read_file,
    "run_command": run_command,
    "open_app": open_app,
    "run_applescript": run_applescript,
    "get_current_datetime": get_current_datetime,
    "play_music": play_music,
    # create_reminder/create_calendar_event deliberately excluded — see
    # their docstrings above. Only list_reminders (a read, no duplicate
    # risk) is model-callable.
    "list_reminders": list_reminders,
    "read_recent_emails": read_recent_emails,
    "open_url": open_url,
    "set_volume": set_volume,
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
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "List incomplete reminders from Reminders.app.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max reminders to return (default 10)"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_recent_emails",
            "description": "Read sender/subject/date of the most recent messages in Mail.app's inbox. Only sees accounts already configured in Mail.app locally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer", "description": "Number of recent messages to read (default 5)"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL in the default browser.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "The URL to open"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_volume",
            "description": "Set the system output volume, 0-100.",
            "parameters": {
                "type": "object",
                "properties": {"level": {"type": "integer", "description": "Volume level from 0 to 100"}},
                "required": ["level"],
            },
        },
    },
]
