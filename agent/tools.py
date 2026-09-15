"""Rocky's tools. write_file/read_file/run_command are sandboxed to
SANDBOX_DIR by resolved-path checking — that stops path traversal
(../../etc/passwd), but run_command's shell=True means shell metacharacters
(cd, absolute paths, pipes) can still act outside SANDBOX_DIR once the
shell is running; see run_command's own docstring below for the actual
guarantee and the mitigation in place. The threat model here is a
hallucinating local model, not an external attacker — there's no network-
exposed input reaching these functions."""

import datetime
import os
import subprocess

SANDBOX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sandbox"))
os.makedirs(SANDBOX_DIR, exist_ok=True)

# Applied to every string interpolated into an AppleScript "..." literal —
# was previously duplicated ad hoc in play_music/create_reminder/
# create_calendar_event (and missing entirely from open_app, a real
# injection gap: an unescaped `name` let a crafted string break out of the
# `tell application "..."` literal and run arbitrary `do shell script`).
# Also escapes newlines, which break a "..." literal just as much as an
# unescaped quote does but weren't handled before.
def _applescript_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# Commands most likely to do real damage if a hallucinated run_command call
# tries to act outside the sandbox — not a security boundary (shell=True
# can't be fully locked down short of a real sandboxing layer like
# sandbox-exec), just a cheap check against the most catastrophic patterns.
_DANGEROUS_COMMAND_MARKERS = ("rm -rf /", "sudo ", ":(){", "dd if=", "mkfs", "> /dev/sd")


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
    """Runs with cwd starting in SANDBOX_DIR — but shell=True means this is
    NOT a hard sandbox: `cd ..`, an absolute path, or a pipe can still act
    outside SANDBOX_DIR, same as typing the command in Terminal yourself.
    The cwd is a convenience default, not an enforced boundary. Refuses a
    short list of the most obviously catastrophic patterns as a cheap
    defense-in-depth check, not a claim of real isolation."""
    lowered = command.lower()
    if any(marker in lowered for marker in _DANGEROUS_COMMAND_MARKERS):
        return "refused: command matches a blocked dangerous pattern"
    result = subprocess.run(
        command, shell=True, cwd=SANDBOX_DIR,
        capture_output=True, text=True, timeout=60,
    )
    out = result.stdout + result.stderr
    return out.strip() or f"(exit {result.returncode}, no output)"


def open_app(name: str) -> str:
    safe_name = _applescript_escape(name)
    subprocess.run(["osascript", "-e", f'tell application "{safe_name}" to activate'], check=False)
    return f"opened {name}"


def run_applescript(script: str) -> str:
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return "AppleScript timed out after 20s (a blocking dialog or infinite loop is the usual cause)"
    return (result.stdout + result.stderr).strip() or f"(exit {result.returncode})"


def get_current_datetime() -> str:
    return datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")


def play_music(query: str) -> str:
    """Searches the local Music.app library for `query` and plays the first
    match. Only finds content actually in the user's library (downloaded/
    synced tracks) — not the full Apple Music streaming catalog."""
    safe_query = _applescript_escape(query)
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
    safe_text = _applescript_escape(text)
    due_clause = ""
    if due_date:
        safe_due = _applescript_escape(due_date)
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
    limit = max(1, int(limit))
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
    safe_title = _applescript_escape(title)
    safe_start = _applescript_escape(start_date)
    if end_date:
        safe_end = _applescript_escape(end_date)
        end_expr = f'date "{safe_end}"'
    else:
        end_expr = f'(date "{safe_start}") + 30 * minutes'
    # "calendar 1" is whatever the OS happens to list first — on a machine
    # with subscribed/read-only calendars (Holidays, Birthdays, etc.) that's
    # not guaranteed to be writable, and `make new event` on a read-only
    # calendar fails outright. Find the first actually-writable one instead.
    script = f'''
    tell application "Calendar"
        set targetCal to missing value
        repeat with c in calendars
            if writable of c then
                set targetCal to c
                exit repeat
            end if
        end repeat
        if targetCal is missing value then return "error: no writable calendar found"
        tell targetCal
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
    count = max(1, int(count))
    # "count of messages of inbox" is a well-known slow call in Mail.app's
    # AppleScript dictionary on any real-sized inbox — querying it up front
    # on every call (to pre-clamp fetchCount) made a normal request time
    # out. Try the direct bounded range first (fast, the common case);
    # only fall back to fetching everything if that range doesn't exist —
    # which only happens when the inbox itself is smaller than `count`,
    # i.e. exactly the case where "everything" is a small, fast set too.
    script = f'''
    tell application "Mail"
        set output to ""
        try
            set msgs to messages 1 thru {count} of inbox
        on error
            set msgs to messages of inbox
        end try
        if (count of msgs) = 0 then return "(inbox is empty)"
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
