#!/bin/bash
# Starts Rocky as a supervised background process via launchd — it will
# automatically restart Rocky if it crashes (non-zero exit), but NOT if you
# quit it normally via the app's Quit action (clean exit 0). Does not run
# at login; only starts when you run this script.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PLIST_DEST="$HOME/Library/LaunchAgents/com.rocky.desktopapp.plist"

cp "$HERE/com.rocky.desktopapp.plist" "$PLIST_DEST"
launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST" 2>/dev/null || true
launchctl kickstart -k "gui/$(id -u)/com.rocky.desktopapp"
echo "Rocky started (supervised — will auto-restart if it crashes)."
echo "Logs: $HERE/rocky.log"
