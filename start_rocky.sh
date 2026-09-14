#!/bin/bash
# Starts Rocky as a supervised background process via launchd — it will
# automatically restart Rocky if it crashes (non-zero exit), but NOT if you
# quit it normally via the app's Quit action (clean exit 0). Does not run
# at login; only starts when you run this script.
set -e
# Hardcoded rather than derived from $(dirname "$0")/$HOME — when macOS
# launches this via double-click/`open` (e.g. from Rocky.app), it runs in a
# fresh, minimal environment where $HOME isn't reliably set and the
# working directory isn't guaranteed either.
HERE="/Users/hemanthsarode/Desktop/PROJECTS/Rocky"
PLIST_DEST="/Users/hemanthsarode/Library/LaunchAgents/com.rocky.desktopapp.plist"

cp "$HERE/com.rocky.desktopapp.plist" "$PLIST_DEST"
launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST" 2>/dev/null || true
launchctl kickstart -k "gui/$(id -u)/com.rocky.desktopapp"
echo "Rocky started (supervised — will auto-restart if it crashes)."
echo "Logs: $HERE/rocky.log"
