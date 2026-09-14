#!/bin/bash
# Stops Rocky and removes it from launchd supervision — it won't come back
# until you run start_rocky.sh again.
launchctl bootout "gui/$(id -u)/com.rocky.desktopapp" 2>&1 || echo "(already stopped)"
echo "Rocky stopped."
