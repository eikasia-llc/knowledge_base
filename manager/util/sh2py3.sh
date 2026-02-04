#!/bin/sh

# Find python interpreter
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
else
    echo "Error: Python not found." >&2
    exit 1
fi

# Determine script details from $0
# Assumed structure: .../bin/<category>/<script_name>
INVOKED_PATH="$0"
INVOKED_DIR="$(dirname "$INVOKED_PATH")"
CATEGORY="$(basename "$INVOKED_DIR")"
SCRIPT_NAME="$(basename "$INVOKED_PATH")"

# Construct path to the python script
# We start from the invoked directory (e.g. bin/manager), go up two levels to root,
# then into category (manager), then the script with .py
# logic: $INVOKED_DIR/../../$CATEGORY/$SCRIPT_NAME.py

PYTHON_SCRIPT="$INVOKED_DIR/../../$CATEGORY/$SCRIPT_NAME.py"

# Execute
exec "$PYTHON_CMD" "$PYTHON_SCRIPT" "$@"
