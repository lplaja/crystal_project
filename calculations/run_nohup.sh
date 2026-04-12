#!/bin/bash

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <python_script.py> [args...]"
    exit 1
fi

PYTHON_SCRIPT="$1"
shift  # remove the script name from arguments

# Extract just the base name for the process title (without .py)
PROCESS_NAME=$(basename "$PYTHON_SCRIPT" .py)

# Run in nohup with custom process name
nohup bash -c "exec -a $PROCESS_NAME python $PYTHON_SCRIPT $*" > "${PROCESS_NAME}.out" 2>&1 &

echo "Running $PYTHON_SCRIPT as $PROCESS_NAME in background. Output -> ${PROCESS_NAME}.out"