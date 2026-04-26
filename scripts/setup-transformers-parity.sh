#!/bin/bash
set -e

# Default values
PYTHON_EXE="python3"
VENV_PATH=".venv-hf-parity"
TRANSFORMERS_SPEC="git+https://github.com/huggingface/transformers.git"
NO_VENV=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --python)
      PYTHON_EXE="$2"
      shift 2
      ;;
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --transformers)
      TRANSFORMERS_SPEC="$2"
      shift 2
      ;;
    --no-venv)
      NO_VENV=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQUIREMENTS="$REPO_ROOT/scripts/hf/requirements-transformers-parity.txt"

if [ ! -f "$REQUIREMENTS" ]; then
    echo "Requirements file not found: $REQUIREMENTS"
    exit 1
fi

if [ "$NO_VENV" = true ]; then
    RESOLVED_PYTHON="$PYTHON_EXE"
else
    RESOLVED_VENV="$VENV_PATH"
    [[ "$RESOLVED_VENV" = /* ]] || RESOLVED_VENV="$REPO_ROOT/$RESOLVED_VENV"

    if [ ! -d "$RESOLVED_VENV" ]; then
        echo "Creating Python venv: $RESOLVED_VENV"
        "$PYTHON_EXE" -m venv "$RESOLVED_VENV"
    fi

    RESOLVED_PYTHON="$RESOLVED_VENV/bin/python3"
fi

echo "Using Python: $RESOLVED_PYTHON"
"$RESOLVED_PYTHON" -m pip install --upgrade pip

"$RESOLVED_PYTHON" -m pip install -r "$REQUIREMENTS"

echo "Installing Transformers: $TRANSFORMERS_SPEC"
"$RESOLVED_PYTHON" -m pip install --upgrade "$TRANSFORMERS_SPEC"

echo "Transformers parity dependencies are installed."
echo "Use: $RESOLVED_PYTHON scripts/hf/transformers_inference.py --help"
