#!/bin/bash
set -e

# Default values
REPO="Qwen/Qwen3.5-0.8B"
OUT_DIR="models/qwen3.5-0.8b"
FULL=false
INSTALL_DEPS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --repo)
      REPO="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --full)
      FULL=true
      shift
      ;;
    --install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOADER="$REPO_ROOT/scripts/hf/download_model.py"

if [ ! -f "$DOWNLOADER" ]; then
    echo "Downloader script not found at $DOWNLOADER"
    exit 1
fi

RESOLVED_OUT_DIR="$OUT_DIR"
[[ "$RESOLVED_OUT_DIR" = /* ]] || RESOLVED_OUT_DIR="$REPO_ROOT/$RESOLVED_OUT_DIR"

# Try to use .venv if it exists
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "Using existing virtual environment at $REPO_ROOT/.venv"
    source "$REPO_ROOT/.venv/bin/activate"
fi

if [ "$INSTALL_DEPS" = true ]; then
    echo "Installing/updating huggingface_hub (<1.0 for transformers compatibility)..."
    python3 -m pip install --upgrade "huggingface_hub<1.0"
else
    if ! python3 -c "import huggingface_hub" &> /dev/null; then
        echo "huggingface_hub is not installed. Re-run with --install-deps or activate your virtual environment."
        exit 1
    fi
fi

ARGS=(
    "$DOWNLOADER"
    "--repo" "$REPO"
    "--dest" "$RESOLVED_OUT_DIR"
)
if [ "$FULL" = true ]; then
    ARGS+=("--full")
fi

echo "Downloading $REPO into $RESOLVED_OUT_DIR"
python3 "${ARGS[@]}"

echo "Model download complete."
