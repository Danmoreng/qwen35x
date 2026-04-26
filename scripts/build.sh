#!/bin/bash
set -e

# Default values
CLEAN=false
USE_NINJA=false
ENABLE_CUDA=true
BUILD_ALL=false
RUN_SMOKE_TEST=false
CONFIGURATION="Release"
BUILD_DIR="build"
TARGET="qwen35x"
PROFILE="configs/qwen3_5_0_8b.profile.json"
SM_VERSION=120
CUDA_ARCHITECTURES="native"
CUDA_VARIANT="0p8b"
KERNEL_BLOCK_SIZE=256
KERNEL_NUM_BLOCKS=82

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN=true
      shift
      ;;
    --ninja)
      USE_NINJA=true
      shift
      ;;
    --no-cuda)
      ENABLE_CUDA=false
      shift
      ;;
    --all)
      BUILD_ALL=true
      shift
      ;;
    --smoke)
      RUN_SMOKE_TEST=true
      shift
      ;;
    --config)
      CONFIGURATION="$2"
      shift 2
      ;;
    --build-dir)
      BUILD_DIR="$2"
      shift 2
      ;;
    --target)
      TARGET="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --sm)
      SM_VERSION="$2"
      shift 2
      ;;
    --arch)
      CUDA_ARCHITECTURES="$2"
      shift 2
      ;;
    --variant)
      CUDA_VARIANT="$2"
      shift 2
      ;;
    --block-size)
      KERNEL_BLOCK_SIZE="$2"
      shift 2
      ;;
    --num-blocks)
      KERNEL_NUM_BLOCKS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOLVED_BUILD_DIR="$BUILD_DIR"
[[ "$RESOLVED_BUILD_DIR" = /* ]] || RESOLVED_BUILD_DIR="$REPO_ROOT/$RESOLVED_BUILD_DIR"

RESOLVED_PROFILE="$PROFILE"
[[ "$RESOLVED_PROFILE" = /* ]] || RESOLVED_PROFILE="$REPO_ROOT/$RESOLVED_PROFILE"

if [ "$CLEAN" = true ] && [ -d "$RESOLVED_BUILD_DIR" ]; then
  echo "Cleaning build directory: $RESOLVED_BUILD_DIR"
  rm -rf "$RESOLVED_BUILD_DIR"
fi

mkdir -p "$RESOLVED_BUILD_DIR"

GENERATOR_ARGS=()
if [ "$USE_NINJA" = true ]; then
  if ! command -v ninja &> /dev/null; then
    echo "Ninja was requested but was not found on PATH."
    exit 1
  fi
  GENERATOR_ARGS=("-G" "Ninja")
  echo "Generator: Ninja"
else
  echo "Generator: Default (Unix Makefiles)"
fi

CUDA_FLAG="OFF"
if [ "$ENABLE_CUDA" = true ]; then
  CUDA_FLAG="ON"
fi

CONFIGURE_ARGS=(
  "-S" "$REPO_ROOT"
  "-B" "$RESOLVED_BUILD_DIR"
  "${GENERATOR_ARGS[@]}"
  "-DCMAKE_CXX_STANDARD=20"
  "-DQWEN35X_ENABLE_CUDA=$CUDA_FLAG"
  "-DQWEN35X_CUDA_ARCHITECTURES=$CUDA_ARCHITECTURES"
  "-DQWEN35X_CUDA_VARIANT=$CUDA_VARIANT"
  "-DQWEN35X_KERNEL_BENCH_BLOCK_SIZE=$KERNEL_BLOCK_SIZE"
  "-DQWEN35X_KERNEL_BENCH_NUM_BLOCKS=$KERNEL_NUM_BLOCKS"
  "-DCMAKE_BUILD_TYPE=$CONFIGURATION"
)

echo "Configuring CMake in: $RESOLVED_BUILD_DIR"
cmake "${CONFIGURE_ARGS[@]}"

RESOLVED_TARGET="$TARGET"
if [ "$BUILD_ALL" = true ]; then
  RESOLVED_TARGET="all"
fi

echo "Building target: $RESOLVED_TARGET ($CONFIGURATION)"
cmake --build "$RESOLVED_BUILD_DIR" --target "$RESOLVED_TARGET" --parallel

EXE_PATH=""
FOR_PATHS=(
  "$RESOLVED_BUILD_DIR/qwen35x"
  "$RESOLVED_BUILD_DIR/bin/qwen35x"
)

for p in "${FOR_PATHS[@]}"; do
  if [ -f "$p" ]; then
    EXE_PATH="$p"
    break
  fi
done

echo "Build success."
if [ -n "$EXE_PATH" ]; then
  echo "Executable: $EXE_PATH"
else
  echo "Executable not found in expected paths."
fi

if [ "$RUN_SMOKE_TEST" = true ]; then
  if [ -z "$EXE_PATH" ]; then
    echo "RunSmokeTest requested, but qwen35x was not found."
    exit 1
  fi
  if [ ! -f "$RESOLVED_PROFILE" ]; then
    echo "RunSmokeTest requested, profile not found: $RESOLVED_PROFILE"
    exit 1
  fi

  echo "Running smoke test..."
  "$EXE_PATH" --profile "$RESOLVED_PROFILE" --sm "$SM_VERSION"
  echo "Smoke test passed."
fi
