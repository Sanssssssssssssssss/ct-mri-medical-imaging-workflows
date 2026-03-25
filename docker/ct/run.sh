#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
IMAGE_NAME="a2mi-ct"

mkdir -p "$PROJECT_ROOT/results"

docker build -f "$SCRIPT_DIR/Dockerfile" -t "$IMAGE_NAME" "$PROJECT_ROOT"
docker run --rm -v "$PROJECT_ROOT/results:/app/results" "$IMAGE_NAME" "$@"
