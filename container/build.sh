#!/usr/bin/env bash
# Stage the build context (engine fork + these container files) and build the image.
# Usage: ./container/build.sh [tag]
set -euo pipefail

TAG="${1:-qwen38-flash-next-exl3:2.50bpw}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # ~/exl3-flash-next
ENGINE="${ENGINE:-$HOME/exl3-qwen38-dflash2/exllamav3}"
CTX="$ROOT/container/.buildctx"

rm -rf "$CTX"
mkdir -p "$CTX"
cp -r "$ENGINE/exllamav3" "$CTX/exllamav3"
cp -r "$ENGINE/examples" "$CTX/examples"
cp "$ROOT/container/entrypoint.sh" "$CTX/entrypoint.sh"
cp "$ROOT/container/Dockerfile" "$CTX/Dockerfile"

docker build -t "$TAG" -f "$CTX/Dockerfile" "$CTX"
echo "built $TAG"

# Smoke (CPU-side): the image imports + the entrypoint refuses cleanly without a model
docker run --rm --entrypoint python "$TAG" -c "import torch, exllamav3; print('import ok')"
