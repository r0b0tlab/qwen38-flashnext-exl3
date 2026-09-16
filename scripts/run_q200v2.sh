#!/usr/bin/env bash
# Q200v2 text-180 run against the local Flash-Next serve (frozen kit, PYTHONPATH shim).
# Usage: ./scripts/run_q200v2.sh <run-id> [base-url] [model-name]
set -euo pipefail

RUN_ID="${1:?run id}"
BASE_URL="${2:-http://127.0.0.1:8890}"
MODEL="${3:-qwen38-flash-next-exl3}"

DEST="/home/am/r0b0bench-q200v2/runs/${RUN_ID}"
mkdir -p "$DEST"
cd "$DEST"

PYTHONPATH=/home/am/r0b0bench-q200v2 python3 /home/am/r0b0bench/subsets/q200v2/scripts/run_quality_set.py \
    --base-url "$BASE_URL" \
    --run-id "$RUN_ID" \
    --set /home/am/r0b0bench/subsets/q200v2/artifacts/quality-text-180-v2.jsonl \
    --model "$MODEL" \
    --max-tokens 8192 \
    --timeout 3600 \
    --workers 1 \
    --image-id sha256:caf1a95b886e50be2218621a79057ce68897f55497721c1344b3c061bbca1a8e \
    --profile-id local-rtx3090-flashnext-2.50bpw \
    --candidate-id qwen38-flash-next-exl3-2.50bpw \
    --admission-config /home/am/r0b0bench-q200v2/admission.json \
    --chat-template-kwargs '{"enable_thinking": true, "thinking": true, "reasoning_effort": "low"}' \
    > run.log 2>&1

echo "exit=$?"
echo "run dir: $DEST"
