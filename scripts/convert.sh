#!/usr/bin/env bash
# THE conversion — Qwen3.8-Flash-Next -> EXL3 2.50 bpw, single quant (plan v6).
# Run from anywhere; logs to notes/convert.log; resumable with `-r`.
set -euo pipefail

WS="/home/am/exl3-flash-next"
ENGINE="${ENGINE:-$HOME/exl3-qwen38-dflash2/exllamav3}"
SRC="$WS/models/qwen38-flash-next-hf"
OUT="$WS/models/qwen38-flash-next-exl3-b250"
WORK="$WS/work/target-b250"
LOG="$WS/notes/convert.log"

RESUME=0
if [ "${1:-}" = "-r" ]; then RESUME=1; fi

cd "$ENGINE"

if [ "$RESUME" = "1" ]; then
    python3 convert.py -w "$WORK" -r >> "$LOG" 2>&1
else
    python3 convert.py \
        -i "$SRC" \
        -o "$OUT" \
        -w "$WORK" \
        -b 2.50 -mb 4 -vb 6 -hq -ngb 3 -cr 250 \
        >> "$LOG" 2>&1
fi
echo "convert exit=$?" >> "$LOG"