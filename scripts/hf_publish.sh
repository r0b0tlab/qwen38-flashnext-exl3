#!/usr/bin/env bash
# Publish the Flash-Next EXL3 model + card to Hugging Face, then verify.
# Usage: ./scripts/hf_publish.sh <model_dir> <hf_repo> <card_file>
set -euo pipefail

MODEL_DIR="${1:?model dir}"
HF_REPO="${2:?hf repo id, e.g. r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw}"
CARD="${3:?card markdown file}"

echo "== create repo =="
hf repo create "$HF_REPO" --repo-type model --public --exist-ok

echo "== upload weights/configs =="
hf upload "$HF_REPO" "$MODEL_DIR" --repo-type model

echo "== upload card =="
hf upload "$HF_REPO" "$CARD" README.md --repo-type model

echo "== verify =="
python3 - "$HF_REPO" <<'EOF'
import json, re, sys, urllib.request
repo = sys.argv[1]
tok = re.search(r'^HF_TOKEN=(.+)$', open('/home/am/.hermes/.env').read(), re.M).group(1).strip()
req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}?blobs=true",
                             headers = {"Authorization": f"Bearer {tok}"})
d = json.load(urllib.request.urlopen(req))
files = [s["rfilename"] for s in d.get("siblings", [])]
card = d.get("cardData") or {}
print("files:", len(files))
print("has ngram_embedding:", any("ngram" in f for f in files))
print("base_model:", card.get("base_model"), "| relation:", card.get("base_model_relation"))
assert any(f.endswith(".safetensors") for f in files), "no safetensors uploaded"
print("VERIFIED")
EOF
