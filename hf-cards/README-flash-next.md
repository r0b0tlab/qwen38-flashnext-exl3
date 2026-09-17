---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
- exl3
- exllamav3
- quantization
- 2-bit
- moe
- mtp
- rtx3090
---



# Qwen3.8-Flash-Next — EXL3 2.50 bpw

EXL3 (ExLlamaV3) quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
tuned for a 24 GB RTX 3090 at the model's full 262,144-token native context with the MTP
head and the vision tower intact.

Conversion: ExLlamaV3 v1.5.0, single pass with
`convert.py -b 2.50 -mb 4 -vb 6 -hq -ngb 3` — 2.50 bpw for the routed experts and
decoder, higher-quality attention/shared modules (`-hq`), vision tower 6 bpw, MTP head
4 bpw, n-gram embedding table 3 bpw (streamed from disk at runtime).

## Resource requirements (RTX 3090, 24 GB)

- Weights on disk: **61 GiB**.
- VRAM at the full 262,144-token context (cq3 cache, MTP on): **20.0-20.7 GB**
  peak, staying ≥ 1.5 GiB below the card's ceiling by design.
- MoE expert offload: run with `-mcs 320 -mct 6` — ~30 GiB of expert weights stay in
  system RAM (59 GB host), experts stream per token over the CPU path; 262k context,
  one sequence slot.
- KV cache (12 full-attention layers) at 262,144 tokens: cq3 ~2.25 GiB (validated).
- The n-gram table streams from disk; no RAM-offload needed (`-ngr` not used).

## Use

- **ExLlamaV3** (v1.5.0-compatible build with MoE CPU-offload support):

  ```bash
  python examples/chat.py -m <this-dir> -mode chatml \
    -cs 262144 -cq 3 -mcs 320 -mct 6 -mtp
  ```

- **Container (click-run)** — auto-downloads this repo on first run:

  ```bash
  docker run --gpus all -v flash-next-models:/models \
    ghcr.io/r0b0tlab/qwen38-flash-next-exl3:2.50bpw
  ```

## Validation (single RTX 3090, 24 GB)

- Decode throughput (MTP on): **38.6 tok/s** (256-token TTFT
  1.9 s); MTP acceptance **4.06**.
- 262,144-token load and 200k-token prefill: **664 tok/s**,
  no OOM.
- Multi-needle NIAH at 262,080 tokens: **PASS at both 33/66% and 33/66/90% (262,080 tokens)**.
- Q200v2 text-180 (frozen kit): **173 correct / 6 incorrect / 1 ungraded (ifeval-023 disclosed at the 8,192-token ceiling); gsm8k 98.75% | hard_reasoning 20/20 | humaneval 100% | ifeval 87.18%; e2e mean 51.2 tok/s, 20.4 GiB VRAM mean**.

Full metrics, harnesses and the served model: https://github.com/r0b0tlab/qwen38-flashnext-exl3