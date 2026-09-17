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
head and the vision tower intact, using MoE expert CPU offload for the 125B model
(6B active, plus a 51B-parameter n-gram table).

Conversion: ExLlamaV3 v1.5.0, single pass with
`convert.py -b 2.50 -mb 4 -vb 6 -hq -ngb 3` — 2.50 bpw for the routed experts and
decoder, higher-quality attention/shared modules (`-hq`), vision tower 6 bpw, MTP head
4 bpw, n-gram embedding table 3 bpw (streamed from disk at runtime).

## Resource requirements (RTX 3090, 24 GB + 59 GB host RAM)

**Disk:**

| artifact | size |
| --- | --- |
| this repo — quantized weights (6 shards) | 41.5 GiB |
| this repo — n-gram embedding table (3 bpw) | 18.5 GiB |
| **this repo — total** | **61 GiB (64.6 GB)** |
| container image (`qwen38-flash-next-exl3:2.50bpw`) | 19.5 GB on disk (6.8 GB compressed) |
| BF16 source (conversion only — not needed to run) | 335.3 GiB / 131 shards |
| peak free space needed to convert (source + output + work dir + safety) | ~468 GiB |

**VRAM (measured on the RTX 3090, one sequence, MTP on):**

| preset | cache | loaded | peak observed |
| --- | --- | --- | --- |
| full 262,144-token context — `-mcs 384` | 270,336 tok, cq3 | 20.0 GB | 20.7 GB (175k-token prefill + decode) |
| short-context max-throughput — `-mcs 320` | ≤ 32k tok, cq3 | 21.3 GB | 22.4 GB (5-prompt sweep) |

Both presets keep ≥ 1.6 GiB below the card's 24 GB ceiling by design.

**KV cache @ 262,144 tokens** (12 full-attention layers; 2 kv heads × 256 dim × 2 (K+V)
= 12,288 values/token ≈ 3.22 B values):

| precision | size |
| --- | --- |
| fp16 | 6.00 GiB |
| 8-bit | 3.00 GiB |
| 6-bit | 2.25 GiB |
| 4-bit | 1.50 GiB |
| **3-bit (cq3 — the validated setting)** | **1.13 GiB** |

**GDN recurrent state** (36 linear-attention layers; 48 value heads × 128 × 128, fp32):

| item | size |
| --- | --- |
| one history row, all 36 layers | ≈ 0.11 GiB |
| with the MTP verify window (4 rows) | ≈ 0.45 GiB |

**Host RAM (59 GB):**

| component | size |
| --- | --- |
| CPU expert tail at `-mcs 320` (tail 320 of 512 experts × 48 layers) | ≈ 22 GiB + ~2 GiB worker rings |
| CPU expert tail at `-mcs 384` (full-context preset) | ≈ 26.5 GiB + ~2 GiB worker rings |
| n-gram table | **streamed from NVMe** — no RAM allocation (`-ngr` not used) |
| measured min MemAvailable during the eval run (mcs 336) | 23.6 GiB |

**Measured serving telemetry** (Q200v2 phase, 2 s cadence, 1,024 samples / 2,078 s):

| metric | mean | max |
| --- | --- | --- |
| board power | 193 W | 271 W |
| GPU temp | 47.6 °C | 56 °C |
| GPU clock | 1,814 MHz | 1,980 MHz |
| VRAM in use | 20.4 GiB | 20.5 GiB |

Throttling: none thermal — only software power-cap entries, as expected on a 3090.

## Use

- **ExLlamaV3** (v1.5.0-compatible build with MoE CPU-offload support):

  ```bash
  # full-context preset (262,144 tokens)
  python examples/chat.py -m <this-dir> -mode chatml \
    -cs 262144 -cq 3 -mcs 384 -mct 6 -mtp
  # short-context max-throughput preset
  python examples/chat.py -m <this-dir> -mode chatml \
    -cs 16384 -cq 3 -mcs 320 -mct 6 -mtp
  ```

- **Container (click-run)** — auto-downloads this repo on first run:

  ```bash
  docker run --gpus all -v flash-next-models:/models \
    ghcr.io/r0b0tlab/qwen38-flash-next-exl3:2.50bpw
  ```

## Results (single RTX 3090, 24 GB)

| metric | value |
| --- | --- |
| decode, MTP on (mcs 320, 5-prompt sweep) | **38.6 tok/s** |
| decode, MTP off (same config) | 27.8 tok/s (+39 % from MTP) |
| TTFT, 256-token prompt | 1.9 s |
| MTP acceptance — GSM8K greedy | 4.06 |
| MTP acceptance — at 175k depth | 1.84 |
| prefill @ 175,000-token prompt | 664 tok/s |
| decode at 175k depth | 20.9 tok/s |
| end-to-end throughput, Q200v2 (n = 180) | mean **51.2** / p50 51.7 / aggregate 52.0 tok/s |
| NIAH 262,080 tokens, 2 needles (33/66 %) | PASS |
| NIAH 262,080 tokens, 3 needles (33/66/90 %) | PASS |

**Quality (frozen Q200v2 kit, 180 rows):** 173 correct / 6 incorrect / 1 ungraded
(179/180 transported; `ifeval-023` disclosed at the 8,192-token ceiling).

| family | result |
| --- | --- |
| gsm8k | 79/80 — 98.75 % |
| hard_reasoning | **20/20** (hash-bound independent review) |
| humaneval | 40/40 — 100 % |
| ifeval | 34/39 — 87.18 % (+1 ceiling-disclosed) |

Full metrics, harnesses, raw JSON and the exact serving commands:
https://github.com/r0b0tlab/qwen38-flashnext-exl3
