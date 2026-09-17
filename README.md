# Qwen3.8-Flash-Next EXL3 2.50 bpw on the RTX 3090

The published EXL3 quant — [`r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw)
— tuned for a 24 GB RTX 3090, running at the model's full 262,144-token native context
with the **MTP head (NEXTN)** and the **vision tower** intact, using **MoE expert CPU
offload** to fit the 125B model (6B active, plus a 51B-parameter n-gram table). Quant of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), licensed per
the parent.

Single-pass conversion: `convert.py -b 2.50 -mb 4 -vb 6 -hq -ngb 3` — 2.50 bpw routed
experts/decoder, higher-rate attention + shared experts (`-hq`), vision 6 bpw, MTP 4 bpw,
n-gram table 3 bpw (streamed from disk at runtime).

## Results

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
| NIAH 262,080 tokens (2 needles, 33/66 %) | PASS |
| NIAH 262,080 tokens (3 needles, 33/66/90 %) | PASS |

Quality (frozen Q200v2 kit): **173 correct / 6 incorrect / 1 ungraded** (179/180
transported; `ifeval-023` disclosed at the 8,192-token ceiling) — gsm8k 98.75 %,
**hard_reasoning 20/20**, humaneval 100 %, ifeval 87.18 %. Raw JSON: `metrics/`.

## Resource requirements (RTX 3090, 24 GB + 59 GB host RAM)

**On disk:**

| artifact | size |
| --- | --- |
| EXL3 2.50 bpw weights (6 shards) | 41.5 GiB |
| n-gram embedding table (3 bpw) | 18.5 GiB |
| **published artifact — total** | **61 GiB (64.6 GB)** |
| container image | 19.5 GB on disk / 6.8 GB compressed |
| BF16 source (conversion only) | 335.3 GiB / 131 shards |
| peak free disk needed to convert | ~468 GiB (source + output + work + safety) |

**VRAM (measured, one sequence, MTP on):**

| preset | cache | loaded | peak observed |
| --- | --- | --- | --- |
| full 262,144-token context (`-mcs 384`) | 270,336 tok, cq3 | 20.0 GB | 20.7 GB (175k prefill + decode) |
| short-ctx max-throughput (`-mcs 320`) | ≤ 32k tok, cq3 | 21.3 GB | 22.4 GB (5-prompt sweep) |

Both presets hold ≥ 1.6 GiB below the 24 GB ceiling.

**KV cache @ 262,144 tokens** (12 full-attention layers; 2 kv heads × 256 dim × 2 (K+V)
= 12,288 values/token ≈ 3.22 B values):

| precision | size |
| --- | --- |
| fp16 | 6.00 GiB |
| 8-bit | 3.00 GiB |
| 6-bit | 2.25 GiB |
| 4-bit | 1.50 GiB |
| **3-bit (cq3 — validated)** | **1.13 GiB** |

**GDN recurrent state** (36 linear-attention layers; 48 value heads × 128 × 128, fp32):

| item | size |
| --- | --- |
| one history row, all 36 layers | ≈ 0.11 GiB |
| with MTP verify window (4 rows) | ≈ 0.45 GiB |

**Host RAM (59 GB):**

| component | size |
| --- | --- |
| CPU expert tail at `-mcs 320` | ≈ 22 GiB + ~2 GiB worker rings |
| CPU expert tail at `-mcs 384` | ≈ 26.5 GiB + ~2 GiB worker rings |
| n-gram table | streamed from NVMe (no RAM allocation; `-ngr` not used) |
| measured min MemAvailable during evals (mcs 336) | 23.6 GiB |

**Measured serving telemetry** (Q200v2 phase, 2 s cadence, 1,024 samples / 2,078 s):
power 193 W mean / 271 W max · temp 47.6 °C mean / 56 °C max · clock 1,814 / 1,980 MHz ·
VRAM 20.4 GiB mean / 20.5 GiB max · no thermal throttling (software power-cap only).

## Layout

- `scripts/` — budget gates, OpenAI-compatible serve (`serve_openai.py`), acceptance /
  long-context / NIAH harnesses, CPU probe, config sweep, digests, eval runner.
- `notes/` — BUDGET / PERF notes, progress log, pipeline logs (convert, smokes, sweeps).
- `container/` — runtime image (click-run; auto-downloads the HF repo on first start).
- `hf-cards/` — the published HuggingFace card + publish script.
- `metrics/` — Q200v2 summary + throughput/telemetry digests + manual evidence + NIAH JSONs.
- Weights (HuggingFace): [r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw).

## Reproduce

```bash
# Engine (v1.5.0-compatible build with MoE CPU-offload support):
git clone -b dflash2-pathway https://github.com/r0b0tlab/exllamav3 exllamav3  # validated checkout

# Convert (single pass; ~hours; requires the 335 GiB BF16 source):
cd exllamav3 && python convert.py \
  -i ../models/qwen38-flash-next-hf -o ../models/qwen38-flash-next-exl3-b250 \
  -w ../work/target-b250 -b 2.50 -mb 4 -vb 6 -hq -ngb 3 -cr 250
```

## Quickstart (host)

```bash
# Interactive chat at full context with MTP + MoE offload:
python examples/chat.py -m ../models/qwen38-flash-next-exl3-b250 -mode chatml \
  -cs 262144 -cq 3 -mcs 384 -mct 6 -mtp

# OpenAI-compatible serve (used by the benchmarks here):
python scripts/serve_openai.py --target ../models/qwen38-flash-next-exl3-b250 \
  --moe-cpu-split 384 --moe-cpu-threads 6 --mtp --cache-tokens 270336 --port 8890
```

## Quickstart (container)

```bash
docker run --gpus all -v flash-next-models:/models \
  ghcr.io/r0b0tlab/qwen38-flash-next-exl3:2.50bpw
```

## How the MoE offload works

`-mcs N` keeps the tail N routed experts of every eligible layer in system RAM (the
experts use the `mul1` codebook, which the CPU path requires — it is the conversion
default). A background worker streams each token's routed experts over pinned buffers,
overlapping the CPU GEMMs with the layer's own GPU expert compute; dynamic hot/cold
placement migrates repeatedly-used experts to VRAM. `-mct` sets the worker threads
(6 = the physical cores here; 12 was measured 50 % slower — SMT contention).

## Validation gates

| gate | where | result |
| --- | --- | --- |
| Budget gates (disk/VRAM/RAM) | `scripts/util_budget.py` | FITS / mcs ≥ 321 / OK |
| No-YaRN native context | config assert + NIAH | PASS — 262,144 native, no rope scaling |
| MTP acceptance | `scripts/acceptance_check.py` | PASS — 4.06 (GSM8K greedy) |
| Vision | `scripts/vision_smoke.py` | PASS — screenshot described correctly |
| 262k load + 175k prefill | `scripts/long_context_check.py` | PASS — 664 tok/s prefill, 20.9 tok/s decode, 20.7 GB peak |
| Multi-needle NIAH 262,080 | `scripts/niah_multikey.py` | PASS ×2 (33/66 % and 33/66/90 %) |
| Q200v2 text-180 (frozen kit) | `scripts/run_q200v2.sh` | PASS — 173/6/1; floors met |
| Perf search (A–I) | `scripts/perf_sweep.py` | winner: mcs 320 / mct 6 / MTP / ndt 4 / cq3 — 38.6 tok/s |

## Licenses

- Engine: ExLlamaV3 MIT (this repository's original code: MIT, see `LICENSE`).
- Weights: Qwen3.8-Flash-Next under the Qwen license.
