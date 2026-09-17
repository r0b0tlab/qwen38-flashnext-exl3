# Qwen3.8-Flash-Next EXL3 2.50 bpw on the RTX 3090



The published EXL3 quant — [`r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw`](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw)
— tuned for a 24 GB RTX 3090, running at the model's full 262,144-token native context
with the **MTP head (NEXTN)** and the **vision tower** intact, using **MoE expert CPU
offload** to fit the 125B-parameter (6B active) model. Quant of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), licensed per
the parent.

Single-pass conversion: `convert.py -b 2.50 -mb 4 -vb 6 -hq -ngb 3` — 2.50 bpw routed
experts/decoder, higher-rate attention + shared experts (`-hq`), vision 6 bpw, MTP 4 bpw,
n-gram table 3 bpw (streamed from disk at runtime).

## Results

| | autoregressive | MTP (this work) |
| --- | --- | --- |
| Acceptance length (GSM8K, greedy) | 1.00 | 4.06 |
| Decode tok/s (RTX 3090, mcs 320) | 27.8 | **38.6** |
| 256-token TTFT | 1.9 s | |

At 200k-token depth: prefill 664 tok/s, decode
20.9 tok/s. Full numbers and raw JSON: `notes/PERF.md`,
`metrics/`.

## Resource requirements (RTX 3090, 24 GB + 59 GB host RAM)

**On disk:**

| artifact | size |
| --- | --- |
| EXL3 2.50 bpw (vision 6, MTP 4, n-gram table 3) | 61 GiB |
| BF16 source (needed for conversion only) | 335.3 GiB |

**VRAM at 262,144-token context (cq3 cache, one sequence, MTP on):**

| state | usage |
| --- | --- |
| loaded and idle (torch) | 20.0 GB |
| peak during long-context requests | 20.7 GB (≥ 1.5 GiB margin kept) |

Budget math behind those numbers:

- routed experts: 192 (128 at the full-context mcs 384 preset) of 512 per layer reside in VRAM; the tail
  `-mcs 320` streams from system RAM through the CPU worker (dynamic hot/cold placement on)
- KV cache (12 full-attention layers, 2 kv heads × 256 dim × 2) at 262,144 tokens:
  cq3 ≈ 2.25 GiB — the validated setting
- n-gram embedding table: streamed from NVMe per token; RAM-offload not used (infeasible
  at this expert split and not measurably beneficial)
- CPU expert tail RAM: ≈ 22 (26 at mcs 384) GiB (+ ~2 GiB ring buffers); 6-core
  Zen 5 host, `-mct 6`

## Layout

- `scripts/` — budget gates, OpenAI-compatible serve (`serve_openai.py`), acceptance /
  long-context / NIAH harnesses, CPU probe, config sweep, digests, eval runner.
- `notes/` — BUDGET / PERF / RESULTS notes and pipeline logs.
- `container/` — runtime image (click-run; auto-downloads the HF repo on first start).
- `hf-cards/` — the published HuggingFace card + publish script.
- `metrics/` — Q200v2 summary + throughput/telemetry digests + NIAH JSONs.
- Weights (HuggingFace): [r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw](https://huggingface.co/r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw)
  — what the container pulls on first run.

## Reproduce

```bash
# Engine (v1.5.0-compatible build with MoE CPU-offload support):
git clone https://github.com/r0b0tlab/exllamav3 exllamav3

# Convert (single pass; ~hours; requires the 335 GiB BF16 source):
cd exllamav3 && python convert.py \
  -i ../models/qwen38-flash-next-hf -o ../models/qwen38-flash-next-exl3-b250 \
  -w ../work/target-b250 -b 2.50 -mb 4 -vb 6 -hq -ngb 3 -cr 250
```

## Quickstart (host)

```bash
# Interactive chat at full context with MTP + MoE offload:
python examples/chat.py -m ../models/qwen38-flash-next-exl3-b250 -mode chatml \
  -cs 262144 -cq 3 -mcs 320 -mct 6 -mtp

# OpenAI-compatible serve (used by the benchmarks here):
python scripts/serve_openai.py --target ../models/qwen38-flash-next-exl3-b250 \
  --moe-cpu-split 320 --moe-cpu-threads 6 --mtp --port 8890
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
placement migrates repeatedly-used experts to VRAM. `-mct` sets the worker threads.

## Validation gates

| gate | where | result |
| --- | --- | --- |
| Budget gates (disk/VRAM/RAM) | `scripts/util_budget.py` | FITS / mcs ≥ 321 / OK |
| No-YaRN native context | config assert + NIAH | PASS — 262,144 native, no rope scaling |
| MTP acceptance | `scripts/acceptance_check.py` | PASS - acceptance 4.06 (GSM8K greedy) |
| Vision | examples/multimodal.py | PASS - screenshot described correctly |
| 262k load + 200k prefill | `scripts/long_context_check.py` | PASS - 175k-token prefill 664 tok/s, decode 20.9 tok/s, 20.7 GB peak |
| Multi-needle NIAH 262,080 | `scripts/niah_multikey.py` | PASS x2 (33/66% and 33/66/90% at 262,080 tokens) |
| Q200v2 text-180 (frozen kit) | `scripts/run_q200v2.sh` | PASS - 173/6/1; floors met (gsm8k 98.75%, humaneval 100%) |

## Licenses

- Engine: ExLlamaV3 MIT (this repository's original code: MIT, see `LICENSE`).
- Weights: Qwen3.8-Flash-Next under the Qwen license.