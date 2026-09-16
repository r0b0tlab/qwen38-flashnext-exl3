# Container — Qwen3.8-Flash-Next EXL3 2.50 bpw (RTX 3090)

Runtime image for the Flash-Next EXL3 quantization with MoE CPU offload (`-mcs/-mct`),
MTP speculative decode, and the vision tower. Based on the same pinned stack as the
Qwen3.8-27B campaign image (CUDA 12.8.1 runtime, torch 2.10.0+cu128, ExLlamaV3 v1.5.0
wheel + source overlay from the `dflash2-pathway` engine fork).

## Build

```bash
./container/build.sh            # stages .buildctx (engine + these files) and builds
```

## Run (click-run)

```bash
docker run --gpus all -v flash-next-models:/models \
  ghcr.io/r0b0tlab/qwen38-flash-next-exl3:2.50bpw
```

First start downloads the model repo into the named volume. Subsequent starts reuse it.

## Flags (env)

| env | default | meaning |
| --- | --- | --- |
| `MCS` | `320` | experts per layer left on CPU (tail N) |
| `MCT` | `6` | CPU worker threads (6-core Zen 5 host) |
| `MTP` | `1` | MTP speculative decode on |
| `CTX` | `262144` | context (native; no rope scaling) |
| `CQ` | `3` | KV cache quant bits |
| `PROMPT` | — | one-shot prompt (exits after one turn) |
| `EXTRA_ARGS` | — | extra chat.py flags, e.g. `-basic -tps` |

## Notes

- GPU passthrough requires the NVIDIA container runtime on the host; the development host
  lacked it, so GPU validation here ran the engine directly (see the results repo).
- `chat.py` is an interactive console app (no HTTP server); use the repo's
  `scripts/serve_openai.py` for an OpenAI-compatible endpoint.