# syntax=docker/dockerfile:1
# Runtime for Qwen3.8-Flash-Next EXL3 (2.50 bpw, MTP + vision) on a single
# 24 GB Ampere-class GPU (developed/validated on RTX 3090 with MoE CPU offload).
#
# Build context (see build.sh — it stages this):
#   Dockerfile (this file), entrypoint.sh, exllamav3/, examples/
#
# Build:
#   ./container/build.sh
#
# Click-run (model auto-downloads to the named volume on first start):
#   docker run --gpus all -v flash-next-models:/models \
#     ghcr.io/r0b0tlab/qwen38-flash-next-exl3:2.50bpw
# With local models:
#   docker run -it --rm --gpus all -v /path/to/models:/models:ro qwen38-flash-next-exl3:2.50bpw
# One-shot:
#   docker run --rm --gpus all -v /path/to/models:/models:ro \
#     -e PROMPT="Explain photosynthesis in one sentence." qwen38-flash-next-exl3:2.50bpw
#
# Runtime-only image: no compiler toolchain, no CUDA toolkit.

FROM ghcr.io/astral-sh/uv:0.9.15 AS uv

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /opt

# Pinned runtime environment (cu128 keeps torch matched to the base image and to
# the exllamav3 wheel). Managed CPython under /opt so the non-root user can read it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON_PREFERENCE=only-managed
RUN uv venv --python 3.13 /opt/venv \
    && for i in 1 2 3 4; do \
        uv pip install --python /opt/venv/bin/python \
            --index-url https://download.pytorch.org/whl/cu128 \
            "torch==2.10.0" \
        && uv pip install --python /opt/venv/bin/python \
            "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.0/exllamav3-1.5.0+cu128.torch2.10.0-cp313-cp313-linux_x86_64.whl" \
            safetensors rich blessed prompt_toolkit pyperclip requests pydantic \
        && break || { echo "install attempt $i failed; retrying"; sleep 15; }; \
    done \
    && uv cache clean

RUN uv pip install --python /opt/venv/bin/python huggingface_hub hf_transfer \
    && uv cache clean

# Source overlay from the engine fork (mul1 codebook + MoE CPU offload paths intact;
# additive over the wheel — the compiled ext .so from the wheel is preserved).
COPY exllamav3/ /opt/venv/lib/python3.13/site-packages/exllamav3/
COPY examples/ /opt/exllamav3/examples/
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

RUN useradd --create-home --uid 10001 runner \
    && mkdir -p /models && chown runner:runner /models
USER runner

ENV PATH="/opt/venv/bin:${PATH}" \
    MODEL_DIR=/models/qwen38-flash-next-exl3 \
    MODE=chatml \
    CTX=262144 \
    CQ=3 \
    MCS=320 \
    MCT=6 \
    MTP=1 \
    HF_HOME=/models/.hf-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN python -V && python -c "import torch, exllamav3; print('smoke ok:', torch.__version__)"

ENTRYPOINT ["/opt/entrypoint.sh"]
