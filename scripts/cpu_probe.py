#!/usr/bin/env python3
"""
CPU expert-path probe (plan Task 4.1): loads the model with ALL experts on CPU
(--moe-cpu-split 512), MTP off, runs fixed prompts greedily, and reports decode
tok/s plus the implied cold-path bandwidth (calibrates the perf model).

Usage:
  python3 scripts/cpu_probe.py --target models/qwen38-flash-next-exl3-b250 \
      --moe-cpu-threads 6 --json-out notes/cpu-probe.json
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ENGINE = os.path.expanduser("~/exl3-qwen38-dflash2/exllamav3")
if os.path.isdir(os.path.join(ENGINE, "exllamav3")):
    sys.path.insert(0, ENGINE)
else:
    sys.path.insert(0, os.path.join(HERE, "..", "exllamav3"))

import torch  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator  # noqa: E402
from serve_openai import apply_offload_config  # noqa: E402

PROMPTS = [
    "Explain in detail how a refrigerator works.",
    "What is 17*23? Show the multiplication steps.",
    "Write a short story about a lighthouse keeper.",
    "Summarize the main causes of the French Revolution.",
    "Write a Python function that reverses a singly linked list.",
]

PER_EXPERT_B = 4.9152e6
BPW = 2.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required = True)
    ap.add_argument("--moe-cpu-split", type = int, default = 512)
    ap.add_argument("--moe-cpu-threads", type = int, default = 6)
    ap.add_argument("--ctx", type = int, default = 4096)
    ap.add_argument("--max-new-tokens", type = int, default = 128)
    ap.add_argument("--json-out", default = None)
    args = ap.parse_args()

    tcfg = Config.from_directory(args.target)
    tcfg = apply_offload_config(tcfg, args)

    model = Model.from_config(tcfg)
    cache = Cache(model, max_num_tokens = args.ctx, max_batch_size = 1)
    model.load(progressbar = False)
    tokenizer = Tokenizer.from_config(tcfg)
    gen = Generator(model, cache, tokenizer)

    from exllamav3.generator.sampler.presets import ArgmaxSampler
    sampler = ArgmaxSampler()
    eos_ids = list(getattr(model.config, "eos_token_id_list", None) or [])

    rows = []
    for p in PROMPTS:
        t0 = time.time()
        _, r = gen.generate(prompt = f"user\n{p}\nassistant\n", max_new_tokens = args.max_new_tokens,
                            sampler = sampler, stop_conditions = eos_ids, return_last_results = True)
        dt = time.time() - t0
        r = r or {}
        nt = r.get("new_tokens", 0)
        rows.append({"prompt": p[:40], "new_tokens": nt, "tok_per_s": nt / dt if nt and dt else 0.0})
        print(f"[probe] {nt} tok in {dt:.1f}s -> {rows[-1]['tok_per_s']:.1f} tok/s", flush=True)

    tps = [x["tok_per_s"] for x in rows if x["tok_per_s"] > 0]
    mean_tps = sum(tps) / max(1, len(tps))
    cold_bw = 48 * 10 * PER_EXPERT_B * BPW / 8 * mean_tps / 1e9
    out = {
        "moe_cpu_split": args.moe_cpu_split,
        "moe_cpu_threads": args.moe_cpu_threads,
        "mean_tok_per_s": round(mean_tps, 2),
        "implied_cold_bw_gb_s": round(cold_bw, 2),
        "rows": rows,
    }
    print(json.dumps(out, indent = 2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent = 2)


if __name__ == "__main__":
    main()
