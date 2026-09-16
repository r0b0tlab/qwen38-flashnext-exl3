#!/usr/bin/env python3
"""
Long-context validation for EXL3 Qwen3.8-Flash-Next: build a ~200k-token prompt
from a public-domain text, prefill at full context with cache quant + MoE CPU
offload (+ optional MTP), decode a short continuation, and print prefill tok/s,
decode tok/s, acceptance, and VRAM.
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ENGINE = os.path.expanduser("~/exl3-qwen38-dflash2/exllamav3")
if os.path.isdir(os.path.join(ENGINE, "exllamav3")):
    sys.path.insert(0, ENGINE)
else:
    sys.path.insert(0, os.path.join(HERE, "..", "exllamav3"))

import torch  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job  # noqa: E402
from exllamav3.cache import CacheLayer_quant  # noqa: E402
from serve_openai import apply_offload_config  # noqa: E402

GUTENBERG_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required = True)
    ap.add_argument("--draft", default = "none", help = "'mtp' or 'none'")
    ap.add_argument("--ctx", type = int, default = 262144)
    ap.add_argument("--cq", default = "3")
    ap.add_argument("--prompt-tokens", type = int, default = 200_000)
    ap.add_argument("--decode-tokens", type = int, default = 200)
    ap.add_argument("--moe-cpu-split", type = int, default = 320)
    ap.add_argument("--moe-cpu-threads", type = int, default = 6)
    ap.add_argument("--draft-tokens", type = int, default = None,
                    help = "Override the draft window width (num_draft_tokens)")
    ap.add_argument("--json-out", default = None)
    args = ap.parse_args()

    tcfg = Config.from_directory(args.target)
    tcfg = apply_offload_config(tcfg, args)

    draft_model = None
    if args.draft == "mtp":
        draft_model = Model.from_config(tcfg, component = "mtp")
    elif args.draft != "none":
        raise SystemExit("flash-next: --draft must be 'mtp' or 'none'")
    max_history = draft_model.caps.get("default_draft_size", 4) if draft_model else 0

    model = Model.from_config(tcfg)
    k_bits = v_bits = int(args.cq)
    cache = Cache(model, max_num_tokens = args.ctx, layer_type = CacheLayer_quant,
                  k_bits = k_bits, v_bits = v_bits, max_history = max_history, max_batch_size = 1)
    model.load(progressbar = False)
    tokenizer = Tokenizer.from_config(tcfg)

    gen_kwargs = {}
    if draft_model is not None:
        draft_cache = Cache(draft_model, max_num_tokens = args.ctx, layer_type = CacheLayer_quant,
                            k_bits = k_bits, v_bits = v_bits, max_batch_size = 1)
        draft_model.load(progressbar = False)
        gen_kwargs = {"draft_model": draft_model, "draft_cache": draft_cache}
        if args.draft_tokens:
            gen_kwargs["num_draft_tokens"] = args.draft_tokens

    gen = Generator(model, cache, tokenizer, **gen_kwargs)

    # reuse the 27B campaign's copy if present, else fetch
    text_path = os.path.expanduser("~/exl3-flash-next/work/longctx.txt")
    prev = os.path.expanduser("~/exl3-qwen38-dflash2/work/longctx.txt")
    if not os.path.exists(text_path):
        os.makedirs(os.path.dirname(text_path), exist_ok = True)
        if os.path.exists(prev):
            shutil.copyfile(prev, text_path)
        else:
            urllib.request.urlretrieve(GUTENBERG_URL, text_path)
    with open(text_path, "r", encoding = "utf-8", errors = "ignore") as f:
        book = f.read()

    ids = tokenizer.encode(book, encode_special_tokens = False)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    prompt_ids = ids[:, : args.prompt_tokens].to(torch.long)
    print(f"prompt tokens: {prompt_ids.shape[1]} (ctx {args.ctx})", flush=True)

    eos_ids = list(getattr(model.config, "eos_token_id_list", None) or [])
    job = Job(
        input_ids = prompt_ids,
        max_new_tokens = args.decode_tokens,
        stop_conditions = eos_ids,
    )
    gen.enqueue(job)

    t0 = time.time()
    first_token_time = None
    result = None
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and first_token_time is None:
                first_token_time = time.time()
            if r.get("eos"):
                result = r
    total_s = time.time() - t0

    free, total = torch.cuda.mem_get_info()
    r = result or {}
    prefill_s = (first_token_time - t0) if first_token_time else None
    decode_s = (total_s - prefill_s) if prefill_s else None
    new_tokens = r.get("new_tokens", 0)
    accepted = r.get("accepted_draft_tokens", 0)
    out = {
        "prompt_tokens": prompt_ids.shape[1],
        "new_tokens": new_tokens,
        "prefill_seconds": round(prefill_s, 1) if prefill_s else None,
        "prefill_tok_per_s": round(prompt_ids.shape[1] / prefill_s, 1) if prefill_s else None,
        "decode_seconds": round(decode_s, 2) if decode_s else None,
        "decode_tok_per_s": round(new_tokens / decode_s, 2) if decode_s else None,
        "acceptance_length": round(new_tokens / max(1, new_tokens - accepted), 3) if new_tokens else None,
        "vram_used_gb": round((total - free) / 1e9, 2),
    }
    print(json.dumps(out, indent = 2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent = 2)

    tail = (result or {}).get("text", "")
    if isinstance(tail, str) and tail:
        print("continuation tail:", repr(tail[-200:]))


if __name__ == "__main__":
    main()
