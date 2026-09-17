#!/usr/bin/env python3
"""
Vision smoke for the EXL3 Qwen3.8-Flash-Next quant (component='vision' path,
mirrors examples/multimodal.py mechanics with CLI flags + MoE CPU offload).

Usage:
  python3 scripts/vision_smoke.py --target models/qwen38-flash-next-exl3-b250 \
      --img /path/to/image.png --moe-cpu-split 320 --moe-cpu-threads 6
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
    sys.path.insert(0, os.path.join(ENGINE, "examples"))
else:
    sys.path.insert(0, os.path.join(HERE, "..", "exllamav3"))

from PIL import Image  # noqa: E402

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job  # noqa: E402
from common import format_prompt, get_stop_conditions  # noqa: E402
from serve_openai import apply_offload_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required = True)
    ap.add_argument("--img", required = True)
    ap.add_argument("--prompt", default = "Describe this image in one sentence.")
    ap.add_argument("--system", default = "You are a helpful assistant.")
    ap.add_argument("--prompt-format", default = "chatml")
    ap.add_argument("--ctx", type = int, default = 8192)
    ap.add_argument("--max-new-tokens", type = int, default = 128)
    ap.add_argument("--moe-cpu-split", type = int, default = 320)
    ap.add_argument("--moe-cpu-threads", type = int, default = 6)
    ap.add_argument("--json-out", default = None)
    args = ap.parse_args()

    config = Config.from_directory(args.target)
    config = apply_offload_config(config, args)

    vision_model = Model.from_config(config, component = "vision")
    vision_model.load(progressbar = False)

    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.ctx)
    model.load(progressbar = False)
    tokenizer = Tokenizer.from_config(config)

    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)

    image = Image.open(args.img)
    t0 = time.time()
    ie = vision_model.get_image_embeddings(tokenizer = tokenizer, image = image)

    placeholders = ie.text_alias + "\n"
    prompt = format_prompt(args.prompt_format, args.system, placeholders + args.prompt)

    input_ids = tokenizer.encode(prompt, encode_special_tokens = True, embeddings = [ie])
    job = Job(
        input_ids = input_ids,
        max_new_tokens = args.max_new_tokens,
        decode_special_tokens = True,
        stop_conditions = get_stop_conditions(args.prompt_format, tokenizer),
        embeddings = [ie],
    )
    generator.enqueue(job)
    captured = ""
    streamed = []
    final = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            t = r.get("text")
            if t:
                streamed.append(t)
                print(t, end = "", flush = True)
            if r.get("eos"):
                final = r
    if final:
        captured = final.get("text") or final.get("full_completion") or "".join(streamed)
    else:
        captured = "".join(streamed)
    new_tokens = int((final or {}).get("new_tokens") or 0)
    dt = time.time() - t0

    out = {
        "target": args.target,
        "img": args.img,
        "caption": captured.strip()[:600],
        "elapsed_s": round(dt, 1),
        "new_tokens": new_tokens,
        "moe_cpu_split": args.moe_cpu_split,
    }
    print(json.dumps(out, indent = 2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent = 2)


if __name__ == "__main__":
    main()
