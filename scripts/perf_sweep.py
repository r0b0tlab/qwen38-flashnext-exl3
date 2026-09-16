#!/usr/bin/env python3
"""
Runtime config sweep for EXL3 Qwen3.8-Flash-Next (plan Task 4.2).

Each row loads the model ONCE in its own subprocess (isolated CUDA context), runs
5 fixed prompts greedily (first prompt ~256 tokens for a real TTFT), and reports
decode tok/s + TTFT + VRAM. Results append to notes/perf.json.

Rows (per plan v6):
  A: mcs 295, mct 6,  MTP on,  cq3   # max hot-cache
  B: mcs 320, mct 6,  MTP on,  cq3   # start preset (comfort)
  C: mcs 320, mct 12, MTP on,  cq3   # SMT vs physical
  D: mcs 360, mct 6,  MTP on,  cq3   # deeper tail
  E: mcs 320, mct 6,  MTP off, cq3   # MTP multiplier measurement
  F: mcs 320, mct 6,  MTP on,  cq4   # KV-vs-experts trade

Usage:
  python3 scripts/perf_sweep.py --target models/... --rows B,E
  python3 scripts/perf_sweep.py --target ... --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

ROWS = {
    "A": {"mcs": 295, "mct": 6,  "mtp": True,  "cq": 3},
    "B": {"mcs": 320, "mct": 6,  "mtp": True,  "cq": 3},
    "C": {"mcs": 320, "mct": 12, "mtp": True,  "cq": 3},
    "D": {"mcs": 360, "mct": 6,  "mtp": True,  "cq": 3},
    "E": {"mcs": 320, "mct": 6,  "mtp": False, "cq": 3},
    "F": {"mcs": 320, "mct": 6,  "mtp": True,  "cq": 4},
}

BASE_ENV = {
    "EXL3_MOE_CPU_PIN": "1",
    "EXL3_MOE_CPU_SWIZZLE": "1",
    "EXL3_MOE_FUSED_DET": "1",
}

UNIT = " The archival record contains neutral observations about weather, tools, books, roads, and ordinary daily events."
PROMPTS = [
    ("What is 17*23? Show the multiplication steps.", 128),
    ("Explain in detail how a refrigerator works.", 128),
    ("Write a short story about a lighthouse keeper.", 128),
    ("Summarize the main causes of the French Revolution.", 128),
    ("Write a Python function that reverses a singly linked list.", 128),
]


def long_prompt(tokenizer_target, approx_tokens = 256):
    """Build a ~256-token contextual prompt with a question at the end."""
    filler = UNIT * (approx_tokens // 18 + 2)
    return (f"user\nContext:{filler}\nQuestion: In one sentence, what kind of record is this?\nassistant\n")


def worker(args):
    os.environ.update(BASE_ENV)
    for kv in args.env_extra or []:
        k, v = kv.split("=", 1)
        os.environ[k] = v

    sys.path.insert(0, HERE)
    ENGINE = os.path.expanduser("~/exl3-qwen38-dflash2/exllamav3")
    sys.path.insert(0, ENGINE if os.path.isdir(os.path.join(ENGINE, "exllamav3")) else os.path.join(HERE, "..", "exllamav3"))

    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator
    from exllamav3.cache import CacheLayer_quant
    from exllamav3.generator.sampler.presets import ArgmaxSampler
    from serve_openai import apply_offload_config

    tcfg = Config.from_directory(args.target)
    oa = type("OA", (), {"moe_cpu_split": args.mcs, "moe_cpu_threads": args.mct})()
    tcfg = apply_offload_config(tcfg, oa)

    draft_model = None
    if args.mtp:
        draft_model = Model.from_config(tcfg, component = "mtp")
    max_history = draft_model.caps.get("default_draft_size", 4) if draft_model else 0

    model = Model.from_config(tcfg)
    cache = Cache(model, max_num_tokens = args.ctx, layer_type = CacheLayer_quant,
                  k_bits = args.cq, v_bits = args.cq, max_history = max_history, max_batch_size = 1)
    t0 = time.time()
    model.load(progressbar = False)
    load_s = time.time() - t0
    tokenizer = Tokenizer.from_config(tcfg)

    gen_kwargs = {}
    if draft_model is not None:
        draft_cache = Cache(draft_model, max_num_tokens = args.ctx, layer_type = CacheLayer_quant,
                            k_bits = args.cq, v_bits = args.cq, max_batch_size = 1)
        draft_model.load(progressbar = False)
        if args.ndt:
            gen_kwargs["num_draft_tokens"] = args.ndt
        gen_kwargs.update({"draft_model": draft_model, "draft_cache": draft_cache})

    gen = Generator(model, cache, tokenizer, **gen_kwargs)
    sampler = ArgmaxSampler()
    eos_ids = list(getattr(model.config, "eos_token_id_list", None) or [])

    results = []

    # TTFT leg: ~256-token prompt, 8 tokens out
    lp = long_prompt(tokenizer)
    t0 = time.time()
    ttft = None
    _, r = gen.generate(prompt = lp, max_new_tokens = 8, sampler = sampler,
                        stop_conditions = eos_ids, return_last_results = True)  # noqa: F841
    ttft = time.time() - t0

    for p, ntok in PROMPTS:
        t0 = time.time()
        _, r = gen.generate(prompt = f"user\n{p}\nassistant\n", max_new_tokens = ntok,
                            sampler = sampler, stop_conditions = eos_ids, return_last_results = True)
        dt = time.time() - t0
        r = r or {}
        nt = r.get("new_tokens", 0)
        results.append({"prompt": p[:40], "new_tokens": nt, "seconds": round(dt, 2),
                        "tok_per_s": round(nt / dt, 2) if nt and dt else 0.0})
        print(f"[sweep] {p[:30]!r} -> {nt} tok, {results[-1]['tok_per_s']} tok/s", flush = True)

    tps = [x["tok_per_s"] for x in results if x["tok_per_s"] > 0]
    free, total = torch.cuda.mem_get_info()
    row = {
        "row": args.row,
        "config": {"mcs": args.mcs, "mct": args.mct, "mtp": args.mtp, "cq": args.cq, "ndt": args.ndt},
        "load_s": round(load_s, 1),
        "ttft_256tok_s": round(ttft, 3),
        "mean_tok_per_s": round(sum(tps) / max(1, len(tps)), 2),
        "vram_used_gb": round((total - free) / 1e9, 2),
        "results": results,
    }
    print("PERF_ROW_JSON:" + json.dumps(row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default = None)
    ap.add_argument("--rows", default = "A,B,C,D,E,F")
    ap.add_argument("--dry-run", action = "store_true")
    ap.add_argument("--json-out", default = None)
    ap.add_argument("--ctx", type = int, default = 4096)
    ap.add_argument("--ndt", type = int, default = None)
    # worker-mode args
    ap.add_argument("--worker", action = "store_true")
    ap.add_argument("--row", default = None)
    ap.add_argument("--mcs", type = int, default = 320)
    ap.add_argument("--mct", type = int, default = 6)
    ap.add_argument("--mtp", action = "store_true")
    ap.add_argument("--cq", type = int, default = 3)
    ap.add_argument("--env-extra", action = "append", default = None)
    args = ap.parse_args()

    if args.dry_run:
        for letter in args.rows.split(","):
            cfg = ROWS[letter.strip()]
            print(f"{letter}: mcs {cfg['mcs']}, mct {cfg['mct']}, mtp {cfg['mtp']}, cq{cfg['cq']}, env {BASE_ENV}")
        return 0

    if args.worker:
        worker(args)
        return 0

    assert args.target, "--target required"
    json_out = args.json_out or os.path.join(HERE, "..", "notes", "perf.json")
    json_out = os.path.abspath(json_out)
    rows = [r.strip() for r in args.rows.split(",") if r.strip()]

    all_rows = []
    if os.path.exists(json_out):
        try:
            all_rows = json.load(open(json_out))
        except Exception:
            all_rows = []

    for letter in rows:
        cfg = ROWS[letter]
        env = {k: v for k, v in os.environ.items()}
        env.update(BASE_ENV)
        cmd = [sys.executable, __file__, "--worker", "--row", letter, "--target", args.target,
               "--mcs", str(cfg["mcs"]), "--mct", str(cfg["mct"]), "--cq", str(cfg["cq"]),
               "--ctx", str(args.ctx)]
        if cfg["mtp"]:
            cmd.append("--mtp")
        if args.ndt:
            cmd += ["--ndt", str(args.ndt)]
        print(f"=== row {letter}: {cfg} ===", flush = True)
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output = True, text = True, env = env)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr[-2000:] if proc.stderr else "")
        row_json = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("PERF_ROW_JSON:"):
                row_json = json.loads(line.split(":", 1)[1])
        if row_json is None:
            row_json = {"row": letter, "config": dict(cfg), "error": f"exit={proc.returncode}"}
        row_json["wall_s"] = round(time.time() - t0, 1)
        all_rows = [r for r in all_rows if r.get("row") != letter] + [row_json]
        with open(json_out, "w") as f:
            json.dump(all_rows, f, indent = 2)
        print(f"=== row {letter} done in {row_json['wall_s']}s: {row_json.get('mean_tok_per_s', 'ERR')} tok/s ===", flush = True)

    print(json.dumps(all_rows, indent = 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
