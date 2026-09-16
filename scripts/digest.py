#!/usr/bin/env python3
"""
Digest generator (mirrors the 27B campaign's formats).

Inputs in a run dir:
  - *.rows.jsonl            (Q200v2 rows) -> throughput-digest.json
  - telemetry.tsv           (2 s sampler) -> telemetry-digest.json

Usage:
  python3 scripts/digest.py <run_dir> [--throughput-only|--telemetry-only]
"""
import argparse
import glob
import json
import os
import statistics
import sys


def throughput(run_dir):
    rows = []
    for path in glob.glob(os.path.join(run_dir, "*.rows.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    usable = []
    for r in rows:
        el = r.get("elapsed_seconds") or r.get("elapsed")
        ct = r.get("completion_tokens")
        if el and ct is not None and not r.get("error"):
            usable.append((ct, el))
    if not usable:
        print("no usable rows", file = sys.stderr)
        return None
    tps = [ct / el for ct, el in usable]
    total_tokens = sum(ct for ct, _ in usable)
    total_time = sum(el for _, el in usable)
    out = {
        "n": len(usable),
        "completion_tokens_total": total_tokens,
        "e2e_tok_s_mean": round(statistics.mean(tps), 2),
        "e2e_tok_s_p50": round(statistics.median(tps), 2),
        "e2e_tok_s_aggregate": round(total_tokens / total_time, 2),
        "method": "completion_tokens / elapsed_seconds per row (PROCEDURES section 4)",
    }
    return out


def telemetry(run_dir, load_only_w = 100.0):
    path = os.path.join(run_dir, "telemetry.tsv")
    if not os.path.exists(path):
        print(f"missing {path}", file = sys.stderr)
        return None
    rows = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
    if not rows:
        return None

    def num(r, k):
        try:
            return float(r[k])
        except Exception:
            return None

    def stat(key):
        vals = [num(r, key) for r in rows]
        vals = [v for v in vals if v is not None]
        return {"mean": round(statistics.mean(vals), 2), "max": round(max(vals), 2)} if vals else None

    loaded = [r for r in rows if (num(r, "power_w") or 0) >= load_only_w]
    span = rows[-1].get("epoch_s") - rows[0].get("epoch_s") if len(rows) > 1 else 0

    def stat_loaded(key):
        vals = [num(r, key) for r in loaded]
        vals = [v for v in vals if v is not None]
        return {"mean": round(statistics.mean(vals), 2), "max": round(max(vals), 2)} if vals else None

    mem_avail = [num(r, "mem_available_kb") for r in rows if num(r, "mem_available_kb") is not None]
    throttle = {}
    for r in rows:
        t = (r.get("throttle") or "?").strip()
        throttle[t] = throttle.get(t, 0) + 1

    out = {
        "samples": len(rows),
        "span_s": int(span) if span else span,
        "load_only": {
            "power_w": stat_loaded("power_w"),
            "temp_c": stat_loaded("temp_c"),
            "util_pct": stat_loaded("util_pct"),
            "gfx_clock_mhz": stat_loaded("gfx_clock_mhz"),
            "vram_mib": stat_loaded("vram_mib"),
        },
        "load_only_threshold_w": load_only_w,
        "min_mem_available_gib": round(min(mem_avail) / 1024 / 1024, 2) if mem_avail else None,
        "throttle_states": throttle,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--throughput-only", action = "store_true")
    ap.add_argument("--telemetry-only", action = "store_true")
    args = ap.parse_args()

    if not args.telemetry_only:
        t = throughput(args.run_dir)
        if t:
            p = os.path.join(args.run_dir, "throughput-digest.json")
            json.dump(t, open(p, "w"), indent = 2)
            print("wrote", p)
    if not args.throughput_only:
        d = telemetry(args.run_dir)
        if d:
            p = os.path.join(args.run_dir, "telemetry-digest.json")
            json.dump(d, open(p, "w"), indent = 2)
            print("wrote", p)


if __name__ == "__main__":
    main()
