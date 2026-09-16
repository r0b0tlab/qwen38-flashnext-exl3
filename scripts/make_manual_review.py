#!/usr/bin/env python3
"""
Prepare the manual-review pair for a Q200v2 run's hard_reasoning rows
(mirrors the 27B campaign's procedure): extracts the rows into
manual-review-input.json and writes a manual-evidence.json skeleton
(ids + content hashes pre-bound) for the reviewer to fill.

Usage:
  python3 scripts/make_manual_review.py <run_dir> [--reviewer hermes-agent]
Outputs (in run_dir):
  manual-review-input.json   — the hard_reasoning rows, full schema
  manual-evidence.json       — skeleton: {schema, dataset_sha256, run_identity_sha256,
                               reviewer, method, rows:[{id, content_sha256, passed, rationale}]}
"""
import argparse
import glob
import json
import os

MANUAL_EVIDENCE_SCHEMA = "r0b0tlab.qwen38.manual_evidence.v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--reviewer", default = "hermes-agent")
    args = ap.parse_args()

    rows = []
    for path in glob.glob(os.path.join(args.run_dir, "*.rows.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    hard = [r for r in rows if r.get("family") == "hard_reasoning"]
    if not hard:
        raise SystemExit("no hard_reasoning rows found")

    dataset_sha = {r.get("dataset_sha256") for r in hard}
    identity_sha = {r.get("identity_sha256") for r in hard}
    dataset_sha256 = dataset_sha.pop() if len(dataset_sha) == 1 else sorted(hard[0].get("dataset_sha256") or [""])[0]
    run_identity_sha256 = identity_sha.pop() if len(identity_sha) == 1 else (hard[0].get("identity_sha256") or "")

    with open(os.path.join(args.run_dir, "manual-review-input.json"), "w") as f:
        json.dump(hard, f, indent = 2)

    evidence = {
        "schema": MANUAL_EVIDENCE_SCHEMA,
        "dataset_sha256": dataset_sha256,
        "run_identity_sha256": run_identity_sha256,
        "reviewer": args.reviewer,
        "method": "Independent item-by-item review of the hard_reasoning rows; "
                  "content bytes hash-bound (content_sha256); no responses regenerated.",
        "rows": [
            {"id": r["id"], "content_sha256": r.get("content_sha256"),
             "passed": None, "rationale": ""}
            for r in hard
        ],
    }
    with open(os.path.join(args.run_dir, "manual-evidence.json"), "w") as f:
        json.dump(evidence, f, indent = 2)

    print(json.dumps({
        "run_dir": os.path.abspath(args.run_dir),
        "hard_reasoning_rows": len(hard),
        "dataset_sha256": dataset_sha256,
        "run_identity_sha256": run_identity_sha256,
        "wrote": ["manual-review-input.json", "manual-evidence.json (skeleton: fill passed/rationale)"],
    }, indent = 2))


if __name__ == "__main__":
    main()
